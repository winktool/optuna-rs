//! Journal 文件存储 — 追加式日志存储
//!
//! 对应 Python `optuna.storages.JournalStorage` + `JournalFileStorage`。
//! 通过追加 JSON 日志行实现持久化，支持断电恢复。
//!
//! # 用法
//! ```ignore
//! use optuna_rs::storage::JournalFileStorage;
//!
//! let storage = JournalFileStorage::new("/path/to/journal.log").unwrap();
//! ```

use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Arc;

use chrono::Utc;
use parking_lot::Mutex;
use serde::{Deserialize, Serialize};

use crate::distributions::{Distribution, ParamValue};
use crate::error::{OptunaError, Result};
use crate::storage::Storage;
use crate::study::{FrozenStudy, StudyDirection};
use crate::trial::{FrozenTrial, TrialState};

// ════════════════════════════════════════════════════════════════════════
// 操作类型枚举 — 对应 Python JournalOperation
// ════════════════════════════════════════════════════════════════════════

/// Journal 日志操作类型
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum JournalOp {
    /// 创建研究
    CreateStudy = 0,
    /// 删除研究
    DeleteStudy = 1,
    /// 设置研究用户属性
    SetStudyUserAttr = 2,
    /// 设置研究系统属性
    SetStudySystemAttr = 3,
    /// 创建试验
    CreateTrial = 4,
    /// 设置试验参数
    SetTrialParam = 5,
    /// 设置试验状态和值
    SetTrialStateValues = 6,
    /// 设置试验中间值
    SetTrialIntermediateValue = 7,
    /// 设置试验用户属性
    SetTrialUserAttr = 8,
    /// 设置试验系统属性
    SetTrialSystemAttr = 9,
}

/// 单条日志记录
///
/// 对齐 Python: 序列化为扁平格式（所有字段在同一层级），
/// 反序列化同时支持扁平格式（Python）和嵌套格式（旧 Rust `data` 包裹）。
#[derive(Debug, Clone)]
pub struct JournalLogEntry {
    /// 操作类型
    pub op_code: i32,
    /// 关联的 study_id
    pub study_id: Option<i64>,
    /// 关联的 trial_id
    pub trial_id: Option<i64>,
    /// 通用数据载荷（JSON）
    pub data: serde_json::Value,
}

// 对齐 Python: 扁平序列化 — 与 Python JournalStorage 日志格式完全兼容
impl Serialize for JournalLogEntry {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error> {
        use serde::ser::SerializeMap;
        let data_fields = self.data.as_object();
        let mut count = 2; // op_code + worker_id
        if self.study_id.is_some() { count += 1; }
        if self.trial_id.is_some() { count += 1; }
        if let Some(obj) = data_fields { count += obj.len(); }

        let mut map = serializer.serialize_map(Some(count))?;
        map.serialize_entry("op_code", &self.op_code)?;
        map.serialize_entry("worker_id", "rust:0")?;
        if let Some(sid) = &self.study_id {
            map.serialize_entry("study_id", sid)?;
        }
        if let Some(tid) = &self.trial_id {
            map.serialize_entry("trial_id", tid)?;
        }
        if let Some(obj) = data_fields {
            for (k, v) in obj {
                map.serialize_entry(k, v)?;
            }
        }
        map.end()
    }
}

// 对齐 Python: 反序列化同时支持扁平（Python）和嵌套（旧 Rust）格式
impl<'de> Deserialize<'de> for JournalLogEntry {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        let map: serde_json::Map<String, serde_json::Value> =
            serde_json::Map::deserialize(deserializer)?;

        let op_code = map.get("op_code")
            .and_then(|v| v.as_i64())
            .unwrap_or(0) as i32;
        let study_id = map.get("study_id").and_then(|v| v.as_i64());
        let trial_id = map.get("trial_id").and_then(|v| v.as_i64());

        // 兼容旧 Rust 嵌套格式: {"data": {...}} 和 Python 扁平格式
        let data = if let Some(data_val) = map.get("data") {
            data_val.clone()
        } else {
            let mut data_map = serde_json::Map::new();
            for (k, v) in &map {
                match k.as_str() {
                    "op_code" | "study_id" | "trial_id" | "worker_id" => continue,
                    _ => { data_map.insert(k.clone(), v.clone()); }
                }
            }
            serde_json::Value::Object(data_map)
        };

        Ok(JournalLogEntry { op_code, study_id, trial_id, data })
    }
}

// ════════════════════════════════════════════════════════════════════════
// 内部状态 — 用于 replay 日志时维护
// ════════════════════════════════════════════════════════════════════════

/// 内部 study 状态
#[derive(Debug, Clone)]
struct StudyState {
    study_id: i64,
    study_name: String,
    directions: Vec<StudyDirection>,
    user_attrs: HashMap<String, serde_json::Value>,
    system_attrs: HashMap<String, serde_json::Value>,
    trial_count: i64,
}

/// 内部 trial 状态
#[derive(Debug, Clone)]
struct TrialInternalState {
    trial_id: i64,
    study_id: i64,
    number: i64,
    state: TrialState,
    values: Option<Vec<f64>>,
    datetime_start: Option<chrono::DateTime<Utc>>,
    datetime_complete: Option<chrono::DateTime<Utc>>,
    params: HashMap<String, ParamValue>,
    distributions: HashMap<String, Distribution>,
    user_attrs: HashMap<String, serde_json::Value>,
    system_attrs: HashMap<String, serde_json::Value>,
    intermediate_values: HashMap<i64, f64>,
}

impl TrialInternalState {
    fn to_frozen(&self) -> FrozenTrial {
        FrozenTrial {
            number: self.number,
            state: self.state,
            values: self.values.clone(),
            datetime_start: self.datetime_start,
            datetime_complete: self.datetime_complete,
            params: self.params.clone(),
            distributions: self.distributions.clone(),
            user_attrs: self.user_attrs.clone(),
            system_attrs: self.system_attrs.clone(),
            intermediate_values: self.intermediate_values.clone(),
            trial_id: self.trial_id,
        }
    }
}

/// 全局内部状态（通过 replay 重建）
#[derive(Debug, Clone)]
struct JournalState {
    /// study_id → StudyState
    studies: HashMap<i64, StudyState>,
    /// trial_id → TrialInternalState
    trials: HashMap<i64, TrialInternalState>,
    /// 下一个 study_id
    next_study_id: i64,
    /// 下一个 trial_id
    next_trial_id: i64,
}

impl JournalState {
    fn new() -> Self {
        Self {
            studies: HashMap::new(),
            trials: HashMap::new(),
            next_study_id: 0,
            next_trial_id: 0,
        }
    }

    /// 回放一条日志记录
    fn replay(&mut self, entry: &JournalLogEntry) -> Result<()> {
        match entry.op_code {
            0 => self.replay_create_study(entry),
            1 => self.replay_delete_study(entry),
            2 => self.replay_set_study_user_attr(entry),
            3 => self.replay_set_study_system_attr(entry),
            4 => self.replay_create_trial(entry),
            5 => self.replay_set_trial_param(entry),
            6 => self.replay_set_trial_state_values(entry),
            7 => self.replay_set_trial_intermediate_value(entry),
            8 => self.replay_set_trial_user_attr(entry),
            9 => self.replay_set_trial_system_attr(entry),
            _ => Ok(()), // 忽略未知操作
        }
    }

    fn replay_create_study(&mut self, entry: &JournalLogEntry) -> Result<()> {
        let study_id = self.next_study_id;
        self.next_study_id += 1;

        let name = entry.data.get("study_name")
            .and_then(|v| v.as_str())
            .unwrap_or("unknown")
            .to_string();

        let directions: Vec<StudyDirection> = entry.data.get("directions")
            .and_then(|v| v.as_array())
            .map(|arr| arr.iter().map(|v| {
                match v.as_i64().unwrap_or(1) {
                    1 => StudyDirection::Minimize,
                    2 => StudyDirection::Maximize,
                    _ => StudyDirection::NotSet,
                }
            }).collect())
            .unwrap_or_else(|| vec![StudyDirection::Minimize]);

        self.studies.insert(study_id, StudyState {
            study_id,
            study_name: name,
            directions,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            trial_count: 0,
        });

        Ok(())
    }

    fn replay_delete_study(&mut self, entry: &JournalLogEntry) -> Result<()> {
        if let Some(sid) = entry.study_id {
            // 删除关联的 trials
            let trial_ids: Vec<i64> = self.trials.iter()
                .filter(|(_, t)| t.study_id == sid)
                .map(|(&id, _)| id)
                .collect();
            for tid in trial_ids {
                self.trials.remove(&tid);
            }
            self.studies.remove(&sid);
        }
        Ok(())
    }

    fn replay_set_study_user_attr(&mut self, entry: &JournalLogEntry) -> Result<()> {
        if let Some(sid) = entry.study_id {
            if let Some(study) = self.studies.get_mut(&sid) {
                // 对齐 Python: 优先读 "user_attr": {k: v} 格式，回退到旧的 "key"/"value" 格式
                if let Some(attr_map) = entry.data.get("user_attr").and_then(|v| v.as_object()) {
                    for (k, v) in attr_map {
                        study.user_attrs.insert(k.clone(), v.clone());
                    }
                } else if let (Some(key), Some(value)) = (
                    entry.data.get("key").and_then(|v| v.as_str()),
                    entry.data.get("value"),
                ) {
                    study.user_attrs.insert(key.to_string(), value.clone());
                }
            }
        }
        Ok(())
    }

    fn replay_set_study_system_attr(&mut self, entry: &JournalLogEntry) -> Result<()> {
        if let Some(sid) = entry.study_id {
            if let Some(study) = self.studies.get_mut(&sid) {
                // 对齐 Python: 优先读 "system_attr": {k: v} 格式，回退到旧的 "key"/"value" 格式
                if let Some(attr_map) = entry.data.get("system_attr").and_then(|v| v.as_object()) {
                    for (k, v) in attr_map {
                        study.system_attrs.insert(k.clone(), v.clone());
                    }
                } else if let (Some(key), Some(value)) = (
                    entry.data.get("key").and_then(|v| v.as_str()),
                    entry.data.get("value"),
                ) {
                    study.system_attrs.insert(key.to_string(), value.clone());
                }
            }
        }
        Ok(())
    }

    fn replay_create_trial(&mut self, entry: &JournalLogEntry) -> Result<()> {
        let study_id = entry.study_id.unwrap_or(0);

        // 对齐 Python: 如果 study 不存在则静默跳过
        if !self.studies.contains_key(&study_id) {
            self.next_trial_id += 1;
            return Ok(());
        }

        let trial_id = self.next_trial_id;
        self.next_trial_id += 1;

        let number = self.studies.get(&study_id)
            .map(|s| s.trial_count)
            .unwrap_or(0);

        if let Some(study) = self.studies.get_mut(&study_id) {
            study.trial_count += 1;
        }

        let state_int = entry.data.get("state")
            .and_then(|v| v.as_i64())
            .unwrap_or(0); // 默认 RUNNING
        let state = match state_int {
            1 => TrialState::Complete,
            2 => TrialState::Pruned,
            3 => TrialState::Fail,
            4 => TrialState::Waiting,
            _ => TrialState::Running,
        };

        // 对齐 Python: 从日志条目读取 distributions（JSON 字符串格式）
        let mut distributions = HashMap::new();
        if let Some(dist_map) = entry.data.get("distributions").and_then(|v| v.as_object()) {
            for (name, dist_val) in dist_map {
                let dist = if let Some(s) = dist_val.as_str() {
                    crate::distributions::json_to_distribution(s).ok()
                } else {
                    serde_json::from_value::<crate::distributions::Distribution>(dist_val.clone()).ok()
                };
                if let Some(d) = dist {
                    distributions.insert(name.clone(), d);
                }
            }
        }

        // 对齐 Python: 从日志条目读取 params（internal_repr float）
        let mut params = HashMap::new();
        if let Some(param_map) = entry.data.get("params").and_then(|v| v.as_object()) {
            for (name, param_val) in param_map {
                if let Some(internal) = param_val.as_f64() {
                    if let Some(dist) = distributions.get(name) {
                        if let Ok(external) = dist.to_external_repr(internal) {
                            params.insert(name.clone(), external);
                        }
                    }
                }
            }
        }

        // 对齐 Python: 从日志条目读取 user_attrs
        let user_attrs: HashMap<String, serde_json::Value> = entry.data.get("user_attrs")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        // 对齐 Python: 从日志条目读取 system_attrs
        let system_attrs: HashMap<String, serde_json::Value> = entry.data.get("system_attrs")
            .and_then(|v| serde_json::from_value(v.clone()).ok())
            .unwrap_or_default();

        // 对齐 Python: 从日志条目读取 intermediate_values
        let mut intermediate_values: HashMap<i64, f64> = HashMap::new();
        if let Some(iv_map) = entry.data.get("intermediate_values").and_then(|v| v.as_object()) {
            for (step_str, val) in iv_map {
                if let (Ok(step), Some(v)) = (step_str.parse::<i64>(), val.as_f64()) {
                    intermediate_values.insert(step, v);
                }
            }
        }

        // 对齐 Python: 从日志条目读取 values
        let values = if let Some(vals) = entry.data.get("values").and_then(|v| v.as_array()) {
            let vs: Vec<f64> = vals.iter().filter_map(|v| v.as_f64()).collect();
            if vs.is_empty() { None } else { Some(vs) }
        } else if let Some(val) = entry.data.get("value").and_then(|v| v.as_f64()) {
            // Python 单目标用 "value"（单数）
            Some(vec![val])
        } else {
            None
        };

        // 对齐 Python: 从日志条目读取 datetime_start
        let datetime_start = entry.data.get("datetime_start")
            .and_then(|v| v.as_str())
            .and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(s).ok()
                    .or_else(|| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f").ok()
                        .map(|ndt| ndt.and_utc().fixed_offset()))
            })
            .map(|dt| dt.with_timezone(&Utc))
            .or_else(|| if state != TrialState::Waiting { Some(Utc::now()) } else { None });

        // 对齐 Python: 从日志条目读取 datetime_complete
        let datetime_complete = entry.data.get("datetime_complete")
            .and_then(|v| v.as_str())
            .and_then(|s| {
                chrono::DateTime::parse_from_rfc3339(s).ok()
                    .or_else(|| chrono::NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f").ok()
                        .map(|ndt| ndt.and_utc().fixed_offset()))
            })
            .map(|dt| dt.with_timezone(&Utc));

        self.trials.insert(trial_id, TrialInternalState {
            trial_id,
            study_id,
            number,
            state,
            values,
            datetime_start,
            datetime_complete,
            params,
            distributions,
            user_attrs,
            system_attrs,
            intermediate_values,
        });

        Ok(())
    }

    fn replay_set_trial_param(&mut self, entry: &JournalLogEntry) -> Result<()> {
        if let Some(tid) = entry.trial_id {
            if let Some(trial) = self.trials.get_mut(&tid) {
                let name = entry.data.get("param_name").and_then(|v| v.as_str());
                // 对齐 Python: 键名为 "param_value_internal"（BUG FIX: 原来用了 "param_value"）
                let value = entry.data.get("param_value_internal").and_then(|v| v.as_f64());
                // 对齐 Python: distribution 可以是 JSON 字符串（Python兼容）或 JSON 对象（Rust serde）
                let dist_result = entry.data.get("distribution").and_then(|v| {
                    if let Some(s) = v.as_str() {
                        // JSON 字符串格式（Python 兼容）
                        crate::distributions::json_to_distribution(s).ok()
                    } else {
                        // JSON 对象格式（Rust serde 序列化）
                        serde_json::from_value::<crate::distributions::Distribution>(v.clone()).ok()
                    }
                });
                if let (Some(name), Some(value), Some(dist)) = (name, value, dist_result) {
                    if let Ok(param_val) = dist.to_external_repr(value) {
                        trial.params.insert(name.to_string(), param_val);
                        trial.distributions.insert(name.to_string(), dist);
                    }
                }
            }
        }
        Ok(())
    }

    fn replay_set_trial_state_values(&mut self, entry: &JournalLogEntry) -> Result<()> {
        if let Some(tid) = entry.trial_id {
            if let Some(trial) = self.trials.get_mut(&tid) {
                if let Some(state_int) = entry.data.get("state").and_then(|v| v.as_i64()) {
                    let new_state = match state_int {
                        0 => TrialState::Running,
                        1 => TrialState::Complete,
                        2 => TrialState::Pruned,
                        3 => TrialState::Fail,
                        4 => TrialState::Waiting,
                        _ => trial.state,
                    };

                    // 对齐 Python: RUNNING → RUNNING（且当前已经是 RUNNING）时拒绝。
                    // 用于多 worker 竞争同一 trial 的场景。
                    if new_state == TrialState::Running && trial.state == TrialState::Running {
                        return Ok(()); // 静默拒绝，不改变状态
                    }

                    trial.state = new_state;

                    // 对齐 Python: 从日志条目读取时间戳（如果有）
                    if trial.state == TrialState::Running && trial.datetime_start.is_none() {
                        trial.datetime_start = entry.data.get("datetime_start")
                            .and_then(|v| v.as_str())
                            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                            .map(|dt| dt.with_timezone(&Utc))
                            .or_else(|| Some(Utc::now()));
                    }

                    if trial.state.is_finished() && trial.datetime_complete.is_none() {
                        trial.datetime_complete = entry.data.get("datetime_complete")
                            .and_then(|v| v.as_str())
                            .and_then(|s| chrono::DateTime::parse_from_rfc3339(s).ok())
                            .map(|dt| dt.with_timezone(&Utc))
                            .or_else(|| Some(Utc::now()));
                    }
                }

                if let Some(vals) = entry.data.get("values").and_then(|v| v.as_array()) {
                    let values: Vec<f64> = vals.iter()
                        .filter_map(|v| v.as_f64())
                        .collect();
                    if !values.is_empty() {
                        trial.values = Some(values);
                    }
                }
            }
        }
        Ok(())
    }

    fn replay_set_trial_intermediate_value(&mut self, entry: &JournalLogEntry) -> Result<()> {
        if let Some(tid) = entry.trial_id {
            if let Some(trial) = self.trials.get_mut(&tid) {
                // 对齐 Python: 键名为 "intermediate_value"，同时兼容旧 Rust 的 "value"
                let value = entry.data.get("intermediate_value")
                    .or_else(|| entry.data.get("value"))
                    .and_then(|v| v.as_f64());
                if let (Some(step), Some(value)) = (
                    entry.data.get("step").and_then(|v| v.as_i64()),
                    value,
                ) {
                    trial.intermediate_values.insert(step, value);
                }
            }
        }
        Ok(())
    }

    fn replay_set_trial_user_attr(&mut self, entry: &JournalLogEntry) -> Result<()> {
        if let Some(tid) = entry.trial_id {
            if let Some(trial) = self.trials.get_mut(&tid) {
                // 对齐 Python: 优先读 "user_attr": {k: v} 格式，回退到旧的 "key"/"value" 格式
                if let Some(attr_map) = entry.data.get("user_attr").and_then(|v| v.as_object()) {
                    for (k, v) in attr_map {
                        trial.user_attrs.insert(k.clone(), v.clone());
                    }
                } else if let (Some(key), Some(value)) = (
                    entry.data.get("key").and_then(|v| v.as_str()),
                    entry.data.get("value"),
                ) {
                    trial.user_attrs.insert(key.to_string(), value.clone());
                }
            }
        }
        Ok(())
    }

    fn replay_set_trial_system_attr(&mut self, entry: &JournalLogEntry) -> Result<()> {
        if let Some(tid) = entry.trial_id {
            if let Some(trial) = self.trials.get_mut(&tid) {
                // 对齐 Python: 优先读 "system_attr": {k: v} 格式，回退到旧的 "key"/"value" 格式
                if let Some(attr_map) = entry.data.get("system_attr").and_then(|v| v.as_object()) {
                    for (k, v) in attr_map {
                        trial.system_attrs.insert(k.clone(), v.clone());
                    }
                } else if let (Some(key), Some(value)) = (
                    entry.data.get("key").and_then(|v| v.as_str()),
                    entry.data.get("value"),
                ) {
                    trial.system_attrs.insert(key.to_string(), value.clone());
                }
            }
        }
        Ok(())
    }
}

// ════════════════════════════════════════════════════════════════════════
// JournalBackend trait — 后端接口抽象
// ════════════════════════════════════════════════════════════════════════

/// Journal 后端接口。
///
/// 对应 Python `optuna.storages.journal.BaseJournalBackend`。
/// 为 JournalStorage 提供读写日志的抽象层，
/// 允许不同后端实现（文件、Redis 等）。
pub trait JournalBackend: Send + Sync {
    /// 追加一条日志记录到后端。
    fn append_log(&self, entry: &JournalLogEntry) -> Result<()>;

    /// 读取从 `log_number_from` 开始的所有日志记录。
    /// 返回 `(entries, next_log_number)`。
    fn read_logs(&self, log_number_from: usize) -> Result<Vec<JournalLogEntry>>;
}

/// 组合式 Journal 存储。
///
/// 对应 Python `optuna.storages.JournalStorage`。
/// 将日志后端（`JournalBackend`）与内存状态（`JournalState`）解耦，
/// 支持不同后端（文件、Redis 等）的即插即用。
pub struct JournalStorage {
    /// 日志后端
    backend: Box<dyn JournalBackend>,
    /// 内部状态（通过 replay 重建）
    state: Arc<Mutex<JournalState>>,
    /// 已同步到的日志编号
    synced_log_count: Mutex<usize>,
}

impl JournalStorage {
    /// 从后端创建 JournalStorage，自动 replay 恢复状态。
    pub fn new(backend: Box<dyn JournalBackend>) -> Result<Self> {
        let mut state = JournalState::new();
        let logs = backend.read_logs(0)?;
        let count = logs.len();
        for entry in &logs {
            let _ = state.replay(entry);
        }
        Ok(Self {
            backend,
            state: Arc::new(Mutex::new(state)),
            synced_log_count: Mutex::new(count),
        })
    }

    /// 从后端同步新增日志（供外部调用）。
    pub fn sync(&self) -> Result<()> {
        let mut count = self.synced_log_count.lock();
        let logs = self.backend.read_logs(*count)?;
        let new_count = logs.len();
        let mut state = self.state.lock();
        for entry in &logs {
            let _ = state.replay(entry);
        }
        *count += new_count;
        Ok(())
    }

    /// 追加日志并更新状态。
    fn write_log(&self, entry: &JournalLogEntry) -> Result<()> {
        self.backend.append_log(entry)?;
        self.state.lock().replay(entry)?;
        *self.synced_log_count.lock() += 1;
        Ok(())
    }
}

impl Storage for JournalStorage {
    fn create_new_study(
        &self,
        directions: &[StudyDirection],
        study_name: Option<&str>,
    ) -> Result<i64> {
        let state = self.state.lock();
        let name = study_name
            .map(String::from)
            .unwrap_or_else(|| format!("no-name-{}", state.next_study_id));

        // 检查名称唯一性
        if state.studies.values().any(|s| s.study_name == name) {
            return Err(OptunaError::DuplicatedStudyError(name));
        }
        let next_id = state.next_study_id;
        drop(state);

        let dir_ints: Vec<serde_json::Value> = directions
            .iter()
            .map(|d| match d {
                StudyDirection::Minimize => serde_json::json!(1),
                StudyDirection::Maximize => serde_json::json!(2),
                _ => serde_json::json!(0),
            })
            .collect();

        self.write_log(&JournalLogEntry {
            op_code: 0,
            study_id: None,
            trial_id: None,
            data: serde_json::json!({
                "study_name": name,
                "directions": dir_ints,
            }),
        })?;

        Ok(next_id)
    }

    fn delete_study(&self, study_id: i64) -> Result<()> {
        self.write_log(&JournalLogEntry {
            op_code: 1,
            study_id: Some(study_id),
            trial_id: None,
            data: serde_json::Value::Null,
        })
    }

    fn set_study_user_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        // 对齐 Python: 使用 "user_attr": {key: value} 格式
        let mut attr = serde_json::Map::new();
        attr.insert(key.to_string(), value);
        self.write_log(&JournalLogEntry {
            op_code: 2,
            study_id: Some(study_id),
            trial_id: None,
            data: serde_json::json!({"user_attr": attr}),
        })
    }

    fn set_study_system_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        // 对齐 Python: 使用 "system_attr": {key: value} 格式
        let mut attr = serde_json::Map::new();
        attr.insert(key.to_string(), value);
        self.write_log(&JournalLogEntry {
            op_code: 3,
            study_id: Some(study_id),
            trial_id: None,
            data: serde_json::json!({"system_attr": attr}),
        })
    }

    fn get_study_id_from_name(&self, study_name: &str) -> Result<i64> {
        let state = self.state.lock();
        state
            .studies
            .values()
            .find(|s| s.study_name == study_name)
            .map(|s| s.study_id)
            .ok_or_else(|| {
                OptunaError::ValueError(format!("study '{study_name}' not found"))
            })
    }

    fn get_study_name_from_id(&self, study_id: i64) -> Result<String> {
        let state = self.state.lock();
        state
            .studies
            .get(&study_id)
            .map(|s| s.study_name.clone())
            .ok_or_else(|| {
                OptunaError::ValueError(format!("study_id {study_id} not found"))
            })
    }

    fn get_study_directions(&self, study_id: i64) -> Result<Vec<StudyDirection>> {
        let state = self.state.lock();
        state
            .studies
            .get(&study_id)
            .map(|s| s.directions.clone())
            .ok_or_else(|| {
                OptunaError::ValueError(format!("study_id {study_id} not found"))
            })
    }

    fn get_study_user_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        let state = self.state.lock();
        state
            .studies
            .get(&study_id)
            .map(|s| s.user_attrs.clone())
            .ok_or_else(|| {
                OptunaError::ValueError(format!("study_id {study_id} not found"))
            })
    }

    fn get_study_system_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        let state = self.state.lock();
        state
            .studies
            .get(&study_id)
            .map(|s| s.system_attrs.clone())
            .ok_or_else(|| {
                OptunaError::ValueError(format!("study_id {study_id} not found"))
            })
    }

    fn get_all_studies(&self) -> Result<Vec<FrozenStudy>> {
        let state = self.state.lock();
        Ok(state
            .studies
            .values()
            .map(|s| FrozenStudy {
                study_id: s.study_id,
                study_name: s.study_name.clone(),
                directions: s.directions.clone(),
                user_attrs: s.user_attrs.clone(),
                system_attrs: s.system_attrs.clone(),
            })
            .collect())
    }

    fn create_new_trial(
        &self,
        study_id: i64,
        template_trial: Option<&FrozenTrial>,
    ) -> Result<i64> {
        let next_id = self.state.lock().next_trial_id;

        let mut data = serde_json::json!({
            "datetime_start": Utc::now().to_rfc3339(),
        });

        if let Some(tmpl) = template_trial {
            data["state"] = serde_json::json!(tmpl.state as i32);

            // 对齐 Python: values 序列化 — 单目标用 "value"，多目标用 "values"
            if let Some(ref vals) = tmpl.values {
                if vals.len() > 1 {
                    data["value"] = serde_json::Value::Null;
                    data["values"] = serde_json::json!(vals);
                } else {
                    data["value"] = serde_json::json!(vals.first());
                    data["values"] = serde_json::Value::Null;
                }
            }

            // 对齐 Python: distributions 序列化为 {name: json_string}
            let mut dist_map = serde_json::Map::new();
            for (name, dist) in &tmpl.distributions {
                if let Ok(json_str) = crate::distributions::distribution_to_json(dist) {
                    dist_map.insert(name.clone(), serde_json::json!(json_str));
                }
            }
            data["distributions"] = serde_json::Value::Object(dist_map);

            // 对齐 Python: params 序列化为 {name: internal_repr_float}
            let mut param_map = serde_json::Map::new();
            for (name, param_val) in &tmpl.params {
                if let Some(dist) = tmpl.distributions.get(name) {
                    if let Ok(internal) = dist.to_internal_repr(param_val) {
                        param_map.insert(name.clone(), serde_json::json!(internal));
                    }
                }
            }
            data["params"] = serde_json::Value::Object(param_map);

            data["user_attrs"] = serde_json::json!(tmpl.user_attrs);
            data["system_attrs"] = serde_json::json!(tmpl.system_attrs);

            // 对齐 Python: intermediate_values 序列化为 {step_str: value}
            let mut iv_map = serde_json::Map::new();
            for (step, val) in &tmpl.intermediate_values {
                iv_map.insert(step.to_string(), serde_json::json!(val));
            }
            data["intermediate_values"] = serde_json::Value::Object(iv_map);

            // 对齐 Python: datetime_start / datetime_complete
            if let Some(ref dt) = tmpl.datetime_start {
                data["datetime_start"] = serde_json::json!(dt.to_rfc3339());
            } else {
                data["datetime_start"] = serde_json::Value::Null;
            }
            if let Some(ref dt) = tmpl.datetime_complete {
                data["datetime_complete"] = serde_json::json!(dt.to_rfc3339());
            }
        }

        self.write_log(&JournalLogEntry {
            op_code: 4,
            study_id: Some(study_id),
            trial_id: None,
            data,
        })?;

        Ok(next_id)
    }

    fn set_trial_param(
        &self,
        trial_id: i64,
        param_name: &str,
        param_value_internal: f64,
        distribution: &Distribution,
    ) -> Result<()> {
        // 对齐 Python: distribution 序列化为 JSON 字符串（distribution_to_json）
        let dist_json = crate::distributions::distribution_to_json(distribution)?;
        self.write_log(&JournalLogEntry {
            op_code: 5,
            study_id: None,
            trial_id: Some(trial_id),
            data: serde_json::json!({
                "param_name": param_name,
                "param_value_internal": param_value_internal,
                "distribution": dist_json,
            }),
        })
    }

    fn set_trial_state_values(
        &self,
        trial_id: i64,
        state: TrialState,
        values: Option<&[f64]>,
    ) -> Result<bool> {
        // 对齐 Python: 先同步后端，检查 trial 是否存在且未完成
        {
            let st = self.state.lock();
            if let Some(trial) = st.trials.get(&trial_id) {
                if trial.state.is_finished() {
                    return Err(OptunaError::UpdateFinishedTrialError(format!(
                        "Cannot update finished trial {trial_id}"
                    )));
                }
                // 对齐 Python/InMemoryStorage: RUNNING 且非 WAITING 的 trial 再次设为 RUNNING 返回 false
                if state == TrialState::Running && trial.state != TrialState::Waiting {
                    return Ok(false);
                }
            } else {
                return Err(OptunaError::ValueError(format!(
                    "trial_id {trial_id} not found"
                )));
            }
        }
        // 对齐 Python: 写入时间戳到日志条目，以便 replay 时恢复
        let now = Utc::now().to_rfc3339();
        let mut data = serde_json::json!({
            "state": state as i32,
            "values": values,
        });
        if state == TrialState::Running {
            data["datetime_start"] = serde_json::json!(now);
        }
        if state.is_finished() {
            data["datetime_complete"] = serde_json::json!(now);
        }
        self.write_log(&JournalLogEntry {
            op_code: 6,
            study_id: None,
            trial_id: Some(trial_id),
            data,
        })?;
        Ok(true)
    }

    fn set_trial_intermediate_value(
        &self,
        trial_id: i64,
        step: i64,
        intermediate_value: f64,
    ) -> Result<()> {
        // 对齐 Python: 使用 "intermediate_value" 键名
        self.write_log(&JournalLogEntry {
            op_code: 7,
            study_id: None,
            trial_id: Some(trial_id),
            data: serde_json::json!({
                "step": step,
                "intermediate_value": intermediate_value,
            }),
        })
    }

    fn set_trial_user_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        // 对齐 Python: 使用 "user_attr": {key: value} 格式
        let mut attr = serde_json::Map::new();
        attr.insert(key.to_string(), value);
        self.write_log(&JournalLogEntry {
            op_code: 8,
            study_id: None,
            trial_id: Some(trial_id),
            data: serde_json::json!({"user_attr": attr}),
        })
    }

    fn set_trial_system_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        // 对齐 Python: 使用 "system_attr": {key: value} 格式
        let mut attr = serde_json::Map::new();
        attr.insert(key.to_string(), value);
        self.write_log(&JournalLogEntry {
            op_code: 9,
            study_id: None,
            trial_id: Some(trial_id),
            data: serde_json::json!({"system_attr": attr}),
        })
    }

    fn get_trial(&self, trial_id: i64) -> Result<FrozenTrial> {
        let state = self.state.lock();
        state
            .trials
            .get(&trial_id)
            .map(|t| t.to_frozen())
            .ok_or_else(|| {
                OptunaError::ValueError(format!("trial_id {trial_id} not found"))
            })
    }

    fn get_all_trials(
        &self,
        study_id: i64,
        states: Option<&[TrialState]>,
    ) -> Result<Vec<FrozenTrial>> {
        let state = self.state.lock();
        let mut trials: Vec<FrozenTrial> = state
            .trials
            .values()
            .filter(|t| {
                t.study_id == study_id
                    && states
                        .map(|ss| ss.contains(&t.state))
                        .unwrap_or(true)
            })
            .map(|t| t.to_frozen())
            .collect();
        // 对齐 Python: 按 trial number 排序保证顺序一致性
        trials.sort_by_key(|t| t.number);
        Ok(trials)
    }
}

// ════════════════════════════════════════════════════════════════════════
// JournalFileBackend — 文件后端实现
// ════════════════════════════════════════════════════════════════════════

/// Journal 文件后端。
///
/// 对应 Python `optuna.storages.journal.JournalFileBackend`。
/// 实现 `JournalBackend` trait，支持文件追加写和行读取。
pub struct JournalFileBackend {
    path: PathBuf,
    file: Arc<Mutex<File>>,
}

impl JournalFileBackend {
    /// 创建或打开文件后端。
    pub fn new(path: &str) -> Result<Self> {
        let path = PathBuf::from(path);
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)
            .map_err(|e| OptunaError::StorageInternalError(format!("打开日志文件失败: {e}")))?;
        Ok(Self {
            path,
            file: Arc::new(Mutex::new(file)),
        })
    }
}

impl JournalBackend for JournalFileBackend {
    fn append_log(&self, entry: &JournalLogEntry) -> Result<()> {
        let json = serde_json::to_string(entry)
            .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
        let mut file = self.file.lock();
        writeln!(file, "{json}")
            .map_err(|e| OptunaError::StorageInternalError(format!("写入日志失败: {e}")))?;
        file.flush()
            .map_err(|e| OptunaError::StorageInternalError(format!("刷新日志失败: {e}")))?;
        Ok(())
    }

    fn read_logs(&self, log_number_from: usize) -> Result<Vec<JournalLogEntry>> {
        let file = File::open(&self.path)
            .map_err(|e| OptunaError::StorageInternalError(format!("读取日志文件失败: {e}")))?;
        let reader = BufReader::new(file);
        let mut entries = Vec::new();
        for (i, line) in reader.lines().enumerate() {
            if i < log_number_from {
                continue;
            }
            if let Ok(line) = line {
                let line = line.trim().to_string();
                if line.is_empty() {
                    continue;
                }
                if let Ok(entry) = serde_json::from_str::<JournalLogEntry>(&line) {
                    entries.push(entry);
                }
            }
        }
        Ok(entries)
    }
}

// ════════════════════════════════════════════════════════════════════════
// JournalFileStorage — 文件后端（原始扁平实现，保留向后兼容）
// ════════════════════════════════════════════════════════════════════════

/// Journal 文件存储 — 追加式 JSON 日志。
///
/// 对应 Python `optuna.storages.JournalFileStorage`。
/// 每个操作追加一行 JSON 到文件，启动时 replay 恢复状态。
pub struct JournalFileStorage {
    /// 日志文件路径
    path: PathBuf,
    /// 内部状态（通过 replay 重建）
    state: Arc<Mutex<JournalState>>,
    /// 日志文件句柄（追加模式）
    file: Arc<Mutex<File>>,
}

impl JournalFileStorage {
    /// 创建或打开 Journal 文件存储。
    ///
    /// 如果文件已存在，会从头 replay 恢复状态。
    pub fn new(path: &str) -> Result<Self> {
        let path = PathBuf::from(path);
        let mut state = JournalState::new();

        // 如果文件已存在，replay 恢复状态
        if path.exists() {
            let file = File::open(&path)
                .map_err(|e| OptunaError::StorageInternalError(
                    format!("打开日志文件失败: {e}")))?;
            let reader = BufReader::new(file);
            for line in reader.lines() {
                if let Ok(line) = line {
                    let line = line.trim();
                    if line.is_empty() { continue; }
                    if let Ok(entry) = serde_json::from_str::<JournalLogEntry>(line) {
                        let _ = state.replay(&entry);
                    }
                }
            }
        }

        // 打开追加模式文件
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .map_err(|e| OptunaError::StorageInternalError(
                format!("打开日志文件失败: {e}")))?;

        Ok(Self {
            path,
            state: Arc::new(Mutex::new(state)),
            file: Arc::new(Mutex::new(file)),
        })
    }

    /// 追加一条日志并更新内部状态
    fn append_log(&self, entry: &JournalLogEntry) -> Result<()> {
        // 写入文件
        let json = serde_json::to_string(entry)
            .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
        let mut file = self.file.lock();
        writeln!(file, "{json}")
            .map_err(|e| OptunaError::StorageInternalError(
                format!("写入日志失败: {e}")))?;
        file.flush()
            .map_err(|e| OptunaError::StorageInternalError(
                format!("刷新日志失败: {e}")))?;

        // 更新状态
        self.state.lock().replay(entry)
    }

    /// 获取日志文件路径
    pub fn log_path(&self) -> &std::path::Path {
        &self.path
    }
}

impl Storage for JournalFileStorage {
    fn create_new_study(
        &self,
        directions: &[StudyDirection],
        study_name: Option<&str>,
    ) -> Result<i64> {
        let name = study_name
            .map(String::from)
            .unwrap_or_else(|| format!("no-name-{}", uuid::Uuid::new_v4()));

        // 检查重名
        {
            let state = self.state.lock();
            for s in state.studies.values() {
                if s.study_name == name {
                    return Err(OptunaError::DuplicatedStudyError(
                        format!("study '{}' already exists", name)));
                }
            }
        }

        let dirs = if directions.is_empty() {
            vec![StudyDirection::Minimize]
        } else {
            directions.to_vec()
        };

        let dir_ints: Vec<i64> = dirs.iter().map(|d| match d {
            StudyDirection::NotSet => 0,
            StudyDirection::Minimize => 1,
            StudyDirection::Maximize => 2,
        }).collect();

        let entry = JournalLogEntry {
            op_code: 0,
            study_id: None,
            trial_id: None,
            data: serde_json::json!({
                "study_name": name,
                "directions": dir_ints,
            }),
        };
        self.append_log(&entry)?;

        // 返回新创建的 study_id
        let state = self.state.lock();
        Ok(state.next_study_id - 1)
    }

    fn delete_study(&self, study_id: i64) -> Result<()> {
        let entry = JournalLogEntry {
            op_code: 1,
            study_id: Some(study_id),
            trial_id: None,
            data: serde_json::Value::Null,
        };
        self.append_log(&entry)
    }

    fn set_study_user_attr(
        &self, study_id: i64, key: &str, value: serde_json::Value,
    ) -> Result<()> {
        // 对齐 Python: 使用 "user_attr": {key: value} 格式
        let mut attr = serde_json::Map::new();
        attr.insert(key.to_string(), value);
        let entry = JournalLogEntry {
            op_code: 2,
            study_id: Some(study_id),
            trial_id: None,
            data: serde_json::json!({"user_attr": attr}),
        };
        self.append_log(&entry)
    }

    fn set_study_system_attr(
        &self, study_id: i64, key: &str, value: serde_json::Value,
    ) -> Result<()> {
        // 对齐 Python: 使用 "system_attr": {key: value} 格式
        let mut attr = serde_json::Map::new();
        attr.insert(key.to_string(), value);
        let entry = JournalLogEntry {
            op_code: 3,
            study_id: Some(study_id),
            trial_id: None,
            data: serde_json::json!({"system_attr": attr}),
        };
        self.append_log(&entry)
    }

    fn get_study_id_from_name(&self, study_name: &str) -> Result<i64> {
        let state = self.state.lock();
        state.studies.values()
            .find(|s| s.study_name == study_name)
            .map(|s| s.study_id)
            .ok_or_else(|| OptunaError::ValueError(
                format!("study '{}' not found", study_name)))
    }

    fn get_study_name_from_id(&self, study_id: i64) -> Result<String> {
        let state = self.state.lock();
        state.studies.get(&study_id)
            .map(|s| s.study_name.clone())
            .ok_or_else(|| OptunaError::ValueError(
                format!("study_id {} not found", study_id)))
    }

    fn get_study_directions(&self, study_id: i64) -> Result<Vec<StudyDirection>> {
        let state = self.state.lock();
        state.studies.get(&study_id)
            .map(|s| s.directions.clone())
            .ok_or_else(|| OptunaError::ValueError(
                format!("study_id {} not found", study_id)))
    }

    fn get_study_user_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        let state = self.state.lock();
        state.studies.get(&study_id)
            .map(|s| s.user_attrs.clone())
            .ok_or_else(|| OptunaError::ValueError(
                format!("study_id {} not found", study_id)))
    }

    fn get_study_system_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        let state = self.state.lock();
        state.studies.get(&study_id)
            .map(|s| s.system_attrs.clone())
            .ok_or_else(|| OptunaError::ValueError(
                format!("study_id {} not found", study_id)))
    }

    fn get_all_studies(&self) -> Result<Vec<FrozenStudy>> {
        let state = self.state.lock();
        Ok(state.studies.values().map(|s| FrozenStudy {
            study_id: s.study_id,
            study_name: s.study_name.clone(),
            directions: s.directions.clone(),
            user_attrs: s.user_attrs.clone(),
            system_attrs: s.system_attrs.clone(),
        }).collect())
    }

    fn create_new_trial(
        &self, study_id: i64, template_trial: Option<&FrozenTrial>,
    ) -> Result<i64> {
        // 对齐 Python: 将所有模板数据写入单条 CREATE_TRIAL 日志条目
        let mut data = serde_json::json!({
            "datetime_start": Utc::now().to_rfc3339(),
        });

        if let Some(tmpl) = template_trial {
            data["state"] = serde_json::json!(tmpl.state as i32);

            // 对齐 Python: values 序列化
            if let Some(ref vals) = tmpl.values {
                if vals.len() > 1 {
                    data["value"] = serde_json::Value::Null;
                    data["values"] = serde_json::json!(vals);
                } else {
                    data["value"] = serde_json::json!(vals.first());
                    data["values"] = serde_json::Value::Null;
                }
            }

            // 对齐 Python: distributions 序列化为 {name: json_string}
            let mut dist_map = serde_json::Map::new();
            for (name, dist) in &tmpl.distributions {
                if let Ok(json_str) = crate::distributions::distribution_to_json(dist) {
                    dist_map.insert(name.clone(), serde_json::json!(json_str));
                }
            }
            data["distributions"] = serde_json::Value::Object(dist_map);

            // 对齐 Python: params 序列化为 {name: internal_repr_float}
            let mut param_map = serde_json::Map::new();
            for (name, param_val) in &tmpl.params {
                if let Some(dist) = tmpl.distributions.get(name) {
                    if let Ok(internal) = dist.to_internal_repr(param_val) {
                        param_map.insert(name.clone(), serde_json::json!(internal));
                    }
                }
            }
            data["params"] = serde_json::Value::Object(param_map);

            data["user_attrs"] = serde_json::json!(tmpl.user_attrs);
            data["system_attrs"] = serde_json::json!(tmpl.system_attrs);

            // 对齐 Python: intermediate_values 序列化为 {step_str: value}
            let mut iv_map = serde_json::Map::new();
            for (step, val) in &tmpl.intermediate_values {
                iv_map.insert(step.to_string(), serde_json::json!(val));
            }
            data["intermediate_values"] = serde_json::Value::Object(iv_map);

            if let Some(ref dt) = tmpl.datetime_start {
                data["datetime_start"] = serde_json::json!(dt.to_rfc3339());
            } else {
                data["datetime_start"] = serde_json::Value::Null;
            }
            if let Some(ref dt) = tmpl.datetime_complete {
                data["datetime_complete"] = serde_json::json!(dt.to_rfc3339());
            }
        }

        let entry = JournalLogEntry {
            op_code: 4,
            study_id: Some(study_id),
            trial_id: None,
            data,
        };
        self.append_log(&entry)?;

        let trial_id = self.state.lock().next_trial_id - 1;
        Ok(trial_id)
    }

    fn set_trial_param(
        &self, trial_id: i64, param_name: &str,
        param_value_internal: f64, distribution: &Distribution,
    ) -> Result<()> {
        let dist_json = crate::distributions::distribution_to_json(distribution)?;

        // 查找 study_id
        let study_id = {
            let state = self.state.lock();
            state.trials.get(&trial_id)
                .map(|t| t.study_id)
                .ok_or_else(|| OptunaError::ValueError(
                    format!("trial {} not found", trial_id)))?
        };

        let entry = JournalLogEntry {
            op_code: 5,
            study_id: Some(study_id),
            trial_id: Some(trial_id),
            data: serde_json::json!({
                "param_name": param_name,
                "param_value_internal": param_value_internal,
                "distribution": dist_json,
            }),
        };
        self.append_log(&entry)
    }

    fn set_trial_state_values(
        &self, trial_id: i64, state: TrialState, values: Option<&[f64]>,
    ) -> Result<bool> {
        // 检查当前状态
        {
            let st = self.state.lock();
            if let Some(trial) = st.trials.get(&trial_id) {
                if trial.state.is_finished() {
                    return Err(OptunaError::UpdateFinishedTrialError(
                        format!("trial {} already finished", trial_id)));
                }
                if state == TrialState::Running && trial.state == TrialState::Running {
                    return Ok(false);
                }
            }
        }

        let study_id = {
            let st = self.state.lock();
            st.trials.get(&trial_id)
                .map(|t| t.study_id)
                .ok_or_else(|| OptunaError::ValueError(
                    format!("trial {} not found", trial_id)))?
        };

        // 对齐 Python: 写入时间戳到日志条目
        let now = Utc::now().to_rfc3339();
        let mut data = serde_json::json!({
            "state": state as i64,
            "values": values.unwrap_or(&[]),
        });
        if state == TrialState::Running {
            data["datetime_start"] = serde_json::json!(now);
        }
        if state.is_finished() {
            data["datetime_complete"] = serde_json::json!(now);
        }

        let entry = JournalLogEntry {
            op_code: 6,
            study_id: Some(study_id),
            trial_id: Some(trial_id),
            data,
        };
        self.append_log(&entry)?;
        Ok(true)
    }

    fn set_trial_intermediate_value(
        &self, trial_id: i64, step: i64, intermediate_value: f64,
    ) -> Result<()> {
        let study_id = {
            let st = self.state.lock();
            st.trials.get(&trial_id)
                .map(|t| t.study_id)
                .ok_or_else(|| OptunaError::ValueError(
                    format!("trial {} not found", trial_id)))?
        };

        // 对齐 Python: 使用 "intermediate_value" 键名
        let entry = JournalLogEntry {
            op_code: 7,
            study_id: Some(study_id),
            trial_id: Some(trial_id),
            data: serde_json::json!({"step": step, "intermediate_value": intermediate_value}),
        };
        self.append_log(&entry)
    }

    fn set_trial_user_attr(
        &self, trial_id: i64, key: &str, value: serde_json::Value,
    ) -> Result<()> {
        let study_id = {
            let st = self.state.lock();
            st.trials.get(&trial_id)
                .map(|t| t.study_id)
                .ok_or_else(|| OptunaError::ValueError(
                    format!("trial {} not found", trial_id)))?
        };

        // 对齐 Python: 使用 "user_attr": {key: value} 格式
        let mut attr = serde_json::Map::new();
        attr.insert(key.to_string(), value);
        let entry = JournalLogEntry {
            op_code: 8,
            study_id: Some(study_id),
            trial_id: Some(trial_id),
            data: serde_json::json!({"user_attr": attr}),
        };
        self.append_log(&entry)
    }

    fn set_trial_system_attr(
        &self, trial_id: i64, key: &str, value: serde_json::Value,
    ) -> Result<()> {
        let study_id = {
            let st = self.state.lock();
            st.trials.get(&trial_id)
                .map(|t| t.study_id)
                .ok_or_else(|| OptunaError::ValueError(
                    format!("trial {} not found", trial_id)))?
        };

        // 对齐 Python: 使用 "system_attr": {key: value} 格式
        let mut attr = serde_json::Map::new();
        attr.insert(key.to_string(), value);
        let entry = JournalLogEntry {
            op_code: 9,
            study_id: Some(study_id),
            trial_id: Some(trial_id),
            data: serde_json::json!({"system_attr": attr}),
        };
        self.append_log(&entry)
    }

    fn get_trial(&self, trial_id: i64) -> Result<FrozenTrial> {
        let state = self.state.lock();
        state.trials.get(&trial_id)
            .map(|t| t.to_frozen())
            .ok_or_else(|| OptunaError::ValueError(
                format!("trial {} not found", trial_id)))
    }

    fn get_all_trials(
        &self, study_id: i64, states: Option<&[TrialState]>,
    ) -> Result<Vec<FrozenTrial>> {
        let state = self.state.lock();
        let mut trials: Vec<FrozenTrial> = state.trials.values()
            .filter(|t| t.study_id == study_id)
            .filter(|t| states.map_or(true, |ss| ss.contains(&t.state)))
            .map(|t| t.to_frozen())
            .collect();
        trials.sort_by_key(|t| t.number);
        Ok(trials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    fn temp_path() -> String {
        let id = uuid::Uuid::new_v4();
        format!("/tmp/optuna_journal_test_{id}.log")
    }

    #[test]
    fn test_journal_create_study() {
        let path = temp_path();
        let storage = JournalFileStorage::new(&path).unwrap();
        let id = storage.create_new_study(
            &[StudyDirection::Minimize], Some("test"),
        ).unwrap();
        assert!(id >= 0); // IDs start from 0（对齐 Python）
        let name = storage.get_study_name_from_id(id).unwrap();
        assert_eq!(name, "test");
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_journal_duplicate_study() {
        let path = temp_path();
        let storage = JournalFileStorage::new(&path).unwrap();
        storage.create_new_study(&[StudyDirection::Minimize], Some("dup")).unwrap();
        assert!(storage.create_new_study(&[StudyDirection::Minimize], Some("dup")).is_err());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_journal_trial_lifecycle() {
        let path = temp_path();
        let storage = JournalFileStorage::new(&path).unwrap();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], Some("lc")).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        let dist = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution {
                low: 0.0, high: 1.0, log: false, step: None,
            },
        );
        storage.set_trial_param(tid, "x", 0.5, &dist).unwrap();
        storage.set_trial_intermediate_value(tid, 0, 0.1).unwrap();
        storage.set_trial_state_values(tid, TrialState::Complete, Some(&[0.5])).unwrap();

        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.state, TrialState::Complete);
        assert_eq!(trial.values.unwrap()[0], 0.5);
        assert!(trial.params.contains_key("x"));
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_journal_replay() {
        let path = temp_path();

        // 第一次会话：创建 study 和 trial
        {
            let storage = JournalFileStorage::new(&path).unwrap();
            let sid = storage.create_new_study(
                &[StudyDirection::Minimize], Some("replay_test"),
            ).unwrap();
            let tid = storage.create_new_trial(sid, None).unwrap();
            storage.set_trial_state_values(tid, TrialState::Complete, Some(&[42.0])).unwrap();
        }

        // 第二次会话：从文件 replay 恢复
        {
            let storage = JournalFileStorage::new(&path).unwrap();
            let sid = storage.get_study_id_from_name("replay_test").unwrap();
            let trials = storage.get_all_trials(sid, None).unwrap();
            assert_eq!(trials.len(), 1);
            assert_eq!(trials[0].state, TrialState::Complete);
            assert_eq!(trials[0].values.as_ref().unwrap()[0], 42.0);
        }

        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_journal_study_attrs() {
        let path = temp_path();
        let storage = JournalFileStorage::new(&path).unwrap();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], Some("attr")).unwrap();
        storage.set_study_user_attr(sid, "k1", serde_json::json!("v1")).unwrap();
        let attrs = storage.get_study_user_attrs(sid).unwrap();
        assert_eq!(attrs["k1"], serde_json::json!("v1"));
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_journal_delete_study() {
        let path = temp_path();
        let storage = JournalFileStorage::new(&path).unwrap();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], Some("del")).unwrap();
        storage.create_new_trial(sid, None).unwrap();
        storage.delete_study(sid).unwrap();
        let studies = storage.get_all_studies().unwrap();
        assert!(studies.is_empty());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_journal_state_filter() {
        let path = temp_path();
        let storage = JournalFileStorage::new(&path).unwrap();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], Some("filt")).unwrap();
        let t1 = storage.create_new_trial(sid, None).unwrap();
        let t2 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(t1, TrialState::Complete, Some(&[1.0])).unwrap();
        storage.set_trial_state_values(t2, TrialState::Fail, None).unwrap();

        let complete = storage.get_all_trials(sid, Some(&[TrialState::Complete])).unwrap();
        assert_eq!(complete.len(), 1);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_journal_storage_running_to_running_returns_false() {
        let path = temp_path();
        let backend = JournalFileBackend::new(&path).unwrap();
        let storage = JournalStorage::new(Box::new(backend)).unwrap();

        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], Some("jr_running"))
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        let updated = storage
            .set_trial_state_values(tid, TrialState::Running, None)
            .unwrap();
        assert!(!updated);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_journal_storage_finished_trial_update_error() {
        let path = temp_path();
        let backend = JournalFileBackend::new(&path).unwrap();
        let storage = JournalStorage::new(Box::new(backend)).unwrap();

        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], Some("jr_finish"))
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(tid, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        let err = storage
            .set_trial_state_values(tid, TrialState::Fail, None)
            .unwrap_err();
        match err {
            OptunaError::UpdateFinishedTrialError(_) => {}
            e => panic!("expected UpdateFinishedTrialError, got {e:?}"),
        }
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_journal_file_backend_read_logs_skips_invalid_lines() {
        let path = temp_path();
        {
            let mut f = File::create(&path).unwrap();
            writeln!(f, "{{\"op_code\":0,\"study_id\":null,\"trial_id\":null,\"data\":{{\"study_name\":\"x\",\"directions\":[1]}}}}")
                .unwrap();
            writeln!(f, "not-json-line").unwrap();
            writeln!(f, "").unwrap();
        }

        let backend = JournalFileBackend::new(&path).unwrap();
        let logs = backend.read_logs(0).unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].op_code, 0);
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_journal_storage_ignores_unknown_op_code_on_replay() {
        let path = temp_path();
        {
            let mut f = File::create(&path).unwrap();
            writeln!(
                f,
                "{{\"op_code\":999,\"study_id\":null,\"trial_id\":null,\"data\":{{}}}}"
            )
            .unwrap();
        }

        // unknown op_code 应被忽略，初始化不应失败
        let storage = JournalFileStorage::new(&path).unwrap();
        let studies = storage.get_all_studies().unwrap();
        assert!(studies.is_empty());
        fs::remove_file(&path).ok();
    }

    #[test]
    fn test_journal_storage_set_trial_state_values_missing_trial_error() {
        let path = temp_path();
        let backend = JournalFileBackend::new(&path).unwrap();
        let storage = JournalStorage::new(Box::new(backend)).unwrap();

        let err = storage
            .set_trial_state_values(123456, TrialState::Complete, Some(&[1.0]))
            .unwrap_err();
        match err {
            OptunaError::ValueError(msg) => {
                assert!(msg.contains("trial_id 123456 not found") || msg.contains("trial 123456 not found"));
            }
            e => panic!("expected ValueError, got {e:?}"),
        }
        fs::remove_file(&path).ok();
    }

    // ── 对齐 Python: replay_create_trial 模板数据恢复测试 ────────────

    #[test]
    fn test_journal_replay_create_trial_with_template() {
        use crate::distributions::{FloatDistribution, Distribution, ParamValue};

        let path = temp_path();
        // 第一次会话：用模板创建 trial
        {
            let storage = JournalFileStorage::new(&path).unwrap();
            let sid = storage.create_new_study(&[StudyDirection::Minimize], Some("tmpl")).unwrap();

            let float_dist = Distribution::FloatDistribution(FloatDistribution {
                low: 0.0, high: 1.0, log: false, step: None,
            });
            let mut params = HashMap::new();
            params.insert("x".to_string(), ParamValue::Float(0.5));
            let mut distributions = HashMap::new();
            distributions.insert("x".to_string(), float_dist);
            let mut user_attrs = HashMap::new();
            user_attrs.insert("ua_key".to_string(), serde_json::json!("ua_val"));
            let mut system_attrs = HashMap::new();
            system_attrs.insert("sa_key".to_string(), serde_json::json!(42));
            let mut intermediate_values = HashMap::new();
            intermediate_values.insert(0, 0.1);
            intermediate_values.insert(1, 0.2);

            let template = FrozenTrial {
                trial_id: -1,
                number: -1,
                state: TrialState::Complete,
                values: Some(vec![0.75]),
                datetime_start: None,
                datetime_complete: None,
                params,
                distributions,
                user_attrs,
                system_attrs,
                intermediate_values,
            };
            let tid = storage.create_new_trial(sid, Some(&template)).unwrap();
            assert_eq!(tid, 0);
        }

        // 第二次会话：从文件 replay 恢复，验证模板数据完整恢复
        {
            let storage = JournalFileStorage::new(&path).unwrap();
            let sid = storage.get_study_id_from_name("tmpl").unwrap();
            let trials = storage.get_all_trials(sid, None).unwrap();
            assert_eq!(trials.len(), 1);
            let t = &trials[0];
            assert_eq!(t.state, TrialState::Complete);
            assert_eq!(t.values.as_ref().unwrap(), &[0.75]);
            assert!(t.params.contains_key("x"));
            assert!(t.distributions.contains_key("x"));
            assert_eq!(t.user_attrs["ua_key"], serde_json::json!("ua_val"));
            assert_eq!(t.system_attrs["sa_key"], serde_json::json!(42));
            assert_eq!(t.intermediate_values[&0], 0.1);
            assert_eq!(t.intermediate_values[&1], 0.2);
        }
        fs::remove_file(&path).ok();
    }

    // ── 对齐 Python: user_attr / system_attr 格式正确 ───────────────

    #[test]
    fn test_journal_attr_format_python_compatible() {
        let path = temp_path();
        // 写入 attrs
        {
            let storage = JournalFileStorage::new(&path).unwrap();
            let sid = storage.create_new_study(&[StudyDirection::Minimize], Some("attr_fmt")).unwrap();
            storage.set_study_user_attr(sid, "sk1", serde_json::json!("sv1")).unwrap();
            storage.set_study_system_attr(sid, "ssk1", serde_json::json!(123)).unwrap();
            let tid = storage.create_new_trial(sid, None).unwrap();
            storage.set_trial_user_attr(tid, "tk1", serde_json::json!("tv1")).unwrap();
            storage.set_trial_system_attr(tid, "tsk1", serde_json::json!(456)).unwrap();
        }

        // 重新 replay 恢复
        {
            let storage = JournalFileStorage::new(&path).unwrap();
            let sid = storage.get_study_id_from_name("attr_fmt").unwrap();

            let study_ua = storage.get_study_user_attrs(sid).unwrap();
            assert_eq!(study_ua["sk1"], serde_json::json!("sv1"));

            let study_sa = storage.get_study_system_attrs(sid).unwrap();
            assert_eq!(study_sa["ssk1"], serde_json::json!(123));

            let trials = storage.get_all_trials(sid, None).unwrap();
            assert_eq!(trials[0].user_attrs["tk1"], serde_json::json!("tv1"));
            assert_eq!(trials[0].system_attrs["tsk1"], serde_json::json!(456));
        }

        // 验证日志文件中的格式符合 Python 规范（user_attr/{k:v} 而不是 key/value）
        {
            let content = fs::read_to_string(&path).unwrap();
            // study user attr 应该用 "user_attr" 格式
            assert!(content.contains("\"user_attr\""), "missing user_attr key format");
            // study system attr 应该用 "system_attr" 格式
            assert!(content.contains("\"system_attr\""), "missing system_attr key format");
            // 不应该出现旧的 "key"/"value" 格式
            // 注意: 只检查 op_code 2,3,8,9 的条目
        }
        fs::remove_file(&path).ok();
    }

    // ── 对齐 Python: 时间戳保留测试 ───────────────────────────────

    #[test]
    fn test_journal_replay_preserves_timestamps() {
        let path = temp_path();
        let now_before = Utc::now();

        {
            let storage = JournalFileStorage::new(&path).unwrap();
            let sid = storage.create_new_study(&[StudyDirection::Minimize], Some("ts")).unwrap();
            let tid = storage.create_new_trial(sid, None).unwrap();
            storage.set_trial_state_values(tid, TrialState::Complete, Some(&[1.0])).unwrap();
        }

        let now_after = Utc::now();

        // replay 后验证时间戳存在且在合理范围内
        {
            let storage = JournalFileStorage::new(&path).unwrap();
            let sid = storage.get_study_id_from_name("ts").unwrap();
            let trials = storage.get_all_trials(sid, None).unwrap();
            let t = &trials[0];
            assert!(t.datetime_start.is_some(), "datetime_start should be preserved");
            assert!(t.datetime_complete.is_some(), "datetime_complete should be preserved");
            let start = t.datetime_start.unwrap();
            let complete = t.datetime_complete.unwrap();
            assert!(start >= now_before && start <= now_after, "datetime_start out of range");
            assert!(complete >= now_before && complete <= now_after, "datetime_complete out of range");
        }
        fs::remove_file(&path).ok();
    }

    // ── 对齐 Python: replay 重复 RUNNING 拒绝测试 ──────────────────

    #[test]
    fn test_journal_replay_duplicate_running_rejection() {
        let path = temp_path();

        // 手动构造日志文件：create study → create trial → Running → Running (重复)
        {
            let mut f = File::create(&path).unwrap();
            // create study
            writeln!(f, "{{\"op_code\":0,\"study_id\":null,\"trial_id\":null,\"data\":{{\"study_name\":\"dup_run\",\"directions\":[1]}}}}").unwrap();
            // create trial (默认 Running)
            writeln!(f, "{{\"op_code\":4,\"study_id\":0,\"trial_id\":null,\"data\":{{\"datetime_start\":\"2024-01-01T00:00:00+00:00\"}}}}").unwrap();
            // 第一个 worker 设为 Running (应被接受 — 此时 trial 默认就是 Running, 但因为 已经 Running 应该被拒绝)
            writeln!(f, "{{\"op_code\":6,\"study_id\":0,\"trial_id\":0,\"data\":{{\"state\":0,\"values\":[],\"datetime_start\":\"2024-01-01T00:01:00+00:00\"}}}}").unwrap();
            // 第二个 worker 也设为 Running (应被静默拒绝)
            writeln!(f, "{{\"op_code\":6,\"study_id\":0,\"trial_id\":0,\"data\":{{\"state\":0,\"values\":[],\"datetime_start\":\"2024-01-01T00:02:00+00:00\"}}}}").unwrap();
            // complete
            writeln!(f, "{{\"op_code\":6,\"study_id\":0,\"trial_id\":0,\"data\":{{\"state\":1,\"values\":[42.0],\"datetime_complete\":\"2024-01-01T00:03:00+00:00\"}}}}").unwrap();
        }

        let storage = JournalFileStorage::new(&path).unwrap();
        let sid = storage.get_study_id_from_name("dup_run").unwrap();
        let trials = storage.get_all_trials(sid, None).unwrap();
        assert_eq!(trials.len(), 1);
        assert_eq!(trials[0].state, TrialState::Complete);
        assert_eq!(trials[0].values.as_ref().unwrap(), &[42.0]);
        // datetime_start 应该来自创建日志（第一个被接受的时间）
        assert!(trials[0].datetime_start.is_some());
        fs::remove_file(&path).ok();
    }

    // ── 对齐 Python: get_best_trial NaN 过滤测试 ────────────────────

    #[test]
    fn test_journal_get_best_trial_nan_filtered() {
        let path = temp_path();
        let storage = JournalFileStorage::new(&path).unwrap();
        let sid = storage.create_new_study(&[StudyDirection::Maximize], Some("nan_test")).unwrap();

        // 创建正常 trial
        let t1 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(t1, TrialState::Complete, Some(&[5.0])).unwrap();

        // 创建 NaN trial
        let t2 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(t2, TrialState::Complete, Some(&[f64::NAN])).unwrap();

        let best = storage.get_best_trial(sid).unwrap();
        assert_eq!(best.values.as_ref().unwrap()[0], 5.0);
        fs::remove_file(&path).ok();
    }

    // ── 对齐 Python: Python 生成的日志文件兼容测试 ──────────────────

    #[test]
    fn test_journal_python_log_format_compatibility() {
        let path = temp_path();

        // 模拟 Python JournalStorage 生成的日志格式
        {
            let mut f = File::create(&path).unwrap();
            // Python create_study 格式
            writeln!(f, "{{\"op_code\":0,\"study_id\":null,\"trial_id\":null,\"data\":{{\"study_name\":\"py_compat\",\"directions\":[1]}}}}").unwrap();
            // Python set_study_user_attr 格式 (user_attr: {k: v})
            writeln!(f, "{{\"op_code\":2,\"study_id\":0,\"trial_id\":null,\"data\":{{\"user_attr\":{{\"py_key\":\"py_val\"}}}}}}").unwrap();
            // Python set_study_system_attr 格式 (system_attr: {k: v})
            writeln!(f, "{{\"op_code\":3,\"study_id\":0,\"trial_id\":null,\"data\":{{\"system_attr\":{{\"sys_key\":42}}}}}}").unwrap();
            // Python create_trial with template 格式
            writeln!(f, "{{\"op_code\":4,\"study_id\":0,\"trial_id\":null,\"data\":{{\"state\":1,\"datetime_start\":\"2024-01-01T00:00:00.000000\",\"datetime_complete\":\"2024-01-01T00:01:00.000000\",\"value\":0.5,\"values\":null,\"distributions\":{{\"x\":\"{{\\\"name\\\":\\\"FloatDistribution\\\",\\\"attributes\\\":{{\\\"low\\\":0.0,\\\"high\\\":1.0,\\\"log\\\":false,\\\"step\\\":null}}}}\"}},\"params\":{{\"x\":0.5}},\"user_attrs\":{{\"u1\":\"v1\"}},\"system_attrs\":{{\"s1\":100}},\"intermediate_values\":{{\"0\":0.1}}}}}}").unwrap();
            // Python set_trial_user_attr 格式 (user_attr: {k: v})
            writeln!(f, "{{\"op_code\":8,\"study_id\":0,\"trial_id\":0,\"data\":{{\"user_attr\":{{\"extra\":\"info\"}}}}}}").unwrap();
            // Python set_trial_system_attr 格式 (system_attr: {k: v})
            writeln!(f, "{{\"op_code\":9,\"study_id\":0,\"trial_id\":0,\"data\":{{\"system_attr\":{{\"sys_extra\":999}}}}}}").unwrap();
        }

        let storage = JournalFileStorage::new(&path).unwrap();
        let sid = storage.get_study_id_from_name("py_compat").unwrap();

        // 验证 study attrs
        let study_ua = storage.get_study_user_attrs(sid).unwrap();
        assert_eq!(study_ua["py_key"], serde_json::json!("py_val"));
        let study_sa = storage.get_study_system_attrs(sid).unwrap();
        assert_eq!(study_sa["sys_key"], serde_json::json!(42));

        // 验证 trial 从模板恢复
        let trials = storage.get_all_trials(sid, None).unwrap();
        assert_eq!(trials.len(), 1);
        let t = &trials[0];
        assert_eq!(t.state, TrialState::Complete);
        assert_eq!(t.values.as_ref().unwrap(), &[0.5]);
        assert!(t.params.contains_key("x"));
        assert!(t.distributions.contains_key("x"));
        assert_eq!(t.user_attrs["u1"], serde_json::json!("v1"));
        // after set_trial_user_attr
        assert_eq!(t.user_attrs["extra"], serde_json::json!("info"));
        assert_eq!(t.system_attrs["s1"], serde_json::json!(100));
        assert_eq!(t.system_attrs["sys_extra"], serde_json::json!(999));
        assert_eq!(t.intermediate_values[&0], 0.1);
        assert!(t.datetime_start.is_some());
        assert!(t.datetime_complete.is_some());

        fs::remove_file(&path).ok();
    }

    // ── 对齐 Python: 验证序列化输出为扁平格式（无 data 包裹） ───────

    #[test]
    fn test_journal_serializes_flat_format() {
        let entry = JournalLogEntry {
            op_code: 0,
            study_id: None,
            trial_id: None,
            data: serde_json::json!({"study_name": "flat", "directions": [1]}),
        };
        let json = serde_json::to_string(&entry).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        // 应无 "data" 字段
        assert!(parsed.get("data").is_none(), "should not have nested 'data' field");
        // 字段应在顶层
        assert_eq!(parsed["op_code"], 0);
        assert_eq!(parsed["study_name"], "flat");
        assert_eq!(parsed["worker_id"], "rust:0");
    }

    // ── 对齐 Python: 反序列化支持 Python 扁平格式 ──────────────────

    #[test]
    fn test_journal_deserialize_python_flat_format() {
        // 模拟 Python 写入的扁平日志行（无 data 包裹）
        let json = r#"{"op_code":0,"worker_id":"abc-123","study_name":"py_flat","directions":[1]}"#;
        let entry: JournalLogEntry = serde_json::from_str(json).unwrap();
        assert_eq!(entry.op_code, 0);
        assert!(entry.study_id.is_none());
        assert_eq!(entry.data.get("study_name").unwrap(), "py_flat");
        assert!(entry.data.get("worker_id").is_none(), "worker_id should not be in data");
    }

    // ── 对齐 Python: Python 扁平日志完整端到端测试 ─────────────────

    #[test]
    fn test_journal_replay_python_flat_log_end_to_end() {
        let path = temp_path();
        // 模拟 Python 实际输出的扁平日志格式（无 data 包裹，有 worker_id）
        {
            let mut f = File::create(&path).unwrap();
            writeln!(f, r#"{{"op_code":0,"worker_id":"uuid-1","study_name":"flat_e2e","directions":[1]}}"#).unwrap();
            writeln!(f, r#"{{"op_code":2,"worker_id":"uuid-1","study_id":0,"user_attr":{{"my_key":"my_val"}}}}"#).unwrap();
            writeln!(f, r#"{{"op_code":4,"worker_id":"uuid-1","study_id":0,"datetime_start":"2024-06-01T12:00:00.000000"}}"#).unwrap();
            writeln!(f, r#"{{"op_code":5,"worker_id":"uuid-1","trial_id":0,"param_name":"lr","param_value_internal":0.01,"distribution":"{{\"name\":\"FloatDistribution\",\"attributes\":{{\"low\":0.001,\"high\":1.0,\"log\":true,\"step\":null}}}}"}}"#).unwrap();
            writeln!(f, r#"{{"op_code":7,"worker_id":"uuid-1","trial_id":0,"step":0,"intermediate_value":0.95}}"#).unwrap();
            writeln!(f, r#"{{"op_code":6,"worker_id":"uuid-1","trial_id":0,"state":1,"values":[0.85],"datetime_complete":"2024-06-01T12:01:00.000000"}}"#).unwrap();
            writeln!(f, r#"{{"op_code":8,"worker_id":"uuid-1","trial_id":0,"user_attr":{{"note":"good"}}}}"#).unwrap();
        }

        let storage = JournalFileStorage::new(&path).unwrap();
        let sid = storage.get_study_id_from_name("flat_e2e").unwrap();

        let study_ua = storage.get_study_user_attrs(sid).unwrap();
        assert_eq!(study_ua["my_key"], serde_json::json!("my_val"));

        let trials = storage.get_all_trials(sid, None).unwrap();
        assert_eq!(trials.len(), 1);
        let t = &trials[0];
        assert_eq!(t.state, TrialState::Complete);
        assert_eq!(t.values.as_ref().unwrap(), &[0.85]);
        assert!(t.params.contains_key("lr"));
        assert!(t.distributions.contains_key("lr"));
        assert_eq!(t.intermediate_values[&0], 0.95);
        assert_eq!(t.user_attrs["note"], serde_json::json!("good"));
        assert!(t.datetime_start.is_some());
        assert!(t.datetime_complete.is_some());
        fs::remove_file(&path).ok();
    }

    // ── 对齐 Python: Rust 写入后 replay 扁平格式正确 ──────────────

    #[test]
    fn test_journal_rust_write_replay_flat() {
        let path = temp_path();
        {
            let storage = JournalFileStorage::new(&path).unwrap();
            let sid = storage.create_new_study(&[StudyDirection::Minimize], Some("flat_rw")).unwrap();
            let tid = storage.create_new_trial(sid, None).unwrap();
            storage.set_trial_intermediate_value(tid, 0, 0.5).unwrap();
            storage.set_trial_state_values(tid, TrialState::Complete, Some(&[1.0])).unwrap();
        }

        // 验证文件内容是扁平格式
        let content = fs::read_to_string(&path).unwrap();
        assert!(!content.contains("\"data\""), "output should not contain nested 'data' field");
        assert!(content.contains("\"worker_id\""), "output should contain 'worker_id'");
        assert!(content.contains("\"intermediate_value\""), "should use 'intermediate_value' key");

        // replay 应成功恢复
        let storage = JournalFileStorage::new(&path).unwrap();
        let sid = storage.get_study_id_from_name("flat_rw").unwrap();
        let trials = storage.get_all_trials(sid, None).unwrap();
        assert_eq!(trials.len(), 1);
        assert_eq!(trials[0].state, TrialState::Complete);
        assert_eq!(trials[0].intermediate_values[&0], 0.5);
        fs::remove_file(&path).ok();
    }
}
