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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JournalLogEntry {
    /// 操作类型
    pub op_code: i32,
    /// 关联的 study_id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub study_id: Option<i64>,
    /// 关联的 trial_id
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trial_id: Option<i64>,
    /// 通用数据载荷（JSON）
    #[serde(default)]
    pub data: serde_json::Value,
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
                if let (Some(key), Some(value)) = (
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
                if let (Some(key), Some(value)) = (
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
            .unwrap_or(0);
        let state = match state_int {
            1 => TrialState::Complete,
            2 => TrialState::Pruned,
            3 => TrialState::Fail,
            4 => TrialState::Waiting,
            _ => TrialState::Running,
        };

        self.trials.insert(trial_id, TrialInternalState {
            trial_id,
            study_id,
            number,
            state,
            values: None,
            datetime_start: if state != TrialState::Waiting { Some(Utc::now()) } else { None },
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
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
                    trial.state = match state_int {
                        0 => TrialState::Running,
                        1 => TrialState::Complete,
                        2 => TrialState::Pruned,
                        3 => TrialState::Fail,
                        4 => TrialState::Waiting,
                        _ => trial.state,
                    };

                    if trial.state == TrialState::Running && trial.datetime_start.is_none() {
                        trial.datetime_start = Some(Utc::now());
                    }

                    if trial.state.is_finished() && trial.datetime_complete.is_none() {
                        trial.datetime_complete = Some(Utc::now());
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
                if let (Some(step), Some(value)) = (
                    entry.data.get("step").and_then(|v| v.as_i64()),
                    entry.data.get("value").and_then(|v| v.as_f64()),
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
                if let (Some(key), Some(value)) = (
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
                if let (Some(key), Some(value)) = (
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
        self.write_log(&JournalLogEntry {
            op_code: 2,
            study_id: Some(study_id),
            trial_id: None,
            data: serde_json::json!({"key": key, "value": value}),
        })
    }

    fn set_study_system_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        self.write_log(&JournalLogEntry {
            op_code: 3,
            study_id: Some(study_id),
            trial_id: None,
            data: serde_json::json!({"key": key, "value": value}),
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

        let data = if let Some(tmpl) = template_trial {
            serde_json::json!({
                "state": tmpl.state as i32,
                "values": tmpl.values,
                "params": tmpl.params,
                "distributions": tmpl.distributions,
                "user_attrs": tmpl.user_attrs,
                "system_attrs": tmpl.system_attrs,
                "intermediate_values": tmpl.intermediate_values,
            })
        } else {
            serde_json::json!({})
        };

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
        self.write_log(&JournalLogEntry {
            op_code: 6,
            study_id: None,
            trial_id: Some(trial_id),
            data: serde_json::json!({
                "state": state as i32,
                "values": values,
            }),
        })?;
        Ok(true)
    }

    fn set_trial_intermediate_value(
        &self,
        trial_id: i64,
        step: i64,
        intermediate_value: f64,
    ) -> Result<()> {
        self.write_log(&JournalLogEntry {
            op_code: 7,
            study_id: None,
            trial_id: Some(trial_id),
            data: serde_json::json!({
                "step": step,
                "value": intermediate_value,
            }),
        })
    }

    fn set_trial_user_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        self.write_log(&JournalLogEntry {
            op_code: 8,
            study_id: None,
            trial_id: Some(trial_id),
            data: serde_json::json!({"key": key, "value": value}),
        })
    }

    fn set_trial_system_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        self.write_log(&JournalLogEntry {
            op_code: 9,
            study_id: None,
            trial_id: Some(trial_id),
            data: serde_json::json!({"key": key, "value": value}),
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
        let entry = JournalLogEntry {
            op_code: 2,
            study_id: Some(study_id),
            trial_id: None,
            data: serde_json::json!({"key": key, "value": value}),
        };
        self.append_log(&entry)
    }

    fn set_study_system_attr(
        &self, study_id: i64, key: &str, value: serde_json::Value,
    ) -> Result<()> {
        let entry = JournalLogEntry {
            op_code: 3,
            study_id: Some(study_id),
            trial_id: None,
            data: serde_json::json!({"key": key, "value": value}),
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
        let state_int = template_trial.map(|t| t.state as i64).unwrap_or(0);

        let entry = JournalLogEntry {
            op_code: 4,
            study_id: Some(study_id),
            trial_id: None,
            data: serde_json::json!({"state": state_int}),
        };
        self.append_log(&entry)?;

        let trial_id = self.state.lock().next_trial_id - 1;

        // 如果有模板，复制参数/属性/值
        if let Some(tmpl) = template_trial {
            for (name, val) in &tmpl.params {
                if let Some(dist) = tmpl.distributions.get(name) {
                    let internal = dist.to_internal_repr(val)?;
                    let dist_json = crate::distributions::distribution_to_json(dist)?;
                    let entry = JournalLogEntry {
                        op_code: 5,
                        study_id: Some(study_id),
                        trial_id: Some(trial_id),
                        data: serde_json::json!({
                            "param_name": name,
                            "param_value_internal": internal,
                            "distribution": dist_json,
                        }),
                    };
                    self.append_log(&entry)?;
                }
            }

            if tmpl.values.is_some() || tmpl.state.is_finished() {
                let vals: Vec<f64> = tmpl.values.as_deref().unwrap_or(&[]).to_vec();
                let entry = JournalLogEntry {
                    op_code: 6,
                    study_id: Some(study_id),
                    trial_id: Some(trial_id),
                    data: serde_json::json!({
                        "state": tmpl.state as i64,
                        "values": vals,
                    }),
                };
                self.append_log(&entry)?;
            }

            for (&step, &v) in &tmpl.intermediate_values {
                let entry = JournalLogEntry {
                    op_code: 7,
                    study_id: Some(study_id),
                    trial_id: Some(trial_id),
                    data: serde_json::json!({"step": step, "value": v}),
                };
                self.append_log(&entry)?;
            }

            for (k, v) in &tmpl.user_attrs {
                let entry = JournalLogEntry {
                    op_code: 8,
                    study_id: Some(study_id),
                    trial_id: Some(trial_id),
                    data: serde_json::json!({"key": k, "value": v}),
                };
                self.append_log(&entry)?;
            }

            for (k, v) in &tmpl.system_attrs {
                let entry = JournalLogEntry {
                    op_code: 9,
                    study_id: Some(study_id),
                    trial_id: Some(trial_id),
                    data: serde_json::json!({"key": k, "value": v}),
                };
                self.append_log(&entry)?;
            }
        }

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

        let entry = JournalLogEntry {
            op_code: 6,
            study_id: Some(study_id),
            trial_id: Some(trial_id),
            data: serde_json::json!({
                "state": state as i64,
                "values": values.unwrap_or(&[]),
            }),
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

        let entry = JournalLogEntry {
            op_code: 7,
            study_id: Some(study_id),
            trial_id: Some(trial_id),
            data: serde_json::json!({"step": step, "value": intermediate_value}),
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

        let entry = JournalLogEntry {
            op_code: 8,
            study_id: Some(study_id),
            trial_id: Some(trial_id),
            data: serde_json::json!({"key": key, "value": value}),
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

        let entry = JournalLogEntry {
            op_code: 9,
            study_id: Some(study_id),
            trial_id: Some(trial_id),
            data: serde_json::json!({"key": key, "value": value}),
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
}
