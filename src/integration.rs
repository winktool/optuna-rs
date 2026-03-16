//! 框架集成模块。
//!
//! 对应 Python `optuna.integration` / `optuna-integration` 包。
//!
//! ## 已实现的集成
//!
//! ### 通用日志回调（无外部依赖）
//! - [`CsvLoggerCallback`] — 将试验结果写入 CSV 文件
//! - [`JsonLoggerCallback`] — 将试验结果写入 JSONL 文件
//! - [`DebugPrintCallback`] — 将试验结果打印到 stderr
//!
//! ### MLflow 集成（feature `mlflow`）
//! - [`MLflowCallback`] — 通过 REST API 将试验记录到 MLflow Tracking Server
//!
//! ### TensorBoard 集成（无外部依赖）
//! - [`TensorBoardCallback`] — 将超参/指标写入 TensorBoard event 文件
//!
//! ### 框架剪枝工具
//! - [`PruningMixin`] — 通用 ML 框架剪枝回调辅助
//!
//! ## 框架集成指南
//! 为 Rust ML 框架创建集成只需实现 `Callback` trait：
//! ```ignore
//! use optuna_rs::{Callback, Study};
//! use optuna_rs::trial::FrozenTrial;
//!
//! struct MyFrameworkCallback;
//!
//! impl Callback for MyFrameworkCallback {
//!     fn on_trial_complete(&self, study: &Study, trial: &FrozenTrial) {
//!         // 记录到你的框架
//!     }
//! }
//! ```

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use parking_lot::Mutex;

use crate::callbacks::Callback;
use crate::error::{OptunaError, Result};
use crate::study::Study;
use crate::trial::FrozenTrial;

// ============================================================================
// CsvLoggerCallback
// ============================================================================

/// 将试验结果写入 CSV 文件的回调。
///
/// 每次试验完成时追加一行到 CSV。
///
/// # 列
/// `number`, `value`, `state`, `datetime_start`, `datetime_complete`,
/// `duration_s`, `params_json`, `user_attrs_json`
pub struct CsvLoggerCallback {
    file: Mutex<std::fs::File>,
}

impl CsvLoggerCallback {
    /// 创建 CSV 日志回调。
    ///
    /// # 参数
    /// * `path` - CSV 文件路径
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
            .map_err(|e| crate::error::OptunaError::StorageInternalError(
                format!("打开 CSV 文件失败: {e}"),
            ))?;

        let needs_header = file.metadata()
            .map(|m| m.len() == 0)
            .unwrap_or(true);

        let cb = Self {
            file: Mutex::new(file),
        };

        if needs_header {
            let f = &mut *cb.file.lock();
            let _ = writeln!(f, "number,value,state,datetime_start,datetime_complete,duration_s,params_json,user_attrs_json");
        }

        Ok(cb)
    }
}

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        format!("\"{}\"", s.replace('"', "\"\""))
    } else {
        s.to_string()
    }
}

impl Callback for CsvLoggerCallback {
    fn on_trial_complete(&self, _study: &Study, trial: &FrozenTrial) {
        let value = trial
            .values
            .as_ref()
            .map(|v| {
                v.iter()
                    .map(|f| f.to_string())
                    .collect::<Vec<_>>()
                    .join(";")
            })
            .unwrap_or_default();

        let duration = trial
            .duration()
            .map(|d| format!("{:.3}", d.num_milliseconds() as f64 / 1000.0))
            .unwrap_or_default();

        let params_json = serde_json::to_string(&trial.params).unwrap_or_default();
        let attrs_json = serde_json::to_string(&trial.user_attrs).unwrap_or_default();

        let line = format!(
            "{},{},{},{},{},{},{},{}",
            trial.number,
            value,
            trial.state,
            trial.datetime_start.map(|dt| dt.to_rfc3339()).unwrap_or_default(),
            trial.datetime_complete.map(|dt| dt.to_rfc3339()).unwrap_or_default(),
            duration,
            csv_escape(&params_json),
            csv_escape(&attrs_json),
        );

        let mut f = self.file.lock();
        let _ = writeln!(f, "{line}");
    }
}

// ============================================================================
// JsonLoggerCallback
// ============================================================================

/// 将试验结果写入 JSONL（JSON Lines）文件的回调。
///
/// 每次试验完成时追加一行 JSON 到文件。
pub struct JsonLoggerCallback {
    file: Mutex<std::fs::File>,
}

impl JsonLoggerCallback {
    /// 创建 JSONL 日志回调。
    pub fn new(path: impl AsRef<Path>) -> Result<Self> {
        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.as_ref())
            .map_err(|e| crate::error::OptunaError::StorageInternalError(
                format!("打开 JSONL 文件失败: {e}"),
            ))?;
        Ok(Self {
            file: Mutex::new(file),
        })
    }
}

impl Callback for JsonLoggerCallback {
    fn on_trial_complete(&self, _study: &Study, trial: &FrozenTrial) {
        #[derive(serde::Serialize)]
        struct TrialLog<'a> {
            number: i64,
            values: &'a Option<Vec<f64>>,
            state: String,
            duration_s: Option<f64>,
            params: &'a std::collections::HashMap<String, crate::distributions::ParamValue>,
            user_attrs: &'a std::collections::HashMap<String, serde_json::Value>,
        }

        let log = TrialLog {
            number: trial.number,
            values: &trial.values,
            state: trial.state.to_string(),
            duration_s: trial.duration().map(|d| d.num_milliseconds() as f64 / 1000.0),
            params: &trial.params,
            user_attrs: &trial.user_attrs,
        };

        if let Ok(line) = serde_json::to_string(&log) {
            let mut f = self.file.lock();
            let _ = writeln!(f, "{line}");
        }
    }
}

// ============================================================================
// DebugPrintCallback
// ============================================================================

/// 调试回调 — 将每次试验结果打印到 stderr。
///
/// 用于开发和调试。
pub struct DebugPrintCallback;

impl Callback for DebugPrintCallback {
    fn on_trial_complete(&self, _study: &Study, trial: &FrozenTrial) {
        eprintln!(
            "[Trial {}] state={:?}, values={:?}, params={:?}",
            trial.number, trial.state, trial.values, trial.params
        );
    }
}

// ============================================================================
// MLflowCallback (feature = "mlflow")
// ============================================================================

/// 将试验记录到 MLflow Tracking Server 的回调。
///
/// 对应 Python `optuna_integration.MLflowCallback`。
///
/// 通过 MLflow REST API 将每次试验的参数、指标和标签记录到 MLflow。
/// 需要运行中的 MLflow Tracking Server。
///
/// # Feature
/// 需要 `mlflow` feature: `cargo build --features mlflow`
///
/// # 示例
/// ```ignore
/// use optuna_rs::integration::MLflowCallback;
///
/// let cb = MLflowCallback::new("http://localhost:5000", None, None);
/// study.optimize(|trial| {
///     let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
///     Ok(x * x)
/// }, Some(100), None, Some(&[&cb])).unwrap();
/// ```
#[cfg(feature = "mlflow")]
pub struct MLflowCallback {
    tracking_uri: String,
    metric_name: Option<String>,
    experiment_name: Option<String>,
    client: reqwest::blocking::Client,
}

#[cfg(feature = "mlflow")]
impl MLflowCallback {
    /// 创建 MLflow 回调。
    ///
    /// # 参数
    /// * `tracking_uri` - MLflow Tracking Server URI (如 `http://localhost:5000`)
    /// * `metric_name` - 指标名称（默认 `"value"`）
    /// * `experiment_name` - 实验名称（默认使用 study 名称）
    pub fn new(
        tracking_uri: &str,
        metric_name: Option<String>,
        experiment_name: Option<String>,
    ) -> Self {
        Self {
            tracking_uri: tracking_uri.trim_end_matches('/').to_string(),
            metric_name,
            experiment_name,
            client: reqwest::blocking::Client::new(),
        }
    }

    fn get_or_create_experiment(&self, name: &str) -> Result<String> {
        // 先尝试按名称获取
        let url = format!(
            "{}/api/2.0/mlflow/experiments/get-by-name",
            self.tracking_uri
        );
        let resp = self
            .client
            .get(&url)
            .query(&[("experiment_name", name)])
            .send()
            .map_err(|e| OptunaError::StorageInternalError(format!("MLflow request failed: {e}")))?;

        if resp.status().is_success() {
            let body: serde_json::Value = resp.json().map_err(|e| {
                OptunaError::StorageInternalError(format!("MLflow JSON parse error: {e}"))
            })?;
            if let Some(id) = body["experiment"]["experiment_id"].as_str() {
                return Ok(id.to_string());
            }
        }

        // 创建新实验
        let url = format!("{}/api/2.0/mlflow/experiments/create", self.tracking_uri);
        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({ "name": name }))
            .send()
            .map_err(|e| OptunaError::StorageInternalError(format!("MLflow request failed: {e}")))?;

        let body: serde_json::Value = resp.json().map_err(|e| {
            OptunaError::StorageInternalError(format!("MLflow JSON parse error: {e}"))
        })?;
        body["experiment_id"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| {
                OptunaError::StorageInternalError("MLflow: failed to create experiment".into())
            })
    }

    fn create_run(&self, experiment_id: &str, trial: &FrozenTrial) -> Result<String> {
        let url = format!("{}/api/2.0/mlflow/runs/create", self.tracking_uri);
        let start_ms = trial
            .datetime_start
            .map(|dt| dt.timestamp_millis())
            .unwrap_or(0);

        let resp = self
            .client
            .post(&url)
            .json(&serde_json::json!({
                "experiment_id": experiment_id,
                "start_time": start_ms,
                "run_name": format!("trial-{}", trial.number),
            }))
            .send()
            .map_err(|e| OptunaError::StorageInternalError(format!("MLflow request failed: {e}")))?;

        let body: serde_json::Value = resp.json().map_err(|e| {
            OptunaError::StorageInternalError(format!("MLflow JSON parse error: {e}"))
        })?;
        body["run"]["info"]["run_id"]
            .as_str()
            .map(|s| s.to_string())
            .ok_or_else(|| {
                OptunaError::StorageInternalError("MLflow: failed to create run".into())
            })
    }

    fn log_batch(
        &self,
        run_id: &str,
        metrics: &[(String, f64)],
        params: &[(String, String)],
        tags: &[(String, String)],
    ) -> Result<()> {
        let url = format!("{}/api/2.0/mlflow/runs/log-batch", self.tracking_uri);
        let ts = chrono::Utc::now().timestamp_millis();

        let metrics_json: Vec<_> = metrics
            .iter()
            .map(|(k, v)| {
                serde_json::json!({
                    "key": k,
                    "value": v,
                    "timestamp": ts,
                    "step": 0
                })
            })
            .collect();

        let params_json: Vec<_> = params
            .iter()
            .map(|(k, v)| serde_json::json!({ "key": k, "value": v }))
            .collect();

        let tags_json: Vec<_> = tags
            .iter()
            .map(|(k, v)| serde_json::json!({ "key": k, "value": v }))
            .collect();

        self.client
            .post(&url)
            .json(&serde_json::json!({
                "run_id": run_id,
                "metrics": metrics_json,
                "params": params_json,
                "tags": tags_json,
            }))
            .send()
            .map_err(|e| {
                OptunaError::StorageInternalError(format!("MLflow log-batch failed: {e}"))
            })?;

        Ok(())
    }

    fn end_run(&self, run_id: &str, status: &str) -> Result<()> {
        let url = format!("{}/api/2.0/mlflow/runs/update", self.tracking_uri);
        let end_ms = chrono::Utc::now().timestamp_millis();
        self.client
            .post(&url)
            .json(&serde_json::json!({
                "run_id": run_id,
                "status": status,
                "end_time": end_ms,
            }))
            .send()
            .map_err(|e| {
                OptunaError::StorageInternalError(format!("MLflow update-run failed: {e}"))
            })?;
        Ok(())
    }
}

#[cfg(feature = "mlflow")]
impl Callback for MLflowCallback {
    fn on_trial_complete(&self, study: &Study, trial: &FrozenTrial) {
        let exp_name = self
            .experiment_name
            .clone()
            .unwrap_or_else(|| study.study_name().to_string());

        let exp_id = match self.get_or_create_experiment(&exp_name) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("[MLflow] experiment error: {e}");
                return;
            }
        };

        let run_id = match self.create_run(&exp_id, trial) {
            Ok(id) => id,
            Err(e) => {
                eprintln!("[MLflow] create run error: {e}");
                return;
            }
        };

        // 收集指标
        let mut metrics: Vec<(String, f64)> = Vec::new();
        if let Some(ref vals) = trial.values {
            if vals.len() == 1 {
                let name = self.metric_name.clone().unwrap_or_else(|| "value".into());
                metrics.push((name, vals[0]));
            } else {
                for (i, v) in vals.iter().enumerate() {
                    let name = format!(
                        "{}_{}",
                        self.metric_name.as_deref().unwrap_or("value"),
                        i
                    );
                    metrics.push((name, *v));
                }
            }
        }

        // 收集参数
        let params: Vec<(String, String)> = trial
            .params
            .iter()
            .map(|(k, v)| (k.clone(), format!("{v:?}")))
            .collect();

        // 收集标签
        let mut tags: Vec<(String, String)> = vec![
            ("trial_number".into(), trial.number.to_string()),
            ("trial_state".into(), trial.state.to_string()),
        ];
        // study directions
        let dirs: Vec<_> = study
            .directions()
            .iter()
            .map(|d| format!("{d:?}"))
            .collect();
        tags.push(("study_directions".into(), dirs.join(",")));
        // user attrs as tags
        for (k, v) in &trial.user_attrs {
            let val_str = match v {
                serde_json::Value::String(s) => s.clone(),
                other => other.to_string(),
            };
            // MLflow tag 值长度限制 5000
            if val_str.len() <= 5000 {
                tags.push((format!("user_attr.{k}"), val_str));
            }
        }

        if let Err(e) = self.log_batch(&run_id, &metrics, &params, &tags) {
            eprintln!("[MLflow] log-batch error: {e}");
        }

        let mlflow_status = match trial.state {
            crate::trial::TrialState::Complete => "FINISHED",
            crate::trial::TrialState::Fail => "FAILED",
            _ => "KILLED",
        };
        if let Err(e) = self.end_run(&run_id, mlflow_status) {
            eprintln!("[MLflow] end-run error: {e}");
        }
    }
}

// ============================================================================
// TensorBoardCallback
// ============================================================================

/// 将超参数和指标写入 TensorBoard event 文件的回调。
///
/// 对应 Python `optuna_integration.TensorBoardCallback`。
///
/// 为每个试验创建子目录，写入与 TensorBoard 兼容的 event 文件。
/// 兼容 `tensorboard --logdir <dirname>` 查看。
///
/// 由于 TF event 文件的二进制格式较复杂，此实现使用简化的 JSON summary
/// 格式，可被 TensorBoard 的 JSON 加载器读取，也可直接用于分析。
///
/// # 示例
/// ```ignore
/// use optuna_rs::integration::TensorBoardCallback;
///
/// let cb = TensorBoardCallback::new("./tb_logs", None);
/// study.optimize(|trial| {
///     let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
///     Ok(x * x)
/// }, Some(100), None, Some(&[&cb])).unwrap();
/// ```
pub struct TensorBoardCallback {
    dirname: PathBuf,
    metric_name: Option<String>,
}

impl TensorBoardCallback {
    /// 创建 TensorBoard 回调。
    ///
    /// # 参数
    /// * `dirname` - TensorBoard 日志目录
    /// * `metric_name` - 指标名称（默认 `"value"`）
    pub fn new(dirname: impl AsRef<Path>, metric_name: Option<String>) -> Self {
        Self {
            dirname: dirname.as_ref().to_path_buf(),
            metric_name,
        }
    }

    /// 写入 TFEvent 格式的 scalar summary。
    ///
    /// TFEvent 格式: length(8字节LE) + crc_length(4字节) + data + crc_data(4字节)
    /// data 是 protobuf 编码的 Event 消息。
    fn write_scalar_event(
        writer: &mut impl Write,
        wall_time: f64,
        step: i64,
        tag: &str,
        value: f32,
    ) -> std::io::Result<()> {
        // 构建 protobuf 手工编码:
        // Event { wall_time: double(field 1), step: int64(field 2), summary: Summary(field 5) }
        // Summary { value: [Value { tag: string(field 1), simple_value: float(field 2) }] }

        // Summary.Value
        let mut sv = Vec::new();
        // field 1: tag (string), wire type 2
        sv.push(0x0A);
        encode_varint(&mut sv, tag.len() as u64);
        sv.extend_from_slice(tag.as_bytes());
        // field 2: simple_value (float), wire type 5
        sv.push(0x15);
        sv.extend_from_slice(&value.to_le_bytes());

        // Summary
        let mut summary = Vec::new();
        // field 1: value (repeated Value), wire type 2
        summary.push(0x0A);
        encode_varint(&mut summary, sv.len() as u64);
        summary.extend_from_slice(&sv);

        // Event
        let mut event = Vec::new();
        // field 1: wall_time (double), wire type 1
        event.push(0x09);
        event.extend_from_slice(&wall_time.to_le_bytes());
        // field 2: step (int64), wire type 0
        event.push(0x10);
        encode_varint(&mut event, step as u64);
        // field 5: summary (Summary), wire type 2
        event.push(0x2A);
        encode_varint(&mut event, summary.len() as u64);
        event.extend_from_slice(&summary);

        // TFRecord 格式
        let len = event.len() as u64;
        let len_bytes = len.to_le_bytes();
        let len_crc = masked_crc32c(&len_bytes);
        let data_crc = masked_crc32c(&event);

        writer.write_all(&len_bytes)?;
        writer.write_all(&len_crc.to_le_bytes())?;
        writer.write_all(&event)?;
        writer.write_all(&data_crc.to_le_bytes())?;

        Ok(())
    }
}

/// Protobuf varint 编码
fn encode_varint(buf: &mut Vec<u8>, mut val: u64) {
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 {
            buf.push(byte);
            break;
        } else {
            buf.push(byte | 0x80);
        }
    }
}

/// CRC32C (Castagnoli) with TF masking
fn masked_crc32c(data: &[u8]) -> u32 {
    let crc = crc32c(data);
    ((crc >> 15) | (crc << 17)).wrapping_add(0xa282ead8)
}

/// Basic CRC32C implementation (Castagnoli polynomial 0x82F63B78)
fn crc32c(data: &[u8]) -> u32 {
    let mut crc: u32 = 0xFFFFFFFF;
    for &byte in data {
        crc ^= byte as u32;
        for _ in 0..8 {
            if crc & 1 != 0 {
                crc = (crc >> 1) ^ 0x82F63B78;
            } else {
                crc >>= 1;
            }
        }
    }
    crc ^ 0xFFFFFFFF
}

impl Callback for TensorBoardCallback {
    fn on_trial_complete(&self, _study: &Study, trial: &FrozenTrial) {
        let trial_dir = self.dirname.join(format!("trial-{}", trial.number));
        if let Err(e) = std::fs::create_dir_all(&trial_dir) {
            eprintln!("[TensorBoard] create dir error: {e}");
            return;
        }

        let event_path = trial_dir.join(format!(
            "events.out.tfevents.{}.optuna",
            chrono::Utc::now().timestamp()
        ));

        let mut file = match std::fs::File::create(&event_path) {
            Ok(f) => std::io::BufWriter::new(f),
            Err(e) => {
                eprintln!("[TensorBoard] create file error: {e}");
                return;
            }
        };

        let wall_time = trial
            .datetime_complete
            .map(|dt| dt.timestamp() as f64 + dt.timestamp_subsec_nanos() as f64 * 1e-9)
            .unwrap_or_else(|| chrono::Utc::now().timestamp() as f64);

        // 写入指标
        if let Some(ref vals) = trial.values {
            if vals.len() == 1 {
                let name = self.metric_name.clone().unwrap_or_else(|| "value".into());
                let _ = Self::write_scalar_event(
                    &mut file,
                    wall_time,
                    trial.number,
                    &name,
                    vals[0] as f32,
                );
            } else {
                for (i, v) in vals.iter().enumerate() {
                    let name = format!(
                        "{}_{}",
                        self.metric_name.as_deref().unwrap_or("value"),
                        i
                    );
                    let _ = Self::write_scalar_event(
                        &mut file,
                        wall_time,
                        trial.number,
                        &name,
                        *v as f32,
                    );
                }
            }
        }

        // 写入参数作为 scalar (数值参数)
        for (k, v) in &trial.params {
            let float_val = match v {
                crate::distributions::ParamValue::Float(f) => *f as f32,
                crate::distributions::ParamValue::Int(i) => *i as f32,
                _ => continue,
            };
            let _ = Self::write_scalar_event(
                &mut file,
                wall_time,
                trial.number,
                &format!("params/{k}"),
                float_val,
            );
        }

        // 同时写入 JSON summary 方便非 TensorBoard 工具解析
        let json_path = trial_dir.join("trial_summary.json");
        let summary = serde_json::json!({
            "trial_number": trial.number,
            "state": trial.state.to_string(),
            "values": trial.values,
            "params": trial.params,
            "user_attrs": trial.user_attrs,
            "datetime_start": trial.datetime_start,
            "datetime_complete": trial.datetime_complete,
        });
        if let Ok(mut f) = std::fs::File::create(&json_path) {
            let _ = f.write_all(serde_json::to_string_pretty(&summary).unwrap_or_default().as_bytes());
        }
    }
}

// ============================================================================
// WandbCallback (Weights & Biases)
// ============================================================================

/// Weights & Biases 集成回调 — 通过可插拔 trait 支持任意 W&B 后端。
///
/// 对应 Python `optuna_integration.WeightsAndBiasesCallback`。
///
/// 由于 Rust 没有官方 W&B SDK，提供 [`WandbLogger`] trait 抽象；
/// 用户可通过 HTTP API 或其他方式实现此 trait。
///
/// # 示例
/// ```ignore
/// use optuna_rs::integration::{WandbCallback, WandbLogger};
///
/// struct MyWandbLogger;
/// impl WandbLogger for MyWandbLogger {
///     fn init(&self, project: &str, name: &str, config: &serde_json::Value) { /* ... */ }
///     fn log(&self, data: &serde_json::Value) { /* ... */ }
///     fn finish(&self) { /* ... */ }
/// }
///
/// let cb = WandbCallback::new(Box::new(MyWandbLogger), None, true);
/// ```
pub trait WandbLogger: Send + Sync {
    /// 初始化一个 W&B run。
    fn init(&self, project: &str, name: &str, config: &serde_json::Value);
    /// 记录一组键值对。
    fn log(&self, data: &serde_json::Value);
    /// 结束当前 run。
    fn finish(&self);
}

/// Weights & Biases 回调。
///
/// 支持两种模式：
/// - `as_multirun = true`（默认）：每个试验创建一个独立 run
/// - `as_multirun = false`：所有试验记录到同一个 run
pub struct WandbCallback {
    logger: Box<dyn WandbLogger>,
    metric_name: Option<String>,
    as_multirun: bool,
    initialized: Mutex<bool>,
}

impl WandbCallback {
    /// 创建 W&B 回调。
    ///
    /// # 参数
    /// * `logger` - W&B 日志实现
    /// * `metric_name` - 指标名称（默认 `"value"`）
    /// * `as_multirun` - 是否每个试验一个 run（默认 true）
    pub fn new(
        logger: Box<dyn WandbLogger>,
        metric_name: Option<String>,
        as_multirun: bool,
    ) -> Self {
        Self {
            logger,
            metric_name,
            as_multirun,
            initialized: Mutex::new(false),
        }
    }
}

impl Callback for WandbCallback {
    fn on_trial_complete(&self, study: &Study, trial: &FrozenTrial) {
        let metric_name = self.metric_name.clone().unwrap_or_else(|| "value".into());

        if self.as_multirun {
            // 每个试验一个 run
            let config = serde_json::to_value(&trial.params).unwrap_or_default();
            self.logger.init(
                study.study_name(),
                &format!("trial-{}", trial.number),
                &config,
            );

            let mut data = serde_json::Map::new();
            data.insert("trial_number".into(), trial.number.into());
            if let Some(ref vals) = trial.values {
                if vals.len() == 1 {
                    data.insert(metric_name.clone(), vals[0].into());
                } else {
                    for (i, v) in vals.iter().enumerate() {
                        data.insert(format!("{metric_name}_{i}"), (*v).into());
                    }
                }
            }
            self.logger.log(&serde_json::Value::Object(data));
            self.logger.finish();
        } else {
            // 所有试验一个 run
            {
                let mut init = self.initialized.lock();
                if !*init {
                    self.logger.init(
                        study.study_name(),
                        "optuna-study",
                        &serde_json::json!({}),
                    );
                    *init = true;
                }
            }

            let mut data = serde_json::Map::new();
            data.insert("trial_number".into(), trial.number.into());
            for (k, v) in &trial.params {
                data.insert(format!("params/{k}"), serde_json::to_value(v).unwrap_or_default());
            }
            if let Some(ref vals) = trial.values {
                if vals.len() == 1 {
                    data.insert(metric_name.clone(), vals[0].into());
                } else {
                    for (i, v) in vals.iter().enumerate() {
                        data.insert(format!("{metric_name}_{i}"), (*v).into());
                    }
                }
            }
            self.logger.log(&serde_json::Value::Object(data));
        }
    }
}

// ============================================================================
// PruningMixin — 通用 ML 框架剪枝辅助
// ============================================================================

/// 通用 ML 框架剪枝辅助工具。
///
/// 对应 Python `optuna_integration` 中的多个 `*PruningCallback`
/// （XGBoost、LightGBM、CatBoost、Keras 等）。
///
/// 在 Rust 中，各 ML 框架的回调机制不同，此结构体提供
/// 统一的 report-and-prune 逻辑，可在任意框架的回调中使用。
///
/// # 示例
/// ```ignore
/// use optuna_rs::integration::PruningMixin;
/// use optuna_rs::trial::Trial;
///
/// fn xgboost_callback(trial: &Trial, epoch: i64, eval_score: f64) -> bool {
///     let mixin = PruningMixin::new("validation-rmse");
///     match mixin.check(trial, epoch, eval_score) {
///         PruneDecision::Continue => false,
///         PruneDecision::Prune => true,
///         PruneDecision::Error(e) => {
///             eprintln!("Pruning check error: {e}");
///             false
///         }
///     }
/// }
/// ```
pub struct PruningMixin {
    /// 指标名称（仅用于日志/调试）
    pub metric_name: String,
}

/// 剪枝决策枚举。
#[derive(Debug)]
pub enum PruneDecision {
    /// 继续训练
    Continue,
    /// 应该剪枝
    Prune,
    /// 检查过程发生错误
    Error(OptunaError),
}

impl PruningMixin {
    /// 创建剪枝辅助工具。
    pub fn new(metric_name: &str) -> Self {
        Self {
            metric_name: metric_name.to_string(),
        }
    }

    /// 报告中间值并检查是否应该剪枝。
    ///
    /// 这是 ML 框架回调中应调用的核心方法。
    ///
    /// # 参数
    /// * `trial` - 当前试验
    /// * `step` - 当前训练步数（epoch）
    /// * `value` - 当前指标值
    pub fn check(
        &self,
        trial: &mut crate::trial::Trial,
        step: i64,
        value: f64,
    ) -> PruneDecision {
        if let Err(e) = trial.report(value, step) {
            return PruneDecision::Error(e);
        }
        match trial.should_prune() {
            Ok(true) => PruneDecision::Prune,
            Ok(false) => PruneDecision::Continue,
            Err(e) => PruneDecision::Error(e),
        }
    }
}

// ============================================================================
// ExperimentTracker — 通用实验跟踪 trait
// ============================================================================

/// 通用实验跟踪器 trait。
///
/// 为 MLflow、W&B、Neptune 等跟踪平台提供统一接口。
/// 用户可实现此 trait 来接入任意跟踪后端。
pub trait ExperimentTracker: Send + Sync {
    /// 记录一组参数。
    fn log_params(&self, params: &HashMap<String, String>);
    /// 记录一组指标。
    fn log_metrics(&self, metrics: &HashMap<String, f64>, step: Option<i64>);
    /// 设置一组标签/元数据。
    fn set_tags(&self, tags: &HashMap<String, String>);
    /// 结束当前实验跟踪。
    fn finish(&self);
}

/// 通用实验跟踪回调。
///
/// 将任意 [`ExperimentTracker`] 实现包装为 Optuna [`Callback`]。
///
/// # 示例
/// ```ignore
/// use optuna_rs::integration::{TrackerCallback, ExperimentTracker};
/// use std::collections::HashMap;
///
/// struct MyTracker;
/// impl ExperimentTracker for MyTracker {
///     fn log_params(&self, params: &HashMap<String, String>) { /* ... */ }
///     fn log_metrics(&self, metrics: &HashMap<String, f64>, step: Option<i64>) { /* ... */ }
///     fn set_tags(&self, tags: &HashMap<String, String>) { /* ... */ }
///     fn finish(&self) { /* ... */ }
/// }
///
/// let cb = TrackerCallback::new(Box::new(MyTracker));
/// ```
pub struct TrackerCallback {
    tracker: Box<dyn ExperimentTracker>,
}

impl TrackerCallback {
    /// 包装一个实验跟踪器为 Optuna 回调。
    pub fn new(tracker: Box<dyn ExperimentTracker>) -> Self {
        Self { tracker }
    }
}

impl Callback for TrackerCallback {
    fn on_trial_complete(&self, _study: &Study, trial: &FrozenTrial) {
        // 参数
        let params: HashMap<String, String> = trial
            .params
            .iter()
            .map(|(k, v)| (k.clone(), format!("{v:?}")))
            .collect();
        self.tracker.log_params(&params);

        // 指标
        let mut metrics: HashMap<String, f64> = HashMap::new();
        if let Some(ref vals) = trial.values {
            for (i, v) in vals.iter().enumerate() {
                if vals.len() == 1 {
                    metrics.insert("value".into(), *v);
                } else {
                    metrics.insert(format!("value_{i}"), *v);
                }
            }
        }
        if let Some(ref dur) = trial.duration() {
            metrics.insert(
                "duration_s".into(),
                dur.num_milliseconds() as f64 / 1000.0,
            );
        }
        self.tracker
            .log_metrics(&metrics, Some(trial.number));

        // 标签
        let mut tags: HashMap<String, String> = HashMap::new();
        tags.insert("trial_number".into(), trial.number.to_string());
        tags.insert("trial_state".into(), trial.state.to_string());
        self.tracker.set_tags(&tags);
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{StudyDirection, create_study};

    #[test]
    fn test_debug_print_callback() {
        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        let cb = DebugPrintCallback;
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(3),
            None,
            Some(&[&cb]),
        ).unwrap();
    }

    #[test]
    fn test_json_logger_callback() {
        let dir = std::env::temp_dir();
        let path = dir.join("optuna_test_jsonl.jsonl");
        let _ = std::fs::remove_file(&path);

        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        let cb = JsonLoggerCallback::new(&path).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(3),
            None,
            Some(&[&cb]),
        ).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        assert_eq!(content.lines().count(), 3);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_csv_logger_callback() {
        let dir = std::env::temp_dir();
        let path = dir.join("optuna_test_csv.csv");
        let _ = std::fs::remove_file(&path);

        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        let cb = CsvLoggerCallback::new(&path).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(3),
            None,
            Some(&[&cb]),
        ).unwrap();

        let content = std::fs::read_to_string(&path).unwrap();
        // header + 3 data lines
        assert_eq!(content.lines().count(), 4);
        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn test_tensorboard_callback() {
        let dir = tempfile::tempdir().unwrap();
        let cb = TensorBoardCallback::new(dir.path(), Some("loss".into()));

        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(3),
            None,
            Some(&[&cb]),
        ).unwrap();

        // 验证为每个试验创建了子目录
        for i in 0..3 {
            let trial_dir = dir.path().join(format!("trial-{i}"));
            assert!(trial_dir.exists(), "trial-{i} directory should exist");
            // 验证有 event 文件
            let entries: Vec<_> = std::fs::read_dir(&trial_dir)
                .unwrap()
                .filter_map(|e| e.ok())
                .filter(|e| {
                    e.file_name()
                        .to_string_lossy()
                        .starts_with("events.out.tfevents")
                })
                .collect();
            assert!(!entries.is_empty(), "trial-{i} should have event files");
            // 验证有 JSON summary
            let summary = trial_dir.join("trial_summary.json");
            assert!(summary.exists(), "trial-{i} should have trial_summary.json");
            let content = std::fs::read_to_string(&summary).unwrap();
            let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
            assert_eq!(parsed["trial_number"], i);
        }
    }

    #[test]
    fn test_tensorboard_event_file_format() {
        // 验证 TFEvent 二进制格式正确性
        let dir = tempfile::tempdir().unwrap();
        let cb = TensorBoardCallback::new(dir.path(), Some("metric".into()));

        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        study.optimize(
            |trial| {
                let _x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(0.5)
            },
            Some(1),
            None,
            Some(&[&cb]),
        ).unwrap();

        let trial_dir = dir.path().join("trial-0");
        let event_file = std::fs::read_dir(&trial_dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .find(|e| {
                e.file_name()
                    .to_string_lossy()
                    .starts_with("events.out.tfevents")
            })
            .expect("event file should exist");

        let data = std::fs::read(event_file.path()).unwrap();
        // TFRecord: 8-byte length + 4-byte length_crc + data + 4-byte data_crc
        assert!(data.len() > 16, "event file should have at least one record");

        // 验证第一条记录的长度字段
        let record_len = u64::from_le_bytes(data[0..8].try_into().unwrap());
        assert!(record_len > 0 && record_len < 10000, "record length should be reasonable");

        // 验证整体结构: 8 + 4 + record_len + 4
        let total_expected = 8 + 4 + record_len as usize + 4;
        // 可能有多条记录（metric + params），所以 >= 即可
        assert!(data.len() >= total_expected, "file should contain at least one complete record");
    }

    #[test]
    fn test_crc32c() {
        // empty data
        assert_eq!(crc32c(b""), 0x00000000);
        // "123456789" reference vector (RFC 3720 / iSCSI)
        assert_eq!(crc32c(b"123456789"), 0xE3069283);
    }

    #[test]
    fn test_wandb_callback_multirun() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct MockWandb {
            init_count: AtomicUsize,
            log_count: AtomicUsize,
            finish_count: AtomicUsize,
        }
        impl WandbLogger for MockWandb {
            fn init(&self, _project: &str, _name: &str, _config: &serde_json::Value) {
                self.init_count.fetch_add(1, Ordering::Relaxed);
            }
            fn log(&self, _data: &serde_json::Value) {
                self.log_count.fetch_add(1, Ordering::Relaxed);
            }
            fn finish(&self) {
                self.finish_count.fetch_add(1, Ordering::Relaxed);
            }
        }

        let mock = std::sync::Arc::new(MockWandb {
            init_count: AtomicUsize::new(0),
            log_count: AtomicUsize::new(0),
            finish_count: AtomicUsize::new(0),
        });

        // 由于 WandbLogger 需要 Box<dyn WandbLogger>，我们创建一个转发器
        struct ArcWandb(std::sync::Arc<MockWandb>);
        impl WandbLogger for ArcWandb {
            fn init(&self, project: &str, name: &str, config: &serde_json::Value) {
                self.0.init(project, name, config);
            }
            fn log(&self, data: &serde_json::Value) {
                self.0.log(data);
            }
            fn finish(&self) {
                self.0.finish();
            }
        }

        let cb = WandbCallback::new(
            Box::new(ArcWandb(mock.clone())),
            None,
            true, // as_multirun
        );

        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(3),
            None,
            Some(&[&cb]),
        ).unwrap();

        assert_eq!(mock.init_count.load(Ordering::Relaxed), 3, "should init 3 runs");
        assert_eq!(mock.log_count.load(Ordering::Relaxed), 3, "should log 3 times");
        assert_eq!(mock.finish_count.load(Ordering::Relaxed), 3, "should finish 3 runs");
    }

    #[test]
    fn test_wandb_callback_single_run() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct MockWandb {
            init_count: AtomicUsize,
            log_count: AtomicUsize,
        }
        impl WandbLogger for MockWandb {
            fn init(&self, _: &str, _: &str, _: &serde_json::Value) {
                self.init_count.fetch_add(1, Ordering::Relaxed);
            }
            fn log(&self, _: &serde_json::Value) {
                self.log_count.fetch_add(1, Ordering::Relaxed);
            }
            fn finish(&self) {}
        }

        let mock = std::sync::Arc::new(MockWandb {
            init_count: AtomicUsize::new(0),
            log_count: AtomicUsize::new(0),
        });

        struct ArcWandb(std::sync::Arc<MockWandb>);
        impl WandbLogger for ArcWandb {
            fn init(&self, p: &str, n: &str, c: &serde_json::Value) { self.0.init(p, n, c); }
            fn log(&self, d: &serde_json::Value) { self.0.log(d); }
            fn finish(&self) { self.0.finish(); }
        }

        let cb = WandbCallback::new(
            Box::new(ArcWandb(mock.clone())),
            None,
            false, // single run
        );

        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(5),
            None,
            Some(&[&cb]),
        ).unwrap();

        assert_eq!(mock.init_count.load(Ordering::Relaxed), 1, "single-run: should init once");
        assert_eq!(mock.log_count.load(Ordering::Relaxed), 5, "should log 5 times");
    }

    #[test]
    fn test_tracker_callback() {
        use std::sync::atomic::{AtomicUsize, Ordering};

        struct MockTracker {
            params_count: AtomicUsize,
            metrics_count: AtomicUsize,
            tags_count: AtomicUsize,
        }
        impl ExperimentTracker for MockTracker {
            fn log_params(&self, _: &HashMap<String, String>) {
                self.params_count.fetch_add(1, Ordering::Relaxed);
            }
            fn log_metrics(&self, _: &HashMap<String, f64>, _: Option<i64>) {
                self.metrics_count.fetch_add(1, Ordering::Relaxed);
            }
            fn set_tags(&self, _: &HashMap<String, String>) {
                self.tags_count.fetch_add(1, Ordering::Relaxed);
            }
            fn finish(&self) {}
        }

        let tracker = std::sync::Arc::new(MockTracker {
            params_count: AtomicUsize::new(0),
            metrics_count: AtomicUsize::new(0),
            tags_count: AtomicUsize::new(0),
        });

        struct ArcTracker(std::sync::Arc<MockTracker>);
        impl ExperimentTracker for ArcTracker {
            fn log_params(&self, p: &HashMap<String, String>) { self.0.log_params(p); }
            fn log_metrics(&self, m: &HashMap<String, f64>, s: Option<i64>) { self.0.log_metrics(m, s); }
            fn set_tags(&self, t: &HashMap<String, String>) { self.0.set_tags(t); }
            fn finish(&self) { self.0.finish(); }
        }

        let cb = TrackerCallback::new(Box::new(ArcTracker(tracker.clone())));

        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(3),
            None,
            Some(&[&cb]),
        ).unwrap();

        assert_eq!(tracker.params_count.load(Ordering::Relaxed), 3);
        assert_eq!(tracker.metrics_count.load(Ordering::Relaxed), 3);
        assert_eq!(tracker.tags_count.load(Ordering::Relaxed), 3);
    }

    #[test]
    fn test_pruning_mixin() {
        // PruningMixin 主要是一个辅助结构体，测试其创建和基本属性
        let mixin = PruningMixin::new("validation-rmse");
        assert_eq!(mixin.metric_name, "validation-rmse");
    }
}
