//! gRPC 存储代理客户端。
//!
//! 对应 Python `optuna.storages.GrpcStorageProxy`。
//! 实现 `Storage` trait，通过 gRPC 连接远程存储。

use std::collections::HashMap;

use parking_lot::Mutex;

use super::proto::storage_service_client::StorageServiceClient;
use super::proto::*;
use crate::distributions::Distribution;
use crate::error::{OptunaError, Result};
use crate::storage::Storage;
use crate::study::{FrozenStudy, StudyDirection};
use crate::trial::{FrozenTrial, TrialState as RustTrialState};

/// gRPC 存储代理客户端。
///
/// 对应 Python `optuna.storages.GrpcStorageProxy`。
/// 通过 gRPC 连接远程存储服务器，实现 `Storage` trait。
///
/// # 示例
/// ```ignore
/// use optuna_rs::storage::grpc::GrpcStorageProxy;
///
/// let proxy = GrpcStorageProxy::new("http://localhost:13000").unwrap();
/// ```
pub struct GrpcStorageProxy {
    client: Mutex<StorageServiceClient<tonic::transport::Channel>>,
    rt: tokio::runtime::Runtime,
}

impl GrpcStorageProxy {
    /// 创建 gRPC 代理。
    pub fn new(endpoint: &str) -> Result<Self> {
        let rt = tokio::runtime::Runtime::new()
            .map_err(|e| OptunaError::StorageInternalError(format!("创建 tokio runtime 失败: {e}")))?;

        let client = rt.block_on(async {
            StorageServiceClient::connect(endpoint.to_string()).await
        }).map_err(|e| OptunaError::StorageInternalError(format!("gRPC 连接失败: {e}")))?;

        Ok(Self {
            client: Mutex::new(client),
            rt,
        })
    }
}

// ── 辅助转换 ──

fn to_study_dir(d: &StudyDirection) -> i32 {
    match d {
        StudyDirection::NotSet => 0,
        StudyDirection::Minimize => 1,
        StudyDirection::Maximize => 2,
    }
}

fn from_study_dir(d: i32) -> StudyDirection {
    match d {
        1 => StudyDirection::Minimize,
        2 => StudyDirection::Maximize,
        _ => StudyDirection::NotSet,
    }
}

fn to_trial_state(s: &RustTrialState) -> i32 {
    match s {
        RustTrialState::Running => 0,
        RustTrialState::Complete => 1,
        RustTrialState::Pruned => 2,
        RustTrialState::Fail => 3,
        RustTrialState::Waiting => 4,
    }
}

fn from_trial_state(s: i32) -> RustTrialState {
    match s {
        0 => RustTrialState::Running,
        1 => RustTrialState::Complete,
        2 => RustTrialState::Pruned,
        3 => RustTrialState::Fail,
        4 => RustTrialState::Waiting,
        _ => RustTrialState::Running,
    }
}

fn status_to_err(s: tonic::Status) -> OptunaError {
    match s.code() {
        tonic::Code::AlreadyExists => OptunaError::DuplicatedStudyError(s.message().to_string()),
        tonic::Code::FailedPrecondition => {
            OptunaError::UpdateFinishedTrialError(s.message().to_string())
        }
        tonic::Code::InvalidArgument => OptunaError::ValueError(s.message().to_string()),
        _ => OptunaError::StorageInternalError(s.message().to_string()),
    }
}

fn proto_to_frozen(m: &TrialMessage) -> FrozenTrial {
    use chrono::DateTime;

    let params: HashMap<String, crate::distributions::ParamValue> = m
        .params
        .iter()
        .map(|(k, &v)| (k.clone(), crate::distributions::ParamValue::Float(v)))
        .collect();

    let distributions: HashMap<String, Distribution> = m
        .distributions
        .iter()
        .filter_map(|(k, v)| serde_json::from_str(v).ok().map(|d| (k.clone(), d)))
        .collect();

    let user_attrs: HashMap<String, serde_json::Value> = m
        .user_attributes
        .iter()
        .filter_map(|(k, v)| serde_json::from_str(v).ok().map(|val| (k.clone(), val)))
        .collect();

    let system_attrs: HashMap<String, serde_json::Value> = m
        .system_attributes
        .iter()
        .filter_map(|(k, v)| serde_json::from_str(v).ok().map(|val| (k.clone(), val)))
        .collect();

    FrozenTrial {
        number: m.number,
        state: from_trial_state(m.state),
        values: if m.values_is_none {
            None
        } else {
            Some(m.values.clone())
        },
        datetime_start: DateTime::parse_from_rfc3339(&m.datetime_start)
            .ok()
            .map(|dt| dt.with_timezone(&chrono::Utc)),
        datetime_complete: DateTime::parse_from_rfc3339(&m.datetime_complete)
            .ok()
            .map(|dt| dt.with_timezone(&chrono::Utc)),
        params,
        distributions,
        user_attrs,
        system_attrs,
        intermediate_values: m.intermediate_values.iter().map(|(&k, &v)| (k, v)).collect(),
        trial_id: m.trial_id,
    }
}

impl Storage for GrpcStorageProxy {
    fn create_new_study(
        &self,
        directions: &[StudyDirection],
        study_name: Option<&str>,
    ) -> Result<i64> {
        let req = CreateStudyRequest {
            directions: directions.iter().map(to_study_dir).collect(),
            study_name: study_name.unwrap_or_default().to_string(),
            study_name_is_none: study_name.is_none(),
        };
        let resp = self
            .rt
            .block_on(self.client.lock().create_new_study(req))
            .map_err(status_to_err)?;
        Ok(resp.into_inner().study_id)
    }

    fn delete_study(&self, study_id: i64) -> Result<()> {
        let req = DeleteStudyRequest { study_id };
        self.rt
            .block_on(self.client.lock().delete_study(req))
            .map_err(status_to_err)?;
        Ok(())
    }

    fn set_study_user_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let req = SetStudyAttributeRequest {
            study_id,
            key: key.to_string(),
            value_json: serde_json::to_string(&value).unwrap_or_default(),
        };
        self.rt
            .block_on(self.client.lock().set_study_user_attribute(req))
            .map_err(status_to_err)?;
        Ok(())
    }

    fn set_study_system_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let req = SetStudyAttributeRequest {
            study_id,
            key: key.to_string(),
            value_json: serde_json::to_string(&value).unwrap_or_default(),
        };
        self.rt
            .block_on(self.client.lock().set_study_system_attribute(req))
            .map_err(status_to_err)?;
        Ok(())
    }

    fn get_study_id_from_name(&self, study_name: &str) -> Result<i64> {
        let req = GetStudyIdFromNameRequest {
            study_name: study_name.to_string(),
        };
        let resp = self
            .rt
            .block_on(self.client.lock().get_study_id_from_name(req))
            .map_err(status_to_err)?;
        Ok(resp.into_inner().study_id)
    }

    fn get_study_name_from_id(&self, study_id: i64) -> Result<String> {
        let req = StudyIdRequest { study_id };
        let resp = self
            .rt
            .block_on(self.client.lock().get_study_name_from_id(req))
            .map_err(status_to_err)?;
        Ok(resp.into_inner().study_name)
    }

    fn get_study_directions(&self, study_id: i64) -> Result<Vec<StudyDirection>> {
        let req = StudyIdRequest { study_id };
        let resp = self
            .rt
            .block_on(self.client.lock().get_study_directions(req))
            .map_err(status_to_err)?;
        Ok(resp
            .into_inner()
            .directions
            .iter()
            .map(|&d| from_study_dir(d))
            .collect())
    }

    fn get_study_user_attrs(
        &self,
        study_id: i64,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let req = StudyIdRequest { study_id };
        let resp = self
            .rt
            .block_on(self.client.lock().get_study_user_attributes(req))
            .map_err(status_to_err)?;
        Ok(resp
            .into_inner()
            .attributes
            .into_iter()
            .map(|(k, v)| {
                let val = serde_json::from_str(&v).unwrap_or(serde_json::Value::String(v));
                (k, val)
            })
            .collect())
    }

    fn get_study_system_attrs(
        &self,
        study_id: i64,
    ) -> Result<HashMap<String, serde_json::Value>> {
        let req = StudyIdRequest { study_id };
        let resp = self
            .rt
            .block_on(self.client.lock().get_study_system_attributes(req))
            .map_err(status_to_err)?;
        Ok(resp
            .into_inner()
            .attributes
            .into_iter()
            .map(|(k, v)| {
                let val = serde_json::from_str(&v).unwrap_or(serde_json::Value::String(v));
                (k, val)
            })
            .collect())
    }

    fn get_all_studies(&self) -> Result<Vec<FrozenStudy>> {
        let resp = self
            .rt
            .block_on(self.client.lock().get_all_studies(Empty {}))
            .map_err(status_to_err)?;
        Ok(resp
            .into_inner()
            .studies
            .iter()
            .map(|s| {
                let user_attrs: HashMap<String, serde_json::Value> = s
                    .user_attributes
                    .iter()
                    .map(|(k, v)| {
                        let val =
                            serde_json::from_str(v).unwrap_or(serde_json::Value::String(v.clone()));
                        (k.clone(), val)
                    })
                    .collect();
                let system_attrs: HashMap<String, serde_json::Value> = s
                    .system_attributes
                    .iter()
                    .map(|(k, v)| {
                        let val =
                            serde_json::from_str(v).unwrap_or(serde_json::Value::String(v.clone()));
                        (k.clone(), val)
                    })
                    .collect();
                FrozenStudy {
                    study_id: s.study_id,
                    study_name: s.study_name.clone(),
                    directions: s.directions.iter().map(|&d| from_study_dir(d)).collect(),
                    user_attrs,
                    system_attrs,
                }
            })
            .collect())
    }

    fn create_new_trial(
        &self,
        study_id: i64,
        template_trial: Option<&FrozenTrial>,
    ) -> Result<i64> {
        let req = CreateTrialRequest {
            study_id,
            template_trial: template_trial.map(|t| {
                super::server::frozen_to_proto(t)
            }),
            template_trial_is_none: template_trial.is_none(),
        };
        let resp = self
            .rt
            .block_on(self.client.lock().create_new_trial(req))
            .map_err(status_to_err)?;
        Ok(resp.into_inner().trial_id)
    }

    fn set_trial_param(
        &self,
        trial_id: i64,
        param_name: &str,
        param_value_internal: f64,
        distribution: &Distribution,
    ) -> Result<()> {
        let req = SetTrialParameterRequest {
            trial_id,
            param_name: param_name.to_string(),
            param_value_internal,
            distribution_json: serde_json::to_string(distribution).unwrap_or_default(),
        };
        self.rt
            .block_on(self.client.lock().set_trial_parameter(req))
            .map_err(status_to_err)?;
        Ok(())
    }

    fn set_trial_state_values(
        &self,
        trial_id: i64,
        state: RustTrialState,
        values: Option<&[f64]>,
    ) -> Result<bool> {
        let req = SetTrialStateValuesRequest {
            trial_id,
            state: to_trial_state(&state),
            values: values.unwrap_or_default().to_vec(),
            values_is_none: values.is_none(),
        };
        let resp = self
            .rt
            .block_on(self.client.lock().set_trial_state_values(req))
            .map_err(status_to_err)?;
        Ok(resp.into_inner().trial_updated)
    }

    fn set_trial_intermediate_value(
        &self,
        trial_id: i64,
        step: i64,
        intermediate_value: f64,
    ) -> Result<()> {
        let req = SetTrialIntermediateValueRequest {
            trial_id,
            step,
            intermediate_value,
        };
        self.rt
            .block_on(
                self.client
                    .lock()
                    .set_trial_intermediate_value(req),
            )
            .map_err(status_to_err)?;
        Ok(())
    }

    fn set_trial_user_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let req = SetTrialAttributeRequest {
            trial_id,
            key: key.to_string(),
            value_json: serde_json::to_string(&value).unwrap_or_default(),
        };
        self.rt
            .block_on(self.client.lock().set_trial_user_attribute(req))
            .map_err(status_to_err)?;
        Ok(())
    }

    fn set_trial_system_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let req = SetTrialAttributeRequest {
            trial_id,
            key: key.to_string(),
            value_json: serde_json::to_string(&value).unwrap_or_default(),
        };
        self.rt
            .block_on(self.client.lock().set_trial_system_attribute(req))
            .map_err(status_to_err)?;
        Ok(())
    }

    fn get_trial(&self, trial_id: i64) -> Result<FrozenTrial> {
        let req = TrialIdRequest { trial_id };
        let resp = self
            .rt
            .block_on(self.client.lock().get_trial(req))
            .map_err(status_to_err)?;
        let msg = resp
            .into_inner()
            .trial
            .ok_or_else(|| OptunaError::StorageInternalError("空 trial 响应".to_string()))?;
        Ok(proto_to_frozen(&msg))
    }

    fn get_all_trials(
        &self,
        study_id: i64,
        states: Option<&[RustTrialState]>,
    ) -> Result<Vec<FrozenTrial>> {
        let req = GetAllTrialsRequest {
            study_id,
            states: states
                .unwrap_or_default()
                .iter()
                .map(to_trial_state)
                .collect(),
            states_is_none: states.is_none(),
            included_trial_ids: vec![],
            trial_id_greater_than: 0,
            use_trial_id_filter: false,
        };
        let resp = self
            .rt
            .block_on(self.client.lock().get_all_trials(req))
            .map_err(status_to_err)?;
        Ok(resp
            .into_inner()
            .trials
            .iter()
            .map(proto_to_frozen)
            .collect())
    }
}

impl GrpcStorageProxy {
    /// 增量获取 trial: 只返回 trial_id > `trial_id_greater_than` 的试验。
    /// 对应 Python gRPC 客户端的增量同步机制。
    pub fn get_trials_incremental(
        &self,
        study_id: i64,
        states: Option<&[RustTrialState]>,
        trial_id_greater_than: i64,
    ) -> Result<Vec<FrozenTrial>> {
        let req = GetAllTrialsRequest {
            study_id,
            states: states
                .unwrap_or_default()
                .iter()
                .map(to_trial_state)
                .collect(),
            states_is_none: states.is_none(),
            included_trial_ids: vec![],
            trial_id_greater_than,
            use_trial_id_filter: true,
        };
        let resp = self
            .rt
            .block_on(self.client.lock().get_all_trials(req))
            .map_err(status_to_err)?;
        Ok(resp
            .into_inner()
            .trials
            .iter()
            .map(proto_to_frozen)
            .collect())
    }

    /// 按 trial_id 列表获取试验。
    /// 对应 Python `_CachedStorage` 的 `included_trial_ids` 用法。
    pub fn get_trials_by_ids(
        &self,
        study_id: i64,
        trial_ids: &[i64],
    ) -> Result<Vec<FrozenTrial>> {
        let req = GetAllTrialsRequest {
            study_id,
            states: vec![],
            states_is_none: true,
            included_trial_ids: trial_ids.to_vec(),
            trial_id_greater_than: 0,
            use_trial_id_filter: false,
        };
        let resp = self
            .rt
            .block_on(self.client.lock().get_all_trials(req))
            .map_err(status_to_err)?;
        Ok(resp
            .into_inner()
            .trials
            .iter()
            .map(proto_to_frozen)
            .collect())
    }
}

// Safety: tonic channels are thread-safe
unsafe impl Send for GrpcStorageProxy {}
unsafe impl Sync for GrpcStorageProxy {}
