//! gRPC 存储代理服务端。
//!
//! 对应 Python `optuna.storages._grpc.server` + `servicer`。

use std::collections::HashMap;
use std::sync::Arc;

use tonic::{Request, Response, Status};

use super::proto::storage_service_server::{StorageService, StorageServiceServer};
use super::proto::*;
use crate::distributions::Distribution;
use crate::error::OptunaError;
use crate::storage::Storage;
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState as RustTrialState};

/// gRPC 服务实现：代理到底层 Storage。
pub struct StorageServiceImpl {
    storage: Arc<dyn Storage>,
}

impl StorageServiceImpl {
    pub fn new(storage: Arc<dyn Storage>) -> Self {
        Self { storage }
    }
}

// ── 辅助转换函数 ──

fn to_study_dir(d: &crate::study::StudyDirection) -> i32 {
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

pub(crate) fn frozen_to_proto(t: &FrozenTrial) -> TrialMessage {
    let params: HashMap<String, f64> = t
        .params
        .iter()
        .map(|(k, v)| {
            let f = match v {
                crate::distributions::ParamValue::Float(f) => *f,
                crate::distributions::ParamValue::Int(i) => *i as f64,
                crate::distributions::ParamValue::Categorical(c) => {
                    match c {
                        crate::distributions::CategoricalChoice::Int(i) => *i as f64,
                        crate::distributions::CategoricalChoice::Float(f) => *f,
                        _ => 0.0,
                    }
                }
            };
            (k.clone(), f)
        })
        .collect();

    let distributions: HashMap<String, String> = t
        .distributions
        .iter()
        .map(|(k, v)| (k.clone(), serde_json::to_string(v).unwrap_or_default()))
        .collect();

    let user_attributes: HashMap<String, String> = t
        .user_attrs
        .iter()
        .map(|(k, v)| (k.clone(), v.to_string()))
        .collect();

    let system_attributes: HashMap<String, String> = t
        .system_attrs
        .iter()
        .map(|(k, v)| (k.clone(), v.to_string()))
        .collect();

    let intermediate_values: HashMap<i64, f64> = t
        .intermediate_values
        .iter()
        .map(|(&k, &v)| (k, v))
        .collect();

    TrialMessage {
        trial_id: t.trial_id,
        number: t.number,
        state: to_trial_state(&t.state),
        values: t.values.clone().unwrap_or_default(),
        values_is_none: t.values.is_none(),
        datetime_start: t
            .datetime_start
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_default(),
        datetime_complete: t
            .datetime_complete
            .map(|dt| dt.to_rfc3339())
            .unwrap_or_default(),
        params,
        distributions,
        user_attributes,
        system_attributes,
        intermediate_values,
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
        .filter_map(|(k, v)| {
            serde_json::from_str(v).ok().map(|d| (k.clone(), d))
        })
        .collect();

    let user_attrs: HashMap<String, serde_json::Value> = m
        .user_attributes
        .iter()
        .filter_map(|(k, v)| {
            serde_json::from_str(v)
                .ok()
                .map(|val| (k.clone(), val))
        })
        .collect();

    let system_attrs: HashMap<String, serde_json::Value> = m
        .system_attributes
        .iter()
        .filter_map(|(k, v)| {
            serde_json::from_str(v)
                .ok()
                .map(|val| (k.clone(), val))
        })
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
        intermediate_values: m
            .intermediate_values
            .iter()
            .map(|(&k, &v)| (k, v))
            .collect(),
        trial_id: m.trial_id,
    }
}

fn optuna_err_to_status(e: OptunaError) -> Status {
    match &e {
        OptunaError::DuplicatedStudyError(_) => Status::already_exists(e.to_string()),
        OptunaError::UpdateFinishedTrialError(_) => {
            Status::failed_precondition(e.to_string())
        }
        OptunaError::ValueError(_) => Status::invalid_argument(e.to_string()),
        OptunaError::TrialPruned => Status::aborted("TrialPruned"),
        _ => Status::internal(e.to_string()),
    }
}

#[tonic::async_trait]
impl StorageService for StorageServiceImpl {
    async fn create_new_study(
        &self,
        request: Request<CreateStudyRequest>,
    ) -> std::result::Result<Response<CreateStudyReply>, Status> {
        let req = request.into_inner();
        let dirs: Vec<StudyDirection> = req.directions.iter().map(|&d| from_study_dir(d)).collect();
        let name = if req.study_name_is_none {
            None
        } else {
            Some(req.study_name.as_str())
        };
        let id = self
            .storage
            .create_new_study(&dirs, name)
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(CreateStudyReply { study_id: id }))
    }

    async fn delete_study(
        &self,
        request: Request<DeleteStudyRequest>,
    ) -> std::result::Result<Response<Empty>, Status> {
        let id = request.into_inner().study_id;
        self.storage.delete_study(id).map_err(optuna_err_to_status)?;
        Ok(Response::new(Empty {}))
    }

    async fn set_study_user_attribute(
        &self,
        request: Request<SetStudyAttributeRequest>,
    ) -> std::result::Result<Response<Empty>, Status> {
        let req = request.into_inner();
        let value: serde_json::Value =
            serde_json::from_str(&req.value_json).map_err(|e| Status::invalid_argument(e.to_string()))?;
        self.storage
            .set_study_user_attr(req.study_id, &req.key, value)
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(Empty {}))
    }

    async fn set_study_system_attribute(
        &self,
        request: Request<SetStudyAttributeRequest>,
    ) -> std::result::Result<Response<Empty>, Status> {
        let req = request.into_inner();
        let value: serde_json::Value =
            serde_json::from_str(&req.value_json).map_err(|e| Status::invalid_argument(e.to_string()))?;
        self.storage
            .set_study_system_attr(req.study_id, &req.key, value)
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(Empty {}))
    }

    async fn get_study_id_from_name(
        &self,
        request: Request<GetStudyIdFromNameRequest>,
    ) -> std::result::Result<Response<StudyIdReply>, Status> {
        let name = request.into_inner().study_name;
        let id = self
            .storage
            .get_study_id_from_name(&name)
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(StudyIdReply { study_id: id }))
    }

    async fn get_study_name_from_id(
        &self,
        request: Request<StudyIdRequest>,
    ) -> std::result::Result<Response<StudyNameReply>, Status> {
        let id = request.into_inner().study_id;
        let name = self
            .storage
            .get_study_name_from_id(id)
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(StudyNameReply { study_name: name }))
    }

    async fn get_study_directions(
        &self,
        request: Request<StudyIdRequest>,
    ) -> std::result::Result<Response<StudyDirectionsReply>, Status> {
        let id = request.into_inner().study_id;
        let dirs = self
            .storage
            .get_study_directions(id)
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(StudyDirectionsReply {
            directions: dirs.iter().map(to_study_dir).collect(),
        }))
    }

    async fn get_study_user_attributes(
        &self,
        request: Request<StudyIdRequest>,
    ) -> std::result::Result<Response<AttributesReply>, Status> {
        let id = request.into_inner().study_id;
        let attrs = self
            .storage
            .get_study_user_attrs(id)
            .map_err(optuna_err_to_status)?;
        let map: HashMap<String, String> = attrs
            .into_iter()
            .map(|(k, v)| (k, v.to_string()))
            .collect();
        Ok(Response::new(AttributesReply { attributes: map }))
    }

    async fn get_study_system_attributes(
        &self,
        request: Request<StudyIdRequest>,
    ) -> std::result::Result<Response<AttributesReply>, Status> {
        let id = request.into_inner().study_id;
        let attrs = self
            .storage
            .get_study_system_attrs(id)
            .map_err(optuna_err_to_status)?;
        let map: HashMap<String, String> = attrs
            .into_iter()
            .map(|(k, v)| (k, v.to_string()))
            .collect();
        Ok(Response::new(AttributesReply { attributes: map }))
    }

    async fn get_all_studies(
        &self,
        _request: Request<Empty>,
    ) -> std::result::Result<Response<GetAllStudiesReply>, Status> {
        let studies = self
            .storage
            .get_all_studies()
            .map_err(optuna_err_to_status)?;
        let messages: Vec<StudyMessage> = studies
            .iter()
            .map(|s| StudyMessage {
                study_id: s.study_id,
                study_name: s.study_name.clone(),
                directions: s.directions.iter().map(to_study_dir).collect(),
                user_attributes: s
                    .user_attrs
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_string()))
                    .collect(),
                system_attributes: s
                    .system_attrs
                    .iter()
                    .map(|(k, v)| (k.clone(), v.to_string()))
                    .collect(),
            })
            .collect();
        Ok(Response::new(GetAllStudiesReply { studies: messages }))
    }

    async fn create_new_trial(
        &self,
        request: Request<CreateTrialRequest>,
    ) -> std::result::Result<Response<TrialIdReply>, Status> {
        let req = request.into_inner();
        let template = if req.template_trial_is_none {
            None
        } else {
            req.template_trial.as_ref().map(proto_to_frozen)
        };
        let id = self
            .storage
            .create_new_trial(req.study_id, template.as_ref())
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(TrialIdReply { trial_id: id }))
    }

    async fn set_trial_parameter(
        &self,
        request: Request<SetTrialParameterRequest>,
    ) -> std::result::Result<Response<Empty>, Status> {
        let req = request.into_inner();
        let dist: Distribution = serde_json::from_str(&req.distribution_json)
            .map_err(|e| Status::invalid_argument(format!("无效分布 JSON: {e}")))?;
        self.storage
            .set_trial_param(req.trial_id, &req.param_name, req.param_value_internal, &dist)
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(Empty {}))
    }

    async fn get_trial_id_from_study_id_trial_number(
        &self,
        request: Request<GetTrialIdRequest>,
    ) -> std::result::Result<Response<TrialIdReply>, Status> {
        let req = request.into_inner();
        let id = self
            .storage
            .get_trial_id_from_study_id_trial_number(req.study_id, req.trial_number)
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(TrialIdReply { trial_id: id }))
    }

    async fn set_trial_state_values(
        &self,
        request: Request<SetTrialStateValuesRequest>,
    ) -> std::result::Result<Response<TrialUpdatedReply>, Status> {
        let req = request.into_inner();
        let state = from_trial_state(req.state);
        let values = if req.values_is_none {
            None
        } else {
            Some(req.values)
        };
        let updated = self
            .storage
            .set_trial_state_values(req.trial_id, state, values.as_deref())
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(TrialUpdatedReply {
            trial_updated: updated,
        }))
    }

    async fn set_trial_intermediate_value(
        &self,
        request: Request<SetTrialIntermediateValueRequest>,
    ) -> std::result::Result<Response<Empty>, Status> {
        let req = request.into_inner();
        self.storage
            .set_trial_intermediate_value(req.trial_id, req.step, req.intermediate_value)
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(Empty {}))
    }

    async fn set_trial_user_attribute(
        &self,
        request: Request<SetTrialAttributeRequest>,
    ) -> std::result::Result<Response<Empty>, Status> {
        let req = request.into_inner();
        let value: serde_json::Value =
            serde_json::from_str(&req.value_json).map_err(|e| Status::invalid_argument(e.to_string()))?;
        self.storage
            .set_trial_user_attr(req.trial_id, &req.key, value)
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(Empty {}))
    }

    async fn set_trial_system_attribute(
        &self,
        request: Request<SetTrialAttributeRequest>,
    ) -> std::result::Result<Response<Empty>, Status> {
        let req = request.into_inner();
        let value: serde_json::Value =
            serde_json::from_str(&req.value_json).map_err(|e| Status::invalid_argument(e.to_string()))?;
        self.storage
            .set_trial_system_attr(req.trial_id, &req.key, value)
            .map_err(optuna_err_to_status)?;
        Ok(Response::new(Empty {}))
    }

    async fn get_trial(
        &self,
        request: Request<TrialIdRequest>,
    ) -> std::result::Result<Response<TrialReply>, Status> {
        let id = request.into_inner().trial_id;
        let trial = self.storage.get_trial(id).map_err(optuna_err_to_status)?;
        Ok(Response::new(TrialReply {
            trial: Some(frozen_to_proto(&trial)),
        }))
    }

    async fn get_all_trials(
        &self,
        request: Request<GetAllTrialsRequest>,
    ) -> std::result::Result<Response<GetAllTrialsReply>, Status> {
        let req = request.into_inner();
        let states = if req.states_is_none {
            None
        } else {
            Some(
                req.states
                    .iter()
                    .map(|&s| from_trial_state(s))
                    .collect::<Vec<_>>(),
            )
        };
        let mut trials = self
            .storage
            .get_all_trials(req.study_id, states.as_deref())
            .map_err(optuna_err_to_status)?;

        // 增量同步: trial_id_greater_than 过滤
        if req.use_trial_id_filter {
            trials.retain(|t| t.trial_id > req.trial_id_greater_than);
        }

        // 增量同步: included_trial_ids 过滤
        if !req.included_trial_ids.is_empty() {
            let ids: std::collections::HashSet<i64> =
                req.included_trial_ids.iter().copied().collect();
            trials.retain(|t| ids.contains(&t.trial_id));
        }

        Ok(Response::new(GetAllTrialsReply {
            trials: trials.iter().map(frozen_to_proto).collect(),
        }))
    }
}

/// 启动 gRPC 代理服务器（阻塞）。
///
/// 对应 Python `optuna.storages.run_grpc_proxy_server()`。
///
/// # 参数
/// * `storage` - 底层存储后端
/// * `host` - 监听地址（默认 "127.0.0.1"）
/// * `port` - 监听端口（默认 13000）
pub async fn run_grpc_proxy_server(
    storage: Arc<dyn Storage>,
    host: &str,
    port: u16,
) -> std::result::Result<(), Box<dyn std::error::Error>> {
    let addr = format!("{host}:{port}").parse()?;
    let service = StorageServiceImpl::new(storage);

    tonic::transport::Server::builder()
        .add_service(StorageServiceServer::new(service))
        .serve(addr)
        .await?;

    Ok(())
}
