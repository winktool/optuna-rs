use std::collections::HashMap;

use chrono::Utc;
use parking_lot::Mutex;
use uuid::Uuid;

use crate::distributions::Distribution;
use crate::error::{OptunaError, Result};
use crate::storage::Storage;
use crate::study::{FrozenStudy, StudyDirection};
use crate::trial::{FrozenTrial, TrialState};

/// Per-study metadata and trial storage.
#[derive(Debug, Clone)]
struct StudyInfo {
    name: String,
    directions: Vec<StudyDirection>,
    trials: Vec<FrozenTrial>,
    param_distribution: HashMap<String, Distribution>,
    user_attrs: HashMap<String, serde_json::Value>,
    system_attrs: HashMap<String, serde_json::Value>,
    best_trial_id: Option<i64>,
}

/// The mutable inner state behind the lock.
#[derive(Debug)]
struct Inner {
    studies: HashMap<i64, StudyInfo>,
    study_name_to_id: HashMap<String, i64>,
    trial_id_to_study_id_and_number: HashMap<i64, (i64, i64)>,
    max_study_id: i64,
    max_trial_id: i64,
}

/// Thread-safe in-memory storage.
///
/// Corresponds to Python `optuna.storages.InMemoryStorage`.
///
/// All state is behind a single `Mutex`. Reads and writes are serialized.
pub struct InMemoryStorage {
    inner: Mutex<Inner>,
}

impl std::fmt::Debug for InMemoryStorage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("InMemoryStorage").finish()
    }
}

impl Default for InMemoryStorage {
    fn default() -> Self {
        Self::new()
    }
}

impl InMemoryStorage {
    /// Create a new empty in-memory storage.
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                studies: HashMap::new(),
                study_name_to_id: HashMap::new(),
                trial_id_to_study_id_and_number: HashMap::new(),
                max_study_id: -1,
                max_trial_id: -1,
            }),
        }
    }

    /// Get the study info, or error if not found.
    fn get_study(inner: &Inner, study_id: i64) -> Result<&StudyInfo> {
        inner.studies.get(&study_id).ok_or_else(|| {
            OptunaError::ValueError(format!("study {study_id} not found"))
        })
    }

    /// Get the study info mutably, or error if not found.
    fn get_study_mut(inner: &mut Inner, study_id: i64) -> Result<&mut StudyInfo> {
        inner.studies.get_mut(&study_id).ok_or_else(|| {
            OptunaError::ValueError(format!("study {study_id} not found"))
        })
    }

    /// Resolve trial_id to (study_id, trial_number).
    fn resolve_trial(inner: &Inner, trial_id: i64) -> Result<(i64, i64)> {
        inner
            .trial_id_to_study_id_and_number
            .get(&trial_id)
            .copied()
            .ok_or_else(|| OptunaError::ValueError(format!("trial {trial_id} not found")))
    }

    /// Check updatability inline (avoids re-acquiring the lock).
    fn check_updatable(trial: &FrozenTrial) -> Result<()> {
        if trial.state.is_finished() {
            return Err(OptunaError::UpdateFinishedTrialError(format!(
                "trial #{} has already finished and cannot be updated",
                trial.number
            )));
        }
        Ok(())
    }

    /// Update the best_trial_id cache for single-objective studies.
    fn update_cache(study: &mut StudyInfo, trial_id: i64) {
        // Only cache for single-objective COMPLETE trials
        if study.directions.len() != 1 {
            return;
        }
        let trial_number = study
            .trials
            .iter()
            .position(|t| t.trial_id == trial_id);
        let trial_number = match trial_number {
            Some(n) => n,
            None => return,
        };
        let trial = &study.trials[trial_number];
        if trial.state != TrialState::Complete {
            return;
        }

        let new_value = match trial.value() {
            Ok(Some(v)) => v,
            _ => return,
        };

        let direction = study.directions[0];
        // 对齐 Python _CachedStorage._update_cache: NaN 值的处理
        let is_better = if let Some(best_id) = study.best_trial_id {
            let best_trial = study.trials.iter().find(|t| t.trial_id == best_id);
            match best_trial.and_then(|t| t.value().ok().flatten()) {
                Some(best_value) => {
                    if best_value.is_nan() {
                        // 当前 best 是 NaN，任何非 NaN 值都更好
                        true
                    } else if new_value.is_nan() {
                        // 新值是 NaN，不替换有效的 best
                        false
                    } else {
                        match direction {
                            StudyDirection::Minimize | StudyDirection::NotSet => new_value < best_value,
                            StudyDirection::Maximize => new_value > best_value,
                        }
                    }
                }
                None => true,
            }
        } else {
            true
        };

        if is_better {
            study.best_trial_id = Some(trial_id);
        }
    }
}

impl Storage for InMemoryStorage {
    fn create_new_study(
        &self,
        directions: &[StudyDirection],
        study_name: Option<&str>,
    ) -> Result<i64> {
        let mut inner = self.inner.lock();
        inner.max_study_id += 1;
        let study_id = inner.max_study_id;

        let name = match study_name {
            Some(n) => {
                if inner.study_name_to_id.contains_key(n) {
                    return Err(OptunaError::DuplicatedStudyError(n.to_string()));
                }
                n.to_string()
            }
            // 对齐 Python: 使用 UUID4 保证全局唯一性
            // Python: study_name = "no-name-" + str(uuid.uuid4())
            None => format!("no-name-{}", Uuid::new_v4()),
        };

        inner.study_name_to_id.insert(name.clone(), study_id);
        inner.studies.insert(
            study_id,
            StudyInfo {
                name,
                directions: directions.to_vec(),
                trials: Vec::new(),
                param_distribution: HashMap::new(),
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                best_trial_id: None,
            },
        );
        Ok(study_id)
    }

    fn delete_study(&self, study_id: i64) -> Result<()> {
        let mut inner = self.inner.lock();
        let study = inner
            .studies
            .remove(&study_id)
            .ok_or_else(|| OptunaError::ValueError(format!("study {study_id} not found")))?;
        inner.study_name_to_id.remove(&study.name);

        // 对齐 Python: 直接从 study.trials 获取 trial_id 删除映射, O(n_trials)
        for trial in &study.trials {
            inner.trial_id_to_study_id_and_number.remove(&trial.trial_id);
        }
        Ok(())
    }

    fn set_study_user_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let study = Self::get_study_mut(&mut inner, study_id)?;
        study.user_attrs.insert(key.to_string(), value);
        Ok(())
    }

    fn set_study_system_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let study = Self::get_study_mut(&mut inner, study_id)?;
        study.system_attrs.insert(key.to_string(), value);
        Ok(())
    }

    fn get_study_id_from_name(&self, study_name: &str) -> Result<i64> {
        let inner = self.inner.lock();
        inner
            .study_name_to_id
            .get(study_name)
            .copied()
            .ok_or_else(|| {
                OptunaError::ValueError(format!("study '{study_name}' not found"))
            })
    }

    fn get_study_name_from_id(&self, study_id: i64) -> Result<String> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;
        Ok(study.name.clone())
    }

    fn get_study_directions(&self, study_id: i64) -> Result<Vec<StudyDirection>> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;
        Ok(study.directions.clone())
    }

    fn get_study_user_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;
        Ok(study.user_attrs.clone())
    }

    fn get_study_system_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;
        Ok(study.system_attrs.clone())
    }

    fn get_all_studies(&self) -> Result<Vec<FrozenStudy>> {
        let inner = self.inner.lock();
        // 对齐 Python: 返回值按 study_id 排序 (Python dict 保持插入顺序等价)
        let mut studies: Vec<FrozenStudy> = inner
            .studies
            .iter()
            .map(|(study_id, info)| FrozenStudy {
                study_id: *study_id,
                study_name: info.name.clone(),
                directions: info.directions.clone(),
                user_attrs: info.user_attrs.clone(),
                system_attrs: info.system_attrs.clone(),
            })
            .collect();
        studies.sort_by_key(|s| s.study_id);
        Ok(studies)
    }

    fn create_new_trial(
        &self,
        study_id: i64,
        template_trial: Option<&FrozenTrial>,
    ) -> Result<i64> {
        let mut inner = self.inner.lock();

        // Get study and compute trial number
        let study = Self::get_study(&inner, study_id)?;
        let trial_number = study.trials.len() as i64;

        inner.max_trial_id += 1;
        let trial_id = inner.max_trial_id;

        let trial = if let Some(template) = template_trial {
            let mut t = template.clone();
            t.trial_id = trial_id;
            t.number = trial_number;
            t
        } else {
            // Create a fresh RUNNING trial
            FrozenTrial {
                number: trial_number,
                state: TrialState::Running,
                values: None,
                datetime_start: Some(Utc::now()),
                datetime_complete: None,
                params: HashMap::new(),
                distributions: HashMap::new(),
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
                trial_id,
            }
        };

        inner
            .trial_id_to_study_id_and_number
            .insert(trial_id, (study_id, trial_number));

        let study = Self::get_study_mut(&mut inner, study_id)?;
        study.trials.push(trial);
        Self::update_cache(study, trial_id);

        Ok(trial_id)
    }

    fn set_trial_param(
        &self,
        trial_id: i64,
        param_name: &str,
        param_value_internal: f64,
        distribution: &Distribution,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;

        let study = Self::get_study(&inner, study_id)?;
        let trial = &study.trials[trial_number as usize];
        Self::check_updatable(trial)?;

        // 对齐 Python check_distribution_compatibility:
        // 同类型同 log 配置允许不同 range（仅警告），CategoricalDistribution 必须完全相同
        if let Some(existing) = study.param_distribution.get(param_name) {
            // 不同分布类型 → 报错
            let same_kind = matches!(
                (existing, distribution),
                (
                    Distribution::FloatDistribution(_),
                    Distribution::FloatDistribution(_)
                ) | (
                    Distribution::IntDistribution(_),
                    Distribution::IntDistribution(_)
                ) | (
                    Distribution::CategoricalDistribution(_),
                    Distribution::CategoricalDistribution(_)
                )
            );
            if !same_kind {
                return Err(OptunaError::ValueError(format!(
                    "Cannot set different distribution kind to the same parameter name '{param_name}'."
                )));
            }
            // Float/Int: log 配置必须一致; Categorical 必须完全相同
            match (existing, distribution) {
                (
                    Distribution::FloatDistribution(old),
                    Distribution::FloatDistribution(new),
                ) => {
                    if old.log != new.log {
                        return Err(OptunaError::ValueError(format!(
                            "Cannot set different log configuration to the same parameter name '{param_name}'."
                        )));
                    }
                }
                (
                    Distribution::IntDistribution(old),
                    Distribution::IntDistribution(new),
                ) => {
                    if old.log != new.log {
                        return Err(OptunaError::ValueError(format!(
                            "Cannot set different log configuration to the same parameter name '{param_name}'."
                        )));
                    }
                }
                (
                    Distribution::CategoricalDistribution(_),
                    Distribution::CategoricalDistribution(_),
                ) => {
                    if existing != distribution {
                        return Err(OptunaError::ValueError(
                            "CategoricalDistribution does not support dynamic value space."
                                .to_string(),
                        ));
                    }
                }
                _ => {}
            }
        }

        let external_value = distribution.to_external_repr(param_value_internal)?;

        let study = Self::get_study_mut(&mut inner, study_id)?;
        study
            .param_distribution
            .insert(param_name.to_string(), distribution.clone());

        let trial = &mut study.trials[trial_number as usize];
        trial.params.insert(param_name.to_string(), external_value);
        trial
            .distributions
            .insert(param_name.to_string(), distribution.clone());

        Ok(())
    }

    fn set_trial_state_values(
        &self,
        trial_id: i64,
        state: TrialState,
        values: Option<&[f64]>,
    ) -> Result<bool> {
        let mut inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;

        let study = Self::get_study(&inner, study_id)?;
        let trial = &study.trials[trial_number as usize];

        // 对齐 Python: 先检查是否已完成（报错），再检查 RUNNING→非WAITING（返回 false）
        Self::check_updatable(trial)?;

        if state == TrialState::Running && trial.state != TrialState::Waiting {
            return Ok(false);
        }

        let study = Self::get_study_mut(&mut inner, study_id)?;
        let trial = &mut study.trials[trial_number as usize];

        // Set timestamps
        if state == TrialState::Running {
            trial.datetime_start = Some(Utc::now());
        }
        if state.is_finished() {
            trial.datetime_complete = Some(Utc::now());
        }

        trial.state = state;
        // 对齐 Python: 仅在 values 非 None 时覆盖（避免清空已有 values）
        if let Some(v) = values {
            trial.values = Some(v.to_vec());
        }

        Self::update_cache(study, trial_id);
        Ok(true)
    }

    fn set_trial_intermediate_value(
        &self,
        trial_id: i64,
        step: i64,
        intermediate_value: f64,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;

        let study = Self::get_study(&inner, study_id)?;
        let trial = &study.trials[trial_number as usize];
        Self::check_updatable(trial)?;

        let study = Self::get_study_mut(&mut inner, study_id)?;
        let trial = &mut study.trials[trial_number as usize];
        trial
            .intermediate_values
            .insert(step, intermediate_value);
        Ok(())
    }

    fn set_trial_user_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;

        let study = Self::get_study(&inner, study_id)?;
        let trial = &study.trials[trial_number as usize];
        Self::check_updatable(trial)?;

        let study = Self::get_study_mut(&mut inner, study_id)?;
        let trial = &mut study.trials[trial_number as usize];
        trial.user_attrs.insert(key.to_string(), value);
        Ok(())
    }

    fn set_trial_system_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        let mut inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;

        // 对齐 Python: check_trial_is_updatable
        let study = Self::get_study(&inner, study_id)?;
        let trial = &study.trials[trial_number as usize];
        Self::check_updatable(trial)?;

        let study = Self::get_study_mut(&mut inner, study_id)?;
        let trial = &mut study.trials[trial_number as usize];
        trial.system_attrs.insert(key.to_string(), value);
        Ok(())
    }

    fn get_trial(&self, trial_id: i64) -> Result<FrozenTrial> {
        let inner = self.inner.lock();
        let (study_id, trial_number) = Self::resolve_trial(&inner, trial_id)?;
        let study = Self::get_study(&inner, study_id)?;
        Ok(study.trials[trial_number as usize].clone())
    }

    fn get_all_trials(
        &self,
        study_id: i64,
        states: Option<&[TrialState]>,
    ) -> Result<Vec<FrozenTrial>> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;

        let trials: Vec<FrozenTrial> = match states {
            None => study.trials.clone(),
            Some(s) => study
                .trials
                .iter()
                .filter(|t| s.contains(&t.state))
                .cloned()
                .collect(),
        };
        Ok(trials)
    }

    fn get_trial_id_from_study_id_trial_number(
        &self,
        study_id: i64,
        trial_number: i64,
    ) -> Result<i64> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;
        study
            .trials
            .get(trial_number as usize)
            .map(|t| t.trial_id)
            .ok_or_else(|| {
                OptunaError::ValueError(format!(
                    "trial number {trial_number} not found in study {study_id}"
                ))
            })
    }

    fn get_trial_number_from_id(&self, trial_id: i64) -> Result<i64> {
        let inner = self.inner.lock();
        let (_, trial_number) = Self::resolve_trial(&inner, trial_id)?;
        Ok(trial_number)
    }

    /// 对齐 Python: 在单一锁作用域内完成 best_trial 查找，避免死锁。
    /// 使用 `best_trial_id` 缓存（由 `update_cache` 维护），O(1) 查找。
    fn get_best_trial(&self, study_id: i64) -> Result<FrozenTrial> {
        let inner = self.inner.lock();
        let study = Self::get_study(&inner, study_id)?;

        // 对齐 Python: 多目标时抛 RuntimeError (Rust 用 OptunaError::RuntimeError)
        if study.directions.len() > 1 {
            return Err(OptunaError::RuntimeError(
                "best trial can be obtained only for single-objective optimization".into(),
            ));
        }

        // 对齐 Python: 使用缓存的 best_trial_id，O(1)
        match study.best_trial_id {
            Some(best_id) => {
                let trial = study.trials.iter()
                    .find(|t| t.trial_id == best_id)
                    .ok_or_else(|| OptunaError::ValueError(
                        "best trial not found in study trials".into(),
                    ))?;
                Ok(trial.clone())
            }
            None => Err(OptunaError::ValueError(
                "no trials are completed yet".into(),
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::ParamValue;

    #[test]
    fn test_create_study() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], Some("test"))
            .unwrap();
        assert_eq!(sid, 0);
        assert_eq!(storage.get_study_name_from_id(sid).unwrap(), "test");
        assert_eq!(
            storage.get_study_directions(sid).unwrap(),
            vec![StudyDirection::Minimize]
        );
    }

    #[test]
    fn test_create_study_auto_name() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let name = storage.get_study_name_from_id(sid).unwrap();
        assert!(name.starts_with("no-name-"));
    }

    #[test]
    fn test_duplicate_study_name() {
        let storage = InMemoryStorage::new();
        storage
            .create_new_study(&[StudyDirection::Minimize], Some("dup"))
            .unwrap();
        assert!(storage
            .create_new_study(&[StudyDirection::Minimize], Some("dup"))
            .is_err());
    }

    #[test]
    fn test_delete_study() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], Some("del"))
            .unwrap();
        storage.delete_study(sid).unwrap();
        assert!(storage.get_study_name_from_id(sid).is_err());
    }

    #[test]
    fn test_create_trial() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.number, 0);
        assert_eq!(trial.state, TrialState::Running);
        assert!(trial.datetime_start.is_some());
    }

    #[test]
    fn test_set_trial_param() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        let dist = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        storage
            .set_trial_param(tid, "x", 0.5, &dist)
            .unwrap();

        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.params.get("x"), Some(&ParamValue::Float(0.5)));
    }

    #[test]
    fn test_set_trial_state_values() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        let ok = storage
            .set_trial_state_values(tid, TrialState::Complete, Some(&[1.0]))
            .unwrap();
        assert!(ok);

        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.state, TrialState::Complete);
        assert_eq!(trial.values, Some(vec![1.0]));
        assert!(trial.datetime_complete.is_some());
    }

    #[test]
    fn test_set_trial_state_running_on_running_rejected() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        // Trial is already RUNNING; trying to set RUNNING again should return false
        let ok = storage
            .set_trial_state_values(tid, TrialState::Running, None)
            .unwrap();
        assert!(!ok);
    }

    #[test]
    fn test_update_finished_trial_rejected() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(tid, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        // Trying to update a finished trial should error
        assert!(storage
            .set_trial_state_values(tid, TrialState::Complete, Some(&[2.0]))
            .is_err());
    }

    #[test]
    fn test_intermediate_values() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_intermediate_value(tid, 0, 0.5).unwrap();
        storage.set_trial_intermediate_value(tid, 1, 0.3).unwrap();

        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.intermediate_values.len(), 2);
        assert_eq!(trial.intermediate_values[&0], 0.5);
        assert_eq!(trial.intermediate_values[&1], 0.3);
    }

    #[test]
    fn test_get_all_trials_filtered() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let t0 = storage.create_new_trial(sid, None).unwrap();
        let _t1 = storage.create_new_trial(sid, None).unwrap();

        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        let all = storage.get_all_trials(sid, None).unwrap();
        assert_eq!(all.len(), 2);

        let complete = storage
            .get_all_trials(sid, Some(&[TrialState::Complete]))
            .unwrap();
        assert_eq!(complete.len(), 1);

        let running = storage
            .get_all_trials(sid, Some(&[TrialState::Running]))
            .unwrap();
        assert_eq!(running.len(), 1);
    }

    #[test]
    fn test_get_best_trial() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();

        let t0 = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[3.0]))
            .unwrap();

        let t1 = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t1, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        let t2 = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t2, TrialState::Complete, Some(&[2.0]))
            .unwrap();

        let best = storage.get_best_trial(sid).unwrap();
        assert_eq!(best.value().unwrap(), Some(1.0));
    }

    #[test]
    fn test_get_best_trial_maximize() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Maximize], None)
            .unwrap();

        let t0 = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        let t1 = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t1, TrialState::Complete, Some(&[3.0]))
            .unwrap();

        let best = storage.get_best_trial(sid).unwrap();
        assert_eq!(best.value().unwrap(), Some(3.0));
    }

    #[test]
    fn test_get_best_trial_no_complete() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        assert!(storage.get_best_trial(sid).is_err());
    }

    #[test]
    fn test_study_user_attrs() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        storage
            .set_study_user_attr(sid, "key", serde_json::json!("value"))
            .unwrap();
        let attrs = storage.get_study_user_attrs(sid).unwrap();
        assert_eq!(attrs.get("key"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn test_trial_user_attrs() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_user_attr(tid, "k", serde_json::json!(42))
            .unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.user_attrs.get("k"), Some(&serde_json::json!(42)));
    }

    #[test]
    fn test_concurrent_access() {
        use std::sync::Arc;
        use std::thread;

        let storage = Arc::new(InMemoryStorage::new());
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();

        let handles: Vec<_> = (0..10)
            .map(|_| {
                let s = Arc::clone(&storage);
                thread::spawn(move || {
                    let tid = s.create_new_trial(sid, None).unwrap();
                    s.set_trial_state_values(tid, TrialState::Complete, Some(&[1.0]))
                        .unwrap();
                })
            })
            .collect();

        for h in handles {
            h.join().unwrap();
        }

        let trials = storage.get_all_trials(sid, None).unwrap();
        assert_eq!(trials.len(), 10);
        assert!(trials.iter().all(|t| t.state == TrialState::Complete));
    }

    #[test]
    fn test_get_n_trials() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();
        let t0 = storage.create_new_trial(sid, None).unwrap();
        storage.create_new_trial(sid, None).unwrap();
        storage
            .set_trial_state_values(t0, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        assert_eq!(storage.get_n_trials(sid, None).unwrap(), 2);
        assert_eq!(
            storage
                .get_n_trials(sid, Some(&[TrialState::Complete]))
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_get_all_studies() {
        let storage = InMemoryStorage::new();
        storage
            .create_new_study(&[StudyDirection::Minimize], Some("a"))
            .unwrap();
        storage
            .create_new_study(&[StudyDirection::Maximize], Some("b"))
            .unwrap();

        let studies = storage.get_all_studies().unwrap();
        assert_eq!(studies.len(), 2);
    }

    #[test]
    fn test_template_trial() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();

        let mut template = FrozenTrial {
            number: 0,
            state: TrialState::Waiting,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        template
            .user_attrs
            .insert("preset".into(), serde_json::json!(true));

        let tid = storage.create_new_trial(sid, Some(&template)).unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.state, TrialState::Waiting);
        assert_eq!(
            trial.user_attrs.get("preset"),
            Some(&serde_json::json!(true))
        );
    }

    #[test]
    fn test_template_trial_preserves_system_attrs() {
        let storage = InMemoryStorage::new();
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], None)
            .unwrap();

        let mut template = FrozenTrial {
            number: 0,
            state: TrialState::Waiting,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        template
            .system_attrs
            .insert("fixed_params".into(), serde_json::json!({"x": 0.5}));

        let tid = storage.create_new_trial(sid, Some(&template)).unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(
            trial.system_attrs.get("fixed_params"),
            Some(&serde_json::json!({"x": 0.5}))
        );
    }

    /// 对齐 Python: set_trial_state_values(values=None) 不应清空已有 values
    #[test]
    fn test_set_trial_state_values_none_preserves_existing() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        // 先设置 Complete + values
        storage.set_trial_state_values(tid, TrialState::Complete, Some(&[1.5])).unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.values.as_ref().unwrap(), &[1.5]);
    }

    /// 对齐 Python: RUNNING → COMPLETE 转换设置 datetime_complete
    #[test]
    fn test_set_trial_state_values_sets_datetime_complete() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        let trial_before = storage.get_trial(tid).unwrap();
        assert!(trial_before.datetime_complete.is_none());

        storage.set_trial_state_values(tid, TrialState::Complete, Some(&[1.0])).unwrap();
        let trial_after = storage.get_trial(tid).unwrap();
        assert!(trial_after.datetime_complete.is_some(),
            "完成后应设置 datetime_complete");
    }

    /// 对齐 Python: 已完成试验不可再更新状态
    #[test]
    fn test_set_trial_state_values_finished_trial_error() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(tid, TrialState::Complete, Some(&[1.0])).unwrap();

        // 再次设置应报错
        let err = storage.set_trial_state_values(tid, TrialState::Complete, Some(&[2.0]));
        assert!(err.is_err(), "已完成试验不应允许更新");
    }

    /// 对齐 Python: set_trial_param / set_trial_intermediate_value / set_trial_user_attr
    #[test]
    fn test_storage_crud_operations() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        // set_trial_param
        let dist = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        storage.set_trial_param(tid, "x", 0.5, &dist).unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.params.get("x"), Some(&ParamValue::Float(0.5)));

        // set_trial_intermediate_value
        storage.set_trial_intermediate_value(tid, 0, 0.8).unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.intermediate_values.get(&0), Some(&0.8));

        // set_trial_user_attr
        storage.set_trial_user_attr(tid, "key", serde_json::json!("val")).unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.user_attrs.get("key"), Some(&serde_json::json!("val")));
    }

    // ========================================================================
    // 分布兼容性检查测试（对齐 Python check_distribution_compatibility）
    // ========================================================================

    /// 同类型 Float 不同 range 应允许（不报错）
    #[test]
    fn test_distribution_compat_float_different_range() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        let dist1 = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        let dist2 = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution::new(0.0, 2.0, false, None).unwrap(),
        );
        storage.set_trial_param(tid, "x", 0.5, &dist1).unwrap();
        // 同类型 Float 不同 range 仍应成功（对齐 Python）
        assert!(storage.set_trial_param(tid, "x", 0.5, &dist2).is_ok());
    }

    /// 同类型 Float log 属性不同应报错
    #[test]
    fn test_distribution_compat_float_different_log() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        let dist1 = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution::new(0.1, 1.0, false, None).unwrap(),
        );
        let dist2 = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution::new(0.1, 1.0, true, None).unwrap(),
        );
        storage.set_trial_param(tid, "x", 0.5, &dist1).unwrap();
        assert!(storage.set_trial_param(tid, "x", 0.5, &dist2).is_err());
    }

    /// 不同类型分布应报错 (Float vs Int)
    #[test]
    fn test_distribution_compat_different_kind() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        let dist_float = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        let dist_int = Distribution::IntDistribution(
            crate::distributions::IntDistribution::new(0, 10, false, 1).unwrap(),
        );
        storage.set_trial_param(tid, "x", 0.5, &dist_float).unwrap();
        let result = storage.set_trial_param(tid, "x", 5.0, &dist_int);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("different distribution kind"));
    }

    /// Categorical 不同选项应报错
    #[test]
    fn test_distribution_compat_categorical_different_choices() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();

        use crate::distributions::{CategoricalChoice, CategoricalDistribution};
        let dist1 = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".into()),
                CategoricalChoice::Str("b".into()),
            ]).unwrap(),
        );
        let dist2 = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".into()),
                CategoricalChoice::Str("c".into()),
            ]).unwrap(),
        );
        storage.set_trial_param(tid, "cat", 0.0, &dist1).unwrap();
        let result = storage.set_trial_param(tid, "cat", 0.0, &dist2);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("dynamic value space"));
    }

    /// 对齐 Python: set_trial_system_attr 在已完成试验上应报错
    #[test]
    fn test_set_trial_system_attr_rejects_finished_trial() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        // WAITING → RUNNING → COMPLETE
        storage.set_trial_state_values(tid, TrialState::Running, None).unwrap();
        storage.set_trial_state_values(tid, TrialState::Complete, Some(&[1.0])).unwrap();
        // 在已完成试验上 set_trial_system_attr 应报 UpdateFinishedTrialError
        let result = storage.set_trial_system_attr(tid, "key", serde_json::json!("val"));
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("already finished"));
    }

    /// 对齐 Python: set_trial_system_attr 在 RUNNING 试验上正常工作
    #[test]
    fn test_set_trial_system_attr_running_ok() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(tid, TrialState::Running, None).unwrap();
        storage.set_trial_system_attr(tid, "test_key", serde_json::json!(42)).unwrap();
        let trial = storage.get_trial(tid).unwrap();
        assert_eq!(trial.system_attrs["test_key"], serde_json::json!(42));
    }

    /// 对齐 Python: get_best_trial 不会死锁（单锁作用域）
    #[test]
    fn test_get_best_trial_override_no_deadlock() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        // 创建两个已完成试验
        let tid1 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(tid1, TrialState::Complete, Some(&[3.0])).unwrap();
        let tid2 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(tid2, TrialState::Complete, Some(&[1.0])).unwrap();
        // 应返回值最小的试验
        let best = storage.get_best_trial(sid).unwrap();
        assert_eq!(best.values, Some(vec![1.0]));
    }

    /// 对齐 Python: get_best_trial override 在 Maximize 方向返回最大值
    #[test]
    fn test_get_best_trial_override_maximize() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Maximize], None).unwrap();
        let tid1 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(tid1, TrialState::Complete, Some(&[3.0])).unwrap();
        let tid2 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(tid2, TrialState::Complete, Some(&[5.0])).unwrap();
        let best = storage.get_best_trial(sid).unwrap();
        assert_eq!(best.values, Some(vec![5.0]));
    }

    /// 对齐 Python: get_best_trial 无已完成试验时报 ValueError
    #[test]
    fn test_get_best_trial_override_no_complete() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let _tid = storage.create_new_trial(sid, None).unwrap();
        // 只有 Running 试验，无 Complete
        let result = storage.get_best_trial(sid);
        assert!(result.is_err());
    }

    /// 对齐 Python: get_best_trial override 多目标时报 ValueError
    #[test]
    fn test_get_best_trial_override_multi_objective_error() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(
            &[StudyDirection::Minimize, StudyDirection::Maximize], None
        ).unwrap();
        let result = storage.get_best_trial(sid);
        assert!(result.is_err());
    }

    /// 对齐 Python: get_best_trial override 处理 NaN 值不 panic（使用 total_cmp）
    #[test]
    fn test_get_best_trial_override_nan_safety() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid1 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(tid1, TrialState::Complete, Some(&[f64::NAN])).unwrap();
        let tid2 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(tid2, TrialState::Complete, Some(&[2.0])).unwrap();
        // 不应 panic
        let best = storage.get_best_trial(sid).unwrap();
        assert_eq!(best.values, Some(vec![2.0]));
    }

    // ── 对齐 Python 审计修复的新测试 ──

    /// 对齐 Python: auto-naming 使用 UUID 保证唯一性
    #[test]
    fn test_auto_name_uniqueness() {
        let storage = InMemoryStorage::new();
        let s1 = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let s2 = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let n1 = storage.get_study_name_from_id(s1).unwrap();
        let n2 = storage.get_study_name_from_id(s2).unwrap();
        assert!(n1.starts_with("no-name-"));
        assert!(n2.starts_with("no-name-"));
        assert_ne!(n1, n2, "UUID 自动名称应唯一");
    }

    /// 对齐 Python: get_all_studies 结果按 study_id 排序
    #[test]
    fn test_get_all_studies_sorted() {
        let storage = InMemoryStorage::new();
        for i in 0..5 {
            storage.create_new_study(&[StudyDirection::Minimize], Some(&format!("s{}", i))).unwrap();
        }
        let studies = storage.get_all_studies().unwrap();
        for i in 1..studies.len() {
            assert!(studies[i].study_id > studies[i - 1].study_id,
                "studies 应按 study_id 排序");
        }
    }

    /// 对齐 Python: get_best_trial 多目标应抛 RuntimeError
    #[test]
    fn test_get_best_trial_multi_objective_runtime_error() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(
            &[StudyDirection::Minimize, StudyDirection::Maximize], None
        ).unwrap();
        let err = storage.get_best_trial(sid).unwrap_err();
        assert!(matches!(err, OptunaError::RuntimeError(_)));
    }

    /// 对齐 Python: get_trial_params/user_attrs/system_attrs 便捷方法
    #[test]
    fn test_storage_convenience_methods() {
        use crate::distributions::{Distribution, FloatDistribution};
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
        let tid = storage.create_new_trial(sid, None).unwrap();
        // 设置参数 (内部表示 f64)
        let dist = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        storage.set_trial_param(tid, "x", 0.5, &dist).unwrap();
        // 设置 user attr
        storage.set_trial_user_attr(tid, "ua_key", serde_json::json!("ua_val")).unwrap();
        // 设置 system attr
        storage.set_trial_system_attr(tid, "sa_key", serde_json::json!("sa_val")).unwrap();

        // 测试便捷方法
        let params = storage.get_trial_params(tid).unwrap();
        assert!(params.contains_key("x"));

        let ua = storage.get_trial_user_attrs(tid).unwrap();
        assert_eq!(ua.get("ua_key"), Some(&serde_json::json!("ua_val")));

        let sa = storage.get_trial_system_attrs(tid).unwrap();
        assert_eq!(sa.get("sa_key"), Some(&serde_json::json!("sa_val")));
    }

    /// 对齐 Python: NaN best 被非 NaN 值替换（Maximize 方向）
    #[test]
    fn test_get_best_trial_nan_maximize() {
        let storage = InMemoryStorage::new();
        let sid = storage.create_new_study(&[StudyDirection::Maximize], None).unwrap();
        let tid1 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(tid1, TrialState::Complete, Some(&[f64::NAN])).unwrap();
        let tid2 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(tid2, TrialState::Complete, Some(&[5.0])).unwrap();
        let best = storage.get_best_trial(sid).unwrap();
        assert_eq!(best.values, Some(vec![5.0]));
    }
}
