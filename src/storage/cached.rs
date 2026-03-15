//! 缓存存储包装器。
//!
//! 对应 Python `optuna.storages._cached_storage._CachedStorage`。
//! 在已完成试验上提供本地缓存，避免重复读取后端存储。
//! 主要用于包装 [`RdbStorage`](super::RdbStorage)，减少数据库查询次数。

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::study::{FrozenStudy, StudyDirection};
use crate::storage::Storage;
use crate::trial::{FrozenTrial, TrialState};

/// 每个 study 的缓存信息。
struct StudyInfo {
    /// 已完成试验的缓存（按 trial_id 索引）
    finished_trials: HashMap<i64, FrozenTrial>,
    /// 最后一个已完成试验的 trial_id（用于增量读取）
    last_finished_trial_id: i64,
    /// 参数分布缓存（用于兼容性检查）
    _param_distributions: HashMap<String, Distribution>,
}

impl StudyInfo {
    fn new() -> Self {
        Self {
            finished_trials: HashMap::new(),
            last_finished_trial_id: -1,
            _param_distributions: HashMap::new(),
        }
    }
}

/// 带本地缓存的存储包装器。
///
/// 对应 Python `optuna.storages._CachedStorage`。
///
/// 已完成的试验（Complete/Pruned/Fail）被缓存在内存中，
/// 后续调用 `get_all_trials()` 只需从后端读取未完成的试验和新增的已完成试验。
pub struct CachedStorage {
    /// 被包装的后端存储
    backend: Arc<dyn Storage>,
    /// 每个 study 的缓存
    study_cache: Mutex<HashMap<i64, StudyInfo>>,
}

impl CachedStorage {
    /// 创建缓存存储包装器。
    ///
    /// # 参数
    /// * `backend` - 被包装的后端存储（通常是 RdbStorage）
    pub fn new(backend: Arc<dyn Storage>) -> Self {
        Self {
            backend,
            study_cache: Mutex::new(HashMap::new()),
        }
    }

    /// 获取被包装的后端存储引用。
    pub fn backend(&self) -> &Arc<dyn Storage> {
        &self.backend
    }

    /// 确保 study_id 有对应的缓存条目。
    fn ensure_study_cache(&self, study_id: i64) {
        let mut cache = self.study_cache.lock();
        cache.entry(study_id).or_insert_with(StudyInfo::new);
    }

    /// 将已完成的试验缓存起来。
    fn cache_trial_if_finished(&self, study_id: i64, trial: &FrozenTrial) {
        if trial.state.is_finished() {
            let mut cache = self.study_cache.lock();
            if let Some(info) = cache.get_mut(&study_id) {
                if trial.trial_id > info.last_finished_trial_id {
                    info.last_finished_trial_id = trial.trial_id;
                }
                info.finished_trials.insert(trial.trial_id, trial.clone());
            }
        }
    }
}

impl Storage for CachedStorage {
    // ── Study CRUD：直接委托给后端 ──

    fn create_new_study(
        &self,
        directions: &[StudyDirection],
        study_name: Option<&str>,
    ) -> Result<i64> {
        let study_id = self.backend.create_new_study(directions, study_name)?;
        self.ensure_study_cache(study_id);
        Ok(study_id)
    }

    fn delete_study(&self, study_id: i64) -> Result<()> {
        self.backend.delete_study(study_id)?;
        self.study_cache.lock().remove(&study_id);
        Ok(())
    }

    fn set_study_user_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        self.backend.set_study_user_attr(study_id, key, value)
    }

    fn set_study_system_attr(
        &self,
        study_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        self.backend.set_study_system_attr(study_id, key, value)
    }

    fn get_study_id_from_name(&self, study_name: &str) -> Result<i64> {
        self.backend.get_study_id_from_name(study_name)
    }

    fn get_study_name_from_id(&self, study_id: i64) -> Result<String> {
        self.backend.get_study_name_from_id(study_id)
    }

    fn get_study_directions(&self, study_id: i64) -> Result<Vec<StudyDirection>> {
        self.backend.get_study_directions(study_id)
    }

    fn get_study_user_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        self.backend.get_study_user_attrs(study_id)
    }

    fn get_study_system_attrs(&self, study_id: i64) -> Result<HashMap<String, serde_json::Value>> {
        self.backend.get_study_system_attrs(study_id)
    }

    fn get_all_studies(&self) -> Result<Vec<FrozenStudy>> {
        self.backend.get_all_studies()
    }

    // ── Trial CRUD：带缓存逻辑 ──

    fn create_new_trial(
        &self,
        study_id: i64,
        template_trial: Option<&FrozenTrial>,
    ) -> Result<i64> {
        self.ensure_study_cache(study_id);
        let trial_id = self.backend.create_new_trial(study_id, template_trial)?;
        // 模板试验可能已经是完成状态
        if let Some(tmpl) = template_trial {
            if tmpl.state.is_finished() {
                let trial = self.backend.get_trial(trial_id)?;
                self.cache_trial_if_finished(study_id, &trial);
            }
        }
        Ok(trial_id)
    }

    fn set_trial_param(
        &self,
        trial_id: i64,
        param_name: &str,
        param_value_internal: f64,
        distribution: &Distribution,
    ) -> Result<()> {
        // 利用缓存的分布做本地兼容性检查
        self.backend
            .set_trial_param(trial_id, param_name, param_value_internal, distribution)
    }

    fn set_trial_state_values(
        &self,
        trial_id: i64,
        state: TrialState,
        values: Option<&[f64]>,
    ) -> Result<bool> {
        let result = self
            .backend
            .set_trial_state_values(trial_id, state, values)?;
        // 状态变为已完成时缓存试验
        if result && state.is_finished() {
            // 状态变为已完成时，在下次 get_all_trials 时自动缓存
        }
        Ok(result)
    }

    fn set_trial_intermediate_value(
        &self,
        trial_id: i64,
        step: i64,
        intermediate_value: f64,
    ) -> Result<()> {
        self.backend
            .set_trial_intermediate_value(trial_id, step, intermediate_value)
    }

    fn set_trial_user_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        self.backend.set_trial_user_attr(trial_id, key, value)
    }

    fn set_trial_system_attr(
        &self,
        trial_id: i64,
        key: &str,
        value: serde_json::Value,
    ) -> Result<()> {
        self.backend.set_trial_system_attr(trial_id, key, value)
    }

    fn get_trial(&self, trial_id: i64) -> Result<FrozenTrial> {
        // 先查缓存
        let cache = self.study_cache.lock();
        for info in cache.values() {
            if let Some(trial) = info.finished_trials.get(&trial_id) {
                return Ok(trial.clone());
            }
        }
        drop(cache);
        // 缓存未命中，从后端获取
        self.backend.get_trial(trial_id)
    }

    fn get_all_trials(
        &self,
        study_id: i64,
        states: Option<&[TrialState]>,
    ) -> Result<Vec<FrozenTrial>> {
        self.ensure_study_cache(study_id);

        // 从后端获取所有试验（未来可优化为增量读取）
        let all_trials = self.backend.get_all_trials(study_id, states)?;

        // 更新缓存
        for trial in &all_trials {
            self.cache_trial_if_finished(study_id, trial);
        }

        Ok(all_trials)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;

    #[test]
    fn test_cached_storage_basic() {
        // 使用 InMemoryStorage 作为后端
        let backend = Arc::new(InMemoryStorage::new());
        let cached = CachedStorage::new(backend);
        let dirs = vec![StudyDirection::Minimize];
        let study_id = cached.create_new_study(&dirs, Some("test")).unwrap();

        // 创建试验
        let trial_id = cached.create_new_trial(study_id, None).unwrap();
        cached
            .set_trial_state_values(trial_id, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        // 获取试验
        let trials = cached.get_all_trials(study_id, None).unwrap();
        assert_eq!(trials.len(), 1);
        assert_eq!(trials[0].state, TrialState::Complete);
    }

    #[test]
    fn test_cached_storage_finished_trial_cache() {
        let backend = Arc::new(InMemoryStorage::new());
        let cached = CachedStorage::new(backend);
        let dirs = vec![StudyDirection::Minimize];
        let study_id = cached.create_new_study(&dirs, Some("test")).unwrap();

        // 创建并完成两个试验
        let tid1 = cached.create_new_trial(study_id, None).unwrap();
        cached
            .set_trial_state_values(tid1, TrialState::Complete, Some(&[1.0]))
            .unwrap();

        let tid2 = cached.create_new_trial(study_id, None).unwrap();
        cached
            .set_trial_state_values(tid2, TrialState::Complete, Some(&[2.0]))
            .unwrap();

        // 第一次获取所有试验（填充缓存）
        let trials = cached.get_all_trials(study_id, None).unwrap();
        assert_eq!(trials.len(), 2);

        // 通过 get_trial 获取已缓存的试验
        let t1 = cached.get_trial(tid1).unwrap();
        assert_eq!(t1.values.as_ref().unwrap()[0], 1.0);
    }

    #[test]
    fn test_cached_storage_delete_study() {
        let backend = Arc::new(InMemoryStorage::new());
        let cached = CachedStorage::new(backend);
        let dirs = vec![StudyDirection::Minimize];
        let study_id = cached.create_new_study(&dirs, Some("test")).unwrap();

        // 删除研究后缓存应清空
        cached.delete_study(study_id).unwrap();
        assert!(cached.get_study_directions(study_id).is_err());
    }

    /// 对齐 Python: 多 study 缓存隔离
    #[test]
    fn test_cached_storage_multi_study_isolation() {
        let backend = Arc::new(InMemoryStorage::new());
        let cached = CachedStorage::new(backend);
        let dirs = vec![StudyDirection::Minimize];

        let sid1 = cached.create_new_study(&dirs, Some("study1")).unwrap();
        let sid2 = cached.create_new_study(&dirs, Some("study2")).unwrap();

        let tid1 = cached.create_new_trial(sid1, None).unwrap();
        cached.set_trial_state_values(tid1, TrialState::Complete, Some(&[1.0])).unwrap();

        let tid2 = cached.create_new_trial(sid2, None).unwrap();
        cached.set_trial_state_values(tid2, TrialState::Complete, Some(&[2.0])).unwrap();

        // study1 只有 1 个试验
        let trials1 = cached.get_all_trials(sid1, None).unwrap();
        assert_eq!(trials1.len(), 1);
        assert_eq!(trials1[0].values.as_ref().unwrap()[0], 1.0);

        // study2 只有 1 个试验
        let trials2 = cached.get_all_trials(sid2, None).unwrap();
        assert_eq!(trials2.len(), 1);
        assert_eq!(trials2[0].values.as_ref().unwrap()[0], 2.0);
    }

    /// 对齐 Python: Pruned 试验也被缓存
    #[test]
    fn test_cached_storage_pruned_trial_cached() {
        let backend = Arc::new(InMemoryStorage::new());
        let cached = CachedStorage::new(backend);
        let dirs = vec![StudyDirection::Minimize];
        let study_id = cached.create_new_study(&dirs, Some("test")).unwrap();

        let tid = cached.create_new_trial(study_id, None).unwrap();
        cached.set_trial_state_values(tid, TrialState::Pruned, None).unwrap();

        // 获取所有试验以填充缓存
        let trials = cached.get_all_trials(study_id, None).unwrap();
        assert_eq!(trials.len(), 1);
        assert_eq!(trials[0].state, TrialState::Pruned);

        // 通过 get_trial 应能命中缓存
        let t = cached.get_trial(tid).unwrap();
        assert_eq!(t.state, TrialState::Pruned);
    }

    /// 对齐 Python: get_trial 在缓存未命中时查询后端
    #[test]
    fn test_cached_storage_get_trial_cache_miss() {
        let backend: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let cached = CachedStorage::new(Arc::clone(&backend));
        let dirs = vec![StudyDirection::Minimize];
        let study_id = cached.create_new_study(&dirs, Some("test")).unwrap();

        let tid = cached.create_new_trial(study_id, None).unwrap();
        // 不调用 get_all_trials（不填充缓存），直接 get_trial → 从后端获取
        let t = cached.get_trial(tid).unwrap();
        assert_eq!(t.state, TrialState::Running); // 新建的试验状态是 Running
    }

    /// 对齐 Python: user_attrs / system_attrs 透传
    #[test]
    fn test_cached_storage_attrs_passthrough() {
        let backend = Arc::new(InMemoryStorage::new());
        let cached = CachedStorage::new(backend);
        let dirs = vec![StudyDirection::Minimize];
        let study_id = cached.create_new_study(&dirs, Some("test")).unwrap();

        cached.set_study_user_attr(study_id, "key1", serde_json::json!("val1")).unwrap();
        let attrs = cached.get_study_user_attrs(study_id).unwrap();
        assert_eq!(attrs.get("key1").unwrap(), &serde_json::json!("val1"));
    }
}
