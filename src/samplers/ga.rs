//! 遗传算法采样器基类。
//!
//! 对应 Python `optuna.samplers._ga._base.BaseGASampler`。
//! 提供代际管理（generation tracking）和种群管理功能，
//! NSGA-II / NSGA-III 等 GA 系采样器可实现此 trait。

use crate::error::Result;
use crate::storage::Storage;
use crate::trial::{FrozenTrial, TrialState};

/// 遗传算法采样器 trait。
///
/// 对应 Python `optuna.samplers.BaseGASampler`。
///
/// 提供代际管理方法的默认实现，子类需实现 [`select_parent`](GaSampler::select_parent)
/// 来定义父代选择策略。所有默认方法都依赖 `generation_key()` 来在 trial/study
/// system_attrs 中持久化代际信息。
///
/// # 示例
///
/// 自定义 GA 采样器只需实现 `select_parent` 和基本属性：
///
/// ```ignore
/// impl GaSampler for MyGaSampler {
///     fn generation_key(&self) -> &str { "MyGaSampler:generation" }
///     fn parent_cache_key_prefix(&self) -> &str { "MyGaSampler:parent:" }
///     fn population_size(&self) -> usize { 50 }
///     fn select_parent(&self, storage: &dyn Storage, study_id: i64, generation: i32)
///         -> Result<Vec<FrozenTrial>> { /* ... */ }
/// }
/// ```
pub trait GaSampler: Send + Sync {
    /// 代际键名，用于 trial system_attrs。
    ///
    /// 对应 Python `_GENERATION_KEY = "{ClassName}:generation"`。
    /// 每个子类应返回唯一的键名，如 `"NSGAIISampler:generation"`。
    fn generation_key(&self) -> &str;

    /// 父代缓存键前缀，用于 study system_attrs。
    ///
    /// 对应 Python `_PARENT_CACHE_KEY_PREFIX = "{ClassName}:parent:"`。
    fn parent_cache_key_prefix(&self) -> &str;

    /// 种群大小。
    fn population_size(&self) -> usize;

    /// 选择父代种群。
    ///
    /// 对应 Python `BaseGASampler.select_parent(study, generation)`。
    /// 由子类实现，返回从当前代选出的父代试验列表。
    /// 结果会被缓存到 study system_attrs 中。
    fn select_parent(
        &self,
        storage: &dyn Storage,
        study_id: i64,
        generation: i32,
    ) -> Result<Vec<FrozenTrial>>;

    /// 获取试验的代际编号。
    ///
    /// 对应 Python `BaseGASampler.get_trial_generation(study, trial)`。
    ///
    /// 如果 trial.system_attrs 中已有代际信息则直接返回；
    /// 否则扫描所有已完成试验计算当前代际编号，并写入 system_attrs。
    fn get_trial_generation(
        &self,
        storage: &dyn Storage,
        study_id: i64,
        trial: &FrozenTrial,
    ) -> Result<i32> {
        // 检查缓存
        if let Some(val) = trial.system_attrs.get(self.generation_key()) {
            if let Some(g) = val.as_i64() {
                return Ok(g as i32);
            }
        }

        let trials = storage.get_all_trials(study_id, Some(&[TrialState::Complete]))?;

        let mut max_generation: i32 = 0;
        let mut max_generation_count: usize = 0;

        for t in trials.iter().rev() {
            let g = t
                .system_attrs
                .get(self.generation_key())
                .and_then(|v| v.as_i64())
                .map(|v| v as i32)
                .unwrap_or(-1);

            if g < max_generation {
                continue;
            } else if g > max_generation {
                max_generation = g;
                max_generation_count = 1;
            } else {
                max_generation_count += 1;
            }
        }

        let generation = if max_generation_count < self.population_size() {
            max_generation
        } else {
            max_generation + 1
        };

        storage.set_trial_system_attr(
            trial.trial_id,
            self.generation_key(),
            serde_json::Value::Number(serde_json::Number::from(generation)),
        )?;

        Ok(generation)
    }

    /// 获取指定代的种群（所有已完成且属于该代的试验）。
    ///
    /// 对应 Python `BaseGASampler.get_population(study, generation)`。
    fn get_population(
        &self,
        storage: &dyn Storage,
        study_id: i64,
        generation: i32,
    ) -> Result<Vec<FrozenTrial>> {
        let trials = storage.get_all_trials(study_id, Some(&[TrialState::Complete]))?;
        Ok(trials
            .into_iter()
            .filter(|t| {
                t.system_attrs
                    .get(self.generation_key())
                    .and_then(|v| v.as_i64())
                    .map(|v| v as i32)
                    == Some(generation)
            })
            .collect())
    }

    /// 获取父代种群（带缓存）。
    ///
    /// 对应 Python `BaseGASampler.get_parent_population(study, generation)`。
    ///
    /// generation == 0 时返回空列表。
    /// 先从 study system_attrs 查找缓存，miss 时调用 `select_parent()` 并缓存。
    fn get_parent_population(
        &self,
        storage: &dyn Storage,
        study_id: i64,
        generation: i32,
    ) -> Result<Vec<FrozenTrial>> {
        if generation == 0 {
            return Ok(vec![]);
        }

        let cache_key = format!("{}{}", self.parent_cache_key_prefix(), generation);
        let sys_attrs = storage.get_study_system_attrs(study_id)?;

        if let Some(cached_ids) = sys_attrs.get(&cache_key) {
            if let Some(ids_arr) = cached_ids.as_array() {
                let parent_ids: std::collections::HashSet<i64> = ids_arr
                    .iter()
                    .filter_map(|v| v.as_i64())
                    .collect();
                let all_trials = storage.get_all_trials(study_id, None)?;
                return Ok(all_trials
                    .into_iter()
                    .filter(|t| parent_ids.contains(&t.trial_id))
                    .collect());
            }
        }

        // Cache miss — 调用 select_parent 并缓存结果
        let parent_population = self.select_parent(storage, study_id, generation)?;
        let parent_ids: Vec<serde_json::Value> = parent_population
            .iter()
            .map(|t| serde_json::Value::Number(serde_json::Number::from(t.trial_id)))
            .collect();

        storage.set_study_system_attr(
            study_id,
            &cache_key,
            serde_json::Value::Array(parent_ids),
        )?;

        Ok(parent_population)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;
    use crate::study::StudyDirection;
    use std::sync::Arc;

    /// 测试用 GaSampler 实现
    struct DummyGaSampler {
        pop_size: usize,
    }
    impl GaSampler for DummyGaSampler {
        fn generation_key(&self) -> &str {
            "DummyGA:generation"
        }
        fn parent_cache_key_prefix(&self) -> &str {
            "DummyGA:parent:"
        }
        fn population_size(&self) -> usize {
            self.pop_size
        }
        fn select_parent(
            &self,
            storage: &dyn Storage,
            study_id: i64,
            generation: i32,
        ) -> Result<Vec<FrozenTrial>> {
            // 返回上一代所有 trial
            self.get_population(storage, study_id, generation - 1)
        }
    }

    /// 辅助：创建 study 并添加 n 个完成 trial
    fn setup(n: usize, pop_size: usize) -> (Arc<InMemoryStorage>, i64, DummyGaSampler) {
        let storage = Arc::new(InMemoryStorage::new());
        let sid = storage
            .create_new_study(&[StudyDirection::Minimize], Some("ga_test"))
            .unwrap();
        for i in 0..n {
            let tid = storage.create_new_trial(sid, None).unwrap();
            storage
                .set_trial_state_values(tid, TrialState::Complete, Some(&[i as f64]))
                .unwrap();
        }
        let sampler = DummyGaSampler { pop_size };
        (storage, sid, sampler)
    }

    /// 对齐 Python: generation 0 时 get_parent_population 返回空
    #[test]
    fn test_parent_population_gen0_empty() {
        let (storage, sid, sampler) = setup(3, 3);
        let parents = sampler.get_parent_population(&*storage, sid, 0).unwrap();
        assert!(parents.is_empty());
    }

    /// 对齐 Python: get_trial_generation 代际计算
    #[test]
    fn test_get_trial_generation() {
        let (storage, sid, sampler) = setup(0, 2);
        // 对齐 Python: get_trial_generation 在 trial 为 RUNNING 时调用
        // 创建第 1 个 trial → RUNNING → gen 0 → Complete
        let t1 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(t1, TrialState::Running, None).unwrap();
        let trial1 = storage.get_trial(t1).unwrap();
        let g1 = sampler.get_trial_generation(&*storage, sid, &trial1).unwrap();
        assert_eq!(g1, 0);
        storage.set_trial_state_values(t1, TrialState::Complete, Some(&[1.0])).unwrap();

        // 创建第 2 个 trial → gen 0（pop_size=2 未满）
        let t2 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(t2, TrialState::Running, None).unwrap();
        let trial2 = storage.get_trial(t2).unwrap();
        let g2 = sampler.get_trial_generation(&*storage, sid, &trial2).unwrap();
        assert_eq!(g2, 0);
        storage.set_trial_state_values(t2, TrialState::Complete, Some(&[2.0])).unwrap();

        // 创建第 3 个 trial → gen 1（pop_size=2 已满，进入下一代）
        let t3 = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(t3, TrialState::Running, None).unwrap();
        let trial3 = storage.get_trial(t3).unwrap();
        let g3 = sampler.get_trial_generation(&*storage, sid, &trial3).unwrap();
        assert_eq!(g3, 1);
        storage.set_trial_state_values(t3, TrialState::Complete, Some(&[3.0])).unwrap();
    }

    /// 对齐 Python: get_population 筛选正确
    #[test]
    fn test_get_population() {
        let (storage, sid, sampler) = setup(0, 2);
        // gen 0 两个 trial: RUNNING → set generation → Complete
        for i in 0..2 {
            let tid = storage.create_new_trial(sid, None).unwrap();
            storage.set_trial_state_values(tid, TrialState::Running, None).unwrap();
            let trial = storage.get_trial(tid).unwrap();
            sampler.get_trial_generation(&*storage, sid, &trial).unwrap();
            storage.set_trial_state_values(tid, TrialState::Complete, Some(&[i as f64])).unwrap();
        }
        let pop0 = sampler.get_population(&*storage, sid, 0).unwrap();
        assert_eq!(pop0.len(), 2);
        let pop1 = sampler.get_population(&*storage, sid, 1).unwrap();
        assert!(pop1.is_empty());
    }

    /// 对齐 Python: get_parent_population 缓存行为
    #[test]
    fn test_parent_population_caching() {
        let (storage, sid, sampler) = setup(0, 2);
        // gen 0: 2 trial, RUNNING → generation → Complete
        for i in 0..2 {
            let tid = storage.create_new_trial(sid, None).unwrap();
            storage.set_trial_state_values(tid, TrialState::Running, None).unwrap();
            let trial = storage.get_trial(tid).unwrap();
            sampler.get_trial_generation(&*storage, sid, &trial).unwrap();
            storage.set_trial_state_values(tid, TrialState::Complete, Some(&[i as f64])).unwrap();
        }
        // gen 1 的父代 = gen 0 的 select_parent 结果
        let parents1 = sampler.get_parent_population(&*storage, sid, 1).unwrap();
        assert_eq!(parents1.len(), 2);
        // 再次调用，走缓存
        let parents1_cached = sampler.get_parent_population(&*storage, sid, 1).unwrap();
        assert_eq!(parents1_cached.len(), 2);
    }

    /// 对齐 Python: generation_key 已缓存时直接返回
    #[test]
    fn test_generation_cached_in_system_attrs() {
        let (storage, sid, sampler) = setup(0, 10);
        let tid = storage.create_new_trial(sid, None).unwrap();
        storage.set_trial_state_values(tid, TrialState::Running, None).unwrap();
        let trial = storage.get_trial(tid).unwrap();
        // 第一次计算并写入（RUNNING 状态）
        let g1 = sampler.get_trial_generation(&*storage, sid, &trial).unwrap();
        // 读回 trial，system_attrs 应有 generation
        let trial2 = storage.get_trial(tid).unwrap();
        let g2 = sampler.get_trial_generation(&*storage, sid, &trial2).unwrap();
        assert_eq!(g1, g2);
    }
}
