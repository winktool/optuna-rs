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
