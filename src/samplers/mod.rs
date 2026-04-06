mod brute_force;
pub mod cmaes;
pub mod ga;
pub mod gp;
pub mod gp_optim_mixed;
mod grid;
pub mod nsgaii;
pub mod nsgaiii;
mod partial_fixed;
pub mod qmc;
pub(crate) mod random;
pub mod tpe;

pub use brute_force::BruteForceSampler;
pub use cmaes::{CmaEsSampler, CmaEsSamplerBuilder, CmaState};
pub use gp::GpSampler;
pub use grid::GridSampler;
pub use nsgaii::{NSGAIISampler, NSGAIISamplerBuilder};
pub use nsgaiii::{NSGAIIISampler, NSGAIIISamplerBuilder};
pub use partial_fixed::PartialFixedSampler;
pub use qmc::{QmcSampler, QmcType};
pub use random::RandomSampler;
pub use tpe::{TpeSampler, TpeSamplerBuilder};

use std::collections::HashMap;

use indexmap::IndexMap;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::trial::{FrozenTrial, TrialState};

/// The sampler trait: decides which parameter values to try next.
///
/// Corresponds to Python `optuna.samplers.BaseSampler`.
pub trait Sampler: Send + Sync {
    /// Infer the search space for relative sampling from completed trials.
    ///
    /// Returns an ordered map from param name to distribution for parameters that
    /// should be sampled together (relative sampling).
    /// 对齐 Python: 返回 IndexMap 保持键排序 (等同 Python dict(sorted(...))).
    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> IndexMap<String, Distribution> {
        let _ = trials;
        IndexMap::new()
    }

    /// Sample parameters jointly in the relative search space.
    ///
    /// Returns a map from param name to internal f64 value.
    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &IndexMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        let _ = (trials, search_space);
        Ok(HashMap::new())
    }

    /// Sample a single parameter independently.
    ///
    /// 对齐 Python `BaseSampler.sample_independent(study, trial, param_name, param_distribution)`。
    /// `trials` 为当前 study 的所有历史试验，用于基于历史信息的采样（如 TPE）。
    fn sample_independent(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64>;

    /// Called before a trial starts (optional hook).
    ///
    /// 对齐 Python `BaseSampler.before_trial(study, trial)`:
    /// 接收 trial_id 和 storage 引用，允许采样器设置 system_attrs（如 GridSampler 的 grid_id）。
    fn before_trial(&self, _trials: &[FrozenTrial], _trial_id: i64, _storage: &dyn crate::storage::Storage) {}

    /// Called after a trial finishes (optional hook).
    fn after_trial(
        &self,
        _trials: &[FrozenTrial],
        _trial: &FrozenTrial,
        _state: TrialState,
        _values: Option<&[f64]>,
    ) {
    }

    /// Whether the sampler wants the study to stop after this trial.
    /// Used by `BruteForceSampler` to signal exhaustion from `after_trial`.
    fn should_stop_study(&self) -> bool {
        false
    }

    /// 对齐 Python `_process_constraints_after_trial`:
    /// 如果采样器设置了约束函数，计算并返回约束值。
    /// 默认返回 None（无约束）。tell() 会将返回的值存储到 storage。
    fn compute_constraints(
        &self,
        _trial: &FrozenTrial,
        _state: TrialState,
    ) -> Option<Vec<f64>> {
        None
    }

    /// 对齐 Python `BaseSampler.reseed_rng(seed)`:
    /// 重新设置随机数生成器种子。用于多线程模式下确保线程间随机性独立。
    /// 默认空操作。
    fn reseed_rng(&self, _seed: u64) {}

    /// 注入 storage 引用。
    ///
    /// 需要状态持久化的采样器（如 CMA-ES）覆盖此方法。
    /// 在 Study 构造时自动调用。默认空操作。
    fn inject_storage(&self, _storage: std::sync::Arc<dyn crate::storage::Storage>, _study_id: i64) {}
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{Distribution, FloatDistribution};
    use crate::trial::FrozenTrial;
    use std::collections::HashMap;

    /// 测试用最小 Sampler 实现：只实现 sample_independent，其余走默认
    struct DummySampler;
    impl Sampler for DummySampler {
        fn sample_independent(
            &self,
            _trials: &[FrozenTrial],
            _trial: &FrozenTrial,
            _param_name: &str,
            _distribution: &Distribution,
        ) -> Result<f64> {
            Ok(0.5)
        }
    }

    fn make_trial() -> FrozenTrial {
        FrozenTrial::new(
            0,
            TrialState::Running,
            None,
            None,
            Some(chrono::Utc::now()),
            None,
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            0,
        )
        .unwrap()
    }

    /// 对齐 Python: infer_relative_search_space 默认返回空
    #[test]
    fn test_default_infer_relative_search_space() {
        let s = DummySampler;
        let space = s.infer_relative_search_space(&[]);
        assert!(space.is_empty());
    }

    /// 对齐 Python: sample_relative 默认返回空
    #[test]
    fn test_default_sample_relative() {
        let s = DummySampler;
        let result = s.sample_relative(&[], &IndexMap::new()).unwrap();
        assert!(result.is_empty());
    }

    /// 对齐 Python: before_trial / after_trial 默认不 panic
    #[test]
    fn test_default_hooks_no_panic() {
        let s = DummySampler;
        let storage = crate::storage::InMemoryStorage::new();
        s.before_trial(&[], 0, &storage);
        let trial = make_trial();
        s.after_trial(&[], &trial, TrialState::Complete, Some(&[1.0]));
    }

    /// 对齐 Python: sample_independent 基本行为
    #[test]
    fn test_sample_independent() {
        let s = DummySampler;
        let trial = make_trial();
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        let val = s.sample_independent(&[], &trial, "x", &dist).unwrap();
        assert_eq!(val, 0.5);
    }

    /// 对齐 Python: reseed_rng 默认空操作不 panic
    #[test]
    fn test_default_reseed_rng() {
        let s = DummySampler;
        s.reseed_rng(42);
        s.reseed_rng(0);
        s.reseed_rng(u64::MAX);
    }

    /// 对齐 Python: should_stop_study 默认返回 false
    #[test]
    fn test_default_should_stop_study() {
        let s = DummySampler;
        assert!(!s.should_stop_study());
    }

    /// 对齐 Python: compute_constraints 默认返回 None
    #[test]
    fn test_default_compute_constraints() {
        let s = DummySampler;
        let trial = make_trial();
        assert!(s.compute_constraints(&trial, TrialState::Complete).is_none());
    }

    /// 对齐 Python: inject_storage 默认空操作不 panic
    #[test]
    fn test_default_inject_storage() {
        let s = DummySampler;
        let storage: std::sync::Arc<dyn crate::storage::Storage> =
            std::sync::Arc::new(crate::storage::InMemoryStorage::new());
        s.inject_storage(storage, 0);
    }

    /// 对齐 Python: after_trial 各种状态都不 panic
    #[test]
    fn test_after_trial_all_states() {
        let s = DummySampler;
        let trial = make_trial();
        s.after_trial(&[], &trial, TrialState::Complete, Some(&[1.0]));
        s.after_trial(&[], &trial, TrialState::Pruned, None);
        s.after_trial(&[], &trial, TrialState::Fail, None);
    }

    /// 对齐 Python: infer_relative_search_space 有历史也返回空
    #[test]
    fn test_infer_relative_with_history() {
        let s = DummySampler;
        let completed = FrozenTrial::new(
            0, TrialState::Complete, Some(1.0), None,
            Some(chrono::Utc::now()), Some(chrono::Utc::now()),
            HashMap::new(), HashMap::new(), HashMap::new(),
            HashMap::new(), HashMap::new(), 0,
        ).unwrap();
        let space = s.infer_relative_search_space(&[completed]);
        assert!(space.is_empty());
    }

    /// 对齐 Python: sample_independent 带 int 分布
    #[test]
    fn test_sample_independent_int_dist() {
        let s = DummySampler;
        let trial = make_trial();
        let dist = Distribution::IntDistribution(
            crate::distributions::IntDistribution::new(0, 10, false, 1).unwrap(),
        );
        let val = s.sample_independent(&[], &trial, "n", &dist).unwrap();
        assert_eq!(val, 0.5);
    }
}
