mod brute_force;
mod cmaes;
pub mod ga;
pub(crate) mod gp;
#[cfg(feature = "gp-lbfgsb")]
pub mod gp_lbfgsb;
mod grid;
pub mod nsgaii;
pub mod nsgaiii;
mod partial_fixed;
pub mod qmc;
pub(crate) mod random;
mod tpe;

pub use brute_force::BruteForceSampler;
pub use cmaes::{CmaEsSampler, CmaEsSamplerBuilder};
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
    fn before_trial(&self, _trials: &[FrozenTrial]) {}

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
        s.before_trial(&[]);
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
}
