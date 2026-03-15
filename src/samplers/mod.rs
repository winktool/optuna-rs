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

use crate::distributions::Distribution;
use crate::error::Result;
use crate::trial::{FrozenTrial, TrialState};

/// The sampler trait: decides which parameter values to try next.
///
/// Corresponds to Python `optuna.samplers.BaseSampler`.
pub trait Sampler: Send + Sync {
    /// Infer the search space for relative sampling from completed trials.
    ///
    /// Returns a map from param name to distribution for parameters that
    /// should be sampled together (relative sampling).
    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> HashMap<String, Distribution> {
        let _ = trials;
        HashMap::new()
    }

    /// Sample parameters jointly in the relative search space.
    ///
    /// Returns a map from param name to internal f64 value.
    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &HashMap<String, Distribution>,
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
}
