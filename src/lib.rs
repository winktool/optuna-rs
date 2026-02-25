//! # optuna-rs
//!
//! A Rust port of [Optuna](https://optuna.org/) — an automatic hyperparameter
//! optimization framework.
//!
//! This crate provides a define-by-run API for hyperparameter optimization,
//! supporting both single-objective and multi-objective optimization with
//! state-of-the-art algorithms.
//!
//! ## Quick Start
//!
//! ```rust
//! use optuna_rs::{create_study, StudyDirection};
//!
//! let study = create_study(None, None, None, None,
//!     Some(StudyDirection::Minimize), None, false).unwrap();
//!
//! study.optimize(|trial| {
//!     let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
//!     Ok(x * x)
//! }, Some(100), None, None).unwrap();
//!
//! println!("Best value: {}", study.best_value().unwrap());
//! ```
//!
//! ## Samplers
//!
//! - [`RandomSampler`] — uniform random sampling
//! - [`TpeSampler`] — Tree-structured Parzen Estimator
//! - [`CmaEsSampler`] — Covariance Matrix Adaptation Evolution Strategy
//! - [`NSGAIISampler`] — NSGA-II for multi-objective optimization
//! - [`NSGAIIISampler`] — NSGA-III for many-objective optimization
//! - [`GridSampler`] — exhaustive grid search
//! - [`QmcSampler`] — quasi-Monte Carlo sampling
//! - [`BruteForceSampler`] — brute-force enumeration
//!
//! ## Pruners
//!
//! - [`MedianPruner`] — prune trials below the median of previous trials
//! - [`PercentilePruner`] — prune trials below a given percentile
//! - [`NopPruner`] — no pruning (default)

pub mod callbacks;
pub mod distributions;
pub mod error;
pub mod importance;
pub mod multi_objective;
pub mod pruners;
pub mod samplers;
pub mod search_space;
pub mod storage;
pub mod study;
pub mod terminators;
pub mod trial;

// Re-export key types at the crate root for convenience.
pub use distributions::{
    CategoricalChoice, CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
    ParamValue,
};
pub use error::{OptunaError, Result};
pub use importance::{get_param_importances, FanovaEvaluator, ImportanceEvaluator};
pub use multi_objective::{
    crowding_distance, dominates, fast_non_dominated_sort, get_pareto_front_trials,
    hypervolume_2d, is_pareto_front,
};
pub use pruners::{MedianPruner, NopPruner, PercentilePruner, Pruner};
pub use samplers::{
    BruteForceSampler, CmaEsSampler, CmaEsSamplerBuilder, GridSampler, NSGAIISampler,
    NSGAIISamplerBuilder, NSGAIIISampler, NSGAIIISamplerBuilder, PartialFixedSampler, QmcSampler,
    RandomSampler, Sampler, TpeSampler, TpeSamplerBuilder,
};
pub use search_space::{IntersectionSearchSpace, SearchSpaceTransform};
pub use storage::{InMemoryStorage, Storage};
pub use study::{create_study, FrozenStudy, Study, StudyDirection};
pub use terminators::{
    MaxTrialsTerminator, NoImprovementTerminator, TargetValueTerminator, Terminator,
};
pub use trial::{FixedTrial, FrozenTrial, Trial, TrialState};
