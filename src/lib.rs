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
//! - [`MedianPruner`] — prune trials below the median of previous trials (default)
//! - [`PercentilePruner`] — prune trials below a given percentile
//! - [`HyperbandPruner`] — multi-bracket successive halving (Hyperband)
//! - [`SuccessiveHalvingPruner`] — asynchronous successive halving (ASHA)
//! - [`ThresholdPruner`] — prune trials outside a value range
//! - [`PatientPruner`] — patience wrapper for other pruners
//! - [`WilcoxonPruner`] — statistical test-based pruning
//! - [`NopPruner`] — no pruning

pub mod artifacts;
pub mod callbacks;
#[cfg(feature = "cli")]
pub mod cli;
pub mod distributions;
pub mod error;
pub mod integration;
pub mod logging;
pub mod progress_bar;
pub mod importance;
pub mod multi_objective;
pub mod pruners;
pub mod samplers;
pub mod search_space;
pub mod storage;
pub mod study;
pub mod terminators;
pub mod testing;
pub mod trial;
#[cfg(feature = "visualization")]
pub mod visualization;
#[cfg(feature = "visualization-matplotlib")]
pub mod visualization_matplotlib;

// Re-export key types at the crate root for convenience.
pub use artifacts::{
    ArtifactMeta, ArtifactNotFound, ArtifactStore, BackoffArtifactStore,
    FileSystemArtifactStore, download_artifact, get_all_artifact_meta_for_study,
    get_all_artifact_meta_for_trial, upload_artifact,
};
pub use callbacks::{
    Callback, MaxTrialsCallback, RetryFailedTrialCallback, TerminatorCallback,
};
pub use distributions::{
    CategoricalChoice, CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
    ParamValue,
};
pub use error::{OptunaError, Result};
pub use importance::{
    get_param_importances, FanovaEvaluator, ImportanceEvaluator,
    MeanDecreaseImpurityEvaluator, PedAnovaEvaluator,
};
pub use multi_objective::{
    CONSTRAINTS_KEY, constrained_dominates, constrained_fast_non_dominated_sort,
    constraint_violation, crowding_distance, dominates, fast_non_dominated_sort,
    get_feasible_trials, get_non_dominated_box_bounds, get_pareto_front_trials, hypervolume,
    hypervolume_2d, is_feasible, is_pareto_front, solve_hssp,
};
pub use pruners::{
    HyperbandPruner, MedianPruner, NopPruner, PatientPruner, PercentilePruner, Pruner,
    SuccessiveHalvingPruner, ThresholdPruner, WilcoxonPruner,
};
pub use samplers::{
    BruteForceSampler, CmaEsSampler, CmaEsSamplerBuilder, GpSampler, GridSampler, NSGAIISampler,
    NSGAIISamplerBuilder, NSGAIIISampler, NSGAIIISamplerBuilder, PartialFixedSampler, QmcSampler,
    RandomSampler, Sampler, TpeSampler, TpeSamplerBuilder,
    ga::GaSampler,
};
pub use search_space::{
    GroupDecomposedSearchSpace, IntersectionSearchSpace, SearchSpaceGroup, SearchSpaceTransform,
};
pub use storage::{
    CachedStorage, InMemoryStorage, JournalBackend, JournalFileBackend, JournalFileStorage,
    JournalStorage, Storage,
    heartbeat::{Heartbeat, HeartbeatHandle, HeartbeatThread, fail_stale_trials, is_heartbeat_enabled, start_heartbeat},
};
#[cfg(feature = "rdb")]
pub use storage::RdbStorage;
#[cfg(feature = "redis-storage")]
pub use storage::JournalRedisBackend;
#[cfg(feature = "s3")]
pub use artifacts::s3::S3ArtifactStore;
#[cfg(feature = "gcs")]
pub use artifacts::gcs::GcsArtifactStore;
#[cfg(feature = "grpc")]
pub use storage::{GrpcStorageProxy, run_grpc_proxy_server};
pub use integration::{
    CsvLoggerCallback, JsonLoggerCallback, DebugPrintCallback,
    TensorBoardCallback, WandbCallback, WandbLogger,
    PruningMixin, PruneDecision,
    ExperimentTracker, TrackerCallback,
};
#[cfg(feature = "mlflow")]
pub use integration::MLflowCallback;
pub use testing::{
    DeterministicSampler, DeterministicPruner, create_frozen_trial,
    fail_objective, pruned_objective, create_storage,
    test_sampler_basic, test_sampler_multi_objective,
    test_storage_crud, test_storage_concurrent,
    prepare_study_with_trials, STORAGE_MODES,
};
pub use study::{
    copy_study, create_study, delete_study, get_all_study_names, get_all_study_summaries,
    load_study, FrozenStudy, Study, StudyDirection,
};
pub use terminators::{
    BestValueStagnationEvaluator, BestValueStagnationTerminator, CrossValidationErrorEvaluator,
    EMMREvaluator, ErrorEvaluator, EvaluatorTerminator, ImprovementEvaluator,
    ImprovementTerminator, MaxTrialsTerminator, MedianErrorEvaluator, NoImprovementTerminator,
    RegretBoundEvaluator, StaticErrorEvaluator, TargetValueTerminator, Terminator,
    report_cross_validation_scores, DEFAULT_MIN_N_TRIALS,
};
pub use trial::{create_trial, FixedTrial, FrozenTrial, Trial, TrialState};
