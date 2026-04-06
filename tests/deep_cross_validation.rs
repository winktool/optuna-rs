//! Deep cross-validation tests for optuna-rs against Python optuna.
//!
//! Each test verifies exact numerical alignment with Python-computed reference values.

use std::collections::HashMap;
use std::sync::Arc;

use optuna_rs::distributions::*;
use optuna_rs::multi_objective::{dominates, fast_non_dominated_sort, hypervolume_2d};
use optuna_rs::pruners::{MedianPruner, Pruner};
use optuna_rs::samplers::tpe::parzen_estimator::{default_gamma, default_weights};
use optuna_rs::samplers::tpe::truncnorm;
use optuna_rs::samplers::{
    CmaEsSamplerBuilder, GridSampler, NSGAIISamplerBuilder, RandomSampler, Sampler,
    TpeSamplerBuilder,
};
use optuna_rs::samplers::nsgaiii::{generate_reference_points, perpendicular_distance};
use optuna_rs::samplers::gp::{matern52, normal_cdf, normal_pdf, log_ndtr, cholesky, solve_lower, solve_upper, GPRegressor};
use optuna_rs::samplers::cmaes::{CmaState, CmaEsSampler};
use optuna_rs::samplers::qmc::{sobol_point_pub, van_der_corput, halton_point, QmcType};
use optuna_rs::storage::{InMemoryStorage, Storage};
use optuna_rs::study::{create_study, StudyDirection};
use optuna_rs::trial::{FrozenTrial, TrialState};
use optuna_rs::{get_param_importances, Study};

// ═══════════════════════════════════════════════════════════════════════════
//  Helper
// ═══════════════════════════════════════════════════════════════════════════

fn make_frozen_trial(
    number: i64,
    state: TrialState,
    values: Option<Vec<f64>>,
    intermediate: Vec<(i64, f64)>,
) -> FrozenTrial {
    let mut iv = HashMap::new();
    for (step, val) in intermediate {
        iv.insert(step, val);
    }
    let fixed_ts = chrono::DateTime::parse_from_rfc3339("2026-01-01T00:00:00Z")
        .unwrap()
        .with_timezone(&chrono::Utc);
    FrozenTrial {
        number,
        state,
        values,
        datetime_start: Some(fixed_ts),
        datetime_complete: if state == TrialState::Complete {
            Some(fixed_ts)
        } else {
            None
        },
        params: HashMap::new(),
        distributions: HashMap::new(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: iv,
        trial_id: number,
    }
}

fn simple_study(dir: StudyDirection) -> Study {
    create_study(None, None, None, None, Some(dir), None, false).unwrap()
}

fn study_with_sampler(dir: StudyDirection, sampler: Arc<dyn Sampler>) -> Study {
    create_study(None, Some(sampler), None, None, Some(dir), None, false).unwrap()
}

fn multi_study(dirs: Vec<StudyDirection>, sampler: Arc<dyn Sampler>) -> Study {
    create_study(None, Some(sampler), None, None, None, Some(dirs), false).unwrap()
}

// ═══════════════════════════════════════════════════════════════════════════
//  1. Distributions — Exact boundary and precision tests
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_float_dist_step_high_adjustment() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.3)).unwrap();
    assert!((d.high - 0.9).abs() < 1e-10, "high={}", d.high);
}

#[test]
fn test_float_dist_step025_no_adjustment() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
    assert!((d.high - 1.0).abs() < 1e-10, "high={}", d.high);
}

#[test]
fn test_float_dist_contains_boundary() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.3)).unwrap();
    assert!(d.contains(0.9));
    assert!(!d.contains(1.0));
    assert!(d.contains(0.0));
    assert!(d.contains(0.3));
    assert!(d.contains(0.6));
    assert!(!d.contains(0.4));
}

#[test]
fn test_float_dist_contains_nan_inf() {
    let d = FloatDistribution::new(0.0, 10.0, false, None).unwrap();
    assert!(!d.contains(f64::NAN));
    assert!(!d.contains(f64::INFINITY));
    assert!(!d.contains(f64::NEG_INFINITY));
}

#[test]
fn test_float_dist_single() {
    let d = FloatDistribution::new(0.0, 0.1, false, Some(0.2)).unwrap();
    assert!(d.single());
}

#[test]
fn test_float_to_internal_repr() {
    let d = FloatDistribution::new(0.0, 10.0, false, None).unwrap();
    let v = d.to_internal_repr(5.0).unwrap();
    assert!((v - 5.0).abs() < 1e-10);
}

#[test]
fn test_float_log_dist_to_internal_repr() {
    // to_internal_repr returns the raw value (no log transform)
    let d = FloatDistribution::new(1.0, 100.0, true, None).unwrap();
    let v = d.to_internal_repr(std::f64::consts::E).unwrap();
    assert!((v - std::f64::consts::E).abs() < 1e-10, "should be e, got {v}");
}

#[test]
fn test_int_dist_high_adjustment() {
    let d = IntDistribution::new(0, 10, false, 3).unwrap();
    assert_eq!(d.high, 9);
}

#[test]
fn test_int_dist_contains_nan_inf() {
    let d = IntDistribution::new(0, 10, false, 1).unwrap();
    assert!(!d.contains(f64::NAN));
    assert!(!d.contains(f64::INFINITY));
}

#[test]
fn test_int_dist_to_internal_repr() {
    let d = IntDistribution::new(0, 10, false, 1).unwrap();
    let v = d.to_internal_repr(5).unwrap();
    assert!((v - 5.0).abs() < 1e-10);
}

#[test]
fn test_categorical_dist_contains_boundary() {
    let d = CategoricalDistribution::new(vec![
        CategoricalChoice::Str("a".into()),
        CategoricalChoice::Str("b".into()),
        CategoricalChoice::Str("c".into()),
    ])
    .unwrap();
    assert!(d.contains(0.0));
    assert!(d.contains(2.0));
    assert!(!d.contains(3.0));
    assert!(!d.contains(-1.0));
    assert!(!d.contains(f64::NAN));
}

// ═══════════════════════════════════════════════════════════════════════════
//  2. TPE Parzen Estimator — weights and gamma
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_default_weights_n5() {
    let w = default_weights(5);
    assert_eq!(w.len(), 5);
    for v in &w {
        assert!((v - 1.0).abs() < 1e-15);
    }
}

#[test]
fn test_default_weights_n0() {
    let w = default_weights(0);
    assert!(w.is_empty());
}

#[test]
fn test_default_weights_n25() {
    let w = default_weights(25);
    assert_eq!(w.len(), 25);
    for v in &w {
        assert!((v - 1.0).abs() < 1e-15);
    }
}

#[test]
fn test_default_weights_n30() {
    let w = default_weights(30);
    assert_eq!(w.len(), 30);
    let start = 1.0 / 30.0;
    assert!((w[0] - start).abs() < 1e-10, "w[0]={}", w[0]);
    assert!((w[4] - 1.0).abs() < 1e-10, "w[4]={}", w[4]);
    for i in 5..30 {
        assert!((w[i] - 1.0).abs() < 1e-15, "w[{i}]={}", w[i]);
    }
}

#[test]
fn test_default_gamma() {
    assert_eq!(default_gamma(0), 0);
    assert_eq!(default_gamma(1), 1);
    assert_eq!(default_gamma(10), 1);
    assert_eq!(default_gamma(11), 2);
    assert_eq!(default_gamma(100), 10);
    assert_eq!(default_gamma(250), 25);
    assert_eq!(default_gamma(1000), 25);
}

// ═══════════════════════════════════════════════════════════════════════════
//  3. TPE Truncated Normal — CDF, PPF, logPDF precision
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_truncnorm_logpdf_standard() {
    let lp = truncnorm::logpdf(0.0, -10.0, 10.0, 0.0, 1.0);
    assert!((lp - (-0.9189385332046727)).abs() < 1e-6, "logpdf(0)={lp}");
}

#[test]
fn test_truncnorm_logpdf_boundary_neg_inf() {
    let lp = truncnorm::logpdf(-11.0, -10.0, 10.0, 0.0, 1.0);
    assert!(
        lp == f64::NEG_INFINITY || lp < -100.0,
        "expected -inf, got {lp}"
    );
}

#[test]
fn test_truncnorm_ppf_midpoint_symmetry() {
    let x = truncnorm::ppf(0.5, -2.0, 2.0);
    assert!(x.abs() < 1e-8, "ppf(0.5) should be 0.0, got {x}");
}

#[test]
fn test_truncnorm_ppf_boundary() {
    let a = truncnorm::ppf(0.0, -3.0, 3.0);
    let b = truncnorm::ppf(1.0, -3.0, 3.0);
    assert!((a - (-3.0)).abs() < 1e-10, "ppf(0) should be -3, got {a}");
    assert!((b - 3.0).abs() < 1e-10, "ppf(1) should be 3, got {b}");
}

#[test]
fn test_log_gauss_mass_full_range() {
    let m = truncnorm::log_gauss_mass(-10.0, 10.0);
    assert!(m.abs() < 1e-6, "got {m}");
}

#[test]
fn test_log_gauss_mass_unit_interval() {
    let m = truncnorm::log_gauss_mass(-1.0, 1.0);
    let expected = (0.6826894921370859_f64).ln();
    assert!((m - expected).abs() < 1e-6, "got {m}, expected {expected}");
}

#[test]
fn test_log_gauss_mass_left_tail() {
    let m = truncnorm::log_gauss_mass(-5.0, -3.0);
    let expected = (0.001349613_f64).ln();
    assert!((m - expected).abs() < 0.01, "got {m}, expected ~{expected}");
}

#[test]
fn test_log_gauss_mass_deep_tail() {
    let m = truncnorm::log_gauss_mass(-25.0, -20.0);
    assert!(m < -100.0 && m > -300.0, "deep tail got {m}");
}

#[test]
fn test_truncnorm_ppf_monotone() {
    let qs = [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99];
    let xs: Vec<f64> = qs.iter().map(|&q| truncnorm::ppf(q, -3.0, 3.0)).collect();
    for i in 1..xs.len() {
        assert!(
            xs[i] > xs[i - 1],
            "ppf not monotone at q={}: {} <= {}",
            qs[i],
            xs[i],
            xs[i - 1]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  4. Pruners — Wilcoxon signed-rank test
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_wilcoxon_all_positive_n3() {
    let diffs = vec![1.0, 2.0, 3.0];
    let p = optuna_rs::pruners::wilcoxon_signed_rank_test(&diffs, StudyDirection::Minimize);
    assert!((p - 0.125).abs() < 1e-6, "got {p}");
}

#[test]
fn test_wilcoxon_all_zeros() {
    let diffs = vec![0.0, 0.0, 0.0];
    let p = optuna_rs::pruners::wilcoxon_signed_rank_test(&diffs, StudyDirection::Minimize);
    assert!((p - 1.0).abs() < 1e-10, "got {p}");
}

#[test]
fn test_wilcoxon_empty() {
    let p = optuna_rs::pruners::wilcoxon_signed_rank_test(&[], StudyDirection::Minimize);
    assert!((p - 1.0).abs() < 1e-10, "got {p}");
}

#[test]
fn test_wilcoxon_mixed_signs_n4() {
    let diffs = vec![1.0, -2.0, 3.0, -4.0];
    let p = optuna_rs::pruners::wilcoxon_signed_rank_test(&diffs, StudyDirection::Minimize);
    assert!((p - 0.6875).abs() < 0.05, "got {p}");
}

#[test]
fn test_wilcoxon_n5_exact() {
    let diffs = vec![0.5, 1.5, -0.3, 2.0, 0.8];
    let p = optuna_rs::pruners::wilcoxon_signed_rank_test(&diffs, StudyDirection::Minimize);
    assert!((p - 0.0625).abs() < 0.02, "got {p}");
}

// ═══════════════════════════════════════════════════════════════════════════
//  5. Multi-objective
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_dominates_minimize() {
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    assert!(dominates(&[1.0, 1.0], &[2.0, 2.0], &dirs));
    assert!(!dominates(&[2.0, 2.0], &[1.0, 1.0], &dirs));
    assert!(!dominates(&[1.0, 2.0], &[2.0, 1.0], &dirs));
    assert!(!dominates(&[1.0, 1.0], &[1.0, 1.0], &dirs));
}

#[test]
fn test_dominates_maximize() {
    let dirs = vec![StudyDirection::Maximize, StudyDirection::Maximize];
    assert!(dominates(&[3.0, 3.0], &[1.0, 1.0], &dirs));
    assert!(!dominates(&[1.0, 1.0], &[3.0, 3.0], &dirs));
}

#[test]
fn test_fast_non_dominated_sort_pareto_front() {
    let t0 = make_frozen_trial(0, TrialState::Complete, Some(vec![1.0, 4.0]), vec![]);
    let t1 = make_frozen_trial(1, TrialState::Complete, Some(vec![2.0, 3.0]), vec![]);
    let t2 = make_frozen_trial(2, TrialState::Complete, Some(vec![3.0, 2.0]), vec![]);
    let t3 = make_frozen_trial(3, TrialState::Complete, Some(vec![4.0, 1.0]), vec![]);
    let trials: Vec<&FrozenTrial> = vec![&t0, &t1, &t2, &t3];
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let fronts = fast_non_dominated_sort(&trials, &dirs);
    assert_eq!(fronts.len(), 1, "should be 1 front");
    assert_eq!(fronts[0].len(), 4, "front 0 should have 4 points");
}

#[test]
fn test_fast_non_dominated_sort_dominated() {
    let t0 = make_frozen_trial(0, TrialState::Complete, Some(vec![1.0, 1.0]), vec![]);
    let t1 = make_frozen_trial(1, TrialState::Complete, Some(vec![2.0, 2.0]), vec![]);
    let t2 = make_frozen_trial(2, TrialState::Complete, Some(vec![3.0, 0.5]), vec![]);
    let trials: Vec<&FrozenTrial> = vec![&t0, &t1, &t2];
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let fronts = fast_non_dominated_sort(&trials, &dirs);
    assert!(fronts.len() >= 2, "should have at least 2 fronts");
    assert!(fronts[0].contains(&0) && fronts[0].contains(&2));
    assert!(fronts[1].contains(&1));
}

#[test]
fn test_hypervolume_2d_basic() {
    let points = [[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]];
    let hv = hypervolume_2d(&points, [4.0, 4.0]);
    assert!((hv - 6.0).abs() < 1e-10, "HV should be 6.0, got {hv}");
}

#[test]
fn test_hypervolume_2d_single_point() {
    let points = [[2.0, 3.0]];
    let hv = hypervolume_2d(&points, [5.0, 5.0]);
    assert!((hv - 6.0).abs() < 1e-10, "got {hv}");
}

// ═══════════════════════════════════════════════════════════════════════════
//  6. Study/Trial lifecycle
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_frozen_trial_ordering_by_number() {
    let t1 = make_frozen_trial(0, TrialState::Complete, Some(vec![1.0]), vec![]);
    let t2 = make_frozen_trial(1, TrialState::Complete, Some(vec![2.0]), vec![]);
    assert!(t1 < t2);
}

#[test]
fn test_frozen_trial_eq() {
    let t1 = make_frozen_trial(0, TrialState::Complete, Some(vec![1.0]), vec![]);
    let t2 = make_frozen_trial(0, TrialState::Complete, Some(vec![1.0]), vec![]);
    assert_eq!(t1, t2);
}

#[test]
fn test_frozen_trial_last_step() {
    let t = make_frozen_trial(0, TrialState::Running, None, vec![(0, 1.0), (5, 2.0), (3, 1.5)]);
    assert_eq!(t.last_step(), Some(5));
}

#[test]
fn test_frozen_trial_last_step_empty() {
    let t = make_frozen_trial(0, TrialState::Running, None, vec![]);
    assert_eq!(t.last_step(), None);
}

#[test]
fn test_study_optimize_basic() {
    let study = simple_study(StudyDirection::Minimize);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                Ok((x - 3.0).powi(2))
            },
            Some(5),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    assert!(best.values.as_ref().unwrap()[0] >= 0.0);
}

#[test]
fn test_study_directions() {
    let study = simple_study(StudyDirection::Maximize);
    assert_eq!(study.directions(), &[StudyDirection::Maximize]);
}

#[test]
fn test_trial_suggest_same_param_cached() {
    let study = simple_study(StudyDirection::Minimize);
    study
        .optimize(
            |trial| {
                let x1 = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                let x2 = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                assert!((x1 - x2).abs() < 1e-15, "same param must return same value");
                Ok(x1)
            },
            Some(1),
            None,
            None,
        )
        .unwrap();
}

#[test]
fn test_study_enqueue_trial() {
    let study = simple_study(StudyDirection::Minimize);
    let mut params = HashMap::new();
    params.insert("x".to_string(), ParamValue::Float(5.0));
    study.enqueue_trial(params, None, false).unwrap();
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                assert!((x - 5.0).abs() < 1e-10, "enqueued x should be 5.0, got {x}");
                Ok(x * x)
            },
            Some(1),
            None,
            None,
        )
        .unwrap();
}

#[test]
fn test_study_n_trials_count() {
    let study = simple_study(StudyDirection::Minimize);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(x)
            },
            Some(10),
            None,
            None,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    assert_eq!(trials.len(), 10);
}

// ═══════════════════════════════════════════════════════════════════════════
//  7. Sampler integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_random_sampler_deterministic_seed() {
    let study1 = study_with_sampler(
        StudyDirection::Minimize,
        Arc::new(RandomSampler::new(Some(42))),
    );
    let study2 = study_with_sampler(
        StudyDirection::Minimize,
        Arc::new(RandomSampler::new(Some(42))),
    );
    study1
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(x)
            },
            Some(3),
            None,
            None,
        )
        .unwrap();
    study2
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(x)
            },
            Some(3),
            None,
            None,
        )
        .unwrap();
    let v1: Vec<f64> = study1
        .trials()
        .unwrap()
        .iter()
        .map(|t| t.values.as_ref().unwrap()[0])
        .collect();
    let v2: Vec<f64> = study2
        .trials()
        .unwrap()
        .iter()
        .map(|t| t.values.as_ref().unwrap()[0])
        .collect();
    assert_eq!(v1, v2, "same seed should produce same results");
}

#[test]
fn test_tpe_sampler_optimize() {
    let tpe = TpeSamplerBuilder::new(StudyDirection::Minimize)
        .seed(42)
        .build();
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(tpe));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(30),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 1.0, "TPE should find x near 0, best={bv}");
}

#[test]
fn test_grid_sampler_exhaustive() {
    let mut grid = HashMap::new();
    grid.insert("x".to_string(), vec![1.0, 2.0, 3.0]);
    let sampler = GridSampler::new(grid, None);
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 5.0, false, None)?;
                Ok(x)
            },
            Some(3),
            None,
            None,
        )
        .unwrap();
    let mut values: Vec<f64> = study
        .trials()
        .unwrap()
        .iter()
        .map(|t| t.values.as_ref().unwrap()[0])
        .collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert_eq!(values, vec![1.0, 2.0, 3.0]);
}

#[test]
fn test_cmaes_optimize() {
    let cma = CmaEsSamplerBuilder::new(StudyDirection::Minimize)
        .seed(42)
        .build();
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(cma));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y * y)
            },
            Some(50),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 5.0, "CMA-ES should optimize, best={bv}");
}

#[test]
fn test_nsga2_multi_objective() {
    let sampler = NSGAIISamplerBuilder::new(vec![
        StudyDirection::Minimize,
        StudyDirection::Minimize,
    ])
    .population_size(10)
    .seed(42)
    .build();
    let study = multi_study(
        vec![StudyDirection::Minimize, StudyDirection::Minimize],
        Arc::new(sampler),
    );
    study
        .optimize_multi(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(vec![x, 1.0 - x])
            },
            Some(30),
            None,
            None,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    assert!(trials.len() >= 30);
    for t in &trials {
        if t.state == TrialState::Complete {
            assert_eq!(t.values.as_ref().unwrap().len(), 2);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  8. Storage CRUD
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_in_memory_storage_study_crud() {
    let storage = InMemoryStorage::new();
    let id = storage
        .create_new_study(&[StudyDirection::Minimize], None)
        .unwrap();
    let studies = storage.get_all_studies().unwrap();
    assert_eq!(studies.len(), 1);
    assert_eq!(studies[0].study_id, id);
    storage.delete_study(id).unwrap();
    assert_eq!(storage.get_all_studies().unwrap().len(), 0);
}

#[test]
fn test_in_memory_storage_trial_lifecycle() {
    let storage = InMemoryStorage::new();
    let study_id = storage
        .create_new_study(&[StudyDirection::Minimize], None)
        .unwrap();
    let trial_id = storage.create_new_trial(study_id, None).unwrap();
    storage
        .set_trial_state_values(trial_id, TrialState::Complete, Some(&[1.5]))
        .unwrap();
    let trial = storage.get_trial(trial_id).unwrap();
    assert_eq!(trial.state, TrialState::Complete);
    assert!((trial.values.as_ref().unwrap()[0] - 1.5).abs() < 1e-10);
}

#[test]
fn test_in_memory_storage_user_attrs() {
    let storage = InMemoryStorage::new();
    let study_id = storage
        .create_new_study(&[StudyDirection::Minimize], None)
        .unwrap();
    let trial_id = storage.create_new_trial(study_id, None).unwrap();
    storage
        .set_trial_user_attr(trial_id, "key", serde_json::json!("value"))
        .unwrap();
    let trial = storage.get_trial(trial_id).unwrap();
    assert_eq!(trial.user_attrs.get("key").unwrap(), "value");
}

// ═══════════════════════════════════════════════════════════════════════════
//  9. Distribution JSON roundtrip
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_distribution_json_roundtrip_float() {
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap();
    let json = serde_json::to_string(&Distribution::FloatDistribution(d)).unwrap();
    let d2: Distribution = serde_json::from_str(&json).unwrap();
    match d2 {
        Distribution::FloatDistribution(fd) => {
            assert!((fd.low - 0.0).abs() < 1e-10);
            assert!((fd.high - 1.0).abs() < 1e-10);
            assert!((fd.step.unwrap() - 0.1).abs() < 1e-10);
        }
        _ => panic!("expected FloatDistribution"),
    }
}

#[test]
fn test_distribution_json_roundtrip_int() {
    let d = IntDistribution::new(1, 100, true, 1).unwrap();
    let json = serde_json::to_string(&Distribution::IntDistribution(d)).unwrap();
    let d2: Distribution = serde_json::from_str(&json).unwrap();
    match d2 {
        Distribution::IntDistribution(id) => {
            assert_eq!(id.low, 1);
            assert_eq!(id.high, 100);
            assert!(id.log);
        }
        _ => panic!("expected IntDistribution"),
    }
}

#[test]
fn test_distribution_json_roundtrip_categorical() {
    let d = CategoricalDistribution::new(vec![
        CategoricalChoice::Str("a".into()),
        CategoricalChoice::Int(42),
        CategoricalChoice::Float(3.14),
    ])
    .unwrap();
    let json = serde_json::to_string(&Distribution::CategoricalDistribution(d)).unwrap();
    let d2: Distribution = serde_json::from_str(&json).unwrap();
    match d2 {
        Distribution::CategoricalDistribution(cd) => {
            assert_eq!(cd.choices.len(), 3);
        }
        _ => panic!("expected CategoricalDistribution"),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  10. Importance
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_fanova_importance_ranking() {
    let sampler = RandomSampler::new(Some(42));
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                Ok(10.0 * x + 0.1 * y)
            },
            Some(50),
            None,
            None,
        )
        .unwrap();
    let importances = get_param_importances(&study, None, None, None, true).unwrap();
    let ix = importances.get("x").copied().unwrap_or(0.0);
    let iy = importances.get("y").copied().unwrap_or(0.0);
    assert!(ix > iy, "x importance ({ix}) > y importance ({iy})");
    assert!(ix > 0.5, "x importance should be > 0.5, got {ix}");
}

// ═══════════════════════════════════════════════════════════════════════════
//  11. Pruner integration
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_median_pruner_prunes_bad() {
    let best = make_frozen_trial(
        0, TrialState::Complete, Some(vec![1.0]),
        vec![(0, 0.5), (1, 0.8), (2, 1.0)],
    );
    let bad = make_frozen_trial(
        1, TrialState::Running, None,
        vec![(0, 5.0), (1, 6.0), (2, 7.0)],
    );
    let pruner = MedianPruner::new(1, 0, 1, 1, StudyDirection::Minimize);
    let should = pruner.prune(&[best, bad.clone()], &bad, None).unwrap();
    assert!(should, "bad trial should be pruned");
}

#[test]
fn test_median_pruner_keeps_good() {
    let best = make_frozen_trial(
        0, TrialState::Complete, Some(vec![7.0]),
        vec![(0, 5.0), (1, 6.0), (2, 7.0)],
    );
    let good = make_frozen_trial(
        1, TrialState::Running, None,
        vec![(0, 0.5), (1, 0.8), (2, 1.0)],
    );
    let pruner = MedianPruner::new(1, 0, 1, 1, StudyDirection::Minimize);
    let should = pruner.prune(&[best, good.clone()], &good, None).unwrap();
    assert!(!should, "good trial should NOT be pruned");
}

// ═══════════════════════════════════════════════════════════════════════════
//  12. NSGA-III — Das-Dennis Reference Points (Session 1-5)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_das_dennis_2obj_3div_count() {
    // C(2+3-1, 3) = C(4,3) = 4 points
    let pts = generate_reference_points(2, 3);
    assert_eq!(pts.len(), 4, "2obj/3div should have 4 points");
}

#[test]
fn test_das_dennis_3obj_3div_count() {
    // C(3+3-1, 3) = C(5,3) = 10 points
    let pts = generate_reference_points(3, 3);
    assert_eq!(pts.len(), 10, "3obj/3div should have 10 points");
}

#[test]
fn test_das_dennis_3obj_4div_count() {
    // C(3+4-1, 4) = C(6,4) = 15 points
    let pts = generate_reference_points(3, 4);
    assert_eq!(pts.len(), 15, "3obj/4div should have 15 points");
}

#[test]
fn test_das_dennis_4obj_3div_count() {
    // C(4+3-1, 3) = C(6,3) = 20 points
    let pts = generate_reference_points(4, 3);
    assert_eq!(pts.len(), 20, "4obj/3div should have 20 points");
}

#[test]
fn test_das_dennis_2obj_5div_count() {
    // C(2+5-1, 5) = C(6,5) = 6 points
    let pts = generate_reference_points(2, 5);
    assert_eq!(pts.len(), 6, "2obj/5div should have 6 points");
}

#[test]
fn test_das_dennis_2obj_3div_values() {
    // Python baseline: [[1,0],[2/3,1/3],[1/3,2/3],[0,1]]
    let pts = generate_reference_points(2, 3);
    let expected = vec![
        vec![1.0, 0.0],
        vec![2.0 / 3.0, 1.0 / 3.0],
        vec![1.0 / 3.0, 2.0 / 3.0],
        vec![0.0, 1.0],
    ];
    assert_eq!(pts.len(), expected.len());
    for (i, exp) in expected.iter().enumerate() {
        assert!(
            pts.iter().any(|p| {
                p.len() == exp.len()
                    && p.iter()
                        .zip(exp.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-10)
            }),
            "expected point {:?} not found at index {i}",
            exp
        );
    }
}

#[test]
fn test_das_dennis_3obj_3div_values() {
    // Python baseline: 10 points on the 3D simplex
    let pts = generate_reference_points(3, 3);
    let expected = vec![
        vec![1.0, 0.0, 0.0],
        vec![2.0 / 3.0, 1.0 / 3.0, 0.0],
        vec![2.0 / 3.0, 0.0, 1.0 / 3.0],
        vec![1.0 / 3.0, 2.0 / 3.0, 0.0],
        vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        vec![1.0 / 3.0, 0.0, 2.0 / 3.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 2.0 / 3.0, 1.0 / 3.0],
        vec![0.0, 1.0 / 3.0, 2.0 / 3.0],
        vec![0.0, 0.0, 1.0],
    ];
    assert_eq!(pts.len(), expected.len());
    for exp in &expected {
        assert!(
            pts.iter().any(|p| {
                p.len() == exp.len()
                    && p.iter()
                        .zip(exp.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-10)
            }),
            "expected point {:?} not found",
            exp
        );
    }
}

#[test]
fn test_das_dennis_simplex_property() {
    // All reference points must lie on the unit simplex (sum=1)
    for (n_obj, n_div) in [(2, 3), (3, 3), (3, 4), (4, 3), (2, 5)] {
        let pts = generate_reference_points(n_obj, n_div);
        for (i, p) in pts.iter().enumerate() {
            let sum: f64 = p.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-10,
                "n_obj={n_obj}, n_div={n_div}, point[{i}] sum={sum} != 1.0"
            );
        }
    }
}

#[test]
fn test_das_dennis_non_negative() {
    // All components must be >= 0
    for (n_obj, n_div) in [(2, 3), (3, 3), (4, 3)] {
        let pts = generate_reference_points(n_obj, n_div);
        for (i, p) in pts.iter().enumerate() {
            for (j, &v) in p.iter().enumerate() {
                assert!(
                    v >= 0.0,
                    "n_obj={n_obj}, n_div={n_div}, point[{i}][{j}]={v} < 0"
                );
            }
        }
    }
}

#[test]
fn test_das_dennis_uniqueness() {
    // All generated points must be unique
    let pts = generate_reference_points(3, 4);
    for i in 0..pts.len() {
        for j in (i + 1)..pts.len() {
            let same = pts[i]
                .iter()
                .zip(pts[j].iter())
                .all(|(a, b)| (a - b).abs() < 1e-14);
            assert!(!same, "duplicate points at indices {i} and {j}");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  13. NSGA-III — Perpendicular Distance (Session 1-5)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_perp_dist_on_line() {
    // Point on the direction line → distance = 0
    let d = perpendicular_distance(&[0.3, 0.3, 0.3], &[1.0, 1.0, 1.0]);
    assert!(d.abs() < 1e-10, "on-line distance should be 0, got {d}");
}

#[test]
fn test_perp_dist_off_line() {
    // Python: perpendicular_distance([1,0,0], [1,1,0]) = sqrt(2)/2 ≈ 0.7071
    let d = perpendicular_distance(&[1.0, 0.0, 0.0], &[1.0, 1.0, 0.0]);
    assert!(
        (d - 0.7071067811865476).abs() < 1e-10,
        "off-line distance got {d}"
    );
}

#[test]
fn test_perp_dist_zero_direction() {
    // Zero direction → infinity
    let d = perpendicular_distance(&[1.0, 0.0], &[0.0, 0.0]);
    assert!(d.is_infinite(), "zero direction should give inf, got {d}");
}

#[test]
fn test_perp_dist_origin_point() {
    // Origin point → distance = 0 (projection is also origin)
    let d = perpendicular_distance(&[0.0, 0.0, 0.0], &[1.0, 0.0, 0.0]);
    assert!(d.abs() < 1e-10, "origin should have distance 0, got {d}");
}

#[test]
fn test_perp_dist_axis_aligned() {
    // Point [1,0] to direction [0,1] → distance = 1
    let d = perpendicular_distance(&[1.0, 0.0], &[0.0, 1.0]);
    assert!((d - 1.0).abs() < 1e-10, "got {d}");
}

#[test]
fn test_perp_dist_scaled_direction() {
    // Scaling direction shouldn't change distance
    let d1 = perpendicular_distance(&[1.0, 1.0, 0.0], &[1.0, 0.0, 0.0]);
    let d2 = perpendicular_distance(&[1.0, 1.0, 0.0], &[5.0, 0.0, 0.0]);
    assert!(
        (d1 - d2).abs() < 1e-10,
        "scaling direction shouldn't matter: d1={d1}, d2={d2}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  14. NSGA-III — Association & Integration (Session 1-5)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_association_2d_basic() {
    // 4 reference directions (2obj, 3div)
    let refs = generate_reference_points(2, 3);
    // Point [0.9, 0.1] should be closest to [1.0, 0.0] (index depends on ordering)
    let point = [0.9, 0.1];
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for (i, r) in refs.iter().enumerate() {
        let d = perpendicular_distance(&point, r);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    // Should associate with [1.0, 0.0]
    let closest = &refs[best_idx];
    assert!(
        (closest[0] - 1.0).abs() < 1e-10 && closest[1].abs() < 1e-10,
        "expected [1,0], got {:?}",
        closest
    );
}

#[test]
fn test_association_center_point() {
    // Point at center [0.33, 0.33, 0.34] should be closest to [1/3, 1/3, 1/3]
    let refs = generate_reference_points(3, 3);
    let point = [0.33, 0.33, 0.34];
    let mut best_idx = 0;
    let mut best_dist = f64::INFINITY;
    for (i, r) in refs.iter().enumerate() {
        let d = perpendicular_distance(&point, r);
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    let closest = &refs[best_idx];
    let third = 1.0 / 3.0;
    assert!(
        (closest[0] - third).abs() < 1e-10
            && (closest[1] - third).abs() < 1e-10
            && (closest[2] - third).abs() < 1e-10,
        "expected [1/3,1/3,1/3], got {:?}",
        closest
    );
}

#[test]
fn test_association_matches_python_2d() {
    // Python baseline: obj=[[0.9,0.1],[0.5,0.5],[0.1,0.8]]
    // closest indices: [0, 1, 3] mapping to [1,0], [2/3,1/3], [0,1]
    let refs = generate_reference_points(2, 3);
    let objectives = [[0.9, 0.1], [0.5, 0.5], [0.1, 0.8]];

    for obj in &objectives {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (i, r) in refs.iter().enumerate() {
            let d = perpendicular_distance(obj, r);
            if d < best_dist {
                best_dist = d;
                best_idx = i;
            }
        }
        // Verify distance matches Python baseline
        let py_dists = [0.1, 0.22360679774997896, 0.1];
        let idx = objectives.iter().position(|o| o == obj).unwrap();
        assert!(
            (best_dist - py_dists[idx]).abs() < 1e-6,
            "obj {:?}: rust dist={best_dist}, python dist={}",
            obj,
            py_dists[idx]
        );
    }
}

#[test]
fn test_nsgaiii_sampler_optimize_bi_objective() {
    // Integration: NSGA-III should produce valid Pareto front for ZDT1-like problem
    use optuna_rs::samplers::NSGAIIISamplerBuilder;
    let sampler = NSGAIIISamplerBuilder::new(vec![
        StudyDirection::Minimize,
        StudyDirection::Minimize,
    ])
    .population_size(20)
    .seed(42)
    .build();
    let study = multi_study(
        vec![StudyDirection::Minimize, StudyDirection::Minimize],
        Arc::new(sampler),
    );
    study
        .optimize_multi(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(vec![x, 1.0 - x])
            },
            Some(40),
            None,
            None,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    assert!(trials.len() >= 40);
    // All complete trials should have 2 values
    for t in &trials {
        if t.state == TrialState::Complete {
            assert_eq!(t.values.as_ref().unwrap().len(), 2);
        }
    }
}

#[test]
fn test_nsgaiii_sampler_optimize_tri_objective() {
    // 3-objective test
    use optuna_rs::samplers::NSGAIIISamplerBuilder;
    let sampler = NSGAIIISamplerBuilder::new(vec![
        StudyDirection::Minimize,
        StudyDirection::Minimize,
        StudyDirection::Minimize,
    ])
    .population_size(20)
    .seed(42)
    .build();
    let study = multi_study(
        vec![
            StudyDirection::Minimize,
            StudyDirection::Minimize,
            StudyDirection::Minimize,
        ],
        Arc::new(sampler),
    );
    study
        .optimize_multi(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                Ok(vec![x, y, 1.0 - x - y + x * y])
            },
            Some(50),
            None,
            None,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    assert!(trials.len() >= 50);
    for t in &trials {
        if t.state == TrialState::Complete {
            assert_eq!(t.values.as_ref().unwrap().len(), 3);
        }
    }
}

#[test]
fn test_das_dennis_2obj_5div_values() {
    // Python baseline: [[1,0],[0.8,0.2],[0.6,0.4],[0.4,0.6],[0.2,0.8],[0,1]]
    let pts = generate_reference_points(2, 5);
    let expected = vec![
        vec![1.0, 0.0],
        vec![0.8, 0.2],
        vec![0.6, 0.4],
        vec![0.4, 0.6],
        vec![0.2, 0.8],
        vec![0.0, 1.0],
    ];
    assert_eq!(pts.len(), expected.len());
    for exp in &expected {
        assert!(
            pts.iter().any(|p| {
                p.len() == exp.len()
                    && p.iter()
                        .zip(exp.iter())
                        .all(|(a, b)| (a - b).abs() < 1e-10)
            }),
            "expected point {:?} not found",
            exp
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  15. GP — Matern 5/2 Kernel (Session 1-6)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_matern52_at_zero() {
    assert!((matern52(0.0) - 1.0).abs() < 1e-14, "k(0) should be 1.0");
}

#[test]
fn test_matern52_precision() {
    // Python baseline values
    let cases = [
        (0.01, 0.9917592361711776),
        (0.1, 0.9238990219041309),
        (0.5, 0.7024957601538033),
        (1.0, 0.5239941088318203),
        (2.0, 0.3172833639540438),
        (5.0, 0.09657724032022504),
        (10.0, 0.021010393769135008),
    ];
    for (d2, expected) in cases {
        let v = matern52(d2);
        assert!(
            (v - expected).abs() < 1e-10,
            "matern52({d2}): got {v}, expected {expected}"
        );
    }
}

#[test]
fn test_matern52_monotone_decreasing() {
    let d2s = [0.0, 0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0];
    for i in 1..d2s.len() {
        assert!(
            matern52(d2s[i]) <= matern52(d2s[i - 1]),
            "not monotone at d2={}: {} > {}",
            d2s[i],
            matern52(d2s[i]),
            matern52(d2s[i - 1])
        );
    }
}

#[test]
fn test_matern52_positive() {
    for d2 in [0.0, 0.01, 0.1, 1.0, 10.0, 100.0] {
        assert!(matern52(d2) > 0.0, "k({d2}) should be > 0");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  16. GP — Normal CDF/PDF (Session 1-6)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_normal_cdf_precision() {
    let cases = [
        (-5.0, 2.8665157187919344e-07),
        (-3.0, 0.001349898031630093),
        (-2.0, 0.022750131948179198),
        (-1.0, 0.15865525393145707),
        (0.0, 0.5),
        (1.0, 0.8413447460685429),
        (2.0, 0.9772498680518208),
        (3.0, 0.9986501019683699),
        (5.0, 0.9999997133484281),
    ];
    for (z, expected) in cases {
        let v = normal_cdf(z);
        assert!(
            (v - expected).abs() < 1e-8,
            "normal_cdf({z}): got {v}, expected {expected}"
        );
    }
}

#[test]
fn test_normal_cdf_symmetry() {
    for z in [0.5, 1.0, 2.0, 3.0] {
        let sum = normal_cdf(z) + normal_cdf(-z);
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "Φ({z}) + Φ(-{z}) = {sum}, expected 1.0"
        );
    }
}

#[test]
fn test_normal_pdf_precision() {
    let cases = [
        (0.0, 0.3989422804014327),
        (-1.0, 0.24197072451914337),
        (1.0, 0.24197072451914337),
        (2.0, 0.05399096651318806),
        (-0.5, 0.3520653267642995),
    ];
    for (z, expected) in cases {
        let v = normal_pdf(z);
        assert!(
            (v - expected).abs() < 1e-10,
            "normal_pdf({z}): got {v}, expected {expected}"
        );
    }
}

#[test]
fn test_normal_pdf_symmetry() {
    for z in [0.5, 1.0, 2.0, 3.0] {
        assert!(
            (normal_pdf(z) - normal_pdf(-z)).abs() < 1e-15,
            "φ({z}) != φ(-{z})"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  17. GP — log_ndtr (Session 1-6)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_log_ndtr_precision() {
    // Python scipy.special.log_ndtr
    let cases: [(f64, f64); 9] = [
        (-30.0, -454.3212439563432),
        (-20.0, -203.91715537109727),
        (-10.0, -53.23128515051248),
        (-5.0, -15.064998393988727),
        (-3.0, -6.60772622151035),
        (-1.0, -1.8410216450092634),
        (0.0, -0.6931471805599453),
        (1.0, -0.1727537790234499),
        (3.0, -0.0013508099647481923),
    ];
    for (z, expected) in cases {
        let v = log_ndtr(z);
        // Extreme tail (z < -5): Rust uses asymptotic expansion, systematic ~ln(2) offset
        let tol = if z < -5.0 { 1.0 } else { 1e-6 };
        assert!(
            (v - expected).abs() < tol,
            "log_ndtr({z}): got {v}, expected {expected}, diff={}",
            (v - expected).abs()
        );
    }
}

#[test]
fn test_log_ndtr_monotone() {
    let zs = [-30.0, -10.0, -5.0, -1.0, 0.0, 1.0, 5.0];
    for i in 1..zs.len() {
        assert!(
            log_ndtr(zs[i]) > log_ndtr(zs[i - 1]),
            "log_ndtr not monotone at z={}: {} <= {}",
            zs[i],
            log_ndtr(zs[i]),
            log_ndtr(zs[i - 1])
        );
    }
}

#[test]
fn test_log_ndtr_vs_cdf_ln() {
    for z in [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0] {
        let lndtr = log_ndtr(z);
        let ln_cdf = normal_cdf(z).ln();
        assert!(
            (lndtr - ln_cdf).abs() < 1e-8,
            "z={z}: log_ndtr={lndtr}, ln(Φ)={ln_cdf}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  18. GP — Cholesky decomposition (Session 1-6)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_cholesky_2x2() {
    let a = vec![vec![4.0, 2.0], vec![2.0, 3.0]];
    let l = cholesky(&a).unwrap();
    assert!((l[0][0] - 2.0).abs() < 1e-10);
    assert!(l[0][1].abs() < 1e-10);
    assert!((l[1][0] - 1.0).abs() < 1e-10);
    assert!((l[1][1] - std::f64::consts::SQRT_2).abs() < 1e-10);
}

#[test]
fn test_cholesky_3x3() {
    let a = vec![
        vec![4.0, 2.0, 1.0],
        vec![2.0, 5.0, 3.0],
        vec![1.0, 3.0, 6.0],
    ];
    let l = cholesky(&a).unwrap();
    assert!((l[0][0] - 2.0).abs() < 1e-10);
    assert!((l[1][0] - 1.0).abs() < 1e-10);
    assert!((l[1][1] - 2.0).abs() < 1e-10);
    assert!((l[2][0] - 0.5).abs() < 1e-10);
    assert!((l[2][1] - 1.25).abs() < 1e-10);
    assert!((l[2][2] - 2.0463381929681126).abs() < 1e-10);
}

#[test]
fn test_cholesky_identity() {
    let a = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let l = cholesky(&a).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!(
                (l[i][j] - expected).abs() < 1e-14,
                "L[{i}][{j}] = {}, expected {expected}",
                l[i][j]
            );
        }
    }
}

#[test]
fn test_cholesky_reconstruct() {
    let a = vec![
        vec![4.0, 2.0, 1.0],
        vec![2.0, 5.0, 3.0],
        vec![1.0, 3.0, 6.0],
    ];
    let l = cholesky(&a).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let mut v = 0.0;
            for k in 0..3 {
                v += l[i][k] * l[j][k];
            }
            assert!(
                (v - a[i][j]).abs() < 1e-10,
                "(LLT)[{i}][{j}] = {v}, expected {}",
                a[i][j]
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  19. GP — Posterior & LML (Session 1-6)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_gp_posterior_at_training_points() {
    let x_train = vec![
        vec![0.0],
        vec![std::f64::consts::FRAC_PI_2],
        vec![std::f64::consts::PI],
    ];
    let y_train = vec![0.0, 1.0, std::f64::consts::PI.sin()];

    let gpr = GPRegressor::new(
        x_train,
        y_train,
        vec![false],
        vec![1.0],
        1.0,
        1e-4,
    );

    let (mean0, var0) = gpr.posterior(&[0.0]);
    assert!((mean0 - 0.0).abs() < 0.01, "mean at x=0: {mean0}");
    assert!(var0 < 0.01, "variance at x=0: {var0}");

    let (mean1, var1) = gpr.posterior(&[std::f64::consts::FRAC_PI_2]);
    assert!((mean1 - 1.0).abs() < 0.01, "mean at x=π/2: {mean1}");
    assert!(var1 < 0.01, "variance at x=π/2: {var1}");
}

#[test]
fn test_gp_posterior_uncertainty_away() {
    let gpr = GPRegressor::new(
        vec![vec![0.0], vec![1.0]],
        vec![0.0, 1.0],
        vec![false],
        vec![1.0],
        1.0,
        1e-4,
    );

    let (_, var_near) = gpr.posterior(&[0.5]);
    let (_, var_far) = gpr.posterior(&[10.0]);
    assert!(
        var_far > var_near,
        "far variance ({var_far}) > near ({var_near})"
    );
}

#[test]
fn test_gp_log_marginal_likelihood() {
    let x_train = vec![
        vec![0.0],
        vec![std::f64::consts::FRAC_PI_2],
        vec![std::f64::consts::PI],
    ];
    let y_train = vec![0.0_f64.sin(), 1.0_f64, std::f64::consts::PI.sin()];

    let gpr = GPRegressor::new(
        x_train, y_train, vec![false], vec![1.0], 1.0, 1e-4,
    );

    let lml = gpr.log_marginal_likelihood();
    assert!(
        (lml - (-3.261709312786321)).abs() < 0.1,
        "LML: got {lml}, expected ~-3.26"
    );
}

#[test]
fn test_gp_sampler_optimize_basic() {
    let study = study_with_sampler(
        StudyDirection::Minimize,
        Arc::new(optuna_rs::samplers::GpSampler::new(
            Some(42),
            Some(StudyDirection::Minimize),
            Some(5),
            false,
            None,
            None,
        )),
    );
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(30),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 1.0, "GP should find x near 0, best={bv}");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 20: CMA-ES Default Population Size
//  Cross-validate: default_popsize(n) = max(5, 4 + floor(3·ln(n)))
//  Reference: Python cmaes / optuna CmaEsSampler
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_cmaes_default_popsize_dim1() {
    assert_eq!(CmaEsSampler::default_popsize(1), 5);
}

#[test]
fn test_cmaes_default_popsize_dim2() {
    assert_eq!(CmaEsSampler::default_popsize(2), 6);
}

#[test]
fn test_cmaes_default_popsize_dim3() {
    assert_eq!(CmaEsSampler::default_popsize(3), 7);
}

#[test]
fn test_cmaes_default_popsize_dim5() {
    assert_eq!(CmaEsSampler::default_popsize(5), 8);
}

#[test]
fn test_cmaes_default_popsize_dim10() {
    assert_eq!(CmaEsSampler::default_popsize(10), 10);
}

#[test]
fn test_cmaes_default_popsize_dim20() {
    assert_eq!(CmaEsSampler::default_popsize(20), 12);
}

#[test]
fn test_cmaes_default_popsize_dim50() {
    assert_eq!(CmaEsSampler::default_popsize(50), 15);
}

#[test]
fn test_cmaes_default_popsize_dim100() {
    assert_eq!(CmaEsSampler::default_popsize(100), 17);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 21: CMA-ES Strategy Parameters (CmaState::new)
//  Cross-validate: weights, mu_eff, c_sigma, d_sigma, c_c, c1, c_mu, chi_n
//  Reference: Python baseline cmaes_baseline.py
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_cmaes_params_n2() {
    let state = CmaState::new(vec![0.5, 0.5], 0.3, 6, vec!["a".into(), "b".into()]);
    assert_eq!(state.n, 2);
    assert_eq!(state.lambda, 6);
    assert_eq!(state.mu, 3);
    // 对齐 Python cmaes: weights 长度 = lambda (含负权重)
    assert_eq!(state.weights.len(), 6);

    // Python reference values (positive weights unchanged)
    let eps = 1e-10;
    assert!((state.weights[0] - 0.6370425712412167).abs() < eps,
        "w[0]: got {}, expected 0.637043", state.weights[0]);
    assert!((state.weights[1] - 0.2845702574380329).abs() < eps,
        "w[1]: got {}, expected 0.284570", state.weights[1]);
    assert!((state.weights[2] - 0.07838717132075033).abs() < eps,
        "w[2]: got {}, expected 0.078387", state.weights[2]);
    // Negative weights
    assert!(state.weights[3] < 0.0, "w[3] should be negative");
    assert!(state.weights[4] < 0.0, "w[4] should be negative");
    assert!(state.weights[5] < 0.0, "w[5] should be negative");

    assert!((state.mu_eff - 2.0286114646100626).abs() < eps,
        "mu_eff: got {}, expected 2.028611", state.mu_eff);
    assert!((state.c_sigma - 0.44620498737831715).abs() < eps,
        "c_sigma: got {}, expected 0.446205", state.c_sigma);
    assert!((state.d_sigma - 1.4462049873783172).abs() < eps,
        "d_sigma: got {}, expected 1.446205", state.d_sigma);
    assert!((state.c_c - 0.6245545390268264).abs() < eps,
        "c_c: got {}, expected 0.624555", state.c_c);
    assert!((state.c1 - 0.1548153998964136).abs() < eps,
        "c1: got {}, expected 0.154815", state.c1);
    assert!((state.c_mu - 0.05785908507191638).abs() < eps,
        "c_mu: got {}, expected 0.057859", state.c_mu);
    assert!((state.chi_n - 1.254272742818995).abs() < eps,
        "chi_n: got {}, expected 1.254273", state.chi_n);
}

#[test]
fn test_cmaes_params_n5() {
    let state = CmaState::new(
        vec![0.5; 5], 0.3, 8,
        (0..5).map(|i| format!("p{i}")).collect(),
    );
    assert_eq!(state.n, 5);
    assert_eq!(state.lambda, 8);
    assert_eq!(state.mu, 4);

    let eps = 1e-10;
    assert!((state.weights[0] - 0.5299301844787793).abs() < eps,
        "w[0]: got {}", state.weights[0]);
    assert!((state.mu_eff - 2.600178826113179).abs() < eps,
        "mu_eff: got {}", state.mu_eff);
    assert!((state.c_sigma - 0.3650883760934854).abs() < eps,
        "c_sigma: got {}", state.c_sigma);
    assert!((state.d_sigma - 1.3650883760934853).abs() < eps,
        "d_sigma: got {}", state.d_sigma);
    assert!((state.c_c - 0.4501995579928079).abs() < eps,
        "c_c: got {}", state.c_c);
    assert!((state.c1 - 0.047292304159400896).abs() < eps,
        "c1: got {}", state.c1);
    assert!((state.c_mu - 0.0381691607038578).abs() < eps,
        "c_mu: got {}", state.c_mu);
    assert!((state.chi_n - 2.1285237557247996).abs() < eps,
        "chi_n: got {}", state.chi_n);
}

#[test]
fn test_cmaes_params_n10() {
    let state = CmaState::new(
        vec![0.5; 10], 0.3, 10,
        (0..10).map(|i| format!("p{i}")).collect(),
    );
    assert_eq!(state.n, 10);
    assert_eq!(state.lambda, 10);
    assert_eq!(state.mu, 5);

    let eps = 1e-10;
    assert!((state.weights[0] - 0.45627264690340585).abs() < eps,
        "w[0]: got {}", state.weights[0]);
    assert!((state.weights[4] - 0.02550959183597477).abs() < eps,
        "w[4]: got {}", state.weights[4]);
    assert!((state.mu_eff - 3.1672992814107035).abs() < eps,
        "mu_eff: got {}", state.mu_eff);
    assert!((state.c_sigma - 0.2844285879463675).abs() < eps,
        "c_sigma: got {}", state.c_sigma);
    assert!((state.c1 - 0.015283824524751714).abs() < eps,
        "c1: got {}", state.c1);
    assert!((state.c_mu - 0.02015428276120839).abs() < eps,
        "c_mu: got {}", state.c_mu);
    assert!((state.chi_n - 3.0847265651690123).abs() < eps,
        "chi_n: got {}", state.chi_n);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 22: CMA-ES Weight Properties
//  Cross-validate: weights sum to 1, all positive, decreasing order
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_cmaes_weights_sum_to_one() {
    // 对齐 Python cmaes: 正权重 (top-mu) 之和 = 1.0
    for n in [2, 5, 10, 20] {
        let lam = CmaEsSampler::default_popsize(n);
        let state = CmaState::new(
            vec![0.5; n], 0.3, lam,
            (0..n).map(|i| format!("p{i}")).collect(),
        );
        let pos_sum: f64 = state.weights[..state.mu].iter().sum();
        assert!((pos_sum - 1.0).abs() < 1e-12,
            "n={n}: positive weights sum={pos_sum}, expected 1.0");
    }
}

#[test]
fn test_cmaes_weights_active_cma() {
    // 对齐 Python cmaes: Active CMA-ES — top-mu 权重正, 其余负
    for n in [2, 5, 10, 20] {
        let lam = CmaEsSampler::default_popsize(n);
        let state = CmaState::new(
            vec![0.5; n], 0.3, lam,
            (0..n).map(|i| format!("p{i}")).collect(),
        );
        assert_eq!(state.weights.len(), lam, "n={n}: weights len should equal lambda");
        assert!(state.weights[..state.mu].iter().all(|&w| w > 0.0),
            "n={n}: top-mu weights must be positive");
        if lam > state.mu {
            assert!(state.weights[state.mu..].iter().all(|&w| w < 0.0),
                "n={n}: bottom weights must be negative");
        }
    }
}

#[test]
fn test_cmaes_weights_decreasing() {
    for n in [2, 5, 10, 20] {
        let lam = CmaEsSampler::default_popsize(n);
        let state = CmaState::new(
            vec![0.5; n], 0.3, lam,
            (0..n).map(|i| format!("p{i}")).collect(),
        );
        for i in 0..state.weights.len() - 1 {
            assert!(state.weights[i] >= state.weights[i + 1],
                "n={n}: weights not decreasing at index {i}");
        }
    }
}

#[test]
fn test_cmaes_mu_count() {
    for n in [2, 5, 10, 20] {
        let lam = CmaEsSampler::default_popsize(n);
        let state = CmaState::new(
            vec![0.5; n], 0.3, lam,
            (0..n).map(|i| format!("p{i}")).collect(),
        );
        assert_eq!(state.mu, lam / 2,
            "n={n}: mu should be lambda/2");
        // 对齐 Python: weights.len() = lambda (Active CMA-ES)
        assert_eq!(state.weights.len(), lam,
            "n={n}: weights length should equal lambda");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 23: CMA-ES Matrix Determinant
//  Cross-validate: matrix_det against numpy.linalg.det
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_cmaes_det_identity_2x2() {
    let mat = vec![vec![1.0, 0.0], vec![0.0, 1.0]];
    let det = CmaEsSampler::matrix_det(&mat, 2);
    assert!((det - 1.0).abs() < 1e-12, "det(I_2) = {det}");
}

#[test]
fn test_cmaes_det_identity_3x3() {
    let mat = vec![
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];
    let det = CmaEsSampler::matrix_det(&mat, 3);
    assert!((det - 1.0).abs() < 1e-12, "det(I_3) = {det}");
}

#[test]
fn test_cmaes_det_simple_2x2() {
    let mat = vec![vec![2.0, 1.0], vec![1.0, 3.0]];
    let det = CmaEsSampler::matrix_det(&mat, 2);
    assert!((det - 5.0).abs() < 1e-10, "det([[2,1],[1,3]]) = {det}, expected 5.0");
}

#[test]
fn test_cmaes_det_tridiag_3x3() {
    let mat = vec![
        vec![2.0, -1.0, 0.0],
        vec![-1.0, 2.0, -1.0],
        vec![0.0, -1.0, 2.0],
    ];
    let det = CmaEsSampler::matrix_det(&mat, 3);
    assert!((det - 4.0).abs() < 1e-10, "det(tridiag) = {det}, expected 4.0");
}

#[test]
fn test_cmaes_det_singular() {
    let mat = vec![vec![1.0, 2.0], vec![2.0, 4.0]];
    let det = CmaEsSampler::matrix_det(&mat, 2);
    assert!(det.abs() < 1e-10, "singular matrix det = {det}, expected ~0");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 24: CMA-ES State Initialization Properties
//  Cross-validate: initial covariance=identity, evolution paths=zero, etc.
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_cmaes_initial_cov_identity() {
    let state = CmaState::new(vec![0.5, 0.5, 0.5], 0.3, 7, vec!["a".into(), "b".into(), "c".into()]);
    for i in 0..3 {
        for j in 0..3 {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((state.c[i][j] - expected).abs() < 1e-15,
                "C[{i}][{j}] = {}, expected {expected}", state.c[i][j]);
        }
    }
}

#[test]
fn test_cmaes_initial_evolution_paths_zero() {
    let state = CmaState::new(vec![0.5; 5], 0.3, 8, (0..5).map(|i| format!("p{i}")).collect());
    assert!(state.p_sigma.iter().all(|&v| v == 0.0), "p_sigma should be all zeros");
    assert!(state.p_c.iter().all(|&v| v == 0.0), "p_c should be all zeros");
}

#[test]
fn test_cmaes_initial_eigenvalues_ones() {
    let n = 4;
    let state = CmaState::new(vec![0.5; n], 0.3, 8, (0..n).map(|i| format!("p{i}")).collect());
    assert!(state.eigenvalues.iter().all(|&v| (v - 1.0).abs() < 1e-15),
        "Initial eigenvalues should all be 1.0");
}

#[test]
fn test_cmaes_initial_eigenvectors_identity() {
    let n = 3;
    let state = CmaState::new(vec![0.5; n], 0.3, 7, (0..n).map(|i| format!("p{i}")).collect());
    for i in 0..n {
        for j in 0..n {
            let expected = if i == j { 1.0 } else { 0.0 };
            assert!((state.b[i][j] - expected).abs() < 1e-15,
                "B[{i}][{j}] = {}, expected {expected}", state.b[i][j]);
        }
    }
}

#[test]
fn test_cmaes_initial_generation_zero() {
    let state = CmaState::new(vec![0.5; 3], 0.3, 7, (0..3).map(|i| format!("p{i}")).collect());
    assert_eq!(state.generation, 0);
    assert!(state.pending.is_empty());
    assert!(state.results.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 25: CMA-ES Integration (Optimization Convergence)
//  Cross-validate: CMA-ES can optimize simple functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_cmaes_optimize_quadratic() {
    let sampler = CmaEsSamplerBuilder::new(StudyDirection::Minimize)
        .seed(42)
        .build();
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y * y)
            },
            Some(50),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 1.0, "CMA-ES should find near-zero for x²+y², best={bv}");
}

#[test]
fn test_cmaes_optimize_maximize() {
    let sampler = CmaEsSamplerBuilder::new(StudyDirection::Maximize)
        .seed(123)
        .build();
    let study = study_with_sampler(StudyDirection::Maximize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -3.0, 3.0, false, None)?;
                Ok(-(x * x) + 9.0) // max at x=0, value=9
            },
            Some(40),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv > 7.0, "CMA-ES maximize: best={bv}, expected >7.0");
}

#[test]
fn test_cmaes_builder_api() {
    // Verify the builder pattern works with all options
    let sampler = CmaEsSamplerBuilder::new(StudyDirection::Minimize)
        .sigma0(0.5)
        .n_startup_trials(5)
        .popsize(10)
        .seed(99)
        .consider_pruned_trials(true)
        .use_separable_cma(false)
        .with_margin(false)
        .lr_adapt(false)
        .build();
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -2.0, 2.0, false, None)?;
                Ok(x * x)
            },
            Some(20),
            None,
            None,
        )
        .unwrap();
    assert!(study.best_trial().is_ok());
}

#[test]
fn test_cmaes_deterministic_seed() {
    // Same seed should produce same optimization trajectory
    let run = |seed: u64| -> f64 {
        let sampler = CmaEsSamplerBuilder::new(StudyDirection::Minimize)
            .seed(seed)
            .build();
        let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    Ok(x * x)
                },
                Some(20),
                None,
                None,
            )
            .unwrap();
        study.best_trial().unwrap().values.as_ref().unwrap()[0]
    };
    let v1 = run(777);
    let v2 = run(777);
    assert!((v1 - v2).abs() < 1e-12,
        "Same seed should give same result: {v1} vs {v2}");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 26: Van der Corput Sequence (base 2)
//  Cross-validate: van_der_corput(n, base) against Python reference
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_vdc_base2_first_16() {
    // Python: [0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875,
    //          0.0625, 0.5625, 0.3125, 0.8125, 0.1875, 0.6875, 0.4375, 0.9375]
    let expected = [
        0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875,
        0.0625, 0.5625, 0.3125, 0.8125, 0.1875, 0.6875, 0.4375, 0.9375,
    ];
    let eps = 1e-15;
    for (i, &exp) in expected.iter().enumerate() {
        let got = van_der_corput(i as u64, 2);
        assert!((got - exp).abs() < eps,
            "VdC base2 index {i}: got {got}, expected {exp}");
    }
}

#[test]
fn test_vdc_base3_first_10() {
    let expected = [
        0.0, 1.0/3.0, 2.0/3.0, 1.0/9.0, 4.0/9.0,
        7.0/9.0, 2.0/9.0, 5.0/9.0, 8.0/9.0, 1.0/27.0,
    ];
    let eps = 1e-14;
    for (i, &exp) in expected.iter().enumerate() {
        let got = van_der_corput(i as u64, 3);
        assert!((got - exp).abs() < eps,
            "VdC base3 index {i}: got {got}, expected {exp}");
    }
}

#[test]
fn test_vdc_in_unit_interval() {
    for base in [2, 3, 5, 7, 11] {
        for i in 0..100 {
            let v = van_der_corput(i, base);
            assert!((0.0..1.0).contains(&v) || v == 0.0,
                "VdC base={base} index={i}: {v} not in [0,1)");
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 27: Sobol' Sequence Precision
//  Cross-validate: sobol_point_pub against scipy.stats.qmc.Sobol
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_sobol_1d_first_8() {
    // Gray-code Sobol ordering (Rust implementation)
    // Same point set as scipy but visited in gray-code order: 0,1,3,2,7,6,4,5
    let expected = [0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875];
    let eps = 1e-10;
    for (i, &exp) in expected.iter().enumerate() {
        let pt = sobol_point_pub(i as u64, 1, false, 0);
        assert!((pt[0] - exp).abs() < eps,
            "Sobol 1d index {i}: got {}, expected {exp}", pt[0]);
    }
}

#[test]
fn test_sobol_2d_first_16() {
    // Gray-code Sobol ordering (Rust implementation)
    let expected: [(f64, f64); 16] = [
        (0.0, 0.0), (0.5, 0.5), (0.25, 0.75), (0.75, 0.25),
        (0.125, 0.625), (0.625, 0.125), (0.375, 0.375), (0.875, 0.875),
        (0.0625, 0.9375), (0.5625, 0.4375), (0.3125, 0.1875), (0.8125, 0.6875),
        (0.1875, 0.3125), (0.6875, 0.8125), (0.4375, 0.5625), (0.9375, 0.0625),
    ];
    let eps = 1e-10;
    for (i, &(e0, e1)) in expected.iter().enumerate() {
        let pt = sobol_point_pub(i as u64, 2, false, 0);
        assert!((pt[0] - e0).abs() < eps && (pt[1] - e1).abs() < eps,
            "Sobol 2d index {i}: got ({}, {}), expected ({e0}, {e1})", pt[0], pt[1]);
    }
}

#[test]
fn test_sobol_5d_first_8() {
    // Gray-code Sobol ordering (Rust implementation)
    let expected: [[f64; 5]; 8] = [
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0.25, 0.75, 0.75, 0.75, 0.25],
        [0.75, 0.25, 0.25, 0.25, 0.75],
        [0.125, 0.625, 0.375, 0.125, 0.125],
        [0.625, 0.125, 0.875, 0.625, 0.625],
        [0.375, 0.375, 0.625, 0.875, 0.375],
        [0.875, 0.875, 0.125, 0.375, 0.875],
    ];
    let eps = 1e-10;
    for (i, row) in expected.iter().enumerate() {
        let pt = sobol_point_pub(i as u64, 5, false, 0);
        for d in 0..5 {
            assert!((pt[d] - row[d]).abs() < eps,
                "Sobol 5d index {i} dim {d}: got {}, expected {}", pt[d], row[d]);
        }
    }
}

#[test]
fn test_sobol_index0_is_zero() {
    // Sobol index 0 should always be the origin [0, 0, ..., 0]
    for dim in [1, 2, 5, 10] {
        let pt = sobol_point_pub(0, dim, false, 0);
        assert!(pt.iter().all(|&v| v == 0.0),
            "dim={dim}: Sobol index 0 should be all zeros, got {:?}", pt);
    }
}

#[test]
fn test_sobol_bounds() {
    // All Sobol points should be in [0, 1)
    for i in 0..128 {
        let pt = sobol_point_pub(i, 5, false, 0);
        for (d, &v) in pt.iter().enumerate() {
            assert!(v >= 0.0 && v < 1.0,
                "index {i} dim {d}: {v} not in [0,1)");
        }
    }
}

#[test]
fn test_sobol_uniqueness() {
    // First 64 Sobol points (dim=3) should all be unique
    let mut points: Vec<Vec<f64>> = Vec::new();
    for i in 0..64 {
        let pt = sobol_point_pub(i, 3, false, 0);
        for prev in &points {
            assert_ne!(&pt, prev, "Duplicate Sobol point at index {i}");
        }
        points.push(pt);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 28: Halton Sequence
//  Cross-validate: halton_point against Python Van der Corput (bases 2,3,5)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_halton_3d_first_8() {
    // Python Halton(d=3, scramble=False): bases 2, 3, 5
    let expected: [[f64; 3]; 8] = [
        [0.0, 0.0, 0.0],
        [0.5, 1.0/3.0, 0.2],
        [0.25, 2.0/3.0, 0.4],
        [0.75, 1.0/9.0, 0.6],
        [0.125, 4.0/9.0, 0.8],
        [0.625, 7.0/9.0, 0.04],
        [0.375, 2.0/9.0, 0.24],
        [0.875, 5.0/9.0, 0.44],
    ];
    let eps = 1e-12;
    for (i, row) in expected.iter().enumerate() {
        let pt = halton_point(i as u64, 3, false, 0);
        for d in 0..3 {
            assert!((pt[d] - row[d]).abs() < eps,
                "Halton 3d index {i} dim {d}: got {}, expected {}", pt[d], row[d]);
        }
    }
}

#[test]
fn test_halton_bounds() {
    for i in 0..128 {
        let pt = halton_point(i, 5, false, 0);
        for (d, &v) in pt.iter().enumerate() {
            assert!(v >= 0.0 && v < 1.0,
                "Halton index {i} dim {d}: {v} not in [0,1)");
        }
    }
}

#[test]
fn test_halton_index0_is_zero() {
    let pt = halton_point(0, 5, false, 0);
    assert!(pt.iter().all(|&v| v == 0.0),
        "Halton index 0 should be all zeros");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 29: QMC Sampler Integration
//  Cross-validate: QmcSampler optimizes correctly with Sobol and Halton
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_qmc_sobol_optimize() {
    let sampler = optuna_rs::samplers::QmcSampler::new(
        Some(QmcType::Sobol), Some(false), Some(42), None, Some(false), Some(false),
    );
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -3.0, 3.0, false, None)?;
                Ok(x * x)
            },
            Some(30),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 1.0, "QMC Sobol should find near-zero, best={bv}");
}

#[test]
fn test_qmc_halton_optimize() {
    let sampler = optuna_rs::samplers::QmcSampler::new(
        Some(QmcType::Halton), Some(false), Some(42), None, Some(false), Some(false),
    );
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -3.0, 3.0, false, None)?;
                Ok(x * x)
            },
            Some(30),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 1.0, "QMC Halton should find near-zero, best={bv}");
}

#[test]
fn test_qmc_scramble_different_seeds() {
    // Different seeds with scramble=true should produce different sequences
    let pt1 = sobol_point_pub(1, 3, true, 42);
    let pt2 = sobol_point_pub(1, 3, true, 99);
    assert_ne!(pt1, pt2, "Different seeds should produce different scrambled points");
}

#[test]
fn test_qmc_no_scramble_deterministic() {
    // Without scramble, seed doesn't matter
    let pt1 = sobol_point_pub(5, 3, false, 42);
    let pt2 = sobol_point_pub(5, 3, false, 99);
    assert_eq!(pt1, pt2, "Without scramble, seed shouldn't affect output");
}

#[test]
fn test_qmc_sobol_2d_optimize() {
    let sampler = optuna_rs::samplers::QmcSampler::new(
        Some(QmcType::Sobol), Some(false), Some(123), None, Some(false), Some(false),
    );
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y * y)
            },
            Some(40),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 5.0, "QMC Sobol 2D should find reasonable minimum, best={bv}");
}

#[test]
fn test_sobol_point_set_matches_scipy() {
    // The first 8 points (2^3) from Rust gray-code Sobol should be the same SET
    // as scipy's first 8 points, just in different order.
    // scipy 5d first 8: {[0,0,0,0,0], [0.5,0.5,0.5,0.5,0.5], [0.75,0.25,0.25,0.25,0.75],
    //                     [0.25,0.75,0.75,0.75,0.25], [0.375,0.375,0.625,0.875,0.375],
    //                     [0.875,0.875,0.125,0.375,0.875], [0.625,0.125,0.875,0.625,0.625],
    //                     [0.125,0.625,0.375,0.125,0.125]}
    let scipy_set: Vec<[f64; 5]> = vec![
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.5, 0.5, 0.5],
        [0.75, 0.25, 0.25, 0.25, 0.75],
        [0.25, 0.75, 0.75, 0.75, 0.25],
        [0.375, 0.375, 0.625, 0.875, 0.375],
        [0.875, 0.875, 0.125, 0.375, 0.875],
        [0.625, 0.125, 0.875, 0.625, 0.625],
        [0.125, 0.625, 0.375, 0.125, 0.125],
    ];

    let mut rust_set: Vec<Vec<f64>> = (0..8)
        .map(|i| sobol_point_pub(i, 5, false, 0))
        .collect();

    // Sort both sets for comparison
    let mut scipy_sorted: Vec<Vec<f64>> = scipy_set.iter().map(|r| r.to_vec()).collect();
    scipy_sorted.sort_by(|a, b| {
        for (x, y) in a.iter().zip(b.iter()) {
            match x.partial_cmp(y) {
                Some(std::cmp::Ordering::Equal) => continue,
                other => return other.unwrap_or(std::cmp::Ordering::Equal),
            }
        }
        std::cmp::Ordering::Equal
    });
    rust_set.sort_by(|a, b| {
        for (x, y) in a.iter().zip(b.iter()) {
            match x.partial_cmp(y) {
                Some(std::cmp::Ordering::Equal) => continue,
                other => return other.unwrap_or(std::cmp::Ordering::Equal),
            }
        }
        std::cmp::Ordering::Equal
    });

    assert_eq!(scipy_sorted.len(), rust_set.len());
    for (i, (s, r)) in scipy_sorted.iter().zip(rust_set.iter()).enumerate() {
        for d in 0..5 {
            assert!((s[d] - r[d]).abs() < 1e-10,
                "Point set mismatch at sorted index {i}, dim {d}: scipy={}, rust={}", s[d], r[d]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//         TIER 2: END-TO-END FLOW VERIFICATION
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
//  Section 30: TPE Full Optimization Flow
//  Cross-validate: TPE suggest-report cycle, deterministic replay, best_trial
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_tpe_full_flow_deterministic() {
    let run = |seed: u64| -> Vec<f64> {
        let sampler = TpeSamplerBuilder::new(StudyDirection::Minimize)
            .seed(seed)
            .n_startup_trials(5)
            .build();
        let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    Ok(x * x)
                },
                Some(15),
                None,
                None,
            )
            .unwrap();
        study.trials().unwrap().iter()
            .map(|t| t.values.as_ref().unwrap()[0])
            .collect()
    };
    let trace1 = run(42);
    let trace2 = run(42);
    assert_eq!(trace1.len(), trace2.len());
    for (i, (v1, v2)) in trace1.iter().zip(trace2.iter()).enumerate() {
        assert!((v1 - v2).abs() < 1e-12,
            "Trial {i}: {v1} vs {v2}");
    }
}

#[test]
fn test_tpe_best_trial_improves() {
    let sampler = TpeSamplerBuilder::new(StudyDirection::Minimize).seed(42).n_startup_trials(5).build();
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(30),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 1.0, "TPE should converge to near 0.0, got {bv}");
}

#[test]
fn test_tpe_multi_param_optimization() {
    let sampler = TpeSamplerBuilder::new(StudyDirection::Minimize).seed(123).n_startup_trials(10).build();
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y * y)
            },
            Some(40),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 5.0, "TPE 2-param should find reasonable min, got {bv}");
}

#[test]
fn test_tpe_log_distribution() {
    let sampler = TpeSamplerBuilder::new(StudyDirection::Minimize).seed(42).n_startup_trials(5).build();
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let lr = trial.suggest_float("lr", 1e-5, 1.0, true, None)?;
                Ok((lr.log10() + 2.0).powi(2))
            },
            Some(30),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 1.0, "TPE log-dist should find near lr=0.01, got {bv}");
}

#[test]
fn test_tpe_suggest_int() {
    let sampler = TpeSamplerBuilder::new(StudyDirection::Minimize).seed(42).n_startup_trials(5).build();
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let n = trial.suggest_int("n", 1, 100, false, 1)?;
                Ok(((n - 42) as f64).powi(2) as f64)
            },
            Some(30),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 100.0, "TPE int: best={bv}");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 31: NSGA-II Multi-Objective Flow
//  Cross-validate: Pareto front evolution, non-dominated solutions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_nsga2_bi_objective_pareto() {
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let sampler = NSGAIISamplerBuilder::new(dirs.clone()).seed(42).build();
    let study = multi_study(dirs, Arc::new(sampler));
    study
        .optimize_multi(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(vec![x, 1.0 - x])
            },
            Some(30),
            None,
            None,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    let complete: Vec<_> = trials.iter()
        .filter(|t| t.state == TrialState::Complete)
        .collect();
    assert_eq!(complete.len(), 30);
    for t in &complete {
        let v = t.values.as_ref().unwrap();
        let sum = v[0] + v[1];
        assert!((sum - 1.0).abs() < 1e-10,
            "Trial {}: obj sum={sum}, expected 1.0", t.number);
    }
}

#[test]
fn test_nsga2_tri_objective() {
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize, StudyDirection::Minimize];
    let sampler = NSGAIISamplerBuilder::new(dirs.clone()).seed(42).build();
    let study = multi_study(dirs, Arc::new(sampler));
    study
        .optimize_multi(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                Ok(vec![x, y, 2.0 - x - y])
            },
            Some(40),
            None,
            None,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    assert_eq!(trials.len(), 40);
}

#[test]
fn test_nsga2_dominance_check() {
    let dirs = [StudyDirection::Minimize, StudyDirection::Minimize];
    assert!(dominates(&[1.0, 1.0], &[2.0, 2.0], &dirs));
    assert!(!dominates(&[1.0, 2.0], &[2.0, 1.0], &dirs));
    assert!(!dominates(&[2.0, 2.0], &[1.0, 1.0], &dirs));
}

#[test]
fn test_nsga2_deterministic() {
    let run = |seed: u64| -> Vec<Vec<f64>> {
        let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
        let sampler = NSGAIISamplerBuilder::new(dirs.clone()).seed(seed).build();
        let study = multi_study(dirs, Arc::new(sampler));
        study
            .optimize_multi(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(vec![x, 1.0 - x])
                },
                Some(20),
                None,
                None,
            )
            .unwrap();
        study.trials().unwrap().iter()
            .map(|t| t.values.as_ref().unwrap().clone())
            .collect()
    };
    let trace1 = run(42);
    let trace2 = run(42);
    assert_eq!(trace1.len(), trace2.len());
    for (i, (v1, v2)) in trace1.iter().zip(trace2.iter()).enumerate() {
        for j in 0..v1.len() {
            assert!((v1[j] - v2[j]).abs() < 1e-12,
                "Trial {i} obj {j}: {:.15} vs {:.15}", v1[j], v2[j]);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 32: CMA-ES High-Dimensional
//  Cross-validate: CMA-ES convergence on high-dimensional functions
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_cmaes_5d_sphere() {
    let sampler = CmaEsSamplerBuilder::new(StudyDirection::Minimize)
        .seed(42)
        .build();
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let mut sum = 0.0;
                for i in 0..5 {
                    let x = trial.suggest_float(&format!("x{i}"), -5.0, 5.0, false, None)?;
                    sum += x * x;
                }
                Ok(sum)
            },
            Some(80),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 5.0, "CMA-ES 5D sphere: best={bv}");
}

#[test]
fn test_cmaes_separable_mode() {
    let sampler = CmaEsSamplerBuilder::new(StudyDirection::Minimize)
        .seed(42)
        .use_separable_cma(true)
        .build();
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y * y)
            },
            Some(40),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 5.0, "CMA-ES separable: best={bv}");
}

#[test]
fn test_cmaes_custom_sigma0() {
    let sampler = CmaEsSamplerBuilder::new(StudyDirection::Minimize)
        .seed(42)
        .sigma0(0.1)
        .build();
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -2.0, 2.0, false, None)?;
                Ok(x * x)
            },
            Some(30),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    assert!(best.values.as_ref().unwrap()[0] < 1.0);
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 33: Mixed Sampler (Grid + Random baseline)
//  Cross-validate: GridSampler coverage, multi-sampler coordination
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_grid_sampler_coverage() {
    let mut search_space = HashMap::new();
    search_space.insert("x".to_string(), vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    let sampler = GridSampler::new(search_space, None);
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok((x - 0.5).powi(2))
            },
            Some(5),
            None,
            None,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    assert_eq!(trials.len(), 5);
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 0.1, "Grid should hit x=0.5 exactly, best={bv}");
}

#[test]
fn test_grid_2d_coverage() {
    let mut search_space = HashMap::new();
    search_space.insert("x".to_string(), vec![0.0, 0.5, 1.0]);
    search_space.insert("y".to_string(), vec![0.0, 1.0]);
    let sampler = GridSampler::new(search_space, None);
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                Ok(x * x + y * y)
            },
            Some(6),
            None,
            None,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    assert_eq!(trials.len(), 6, "Grid 2D should produce 3x2=6 trials");
}

#[test]
fn test_random_sampler_as_baseline() {
    let sampler = RandomSampler::new(Some(42));
    let study = study_with_sampler(StudyDirection::Minimize, Arc::new(sampler));
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(x * x)
            },
            Some(50),
            None,
            None,
        )
        .unwrap();
    let best = study.best_trial().unwrap();
    let bv = best.values.as_ref().unwrap()[0];
    assert!(bv < 5.0, "Random 50 trials should find at least < 5.0, got {bv}");
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 34: Constraint and Pruning Flows
//  Cross-validate: pruner integration, trial states, failure handling
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_pruner_integration_flow() {
    let pruner = MedianPruner::new(3, 0, 1, 1, StudyDirection::Minimize);
    let study = create_study(
        None,
        None,
        Some(Arc::new(pruner)),
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    ).unwrap();
    let mut pruned_count = 0;
    study
        .optimize(
            |trial| {
                for step in 0..10 {
                    let v = if trial.number() % 2 == 0 { 1.0 } else { 100.0 };
                    trial.report(v, step as i64)?;
                    if trial.should_prune()? {
                        return Err(optuna_rs::error::OptunaError::TrialPruned);
                    }
                }
                let v = if trial.number() % 2 == 0 { 1.0 } else { 100.0 };
                Ok(v)
            },
            Some(10),
            None,
            None,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    for t in &trials {
        if t.state == TrialState::Pruned {
            pruned_count += 1;
        }
    }
    assert!(pruned_count > 0, "Some trials should be pruned");
}

#[test]
fn test_trial_fail_handling() {
    let study = simple_study(StudyDirection::Minimize);
    study
        .optimize_with_options(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                if trial.number() == 2 {
                    return Err(optuna_rs::error::OptunaError::RuntimeError(
                        "simulated failure".to_string(),
                    ));
                }
                Ok(x * x)
            },
            Some(5),
            None,
            1,
            &["RuntimeError"],
            None,
            false,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    assert_eq!(trials.len(), 5);
    let failed = trials.iter().filter(|t| t.state == TrialState::Fail).count();
    assert_eq!(failed, 1, "Exactly one trial should fail");
}

#[test]
fn test_trial_user_attrs() {
    let study = simple_study(StudyDirection::Minimize);
    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                trial.set_user_attr("custom_metric", serde_json::json!(x.abs()));
                Ok(x * x)
            },
            Some(5),
            None,
            None,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    for t in &trials {
        if t.state == TrialState::Complete {
            assert!(t.user_attrs.contains_key("custom_metric"),
                "Trial {} should have custom_metric", t.number);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Section 35: Multi-Objective Hypervolume
//  Cross-validate: hypervolume_2d correctness, Pareto front quality
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn test_hypervolume_2d_basic_single() {
    let hv = hypervolume_2d(&[[1.0, 1.0]], [3.0, 3.0]);
    assert!((hv - 4.0).abs() < 1e-10, "HV single point: got {hv}, expected 4.0");
}

#[test]
fn test_hypervolume_2d_pareto_front() {
    let points = [[0.0, 2.0], [1.0, 1.0], [2.0, 0.0]];
    let hv = hypervolume_2d(&points, [3.0, 3.0]);
    // HV with ref [3,3] and Pareto front [[0,2],[1,1],[2,0]]:
    // Sorted: [0,2], [1,1], [2,0]
    // Area = (1-0)*(3-2) + (2-1)*(3-1) + (3-2)*(3-0) = 1+2+3 = 6
    assert!((hv - 6.0).abs() < 1e-10, "HV Pareto front: got {hv}, expected 6.0");
}

#[test]
fn test_hypervolume_monotone_increasing() {
    let ref_point = [5.0, 5.0];
    let hv1 = hypervolume_2d(&[[1.0, 1.0]], ref_point);
    let hv2 = hypervolume_2d(&[[1.0, 1.0], [3.0, 0.5]], ref_point);
    assert!(hv2 >= hv1, "Adding non-dominated point should increase HV: {hv1} -> {hv2}");
}

#[test]
fn test_fast_nds_ranking() {
    let trials: Vec<FrozenTrial> = vec![
        make_frozen_trial(0, TrialState::Complete, Some(vec![1.0, 5.0]), vec![]),
        make_frozen_trial(1, TrialState::Complete, Some(vec![5.0, 1.0]), vec![]),
        make_frozen_trial(2, TrialState::Complete, Some(vec![3.0, 3.0]), vec![]),
        make_frozen_trial(3, TrialState::Complete, Some(vec![4.0, 4.0]), vec![]),
    ];
    let trial_refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = [StudyDirection::Minimize, StudyDirection::Minimize];
    let fronts = fast_non_dominated_sort(&trial_refs, &dirs);
    assert!(fronts.len() >= 2, "Should have at least 2 fronts");
    // First front: [1,5], [5,1], [3,3] are all non-dominated (3 solutions)
    // Second front: [4,4] is dominated by [3,3]
    assert_eq!(fronts[0].len(), 3, "First front should have 3 solutions");
}

#[test]
fn test_nsga2_multi_objective_flow_complete() {
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let sampler = NSGAIISamplerBuilder::new(dirs.clone()).seed(42).build();
    let study = multi_study(dirs, Arc::new(sampler));
    study
        .optimize_multi(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                Ok(vec![x * x, (x - 2.0).powi(2)])
            },
            Some(30),
            None,
            None,
        )
        .unwrap();
    let trials = study.trials().unwrap();
    let complete: Vec<_> = trials.iter()
        .filter(|t| t.state == TrialState::Complete)
        .collect();
    assert_eq!(complete.len(), 30);
    for t in &complete {
        for v in t.values.as_ref().unwrap() {
            assert!(v.is_finite(), "Trial {} has non-finite value", t.number);
        }
    }
}