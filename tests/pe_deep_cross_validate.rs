//! Deep cross-validation tests for ParzenEstimator.
//!
//! Tests edge cases and scenarios not covered by tpe_cross_validate.rs:
//!   - Large n observations with ramp weights
//!   - Boundary observations
//!   - Very close observations
//!   - Higher prior_weight
//!   - Integer log-scale with log_pdf
//!   - Float with step and log_pdf
//!   - 5-choice categorical
//!   - MOTPE hypervolume contribution weights
//!   - Reference point calculation

use indexmap::IndexMap;
use std::collections::HashMap;

use optuna_rs::distributions::*;
use optuna_rs::multi_objective;
use optuna_rs::samplers::tpe::parzen_estimator::{
    ParzenEstimator, ParzenEstimatorParameters,
};

const TOL: f64 = 1e-8;
const LOGPDF_TOL: f64 = 1e-5;

fn load_baseline() -> serde_json::Value {
    let data = include_str!("pe_deep_baseline.json");
    serde_json::from_str(data).expect("Failed to parse pe_deep_baseline.json")
}

fn get_f64_vec(b: &serde_json::Value, key: &str) -> Vec<f64> {
    b[key]
        .as_array()
        .unwrap_or_else(|| panic!("Key '{key}' not an array"))
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect()
}

fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    if expected.is_infinite() {
        assert!(
            actual.is_infinite() && actual.signum() == expected.signum(),
            "{msg}: expected {expected}, got {actual}"
        );
        return;
    }
    let diff = (actual - expected).abs();
    let denom = expected.abs().max(1.0);
    assert!(
        diff / denom < tol,
        "{msg}: expected {expected}, got {actual}, diff={diff}"
    );
}

fn assert_vec_close(actual: &[f64], expected: &[f64], tol: f64, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{msg}: length {}, expected {}",
        actual.len(),
        expected.len()
    );
    for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
        assert_close(a, e, tol, &format!("{msg}[{i}]"));
    }
}

fn default_params() -> ParzenEstimatorParameters {
    ParzenEstimatorParameters::default()
}

// ═══════════════════════════════════════════════════════════════════════════
//  1. Large n=30 — ramp weights + sigma computation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_large_n30_mus() {
    let b = load_baseline();
    let py_obs = get_f64_vec(&b, "large_obs");
    let py_mus = get_f64_vec(&b, "large_mus");

    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), py_obs);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);
    let (mus, _, _, _) = pe.numerical_kernels("x").unwrap();
    assert_vec_close(&mus, &py_mus, TOL, "large_n30_mus");
}

#[test]
fn cv_large_n30_sigmas() {
    let b = load_baseline();
    let py_obs = get_f64_vec(&b, "large_obs");
    let py_sigmas = get_f64_vec(&b, "large_sigmas");

    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), py_obs);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);
    let (_, sigmas, _, _) = pe.numerical_kernels("x").unwrap();
    assert_vec_close(&sigmas, &py_sigmas, TOL, "large_n30_sigmas");
}

#[test]
fn cv_large_n30_weights() {
    let b = load_baseline();
    let py_obs = get_f64_vec(&b, "large_obs");
    let py_weights = get_f64_vec(&b, "large_weights");

    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), py_obs);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);
    assert_vec_close(pe.weights(), &py_weights, TOL, "large_n30_weights");
}

#[test]
fn cv_large_n30_logpdf() {
    let b = load_baseline();
    let py_obs = get_f64_vec(&b, "large_obs");
    let py_logpdf = get_f64_vec(&b, "large_logpdf");

    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), py_obs);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);
    let samples = HashMap::from([("x".to_string(), vec![1.0, 3.0, 5.0, 7.0, 9.0])]);
    let logpdf = pe.log_pdf(&samples);
    assert_vec_close(&logpdf, &py_logpdf, LOGPDF_TOL, "large_n30_logpdf");
}

// ═══════════════════════════════════════════════════════════════════════════
//  2. Boundary observations — obs at distribution edges
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_boundary_observations() {
    let b = load_baseline();
    let py_mus = get_f64_vec(&b, "boundary_mus");
    let py_sigmas = get_f64_vec(&b, "boundary_sigmas");

    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![0.0, 10.0]);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);

    let (mus, sigmas, _, _) = pe.numerical_kernels("x").unwrap();
    assert_vec_close(&mus, &py_mus, TOL, "boundary_mus");
    assert_vec_close(&sigmas, &py_sigmas, TOL, "boundary_sigmas");
}

// ═══════════════════════════════════════════════════════════════════════════
//  3. Very close observations — tests magic clip floor
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_very_close_observations() {
    let b = load_baseline();
    let py_mus = get_f64_vec(&b, "veryclose_mus");
    let py_sigmas = get_f64_vec(&b, "veryclose_sigmas");

    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![5.0, 5.0 + 1e-10]);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);

    let (mus, sigmas, _, _) = pe.numerical_kernels("x").unwrap();
    assert_vec_close(&mus, &py_mus, TOL, "veryclose_mus");
    assert_vec_close(&sigmas, &py_sigmas, TOL, "veryclose_sigmas");
}

// ═══════════════════════════════════════════════════════════════════════════
//  4. Single observation — prior dominates
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_single_observation_deep() {
    let b = load_baseline();
    let py_mus = get_f64_vec(&b, "single_mus");
    let py_sigmas = get_f64_vec(&b, "single_sigmas");
    let py_weights = get_f64_vec(&b, "single_weights");

    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![3.0]);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);

    let (mus, sigmas, _, _) = pe.numerical_kernels("x").unwrap();
    assert_vec_close(&mus, &py_mus, TOL, "single_mus");
    assert_vec_close(&sigmas, &py_sigmas, TOL, "single_sigmas");
    assert_vec_close(pe.weights(), &py_weights, TOL, "single_weights");
}

// ═══════════════════════════════════════════════════════════════════════════
//  5. Higher prior_weight=3.0 — prior kernel dominates
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_prior_weight_3() {
    let b = load_baseline();
    let py_weights = get_f64_vec(&b, "pw3_weights");
    let py_sigmas = get_f64_vec(&b, "pw3_sigmas");

    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![2.0, 5.0, 8.0]);
    let mut params = default_params();
    params.prior_weight = 3.0;
    let pe = ParzenEstimator::new(&obs, &ss, &params, None, None);

    assert_vec_close(pe.weights(), &py_weights, TOL, "pw3_weights");
    let (_, sigmas, _, _) = pe.numerical_kernels("x").unwrap();
    assert_vec_close(&sigmas, &py_sigmas, TOL, "pw3_sigmas");
}

// ═══════════════════════════════════════════════════════════════════════════
//  6. Integer log-scale — mus, sigmas, and log_pdf
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_int_log_deep_kernels() {
    let b = load_baseline();
    let py_mus = get_f64_vec(&b, "intlog_mus");
    let py_sigmas = get_f64_vec(&b, "intlog_sigmas");

    let mut ss = IndexMap::new();
    ss.insert(
        "depth".to_string(),
        Distribution::IntDistribution(IntDistribution::new(1, 100, true, 1).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("depth".to_string(), vec![5.0, 10.0, 30.0, 50.0]);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);

    let (mus, sigmas, _, _) = pe.numerical_kernels("depth").unwrap();
    assert_vec_close(&mus, &py_mus, TOL, "intlog_mus");
    assert_vec_close(&sigmas, &py_sigmas, TOL, "intlog_sigmas");
}

#[test]
fn cv_int_log_deep_logpdf() {
    let b = load_baseline();
    let py_logpdf = get_f64_vec(&b, "intlog_logpdf");

    let mut ss = IndexMap::new();
    ss.insert(
        "depth".to_string(),
        Distribution::IntDistribution(IntDistribution::new(1, 100, true, 1).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("depth".to_string(), vec![5.0, 10.0, 30.0, 50.0]);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);

    let samples = HashMap::from([("depth".to_string(), vec![1.0, 5.0, 10.0, 50.0, 100.0])]);
    let logpdf = pe.log_pdf(&samples);
    assert_vec_close(&logpdf, &py_logpdf, LOGPDF_TOL, "intlog_logpdf");
}

// ═══════════════════════════════════════════════════════════════════════════
//  7. Float with step — discretized continuous
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_step_float_kernels() {
    let b = load_baseline();
    let py_mus = get_f64_vec(&b, "step_float_mus");
    let py_sigmas = get_f64_vec(&b, "step_float_sigmas");

    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap(),
        ),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![0.2, 0.5, 0.8]);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);

    let (mus, sigmas, _, _) = pe.numerical_kernels("x").unwrap();
    assert_vec_close(&mus, &py_mus, TOL, "step_float_mus");
    assert_vec_close(&sigmas, &py_sigmas, TOL, "step_float_sigmas");
}

#[test]
fn cv_step_float_logpdf() {
    let b = load_baseline();
    let py_logpdf = get_f64_vec(&b, "step_float_logpdf");

    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap(),
        ),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![0.2, 0.5, 0.8]);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);

    let samples = HashMap::from([("x".to_string(), vec![0.0, 0.3, 0.5, 0.8, 1.0])]);
    let logpdf = pe.log_pdf(&samples);
    assert_vec_close(&logpdf, &py_logpdf, LOGPDF_TOL, "step_float_logpdf");
}

// ═══════════════════════════════════════════════════════════════════════════
//  8. Categorical with 5 choices
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_categorical_5_choices() {
    let b = load_baseline();
    let py_weights: Vec<Vec<f64>> = b["cat5_weights"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();

    let mut ss = IndexMap::new();
    ss.insert(
        "c".to_string(),
        Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".into()),
                CategoricalChoice::Str("b".into()),
                CategoricalChoice::Str("c".into()),
                CategoricalChoice::Str("d".into()),
                CategoricalChoice::Str("e".into()),
            ])
            .unwrap(),
        ),
    );
    let mut obs = HashMap::new();
    obs.insert("c".to_string(), vec![0.0, 1.0, 1.0, 2.0, 4.0]);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);

    let cat_w = pe.categorical_kernels("c").unwrap();
    assert_eq!(cat_w.len(), py_weights.len(), "cat5 kernel count");
    for (i, (r, p)) in cat_w.iter().zip(py_weights.iter()).enumerate() {
        assert_vec_close(r, p, TOL, &format!("cat5_weights[{i}]"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  9. MOTPE — Hypervolume contribution weights
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_motpe_hv_contributions() {
    let b = load_baseline();
    let py_hv = b["motpe_full_hv"].as_f64().unwrap();
    let py_contribs = get_f64_vec(&b, "motpe_contribs");
    let py_weights = get_f64_vec(&b, "motpe_weights");

    let loss_values = vec![
        vec![1.0, 5.0],
        vec![2.0, 3.0],
        vec![3.0, 2.0],
        vec![5.0, 1.0],
    ];
    let ref_point = vec![5.5, 5.5];

    // Full hypervolume
    let full_hv = multi_objective::hypervolume(&loss_values, &ref_point);
    assert_close(full_hv, py_hv, TOL, "motpe_full_hv");

    // LOO contributions
    let mut contribs = Vec::new();
    for i in 0..4 {
        let without: Vec<Vec<f64>> = loss_values
            .iter()
            .enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, v)| v.clone())
            .collect();
        let hv_without = multi_objective::hypervolume(&without, &ref_point);
        contribs.push(full_hv - hv_without);
    }
    assert_vec_close(&contribs, &py_contribs, TOL, "motpe_contribs");

    // Normalized weights
    let max_c = contribs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let eps = 1e-12;
    let weights: Vec<f64> = contribs.iter().map(|&c| (c / max_c.max(eps)).max(eps)).collect();
    assert_vec_close(&weights, &py_weights, TOL, "motpe_weights");
}

// ═══════════════════════════════════════════════════════════════════════════
//  10. Reference point calculation
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_reference_point() {
    let b = load_baseline();
    let py_ref = get_f64_vec(&b, "refpoint_mixed");

    // Simulate: loss_values = [[1,-2],[3,0],[2,4]]
    // worst = [3, 4], ref = [1.1*3, 1.1*4] = [3.3, 4.4]
    // But column 1: worst = max(-2, 0, 4) = 4. And 0 is not the worst.
    // 0 > 0? No, so ref[1] for worst=4: 1.1*4 = 4.4
    // worst[0] = 3 > 0 → 1.1*3 = 3.3
    let loss_values = vec![vec![1.0, -2.0], vec![3.0, 0.0], vec![2.0, 4.0]];
    let n_objectives = 2;
    let mut worst = vec![f64::NEG_INFINITY; n_objectives];
    for vals in &loss_values {
        for (i, &v) in vals.iter().enumerate() {
            if v > worst[i] {
                worst[i] = v;
            }
        }
    }
    let eps = 1e-12;
    let ref_point: Vec<f64> = worst
        .iter()
        .map(|&w| {
            if w == 0.0 {
                eps
            } else if w > 0.0 {
                1.1 * w
            } else {
                0.9 * w
            }
        })
        .collect();
    assert_vec_close(&ref_point, &py_ref, TOL, "refpoint_mixed");
}

// ═══════════════════════════════════════════════════════════════════════════
//  11. Invariant: log_pdf finite for in-range samples
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_logpdf_finiteness_invariant() {
    // Float
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![1.0, 5.0, 9.0]);
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);
    let samples = HashMap::from([("x".to_string(), vec![0.001, 5.0, 9.999])]);
    for &v in pe.log_pdf(&samples).iter() {
        assert!(v.is_finite(), "log_pdf should be finite, got {v}");
    }

    // Log
    let mut ss_log = IndexMap::new();
    ss_log.insert(
        "lr".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.001, 1.0, true, None).unwrap()),
    );
    let mut obs_log = HashMap::new();
    obs_log.insert("lr".to_string(), vec![0.01, 0.1]);
    let pe_log = ParzenEstimator::new(&obs_log, &ss_log, &default_params(), None, None);
    let samples_log = HashMap::from([("lr".to_string(), vec![0.001, 0.05, 1.0])]);
    for &v in pe_log.log_pdf(&samples_log).iter() {
        assert!(v.is_finite(), "log_pdf (log-scale) should be finite, got {v}");
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  12. Invariant: weights sum to 1.0
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_weights_sum_to_one() {
    for n_obs in [0usize, 1, 3, 10, 25, 30, 50] {
        let mut ss = IndexMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::FloatDistribution(
                FloatDistribution::new(0.0, 10.0, false, None).unwrap(),
            ),
        );
        let obs_vals: Vec<f64> = (0..n_obs).map(|i| i as f64 * 10.0 / (n_obs.max(1) as f64)).collect();
        let mut obs = HashMap::new();
        obs.insert("x".to_string(), obs_vals);
        let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);

        let sum: f64 = pe.weights().iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "weights should sum to 1.0 for n_obs={n_obs}, got {sum}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  13. Invariant: n_kernels = n_obs + 1 (prior)
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_kernel_count_invariant() {
    for n_obs in [0usize, 1, 5, 10, 30] {
        let mut ss = IndexMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::FloatDistribution(
                FloatDistribution::new(0.0, 10.0, false, None).unwrap(),
            ),
        );
        let obs_vals: Vec<f64> = (0..n_obs).map(|i| 1.0 + i as f64).collect();
        let mut obs = HashMap::new();
        obs.insert("x".to_string(), obs_vals);
        let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);

        let expected = if n_obs == 0 { 1 } else { n_obs + 1 };
        assert_eq!(
            pe.weights().len(),
            expected,
            "n_kernels should be {} for n_obs={n_obs}",
            expected
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  14. Invariant: sigmas always > 0 and within [minsigma, maxsigma]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_sigma_bounds_invariant() {
    for n_obs in [1usize, 3, 10, 30] {
        let mut ss = IndexMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::FloatDistribution(
                FloatDistribution::new(0.0, 10.0, false, None).unwrap(),
            ),
        );
        let obs_vals: Vec<f64> = (0..n_obs).map(|i| i as f64 * 10.0 / n_obs as f64).collect();
        let mut obs = HashMap::new();
        obs.insert("x".to_string(), obs_vals);
        let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, None);

        let (_, sigmas, low, high) = pe.numerical_kernels("x").unwrap();
        let range = high - low;
        let n_kernels = n_obs + 1;
        let min_sigma = range / (100.0_f64).min(1.0 + n_kernels as f64);

        for (i, &s) in sigmas.iter().enumerate() {
            assert!(
                s > 0.0,
                "sigma[{i}] should be > 0, got {s} for n_obs={n_obs}"
            );
            // Prior sigma = range, observation sigmas clipped to [min_sigma, range]
            assert!(
                s <= range + 1e-10,
                "sigma[{i}] should be <= range={range}, got {s} for n_obs={n_obs}"
            );
            if i < sigmas.len() - 1 {
                // Observation sigmas (not prior)
                assert!(
                    s >= min_sigma - 1e-10,
                    "sigma[{i}] should be >= min_sigma={min_sigma}, got {s} for n_obs={n_obs}"
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  15. Custom weights function
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_custom_weights_function() {
    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![2.0, 5.0, 8.0]);

    // Custom: linearly decreasing weights
    let custom_fn = |n: usize| -> Vec<f64> {
        (0..n).map(|i| (n - i) as f64).collect()
    };
    let pe = ParzenEstimator::new(&obs, &ss, &default_params(), None, Some(&custom_fn));

    // weights = [3, 2, 1] + [prior_weight=1.0] = [3, 2, 1, 1]
    // normalized: [3/7, 2/7, 1/7, 1/7]
    let expected: Vec<f64> = vec![3.0 / 7.0, 2.0 / 7.0, 1.0 / 7.0, 1.0 / 7.0];
    assert_vec_close(pe.weights(), &expected, TOL, "custom_weights");
}
