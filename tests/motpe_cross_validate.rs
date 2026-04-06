//! MOTPE HV 贡献权重交叉验证测试。
//!
//! 金标准值由 `tests/golden_motpe_weights.py` 使用 Python optuna 生成。
//! 验证 TpeSampler::calculate_mo_weights 的精确输出。

use std::collections::HashMap;

use optuna_rs::samplers::tpe::TpeSampler;
use optuna_rs::study::StudyDirection;
use optuna_rs::trial::{FrozenTrial, TrialState};

const TOL: f64 = 1e-8;

fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    if expected.is_nan() {
        assert!(actual.is_nan(), "{msg}: expected NaN, got {actual}");
        return;
    }
    let diff = (actual - expected).abs();
    let denom = expected.abs().max(1.0);
    assert!(
        diff / denom < tol,
        "{msg}: expected {expected}, got {actual}, rel_diff={:.2e}",
        diff / denom
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

fn make_mo_trial(number: i64, values: &[f64]) -> FrozenTrial {
    FrozenTrial {
        number,
        trial_id: number,
        state: TrialState::Complete,
        values: Some(values.to_vec()),
        datetime_start: None,
        datetime_complete: None,
        params: HashMap::new(),
        distributions: HashMap::new(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  1. 2D minimize — 4 Pareto front trials
//     Python: weights = [0.2, 1.0, 0.25, 0.1]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_motpe_weights_2d_4pareto() {
    let trials: Vec<FrozenTrial> = vec![
        make_mo_trial(0, &[1.0, 4.0]),
        make_mo_trial(1, &[2.0, 2.0]),
        make_mo_trial(2, &[3.0, 1.5]),
        make_mo_trial(3, &[4.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let directions = [StudyDirection::Minimize, StudyDirection::Minimize];

    let weights = TpeSampler::calculate_mo_weights(&refs, &directions, false);
    let expected = [0.20000000000000018, 1.0, 0.25, 0.10000000000000009];
    assert_vec_close(&weights, &expected, TOL, "2d_4pareto");
}

// ═══════════════════════════════════════════════════════════════════════════
//  2. 2D minimize — 3 Pareto + 1 dominated
//     Python: weights = [0.1, 1.0, 1e-12, 0.1]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_motpe_weights_2d_with_dominated() {
    let trials: Vec<FrozenTrial> = vec![
        make_mo_trial(0, &[1.0, 4.0]),
        make_mo_trial(1, &[2.0, 2.0]),
        make_mo_trial(2, &[3.0, 3.0]), // dominated by [2,2]
        make_mo_trial(3, &[4.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let directions = [StudyDirection::Minimize, StudyDirection::Minimize];

    let weights = TpeSampler::calculate_mo_weights(&refs, &directions, false);
    let expected = [0.10000000000000009, 1.0, 1e-12, 0.10000000000000009];
    assert_vec_close(&weights, &expected, TOL, "2d_dominated");

    // Dominated trial [3,3] should have EPS weight
    assert!(weights[2] < 1e-10, "dominated should have EPS weight, got {}", weights[2]);
}

// ═══════════════════════════════════════════════════════════════════════════
//  3. Single trial — trivially returns [1.0]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_motpe_weights_single() {
    let trials = vec![make_mo_trial(0, &[2.0, 3.0])];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let directions = [StudyDirection::Minimize, StudyDirection::Minimize];

    let weights = TpeSampler::calculate_mo_weights(&refs, &directions, false);
    assert_eq!(weights, vec![1.0]);
}

// ═══════════════════════════════════════════════════════════════════════════
//  4. 3D minimize — 4 Pareto trials
//     Python: weights = [0.2202, 0.4954, 1.0, 0.1835]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_motpe_weights_3d() {
    let trials: Vec<FrozenTrial> = vec![
        make_mo_trial(0, &[1.0, 4.0, 3.0]),
        make_mo_trial(1, &[2.0, 1.0, 4.0]),
        make_mo_trial(2, &[3.0, 3.0, 1.0]),
        make_mo_trial(3, &[4.0, 2.0, 2.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let directions = [
        StudyDirection::Minimize,
        StudyDirection::Minimize,
        StudyDirection::Minimize,
    ];

    let weights = TpeSampler::calculate_mo_weights(&refs, &directions, false);
    let expected = [0.2201834862385323, 0.4954128440366972, 1.0, 0.18348623853211024];
    assert_vec_close(&weights, &expected, TOL, "3d_4pareto");
}

// ═══════════════════════════════════════════════════════════════════════════
//  5. 2D maximize — 3 Pareto trials
//     Python: weights = [0.2, 1.0, 0.2]
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_motpe_weights_maximize() {
    let trials: Vec<FrozenTrial> = vec![
        make_mo_trial(0, &[4.0, 1.0]),
        make_mo_trial(1, &[2.0, 2.0]),
        make_mo_trial(2, &[1.0, 4.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let directions = [StudyDirection::Maximize, StudyDirection::Maximize];

    let weights = TpeSampler::calculate_mo_weights(&refs, &directions, false);
    let expected = [0.1999999999999999, 1.0, 0.1999999999999999];
    assert_vec_close(&weights, &expected, TOL, "2d_maximize");

    // Symmetry: first and last should be equal
    assert_close(weights[0], weights[2], TOL, "maximize_symmetry");
}

// ═══════════════════════════════════════════════════════════════════════════
//  6. Empty trials
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_motpe_weights_empty() {
    let trials: Vec<&FrozenTrial> = vec![];
    let directions = [StudyDirection::Minimize, StudyDirection::Minimize];
    let weights = TpeSampler::calculate_mo_weights(&trials, &directions, false);
    assert!(weights.is_empty());
}

// ═══════════════════════════════════════════════════════════════════════════
//  7. Max contributor has weight 1.0 invariant
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_motpe_weights_max_is_one() {
    let trials: Vec<FrozenTrial> = vec![
        make_mo_trial(0, &[1.0, 4.0]),
        make_mo_trial(1, &[2.0, 2.0]),
        make_mo_trial(2, &[3.0, 1.5]),
        make_mo_trial(3, &[4.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let directions = [StudyDirection::Minimize, StudyDirection::Minimize];

    let weights = TpeSampler::calculate_mo_weights(&refs, &directions, false);
    let max_weight = weights.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    assert_close(max_weight, 1.0, TOL, "max_weight_is_1");
}

// ═══════════════════════════════════════════════════════════════════════════
//  8. All weights > 0 invariant
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn cv_motpe_weights_all_positive() {
    let trials: Vec<FrozenTrial> = vec![
        make_mo_trial(0, &[1.0, 4.0]),
        make_mo_trial(1, &[2.0, 2.0]),
        make_mo_trial(2, &[3.0, 3.0]), // dominated
        make_mo_trial(3, &[4.0, 1.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let directions = [StudyDirection::Minimize, StudyDirection::Minimize];

    let weights = TpeSampler::calculate_mo_weights(&refs, &directions, false);
    for (i, &w) in weights.iter().enumerate() {
        assert!(w > 0.0, "weight[{i}] should be > 0, got {w}");
    }
}
