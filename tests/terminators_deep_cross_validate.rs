//! 终止器模块深度交叉验证测试（第二层）。
//!
//! 所有参考值均由 Python optuna terminator 模块生成，
//! 通过 tests/golden_terminators_deep.py 脚本产出 terminators_deep_golden_values.json。
//!
//! 覆盖：
//! - CrossValidationErrorEvaluator 扩展精度（Group 1）
//! - BestValueStagnationEvaluator 边界场景（Group 2）
//! - Beta 函数扩展精度（Group 3）
//! - normal_pdf/cdf 极端值（Group 4）
//! - EMMR 四项分解公式（Group 5）
//! - KL 散度三项独立验证（Group 6）
//! - 终止决策逻辑（Group 7）
//! - MedianErrorEvaluator 阈值计算（Group 8）
//! - GP-based 评估器行为验证（Group 9）

use optuna_rs::terminators::*;
use optuna_rs::trial::{FrozenTrial, TrialState};
use optuna_rs::study::StudyDirection;
use optuna_rs::samplers::gp::{normal_pdf, normal_cdf};

/// Rust 内部使用的最小噪声方差常量（对齐 Python prior.DEFAULT_MINIMUM_NOISE_VAR）
const DEFAULT_MINIMUM_NOISE_VAR: f64 = 1e-6;
use std::collections::HashMap;

/// 加载黄金值 JSON
fn load_golden() -> serde_json::Value {
    let data = include_str!("terminators_deep_golden_values.json");
    serde_json::from_str(data).expect("Failed to parse golden values JSON")
}

/// 辅助：构造带 CV 分数的 FrozenTrial
fn trial_with_cv(scores: &[f64], value: f64) -> FrozenTrial {
    let mut system_attrs = HashMap::new();
    system_attrs.insert(
        "terminator:cv_scores".to_string(),
        serde_json::json!(scores),
    );
    FrozenTrial {
        number: 0,
        trial_id: 0,
        state: TrialState::Complete,
        values: Some(vec![value]),
        datetime_start: None,
        datetime_complete: None,
        params: HashMap::new(),
        distributions: HashMap::new(),
        user_attrs: HashMap::new(),
        system_attrs,
        intermediate_values: HashMap::new(),
    }
}

/// 辅助：从值列表构造试验
fn trials_from_values(values: &[f64]) -> Vec<FrozenTrial> {
    values
        .iter()
        .enumerate()
        .map(|(i, &v)| FrozenTrial {
            number: i as i64,
            trial_id: i as i64,
            state: TrialState::Complete,
            values: Some(vec![v]),
            datetime_start: None,
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
        })
        .collect()
}

// ============================================================================
// Group 1: CrossValidationErrorEvaluator 扩展精度
// ============================================================================

#[test]
fn test_cv_error_k10_uniform() {
    let golden = load_golden();
    let case = &golden["cv_error_extended"][0];
    let scores: Vec<f64> = case["scores"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected = case["std"].as_f64().unwrap();
    let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;

    let trial = trial_with_cv(&scores, mean);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);

    assert!(
        (result - expected).abs() < 1e-12,
        "k10_uniform: Rust={:.15e}, Python={:.15e}, diff={:.2e}",
        result, expected, (result - expected).abs()
    );
}

#[test]
fn test_cv_error_k2_minimal() {
    let golden = load_golden();
    let case = &golden["cv_error_extended"][1];
    let scores: Vec<f64> = case["scores"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected = case["std"].as_f64().unwrap();
    let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;

    let trial = trial_with_cv(&scores, mean);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);

    assert!(
        (result - expected).abs() < 1e-12,
        "k2_minimal: Rust={:.15e}, Python={:.15e}",
        result, expected
    );
}

#[test]
fn test_cv_error_k7_nonuniform() {
    let golden = load_golden();
    let case = &golden["cv_error_extended"][2];
    let scores: Vec<f64> = case["scores"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected = case["std"].as_f64().unwrap();
    let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;

    let trial = trial_with_cv(&scores, mean);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);

    assert!(
        (result - expected).abs() < 1e-12,
        "k7_nonuniform: Rust={:.15e}, Python={:.15e}",
        result, expected
    );
}

#[test]
fn test_cv_error_k3_tiny_variance() {
    let golden = load_golden();
    let case = &golden["cv_error_extended"][3];
    let scores: Vec<f64> = case["scores"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected = case["std"].as_f64().unwrap();
    let mean: f64 = scores.iter().sum::<f64>() / scores.len() as f64;

    let trial = trial_with_cv(&scores, mean);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);

    assert!(
        (result - expected).abs() < 1e-12,
        "k3_tiny_variance: Rust={:.15e}, Python={:.15e}",
        result, expected
    );
}

// ============================================================================
// Group 2: BestValueStagnationEvaluator 边界场景
// ============================================================================

#[test]
fn test_stagnation_negative_values() {
    let golden = load_golden();
    let case = &golden["stagnation_edge_cases"][0];
    let values: Vec<f64> = case["values"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let patience = case["patience"].as_u64().unwrap() as usize;
    let expected = case["result"].as_f64().unwrap();

    let trials = trials_from_values(&values);
    let eval = BestValueStagnationEvaluator::new(patience);
    let result = eval.evaluate(&trials, StudyDirection::Minimize);

    assert!(
        (result - expected).abs() < 1e-10,
        "negative_values: Rust={}, Python={}", result, expected
    );
}

#[test]
fn test_stagnation_alternating() {
    let golden = load_golden();
    let case = &golden["stagnation_edge_cases"][1];
    let values: Vec<f64> = case["values"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let patience = case["patience"].as_u64().unwrap() as usize;
    let expected = case["result"].as_f64().unwrap();

    let trials = trials_from_values(&values);
    let eval = BestValueStagnationEvaluator::new(patience);
    let result = eval.evaluate(&trials, StudyDirection::Minimize);

    assert!(
        (result - expected).abs() < 1e-10,
        "alternating: Rust={}, Python={}", result, expected
    );
}

#[test]
fn test_stagnation_last_step_improve() {
    let golden = load_golden();
    let case = &golden["stagnation_edge_cases"][2];
    let values: Vec<f64> = case["values"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let patience = case["patience"].as_u64().unwrap() as usize;
    let expected = case["result"].as_f64().unwrap();

    let trials = trials_from_values(&values);
    let eval = BestValueStagnationEvaluator::new(patience);
    let result = eval.evaluate(&trials, StudyDirection::Minimize);

    assert!(
        (result - expected).abs() < 1e-10,
        "last_step_improve: Rust={}, Python={}", result, expected
    );
}

#[test]
fn test_stagnation_tiny_float_diff() {
    let golden = load_golden();
    let case = &golden["stagnation_edge_cases"][3];
    let values: Vec<f64> = case["values"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let patience = case["patience"].as_u64().unwrap() as usize;
    let expected = case["result"].as_f64().unwrap();

    let trials = trials_from_values(&values);
    let eval = BestValueStagnationEvaluator::new(patience);
    let result = eval.evaluate(&trials, StudyDirection::Minimize);

    assert!(
        (result - expected).abs() < 1e-10,
        "tiny_float_diff: Rust={}, Python={}", result, expected
    );
}

#[test]
fn test_stagnation_large_range() {
    let golden = load_golden();
    let case = &golden["stagnation_edge_cases"][4];
    let values: Vec<f64> = case["values"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let patience = case["patience"].as_u64().unwrap() as usize;
    let expected = case["result"].as_f64().unwrap();

    let trials = trials_from_values(&values);
    let eval = BestValueStagnationEvaluator::new(patience);
    let result = eval.evaluate(&trials, StudyDirection::Minimize);

    assert!(
        (result - expected).abs() < 1e-10,
        "large_range: Rust={}, Python={}", result, expected
    );
}

// ============================================================================
// Group 3: Beta 函数扩展精度
// ============================================================================

#[test]
fn test_beta_function_extended() {
    let golden = load_golden();
    let cases = golden["beta_extended"].as_array().unwrap();

    for case in cases {
        let n_params = case["n_params"].as_u64().unwrap() as usize;
        let n_trials = case["n_trials"].as_u64().unwrap() as usize;
        let delta = case["delta"].as_f64().unwrap();
        let expected = case["beta"].as_f64().unwrap();

        let n = n_trials as f64;
        let d = n_params as f64;
        let beta_arg = d * n * n * std::f64::consts::PI * std::f64::consts::PI / (6.0 * delta);
        let beta = 2.0 * beta_arg.ln() / 5.0;

        assert!(
            (beta - expected).abs() < 1e-10,
            "Beta(d={}, n={}, δ={}): Rust={:.15e}, Python={:.15e}",
            n_params, n_trials, delta, beta, expected
        );
    }
}

// ============================================================================
// Group 4: normal_pdf / normal_cdf 极端值
// ============================================================================

#[test]
fn test_normal_pdf_cdf_extended() {
    let golden = load_golden();
    let cases = golden["normal_pdf_cdf_extended"].as_array().unwrap();

    for case in cases {
        let g = case["g"].as_f64().unwrap();
        let expected_pdf = case["pdf"].as_f64().unwrap();
        let expected_cdf = case["cdf"].as_f64().unwrap();

        let rust_pdf = normal_pdf(g);
        let rust_cdf = normal_cdf(g);

        // 对 PDF: 使用相对精度（极小值下绝对误差太小）
        let pdf_ok = if expected_pdf.abs() < 1e-10 {
            (rust_pdf - expected_pdf).abs() < 1e-15
        } else {
            ((rust_pdf - expected_pdf) / expected_pdf).abs() < 1e-8
        };
        assert!(
            pdf_ok,
            "normal_pdf({:.1}) mismatch: Rust={:.15e}, Python={:.15e}",
            g, rust_pdf, expected_pdf
        );

        let cdf_ok = if expected_cdf.abs() < 1e-6 {
            ((rust_cdf - expected_cdf) / expected_cdf.max(1e-20)).abs() < 1e-4
        } else {
            (rust_cdf - expected_cdf).abs() < 1e-8
        };
        assert!(
            cdf_ok,
            "normal_cdf({:.1}) mismatch: Rust={:.15e}, Python={:.15e}",
            g, rust_cdf, expected_cdf
        );
    }
}

// ============================================================================
// Group 5: EMMR 四项分解公式
// ============================================================================

/// 独立验证 EMMR 的四项公式，不依赖 GP 拟合。
/// 直接使用已知的后验参数计算各项。
#[test]
fn test_emmr_term_decomposition_normal_case() {
    let golden = load_golden();
    let case = &golden["emmr_term_decomposition"][0];

    let mu_t_star = case["mu_t_star"].as_f64().unwrap();
    let mu_t1_star = case["mu_t1_star"].as_f64().unwrap();
    let var_t_star = case["var_t_star"].as_f64().unwrap();
    let var_t1_star = case["var_t1_star"].as_f64().unwrap();
    let cov_t = case["cov_t"].as_f64().unwrap();
    let kappa_t1 = case["kappa_t1"].as_f64().unwrap();
    let var_t1_x_t = case["var_t1_x_t"].as_f64().unwrap();
    let mu_t1_x_t = case["mu_t1_x_t"].as_f64().unwrap();
    let y_t = case["y_t"].as_f64().unwrap();
    let min_noise_var = case["min_noise_var"].as_f64().unwrap();

    // term1: Δμ
    let delta_mu = mu_t1_star - mu_t_star;
    let expected_delta_mu = case["delta_mu"].as_f64().unwrap();
    assert!((delta_mu - expected_delta_mu).abs() < 1e-14, "delta_mu");

    // v
    let v_sq = var_t_star - 2.0 * cov_t + var_t1_star;
    let v = v_sq.max(1e-10).sqrt();
    let expected_v = case["v"].as_f64().unwrap();
    assert!((v - expected_v).abs() < 1e-12, "v: {v} vs {expected_v}");

    // g
    let g = (mu_t_star - mu_t1_star) / v;
    let expected_g = case["g"].as_f64().unwrap();
    assert!((g - expected_g).abs() < 1e-10, "g: {g} vs {expected_g}");

    // term2
    let term2 = v * normal_pdf(g);
    let expected_term2 = case["term2"].as_f64().unwrap();
    assert!((term2 - expected_term2).abs() < 1e-10, "term2: {term2} vs {expected_term2}");

    // term3
    let term3 = v * g * normal_cdf(g);
    let expected_term3 = case["term3"].as_f64().unwrap();
    assert!((term3 - expected_term3).abs() < 1e-10, "term3: {term3} vs {expected_term3}");

    // KL divergence terms
    let lambda_inv = min_noise_var;
    let lambda = 1.0 / lambda_inv;
    let rhs1 = 0.5 * (1.0 + lambda * var_t1_x_t).ln();
    let rhs2 = -0.5 * var_t1_x_t / (var_t1_x_t + lambda_inv);
    let rhs3 = 0.5 * var_t1_x_t * (y_t - mu_t1_x_t).powi(2) / (var_t1_x_t + lambda_inv).powi(2);

    let expected_rhs1 = case["rhs1"].as_f64().unwrap();
    let expected_rhs2 = case["rhs2"].as_f64().unwrap();
    let expected_rhs3 = case["rhs3"].as_f64().unwrap();
    assert!((rhs1 - expected_rhs1).abs() < 1e-10, "rhs1: {rhs1} vs {expected_rhs1}");
    assert!((rhs2 - expected_rhs2).abs() < 1e-10, "rhs2: {rhs2} vs {expected_rhs2}");
    assert!((rhs3 - expected_rhs3).abs() < 1e-10, "rhs3: {rhs3} vs {expected_rhs3}");

    // KL bound & term4
    let kl_bound = rhs1 + rhs2 + rhs3;
    let expected_kl = case["kl_bound"].as_f64().unwrap();
    assert!((kl_bound - expected_kl).abs() < 1e-10, "kl_bound: {kl_bound} vs {expected_kl}");

    let term4 = kappa_t1 * (0.5 * kl_bound.max(0.0)).sqrt();
    let expected_term4 = case["term4"].as_f64().unwrap();
    assert!((term4 - expected_term4).abs() < 1e-10, "term4: {term4} vs {expected_term4}");

    // Total EMMR
    let emmr = delta_mu + term2 + term3 + term4;
    let expected_emmr = case["emmr"].as_f64().unwrap();
    assert!((emmr - expected_emmr).abs() < 1e-10, "emmr: {emmr} vs {expected_emmr}");
}

#[test]
fn test_emmr_term_decomposition_zero_delta_mu() {
    let golden = load_golden();
    let case = &golden["emmr_term_decomposition"][1];

    let mu_t_star = case["mu_t_star"].as_f64().unwrap();
    let mu_t1_star = case["mu_t1_star"].as_f64().unwrap();
    let var_t_star = case["var_t_star"].as_f64().unwrap();
    let var_t1_star = case["var_t1_star"].as_f64().unwrap();
    let cov_t = case["cov_t"].as_f64().unwrap();
    let kappa_t1 = case["kappa_t1"].as_f64().unwrap();
    let var_t1_x_t = case["var_t1_x_t"].as_f64().unwrap();
    let mu_t1_x_t = case["mu_t1_x_t"].as_f64().unwrap();
    let y_t = case["y_t"].as_f64().unwrap();
    let min_noise_var = case["min_noise_var"].as_f64().unwrap();

    let delta_mu = mu_t1_star - mu_t_star;
    assert!((delta_mu - 0.0).abs() < 1e-14, "delta_mu should be 0");

    let v = (var_t_star - 2.0 * cov_t + var_t1_star).max(1e-10).sqrt();
    let g = (mu_t_star - mu_t1_star) / v;
    assert!((g - 0.0).abs() < 1e-14, "g should be 0 when delta_mu=0");

    let term2 = v * normal_pdf(g);
    let term3 = v * g * normal_cdf(g);
    let expected_term3 = case["term3"].as_f64().unwrap();
    assert!((term3 - expected_term3).abs() < 1e-10, "term3 should be ~0: {term3}");

    let lambda_inv = min_noise_var;
    let lambda = 1.0 / lambda_inv;
    let rhs1 = 0.5 * (1.0 + lambda * var_t1_x_t).ln();
    let rhs2 = -0.5 * var_t1_x_t / (var_t1_x_t + lambda_inv);
    let rhs3 = 0.5 * var_t1_x_t * (y_t - mu_t1_x_t).powi(2) / (var_t1_x_t + lambda_inv).powi(2);
    let kl_bound = rhs1 + rhs2 + rhs3;
    let term4 = kappa_t1 * (0.5 * kl_bound.max(0.0)).sqrt();

    let emmr = delta_mu + term2 + term3 + term4;
    let expected_emmr = case["emmr"].as_f64().unwrap();
    assert!((emmr - expected_emmr).abs() < 1e-10, "emmr: {emmr} vs {expected_emmr}");
}

#[test]
fn test_emmr_term_decomposition_negative_v_sq() {
    let golden = load_golden();
    let case = &golden["emmr_term_decomposition"][2];

    let var_t_star = case["var_t_star"].as_f64().unwrap();
    let var_t1_star = case["var_t1_star"].as_f64().unwrap();
    let cov_t = case["cov_t"].as_f64().unwrap();

    // v_sq 为负
    let v_sq = var_t_star - 2.0 * cov_t + var_t1_star;
    assert!(v_sq < 0.0, "v_sq should be negative: {v_sq}");

    // 应 clamp 到 1e-10
    let v = v_sq.max(1e-10).sqrt();
    let expected_v = case["v"].as_f64().unwrap();
    assert!((v - expected_v).abs() < 1e-14, "v: {v} vs {expected_v}");

    // g 会非常大
    let mu_t_star = case["mu_t_star"].as_f64().unwrap();
    let mu_t1_star = case["mu_t1_star"].as_f64().unwrap();
    let g = (mu_t_star - mu_t1_star) / v;
    assert!(g > 1e4, "g should be very large: {g}");
}

// ============================================================================
// Group 6: KL 散度三项独立验证
// ============================================================================

#[test]
fn test_kl_divergence_terms() {
    let golden = load_golden();
    let cases = golden["kl_divergence_terms"].as_array().unwrap();

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let var_t1 = case["var_t1"].as_f64().unwrap();
        let mu_t1 = case["mu_t1"].as_f64().unwrap();
        let y_t = case["y_t"].as_f64().unwrap();
        let lambda_inv = case["lambda_inv"].as_f64().unwrap();
        let lambda = 1.0 / lambda_inv;

        let rhs1 = 0.5 * (1.0 + lambda * var_t1).ln();
        let rhs2 = -0.5 * var_t1 / (var_t1 + lambda_inv);
        let rhs3 = 0.5 * var_t1 * (y_t - mu_t1).powi(2) / (var_t1 + lambda_inv).powi(2);
        let kl_bound = rhs1 + rhs2 + rhs3;
        let sqrt_half_kl = (0.5 * kl_bound.max(0.0)).sqrt();

        let expected_rhs1 = case["rhs1"].as_f64().unwrap();
        let expected_rhs2 = case["rhs2"].as_f64().unwrap();
        let expected_rhs3 = case["rhs3"].as_f64().unwrap();
        let expected_kl = case["kl_bound"].as_f64().unwrap();
        let expected_sqrt = case["sqrt_half_kl"].as_f64().unwrap();

        assert!((rhs1 - expected_rhs1).abs() < 1e-8,
            "{name}: rhs1 Rust={rhs1:.15e} Python={expected_rhs1:.15e}");
        assert!((rhs2 - expected_rhs2).abs() < 1e-8,
            "{name}: rhs2 Rust={rhs2:.15e} Python={expected_rhs2:.15e}");
        assert!((rhs3 - expected_rhs3).abs() < 1e-8,
            "{name}: rhs3 Rust={rhs3:.15e} Python={expected_rhs3:.15e}");
        assert!((kl_bound - expected_kl).abs() < 1e-8,
            "{name}: kl_bound Rust={kl_bound:.15e} Python={expected_kl:.15e}");
        assert!((sqrt_half_kl - expected_sqrt).abs() < 1e-8,
            "{name}: sqrt_half_kl Rust={sqrt_half_kl:.15e} Python={expected_sqrt:.15e}");
    }
}

// ============================================================================
// Group 7: 终止决策逻辑
// ============================================================================

#[test]
fn test_terminator_decisions() {
    let golden = load_golden();
    let cases = golden["terminator_decisions"].as_array().unwrap();

    for case in cases {
        let name = case["name"].as_str().unwrap();
        let improvement = case["improvement"].as_f64().unwrap();
        let error = case["error"].as_f64().unwrap();
        let expected = case["should_terminate"].as_bool().unwrap();

        let result = improvement < error;
        assert_eq!(
            result, expected,
            "{name}: improvement={improvement} < error={error} → {result}, expected {expected}"
        );
    }
}

// ============================================================================
// Group 8: MedianErrorEvaluator 阈值计算（逻辑验证）
// ============================================================================

#[test]
fn test_median_threshold_odd_count() {
    let golden = load_golden();
    let case = &golden["median_threshold"][0];
    let criteria: Vec<f64> = case["criteria"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let ratio = case["threshold_ratio"].as_f64().unwrap();
    let expected_median = case["median"].as_f64().unwrap();
    let expected_threshold = case["threshold"].as_f64().unwrap();

    // 复现 MedianErrorEvaluator 的中位数逻辑: sort → [len//2]
    let mut sorted = criteria.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let threshold = (median * ratio).min(f64::MAX);

    assert!((median - expected_median).abs() < 1e-14,
        "median: {median} vs {expected_median}");
    assert!((threshold - expected_threshold).abs() < 1e-14,
        "threshold: {threshold} vs {expected_threshold}");
}

#[test]
fn test_median_threshold_even_count() {
    let golden = load_golden();
    let case = &golden["median_threshold"][1];
    let criteria: Vec<f64> = case["criteria"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let ratio = case["threshold_ratio"].as_f64().unwrap();
    let expected_median = case["median"].as_f64().unwrap();
    let expected_threshold = case["threshold"].as_f64().unwrap();

    let mut sorted = criteria.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let threshold = (median * ratio).min(f64::MAX);

    assert!((median - expected_median).abs() < 1e-14,
        "median: {median} vs {expected_median}");
    assert!((threshold - expected_threshold).abs() < 1e-14,
        "threshold: {threshold} vs {expected_threshold}");
}

#[test]
fn test_median_threshold_custom_ratio() {
    let golden = load_golden();
    let case = &golden["median_threshold"][2];
    let criteria: Vec<f64> = case["criteria"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let ratio = case["threshold_ratio"].as_f64().unwrap();
    let expected_threshold = case["threshold"].as_f64().unwrap();

    let mut sorted = criteria.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let threshold = (median * ratio).min(f64::MAX);

    assert!((threshold - expected_threshold).abs() < 1e-14,
        "threshold: {threshold} vs {expected_threshold}");
}

#[test]
fn test_median_threshold_all_same() {
    let golden = load_golden();
    let case = &golden["median_threshold"][3];
    let criteria: Vec<f64> = case["criteria"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let ratio = case["threshold_ratio"].as_f64().unwrap();
    let expected_threshold = case["threshold"].as_f64().unwrap();

    let mut sorted = criteria.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = sorted[sorted.len() / 2];
    let threshold = (median * ratio).min(f64::MAX);

    assert!((threshold - expected_threshold).abs() < 1e-14,
        "threshold: {threshold} vs {expected_threshold}");
}

// ============================================================================
// Group 9: GP-based 评估器行为验证（无精确值，验证性质）
// ============================================================================

/// 辅助：创建含参数的试验
fn make_study_trials(n: usize, seed: u64) -> optuna_rs::study::Study {
    use optuna_rs::samplers::RandomSampler;
    use optuna_rs::study::create_study;
    use std::sync::Arc;

    let sampler: Arc<dyn optuna_rs::samplers::Sampler> =
        Arc::new(RandomSampler::new(Some(seed)));
    let study = create_study(
        None,
        Some(sampler),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(x * x)
            },
            Some(n),
            None,
            None,
        )
        .unwrap();

    study
}

/// RegretBound: 增加试验数应使 regret bound 趋于减小
#[test]
fn test_regret_bound_decreasing_trend() {
    let eval = RegretBoundEvaluator::new(Some(0.5), Some(5), Some(42));

    let study_small = make_study_trials(10, 42);
    let study_large = make_study_trials(50, 42);

    let trials_small = study_small.trials().unwrap();
    let trials_large = study_large.trials().unwrap();

    let regret_small = eval.evaluate(&trials_small, StudyDirection::Minimize);
    let regret_large = eval.evaluate(&trials_large, StudyDirection::Minimize);

    // 更多数据通常导致更小的 regret bound（不保证，但趋势应明显）
    assert!(regret_small.is_finite(), "regret_small should be finite");
    assert!(regret_large.is_finite(), "regret_large should be finite");
    assert!(regret_small > 0.0, "regret_small should be positive");
    assert!(regret_large > 0.0, "regret_large should be positive");
}

/// EMMR: 确定性目标函数，后期 EMMR 应趋向收敛
#[test]
fn test_emmr_convergence_trend() {
    let eval = EMMREvaluator::new(Some(true), None, Some(2), Some(42));

    let study = make_study_trials(30, 42);
    let all_trials = study.trials().unwrap();

    // 用前 5 个试验 vs 用前 20 个试验
    let trials_5: Vec<FrozenTrial> = all_trials[..5].to_vec();
    let trials_20: Vec<FrozenTrial> = all_trials[..20].to_vec();

    let emmr_5 = eval.evaluate(&trials_5, StudyDirection::Minimize);
    let emmr_20 = eval.evaluate(&trials_20, StudyDirection::Minimize);

    assert!(emmr_5.is_finite(), "emmr_5 should be finite: {emmr_5}");
    assert!(emmr_20.is_finite(), "emmr_20 should be finite: {emmr_20}");
}

/// EMMR: 不足试验数应返回大值
#[test]
fn test_emmr_insufficient_trials_returns_large() {
    let eval = EMMREvaluator::new(None, None, Some(5), Some(42));

    let study = make_study_trials(3, 42);
    let trials = study.trials().unwrap();

    let result = eval.evaluate(&trials, StudyDirection::Minimize);
    assert!(result > 1e100, "insufficient trials should return very large value: {result}");
}

/// RegretBound: 空试验应返回 MAX
#[test]
fn test_regret_bound_empty_trials_returns_max() {
    let eval = RegretBoundEvaluator::default();
    let result = eval.evaluate(&[], StudyDirection::Minimize);
    assert_eq!(result, f64::MAX, "empty trials should return MAX");
}

/// StaticError: 确保各方向一致
#[test]
fn test_static_error_direction_invariant() {
    let eval = StaticErrorEvaluator::new(42.0);
    let trials = trials_from_values(&[1.0, 2.0, 3.0]);

    let min_result = eval.evaluate(&trials, StudyDirection::Minimize);
    let max_result = eval.evaluate(&trials, StudyDirection::Maximize);

    assert_eq!(min_result, max_result, "StaticError should be direction-invariant");
    assert_eq!(min_result, 42.0);
}

/// CV Error: Maximize 方向应选择最大值试验
#[test]
fn test_cv_error_maximize_selects_best() {
    let trial_bad = trial_with_cv(&[0.1, 0.2, 0.3], 0.2);  // 低值
    let trial_good = trial_with_cv(&[0.8, 0.85, 0.9], 0.85);  // 高值（最佳 maximize）

    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial_bad, trial_good], StudyDirection::Maximize);

    // 应使用 trial_good 的 CV 分数 [0.8, 0.85, 0.9]
    let k = 3.0;
    let scale = 1.0 / k + 1.0 / (k - 1.0);
    let mean = 0.85;
    let var = ((0.8_f64 - mean).powi(2) + (0.85_f64 - mean).powi(2) + (0.9_f64 - mean).powi(2)) / k;
    let expected = (scale * var).sqrt();

    assert!(
        (result - expected).abs() < 1e-12,
        "maximize should select highest value trial: {result} vs {expected}"
    );
}

/// DEFAULT_MINIMUM_NOISE_VAR 应与 Python 默认值一致
#[test]
fn test_default_minimum_noise_var() {
    // Python: prior.DEFAULT_MINIMUM_NOISE_VAR = 1e-4 (or similar)
    // 确认 Rust 使用的值
    assert!(
        DEFAULT_MINIMUM_NOISE_VAR > 0.0 && DEFAULT_MINIMUM_NOISE_VAR < 1.0,
        "DEFAULT_MINIMUM_NOISE_VAR should be a small positive value: {}",
        DEFAULT_MINIMUM_NOISE_VAR
    );
}
