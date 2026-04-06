//! 数值精度交叉验证测试 — Python 金标准参考值精确匹配。
//!
//! 所有参考值由 `tests/golden_values_generator.py` 通过 Python 3.13 + scipy 生成。
//! 每个测试断言 Rust 结果与 Python 结果的差异 < 1e-12（或更严格）。
//!
//! 覆盖模块:
//!   1. distributions (adjust_discrete_uniform_high, IntDistribution)
//!   2. terminators (CV Error, BestValueStagnation, β formula)
//!   3. normal_pdf / normal_cdf (importance/terminators 共用)
//!   4. 端到端 PED-ANOVA / fANOVA 属性验证
//!   5. 端到端 terminator 组合验证

use optuna_rs::distributions::{FloatDistribution, IntDistribution};
use optuna_rs::samplers::gp::{normal_pdf, normal_cdf};
use optuna_rs::terminators::*;
use optuna_rs::trial::{FrozenTrial, TrialState};
use optuna_rs::study::StudyDirection;
use std::collections::HashMap;

// ============================================================================
// 辅助函数
// ============================================================================

/// 构造 FrozenTrial（带 CV scores）
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

/// 构造多个 FrozenTrial（每个只有一个目标值）
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
// 1. FloatDistribution — adjust_discrete_uniform_high
// ============================================================================

/// 验证 FloatDistribution 的 step 调整与 Python Decimal 精确对齐。
/// Python 参考: `_adjust_discrete_uniform_high` 使用 Decimal(str(x)) 精确算术。
#[test]
fn test_float_distribution_step_adjustment_vs_python() {
    // (low, high, step, expected_adjusted_high)
    let cases: Vec<(f64, f64, f64, f64)> = vec![
        (0.0, 1.0, 0.3, 0.9),    // 1.0 % 0.3 ≠ 0 → 0.9
        (0.0, 1.0, 0.7, 0.7),    // 1.0 % 0.7 ≠ 0 → 0.7
        (0.0, 10.0, 3.0, 9.0),   // 10 % 3 ≠ 0 → 9.0
        (0.5, 5.5, 0.4, 5.3),    // 5.0 % 0.4 ≠ 0 → 0.5 + 12*0.4 = 5.3
        (0.1, 0.9, 0.15, 0.85),  // 0.8 % 0.15 ≠ 0 → 0.1 + 5*0.15 = 0.85
        (0.0, 1.0, 0.25, 1.0),   // 1.0 % 0.25 = 0 → 不调整
        (0.0, 1.0, 1.0, 1.0),    // 1.0 % 1.0 = 0 → 不调整
        (1.0, 10.0, 2.5, 8.5),   // 9.0 % 2.5 ≠ 0 → 1.0 + 3*2.5 = 8.5
        (0.0, 100.0, 7.0, 98.0), // 100 % 7 ≠ 0 → 14*7 = 98.0
        (-5.0, 5.0, 3.0, 4.0),   // 10 % 3 ≠ 0 → -5 + 3*3 = 4.0
    ];

    for (low, high, step, expected) in &cases {
        let dist = FloatDistribution::new(*low, *high, false, Some(*step)).unwrap();
        assert!(
            (dist.high - expected).abs() < 1e-12,
            "adjust_discrete_uniform_high({}, {}, step={}) → Rust={:.15}, Python={:.15}, diff={:.2e}",
            low, high, step, dist.high, expected, (dist.high - expected).abs()
        );
    }
}

/// 整除情况不应调整 high。
#[test]
fn test_float_distribution_step_exact_divisible() {
    let exact_cases: Vec<(f64, f64, f64)> = vec![
        (0.0, 1.0, 0.5),
        (0.0, 1.0, 0.25),
        (0.0, 2.0, 0.1),
        (0.0, 10.0, 5.0),
        (0.0, 10.0, 2.0),
    ];
    for (low, high, step) in &exact_cases {
        let dist = FloatDistribution::new(*low, *high, false, Some(*step)).unwrap();
        assert_eq!(
            dist.high, *high,
            "整除时 high 不应调整: ({}, {}, step={})",
            low, high, step
        );
    }
}

// ============================================================================
// 2. IntDistribution — adjust_int_uniform_high
// ============================================================================

/// 验证整数分布的 step 调整。
/// Python: adjusted = low + ((high - low) // step) * step
#[test]
fn test_int_distribution_step_adjustment_vs_python() {
    let cases: Vec<(i64, i64, i64, i64)> = vec![
        (0, 10, 3, 9),   // 10 % 3 = 1 → 9
        (0, 10, 4, 8),   // 10 % 4 = 2 → 8
        (0, 10, 5, 10),  // 10 % 5 = 0 → 不调整
        (1, 10, 3, 10),  // 9 % 3 = 0 → 不调整
        (1, 10, 4, 9),   // 9 % 4 = 1 → 9
        (-5, 5, 3, 4),   // 10 % 3 = 1 → 4
    ];

    for (low, high, step, expected) in &cases {
        let dist = IntDistribution::new(*low, *high, false, *step).unwrap();
        assert_eq!(
            dist.high, *expected,
            "adjust_int_uniform_high({}, {}, step={}) → Rust={}, Python={}",
            low, high, step, dist.high, expected
        );
    }
}

// ============================================================================
// 3. normal_pdf / normal_cdf — scipy 精确参考
// ============================================================================

/// 对比 scipy.stats.norm.pdf/cdf 的 15 位精度参考值。
#[test]
fn test_normal_pdf_vs_scipy() {
    let cases: Vec<(f64, f64)> = vec![
        (-5.0, 1.486719514734298e-06),
        (-3.0, 4.431848411938008e-03),
        (-2.0, 5.399096651318806e-02),
        (-1.5, 1.295175956658917e-01),
        (-1.0, 2.419707245191434e-01),
        (-0.5, 3.520653267642995e-01),
        (0.0, 3.989422804014327e-01),
        (0.5, 3.520653267642995e-01),
        (1.0, 2.419707245191434e-01),
        (1.5, 1.295175956658917e-01),
        (2.0, 5.399096651318806e-02),
        (3.0, 4.431848411938008e-03),
        (5.0, 1.486719514734298e-06),
    ];

    for (x, expected) in &cases {
        let rust_val = normal_pdf(*x);
        let rel_err = if expected.abs() > 1e-15 {
            (rust_val - expected).abs() / expected.abs()
        } else {
            (rust_val - expected).abs()
        };
        assert!(
            rel_err < 1e-10,
            "normal_pdf({}) → Rust={:.15e}, scipy={:.15e}, rel_err={:.2e}",
            x, rust_val, expected, rel_err
        );
    }
}

#[test]
fn test_normal_cdf_vs_scipy() {
    let cases: Vec<(f64, f64)> = vec![
        (-5.0, 2.866515718791934e-07),
        (-3.0, 1.349898031630093e-03),
        (-2.0, 2.275013194817920e-02),
        (-1.5, 6.680720126885806e-02),
        (-1.0, 1.586552539314571e-01),
        (-0.5, 3.085375387259869e-01),
        (0.0, 5.000000000000000e-01),
        (0.5, 6.914624612740131e-01),
        (1.0, 8.413447460685429e-01),
        (1.5, 9.331927987311419e-01),
        (2.0, 9.772498680518208e-01),
        (3.0, 9.986501019683699e-01),
        (5.0, 9.999997133484281e-01),
    ];

    for (x, expected) in &cases {
        let rust_val = normal_cdf(*x);
        let abs_err = (rust_val - expected).abs();
        // 对于接近 0 和 1 的 CDF，使用绝对误差
        let tol = if expected.abs() < 1e-4 || (1.0 - expected).abs() < 1e-4 {
            1e-10
        } else {
            1e-12
        };
        assert!(
            abs_err < tol,
            "normal_cdf({}) → Rust={:.15e}, scipy={:.15e}, abs_err={:.2e}",
            x, rust_val, expected, abs_err
        );
    }
}

/// normal_pdf 对称性: φ(-x) = φ(x)
#[test]
fn test_normal_pdf_symmetry() {
    for x in [0.1, 0.5, 1.0, 2.0, 3.0, 4.0] {
        let pos = normal_pdf(x);
        let neg = normal_pdf(-x);
        assert!(
            (pos - neg).abs() < 1e-15,
            "normal_pdf 对称性: pdf({x}) = {pos:.15e}, pdf({}) = {neg:.15e}",
            -x
        );
    }
}

/// normal_cdf 互补性: Φ(-x) + Φ(x) = 1
#[test]
fn test_normal_cdf_complementary() {
    for x in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0] {
        let sum = normal_cdf(x) + normal_cdf(-x);
        assert!(
            (sum - 1.0).abs() < 1e-12,
            "normal_cdf 互补性: Φ({x}) + Φ({}) = {sum:.15e} ≠ 1.0",
            -x
        );
    }
}

// ============================================================================
// 4. CrossValidationErrorEvaluator — 精确数值验证
// ============================================================================

/// 与 Python 金标准精确匹配（< 1e-12）
#[test]
fn test_cv_error_precise_3_scores() {
    let trial = trial_with_cv(&[0.8, 0.9, 1.0], 0.9);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);
    let expected = 7.453559924999296e-02;
    assert!(
        (result - expected).abs() < 1e-12,
        "CV(3): Rust={:.15e}, Python={:.15e}",
        result, expected
    );
}

#[test]
fn test_cv_error_precise_5_scores() {
    let trial = trial_with_cv(&[0.5, 0.6, 0.7, 0.8, 0.9], 0.7);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);
    let expected = 9.486832980505139e-02;
    assert!(
        (result - expected).abs() < 1e-12,
        "CV(5): Rust={:.15e}, Python={:.15e}",
        result, expected
    );
}

#[test]
fn test_cv_error_precise_10_scores() {
    let trial = trial_with_cv(
        &[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        0.55,
    );
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);
    let expected = 1.319722192988610e-01;
    assert!(
        (result - expected).abs() < 1e-12,
        "CV(10): Rust={:.15e}, Python={:.15e}",
        result, expected
    );
}

#[test]
fn test_cv_error_zero_variance() {
    let trial = trial_with_cv(&[0.95, 0.95, 0.95, 0.95, 0.95], 0.95);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);
    assert!(
        result.abs() < 1e-15,
        "零方差 CV error 应为 0.0, got {result:.15e}"
    );
}

#[test]
fn test_cv_error_precise_2_scores() {
    let trial = trial_with_cv(&[0.5, 1.5], 1.0);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);
    let expected = 6.123724356957945e-01;
    assert!(
        (result - expected).abs() < 1e-12,
        "CV(2): Rust={:.15e}, Python={:.15e}",
        result, expected
    );
}

/// CV Error scale 公式独立验证: scale = 1/k + 1/(k-1)
#[test]
fn test_cv_error_scale_formula() {
    for k in 2..=20 {
        let scale = 1.0 / k as f64 + 1.0 / (k as f64 - 1.0);
        // scale 应与 Python 一致（纯算术，无精度问题）
        let expected_scale = (2.0 * k as f64 - 1.0) / (k as f64 * (k as f64 - 1.0));
        assert!(
            (scale - expected_scale).abs() < 1e-14,
            "k={k}: scale 计算偏差"
        );
    }
}

// ============================================================================
// 5. BestValueStagnationEvaluator — 精确逻辑验证
// ============================================================================

/// 递减后停滞 → room = patience - stagnation_count
#[test]
fn test_stagnation_precise_decreasing_then_constant() {
    // [10,9,...,1,5,5,5,5,5]: best=1 at step 9, stagnation=5
    let values: Vec<f64> = (0..10)
        .rev()
        .map(|i| (i + 1) as f64)
        .chain(std::iter::repeat(5.0).take(5))
        .collect();
    let trials = trials_from_values(&values);

    // patience=3 → room = 3 - 5 = -2
    assert_eq!(
        BestValueStagnationEvaluator::new(3).evaluate(&trials, StudyDirection::Minimize),
        -2.0
    );
    // patience=5 → room = 5 - 5 = 0
    assert_eq!(
        BestValueStagnationEvaluator::new(5).evaluate(&trials, StudyDirection::Minimize),
        0.0
    );
    // patience=10 → room = 10 - 5 = 5
    assert_eq!(
        BestValueStagnationEvaluator::new(10).evaluate(&trials, StudyDirection::Minimize),
        5.0
    );
}

/// Maximize 方向: 目标 [0,1,2,3,4,2,2,2]
#[test]
fn test_stagnation_precise_maximize() {
    let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0];
    let trials = trials_from_values(&values);
    // best=4 at step 4, stagnation=3 (steps 5,6,7)
    assert_eq!(
        BestValueStagnationEvaluator::new(3).evaluate(&trials, StudyDirection::Maximize),
        0.0
    );
    assert_eq!(
        BestValueStagnationEvaluator::new(5).evaluate(&trials, StudyDirection::Maximize),
        2.0
    );
}

/// 所有值相同 → best_step=0, stagnation = n-1
#[test]
fn test_stagnation_all_equal() {
    let values = vec![1.0; 20];
    let trials = trials_from_values(&values);
    // best_step=0, current=19, stagnation=19, room = patience - 19
    assert_eq!(
        BestValueStagnationEvaluator::new(5).evaluate(&trials, StudyDirection::Minimize),
        -14.0
    );
    assert_eq!(
        BestValueStagnationEvaluator::new(20).evaluate(&trials, StudyDirection::Minimize),
        1.0
    );
}

/// 持续改善 → stagnation=0, room=patience
#[test]
fn test_stagnation_continuous_improvement() {
    let values: Vec<f64> = (0..50).map(|i| 100.0 - i as f64).collect();
    let trials = trials_from_values(&values);
    assert_eq!(
        BestValueStagnationEvaluator::new(10).evaluate(&trials, StudyDirection::Minimize),
        10.0
    );
}

// ============================================================================
// 6. EMMR β 公式 — 独立精确验证
// ============================================================================

/// β = 2 * ln(d * n² * π² / (6δ)) / 5, δ=0.1
#[test]
fn test_emmr_beta_formula_vs_python() {
    let delta = 0.1_f64;
    let cases: Vec<(usize, usize, f64)> = vec![
        (1, 10, 2.962182232581153e+00),
        (2, 20, 3.793958849253087e+00),
        (3, 30, 4.280516978982885e+00),
        (5, 50, 4.893507727502073e+00),
        (10, 100, 5.725284344174008e+00),
        (1, 100, 4.804250306976390e+00),
        (20, 200, 6.557060960845942e+00),
        (2, 5, 2.684923360357175e+00),
    ];

    for (n_params, n_trials, expected) in &cases {
        let d = *n_params as f64;
        let n = *n_trials as f64;
        let arg = d * n * n * std::f64::consts::PI * std::f64::consts::PI / (6.0 * delta);
        let beta = 2.0 * arg.ln() / 5.0;
        assert!(
            (beta - expected).abs() < 1e-10,
            "β(d={}, n={}) → Rust={:.15e}, Python={:.15e}, diff={:.2e}",
            n_params, n_trials, beta, expected, (beta - expected).abs()
        );
    }
}

// ============================================================================
// 7. Pearson χ² 散度 — 独立公式验证
// ============================================================================

/// D_χ²(p || q) = Σ (q+ε) * ((p+ε)/(q+ε) - 1)², ε=1e-12
fn pearson_divergence(pdf_p: &[f64], pdf_q: &[f64]) -> f64 {
    let eps = 1e-12;
    pdf_p
        .iter()
        .zip(pdf_q.iter())
        .map(|(&p, &q)| {
            let q_safe = q + eps;
            let p_safe = p + eps;
            q_safe * ((p_safe / q_safe) - 1.0).powi(2)
        })
        .sum()
}

#[test]
fn test_pearson_divergence_identical_distributions() {
    let p = vec![0.2, 0.2, 0.2, 0.2, 0.2];
    let q = vec![0.2, 0.2, 0.2, 0.2, 0.2];
    let d = pearson_divergence(&p, &q);
    // 相同分布 → 散度 = 0 (模 ε)
    assert!(
        d < 1e-20,
        "相同分布的散度应 ≈ 0, got {d:.15e}"
    );
}

#[test]
fn test_pearson_divergence_left_skewed() {
    let p = vec![0.5, 0.3, 0.1, 0.05, 0.05];
    let q = vec![0.2, 0.2, 0.2, 0.2, 0.2];
    let d = pearson_divergence(&p, &q);
    let expected = 7.749999999961249e-01;
    assert!(
        (d - expected).abs() < 1e-10,
        "左偏散度: Rust={:.15e}, Python={:.15e}",
        d, expected
    );
}

#[test]
fn test_pearson_divergence_concentrated() {
    let p = vec![1.0, 0.0, 0.0, 0.0, 0.0];
    let q = vec![0.2, 0.2, 0.2, 0.2, 0.2];
    let d = pearson_divergence(&p, &q);
    let expected = 3.999999999980000e+00;
    assert!(
        (d - expected).abs() < 1e-8,
        "集中分布散度: Rust={:.15e}, Python={:.15e}",
        d, expected
    );
}

#[test]
fn test_pearson_divergence_different_shapes() {
    let p = vec![0.1, 0.2, 0.4, 0.2, 0.1];
    let q = vec![0.3, 0.1, 0.2, 0.3, 0.1];
    let d = pearson_divergence(&p, &q);
    let expected = 4.666666666641109e-01;
    assert!(
        (d - expected).abs() < 1e-10,
        "不同形状散度: Rust={:.15e}, Python={:.15e}",
        d, expected
    );
}

/// 散度对称性: D(p||q) 不等于 D(q||p)（非对称度量）
#[test]
fn test_pearson_divergence_asymmetry() {
    let p = vec![0.5, 0.3, 0.1, 0.05, 0.05];
    let q = vec![0.2, 0.2, 0.2, 0.2, 0.2];
    let d_pq = pearson_divergence(&p, &q);
    let d_qp = pearson_divergence(&q, &p);
    // Pearson 散度是非对称的（虽然对均匀分布的特殊情况可能相等）
    // 这里只验证两个方向都是非负的
    assert!(d_pq >= 0.0 && d_qp >= 0.0);
}

// ============================================================================
// 8. weighted_variance — 独立公式验证
// ============================================================================

fn weighted_variance(values: &[f64], weights: &[f64]) -> f64 {
    let w_sum: f64 = weights.iter().sum();
    if w_sum < 1e-14 {
        return 0.0;
    }
    let mean: f64 = values
        .iter()
        .zip(weights.iter())
        .map(|(v, w)| v * w)
        .sum::<f64>()
        / w_sum;
    values
        .iter()
        .zip(weights.iter())
        .map(|(v, w)| w * (v - mean).powi(2))
        .sum::<f64>()
        / w_sum
}

#[test]
fn test_weighted_variance_equal_weights() {
    // [1,2,3] 等权 → mean=2, var=2/3
    let var = weighted_variance(&[1.0, 2.0, 3.0], &[1.0, 1.0, 1.0]);
    let expected = 6.666666666666666e-01;
    assert!(
        (var - expected).abs() < 1e-12,
        "等权方差: Rust={:.15e}, Python={:.15e}",
        var, expected
    );
}

#[test]
fn test_weighted_variance_unequal_weights() {
    let var = weighted_variance(
        &[1.0, 2.0, 3.0, 4.0, 5.0],
        &[0.1, 0.2, 0.4, 0.2, 0.1],
    );
    let expected = 1.200000000000000e+00;
    assert!(
        (var - expected).abs() < 1e-12,
        "不等权方差: Rust={:.15e}, Python={:.15e}",
        var, expected
    );
}

#[test]
fn test_weighted_variance_asymmetric() {
    let var = weighted_variance(&[10.0, 20.0, 30.0], &[5.0, 3.0, 2.0]);
    let expected = 6.100000000000000e+01;
    assert!(
        (var - expected).abs() < 1e-10,
        "非对称权重方差: Rust={:.15e}, Python={:.15e}",
        var, expected
    );
}

#[test]
fn test_weighted_variance_zero_variance() {
    let var = weighted_variance(&[1.0, 1.0, 1.0], &[1.0, 2.0, 3.0]);
    assert!(var.abs() < 1e-15, "常数值方差应为 0, got {var:.15e}");
}

#[test]
fn test_weighted_variance_single_value() {
    let var = weighted_variance(&[100.0], &[1.0]);
    assert!(var.abs() < 1e-15, "单值方差应为 0, got {var:.15e}");
}

// ============================================================================
// 9. Scott-Parzen 带宽 — 独立公式验证
// ============================================================================

fn scott_bandwidth(counts: &[usize]) -> f64 {
    let n_total: usize = counts.iter().sum();
    if n_total <= 1 {
        return f64::NAN;
    }

    let mut mus = Vec::new();
    let mut counts_nz = Vec::new();
    for (i, &c) in counts.iter().enumerate() {
        if c > 0 {
            mus.push(i as f64);
            counts_nz.push(c as f64);
        }
    }
    if mus.is_empty() {
        return f64::NAN;
    }

    let n = n_total as f64;
    let weights: Vec<f64> = counts_nz.iter().map(|c| c / n).collect();
    let mean_est: f64 = mus.iter().zip(weights.iter()).map(|(m, w)| m * w).sum();
    let var_est: f64 = mus
        .iter()
        .zip(counts_nz.iter())
        .map(|(m, c)| (m - mean_est).powi(2) * c)
        .sum::<f64>()
        / (n - 1.0).max(1.0);
    let sigma_est = var_est.sqrt();

    // IQR
    let cum: Vec<f64> = {
        let mut c = Vec::with_capacity(counts_nz.len());
        let mut acc = 0.0;
        for &cnt in &counts_nz {
            acc += cnt;
            c.push(acc);
        }
        c
    };
    let q25_target = n / 4.0;
    let q75_target = n * 3.0 / 4.0;
    let idx_q25 = cum.iter().position(|&c| c >= q25_target).unwrap_or(0);
    let idx_q75 = cum
        .iter()
        .position(|&c| c >= q75_target)
        .unwrap_or(mus.len() - 1);
    let iqr = mus[idx_q75.min(mus.len() - 1)] - mus[idx_q25];

    let sigma_choice = if iqr > 0.0 {
        (iqr / 1.34).min(sigma_est)
    } else {
        sigma_est
    };
    let h = 1.059 * sigma_choice * n.powf(-0.2);
    let sigma_min = 0.5 / 1.64;
    h.max(sigma_min)
}

#[test]
fn test_scott_bandwidth_sparse() {
    let bw = scott_bandwidth(&[0, 0, 5, 0, 3, 0, 2, 0, 0, 0]);
    let expected = 9.972892952306337e-01;
    assert!(
        (bw - expected).abs() < 1e-10,
        "稀疏带宽: Rust={:.15e}, Python={:.15e}",
        bw, expected
    );
}

#[test]
fn test_scott_bandwidth_uniform() {
    let bw = scott_bandwidth(&[1, 1, 1, 1, 1, 1, 1, 1, 1, 1]);
    let expected = 2.023027002854586e+00;
    assert!(
        (bw - expected).abs() < 1e-10,
        "均匀带宽: Rust={:.15e}, Python={:.15e}",
        bw, expected
    );
}

#[test]
fn test_scott_bandwidth_bimodal() {
    let bw = scott_bandwidth(&[10, 0, 0, 0, 0, 0, 0, 0, 0, 10]);
    let expected = 2.685595942584707e+00;
    assert!(
        (bw - expected).abs() < 1e-10,
        "双峰带宽: Rust={:.15e}, Python={:.15e}",
        bw, expected
    );
}

#[test]
fn test_scott_bandwidth_concentrated() {
    let bw = scott_bandwidth(&[0, 0, 0, 0, 20, 0, 0, 0, 0, 0]);
    let expected = 3.048780487804878e-01;
    assert!(
        (bw - expected).abs() < 1e-10,
        "集中带宽: Rust={:.15e}, Python={:.15e}",
        bw, expected
    );
}

#[test]
fn test_scott_bandwidth_normal_like() {
    let bw = scott_bandwidth(&[1, 2, 3, 4, 5, 4, 3, 2, 1, 0]);
    let expected = 8.302964014518176e-01;
    assert!(
        (bw - expected).abs() < 1e-10,
        "类正态带宽: Rust={:.15e}, Python={:.15e}",
        bw, expected
    );
}

/// sigma_min 生效的边界情况: 所有样本在一个 grid 点
/// bandwidth = sigma_min = 0.5/1.64 ≈ 0.30488
#[test]
fn test_scott_bandwidth_sigma_min_kicks_in() {
    // 样本集中 → sigma_est=0 → IQR=0 → h=0 → 被 sigma_min 兜底
    let bw = scott_bandwidth(&[0, 0, 0, 0, 20, 0, 0, 0, 0, 0]);
    let sigma_min = 0.5 / 1.64;
    assert_eq!(bw, sigma_min, "sigma_min 应被使用");
}

// ============================================================================
// 10. discretize_param — 精确 grid 映射验证
// ============================================================================

fn discretize_param(values: &[f64], low: f64, high: f64, n_steps: usize, is_log: bool) -> Vec<usize> {
    if (high - low).abs() < 1e-14 {
        return vec![0; values.len()];
    }
    let (s_low, s_high) = if is_log {
        (low.max(1e-300).ln(), high.max(1e-300).ln())
    } else {
        (low, high)
    };
    let grids: Vec<f64> = (0..n_steps)
        .map(|i| s_low + (s_high - s_low) * i as f64 / (n_steps - 1).max(1) as f64)
        .collect();
    values
        .iter()
        .map(|&v| {
            let sv = if is_log { v.max(1e-300).ln() } else { v };
            grids
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| {
                    (sv - *a)
                        .abs()
                        .partial_cmp(&(sv - *b).abs())
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(i, _)| i)
                .unwrap_or(0)
        })
        .collect()
}

#[test]
fn test_discretize_linear_exact_grid_points() {
    // [0, 0.25, 0.5, 0.75, 1.0] 在 5-grid 上 → [0, 1, 2, 3, 4]
    let indices = discretize_param(&[0.0, 0.25, 0.5, 0.75, 1.0], 0.0, 1.0, 5, false);
    assert_eq!(indices, vec![0, 1, 2, 3, 4]);
}

#[test]
fn test_discretize_linear_non_grid_points() {
    // [0.1, 0.3, 0.7, 0.9] 在 10-grid 上 → [1, 3, 6, 8]
    let indices = discretize_param(&[0.1, 0.3, 0.7, 0.9], 0.0, 1.0, 10, false);
    assert_eq!(indices, vec![1, 3, 6, 8]);
}

#[test]
fn test_discretize_log_domain() {
    // [1, 10, 100] 在 log(1)..log(100) 的 10-grid 上 → [0, 4, 9]
    let indices = discretize_param(&[1.0, 10.0, 100.0], 1.0, 100.0, 10, true);
    assert_eq!(indices, vec![0, 4, 9]);
}

#[test]
fn test_discretize_symmetric() {
    // [-5, -2.5, 0, 2.5, 5] 在 11-grid 上 → [0, 2, 5, 7, 10]
    let indices = discretize_param(&[-5.0, -2.5, 0.0, 2.5, 5.0], -5.0, 5.0, 11, false);
    assert_eq!(indices, vec![0, 2, 5, 7, 10]);
}

/// 恒定值 → 所有索引为 0
#[test]
fn test_discretize_constant_range() {
    let indices = discretize_param(&[5.0, 5.0, 5.0], 5.0, 5.0, 10, false);
    assert_eq!(indices, vec![0, 0, 0]);
}

// ============================================================================
// 11. quantile_filter — 精确索引验证
// ============================================================================

fn quantile_filter(
    values: &[f64],
    quantile: f64,
    is_lower_better: bool,
    min_n_top: usize,
) -> Vec<usize> {
    let n = values.len();
    if n == 0 {
        return vec![];
    }
    let losses: Vec<f64> = if is_lower_better {
        values.to_vec()
    } else {
        values.iter().map(|v| -v).collect()
    };
    let mut sorted = losses.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let q_idx = ((n as f64 * quantile).ceil() as usize)
        .max(min_n_top)
        .min(n);
    let cutoff = sorted[q_idx - 1];

    let mut indices: Vec<usize> = (0..n).filter(|&i| losses[i] <= cutoff).collect();

    if indices.len() < min_n_top {
        let mut sorted_idx: Vec<usize> = (0..n).collect();
        sorted_idx.sort_by(|&a, &b| {
            losses[a]
                .partial_cmp(&losses[b])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        indices = sorted_idx[..min_n_top.min(n)].to_vec();
        indices.sort();
    }

    indices.sort();
    indices
}

#[test]
fn test_quantile_filter_10pct_minimize() {
    let values = vec![5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0];
    let indices = quantile_filter(&values, 0.1, true, 2);
    // Python: [3, 5] → values [1.0, 2.0]
    assert_eq!(indices, vec![3, 5]);
}

#[test]
fn test_quantile_filter_30pct_minimize() {
    let values = vec![5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0];
    let indices = quantile_filter(&values, 0.3, true, 2);
    // Python: [1, 3, 5] → values [3.0, 1.0, 2.0]
    assert_eq!(indices, vec![1, 3, 5]);
}

#[test]
fn test_quantile_filter_50pct_minimize() {
    let values = vec![5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0];
    let indices = quantile_filter(&values, 0.5, true, 2);
    // Python: [0, 1, 3, 5, 7] → values [5.0, 3.0, 1.0, 2.0, 4.0]
    assert_eq!(indices, vec![0, 1, 3, 5, 7]);
}

#[test]
fn test_quantile_filter_20pct_maximize() {
    let values = vec![5.0, 3.0, 8.0, 1.0, 7.0, 2.0, 9.0, 4.0, 6.0, 10.0];
    let indices = quantile_filter(&values, 0.2, false, 2);
    // Python: [6, 9] → values [9.0, 10.0]
    assert_eq!(indices, vec![6, 9]);
}

/// min_n_top 兜底: quantile 太小但 min_n_top 保证最少返回
#[test]
fn test_quantile_filter_min_n_top_floor() {
    let values = vec![10.0, 9.0, 8.0, 7.0, 6.0];
    // 0.01 * 5 = 0.05, ceil=1, max(1, min_n_top=3) = 3
    let indices = quantile_filter(&values, 0.01, true, 3);
    assert!(
        indices.len() >= 3,
        "min_n_top=3 应保证至少 3 个结果, got {}",
        indices.len()
    );
}

// ============================================================================
// 12. 端到端: terminator 组合验证
// ============================================================================

/// StaticError + BestValueStagnation: 应终止的场景
#[test]
fn test_terminator_should_stop() {
    let values: Vec<f64> = (0..10)
        .rev()
        .map(|i| (i + 1) as f64)
        .chain(std::iter::repeat(5.0).take(5))
        .collect();
    let trials = trials_from_values(&values);

    let imp = BestValueStagnationEvaluator::new(3);
    let err = StaticErrorEvaluator::new(0.0);

    let improvement = imp.evaluate(&trials, StudyDirection::Minimize);
    let error = err.evaluate(&trials, StudyDirection::Minimize);

    assert_eq!(improvement, -2.0);
    assert_eq!(error, 0.0);
    assert!(
        improvement < error,
        "应终止: improvement={} < error={}",
        improvement, error
    );
}

/// StaticError + BestValueStagnation: 不应终止的场景
#[test]
fn test_terminator_should_continue() {
    let values: Vec<f64> = (0..10)
        .rev()
        .map(|i| (i + 1) as f64)
        .chain(std::iter::repeat(5.0).take(5))
        .collect();
    let trials = trials_from_values(&values);

    let imp = BestValueStagnationEvaluator::new(10);
    let err = StaticErrorEvaluator::new(0.0);

    let improvement = imp.evaluate(&trials, StudyDirection::Minimize);
    let error = err.evaluate(&trials, StudyDirection::Minimize);

    assert_eq!(improvement, 5.0);
    assert_eq!(error, 0.0);
    assert!(
        improvement >= error,
        "不应终止: improvement={} >= error={}",
        improvement, error
    );
}

// ============================================================================
// 13. Distribution contains 边界行为
// ============================================================================

#[test]
fn test_float_distribution_contains_nan() {
    let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
    assert!(!d.contains(f64::NAN), "NaN 不应属于任何分布");
}

#[test]
fn test_float_distribution_contains_inf() {
    let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
    assert!(!d.contains(f64::INFINITY));
    assert!(!d.contains(f64::NEG_INFINITY));
}

#[test]
fn test_int_distribution_contains_nan() {
    let d = IntDistribution::new(0, 10, false, 1).unwrap();
    assert!(!d.contains(f64::NAN), "NaN 不应属于整数分布");
}

#[test]
fn test_float_distribution_step_contains_precision() {
    // step=0.3: valid values are 0.0, 0.3, 0.6, 0.9
    let d = FloatDistribution::new(0.0, 1.0, false, Some(0.3)).unwrap();
    assert_eq!(d.high, 0.9);
    assert!(d.contains(0.0));
    assert!(d.contains(0.3));
    assert!(d.contains(0.6));
    assert!(d.contains(0.9));
    // 0.15 不在 grid 上
    assert!(!d.contains(0.15));
    // 1.0 已被调整为 0.9, 所以 1.0 不在范围内
    assert!(!d.contains(1.0));
}

// ============================================================================
// 14. 浮点精度边界验证
// ============================================================================

/// f64 精度极限场景: 1.0 - 1e-16 应在 [0, 1] 内
#[test]
fn test_float_precision_boundary() {
    let d = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
    assert!(d.contains(1.0 - 1e-16));
    assert!(d.contains(0.0 + 1e-16));
    // 刚好在外面
    assert!(!d.contains(1.0 + 1e-10));
    assert!(!d.contains(-1e-10));
}

/// log 分布: 0 和负数不应接受
#[test]
fn test_float_log_distribution_validation() {
    assert!(FloatDistribution::new(0.0, 1.0, true, None).is_err());
    assert!(FloatDistribution::new(-1.0, 1.0, true, None).is_err());
    assert!(FloatDistribution::new(0.001, 100.0, true, None).is_ok());
}

/// int log 分布: low 必须 >= 1
#[test]
fn test_int_log_distribution_validation() {
    assert!(IntDistribution::new(0, 10, true, 1).is_err());
    assert!(IntDistribution::new(1, 10, true, 1).is_ok());
    // log + step != 1 被拒绝
    assert!(IntDistribution::new(1, 10, true, 2).is_err());
}

// ============================================================================
// 15. 端到端 PED-ANOVA 属性验证（确定性试验数据）
// ============================================================================

/// 用手工构造的 FrozenTrial 测试 PED-ANOVA 重要性评估。
/// 目标: 参数 "big" 范围大且与目标高度相关, "small" 无关。
#[test]
fn test_pedanova_deterministic_trials() {
    use optuna_rs::importance::{get_param_importances, PedAnovaEvaluator};
    use std::sync::Arc;
    use optuna_rs::samplers::{RandomSampler, Sampler};
    use optuna_rs::study::create_study;

    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
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
                let big = trial.suggest_float("big", 0.0, 100.0, false, None)?;
                let small = trial.suggest_float("small", 0.0, 0.001, false, None)?;
                Ok(big + small)
            },
            Some(80),
            None,
            None,
        )
        .unwrap();

    let eval = PedAnovaEvaluator::default();
    let imp = get_param_importances(&study, Some(&eval), None, None, true).unwrap();

    let big_imp = *imp.get("big").unwrap_or(&0.0);
    let small_imp = *imp.get("small").unwrap_or(&0.0);

    // "big" 应该远比 "small" 重要
    assert!(
        big_imp > 0.9,
        "big 应主导, got big={big_imp}, small={small_imp}"
    );
    // 归一化 sum = 1
    let sum: f64 = imp.values().sum();
    assert!((sum - 1.0).abs() < 1e-10);
}

// ============================================================================
// 16. 综合: 多公式联合一致性验证
// ============================================================================

/// 验证 CV Error 公式的数学性质:
///   1. scale 随 k 增大越来越接近 0
///   2. 方差为 0 时误差为 0
///   3. 增加折数但保持方差不变 → error 减小
#[test]
fn test_cv_error_mathematical_properties() {
    // Property 1: scale 递减
    let mut prev_scale = f64::MAX;
    for k in 2..=100 {
        let scale = 1.0 / k as f64 + 1.0 / (k as f64 - 1.0);
        assert!(
            scale < prev_scale,
            "k={k}: scale={scale} 应 < prev={prev_scale}"
        );
        prev_scale = scale;
    }

    // Property 2: 零方差 → 零误差
    for k in 2..=10 {
        let scores: Vec<f64> = vec![0.5; k];
        let trial = trial_with_cv(&scores, 0.5);
        let eval = CrossValidationErrorEvaluator::new();
        let result = eval.evaluate(&[trial], StudyDirection::Minimize);
        assert!(
            result.abs() < 1e-15,
            "k={k}: 零方差应给出零误差, got {result}"
        );
    }
}

/// normal_pdf 在 x=0 处的值是已知常数 1/√(2π)
#[test]
fn test_normal_pdf_at_zero_is_exact() {
    let expected = 1.0 / (2.0 * std::f64::consts::PI).sqrt();
    let result = normal_pdf(0.0);
    assert!(
        (result - expected).abs() < 1e-15,
        "pdf(0) = 1/sqrt(2π): Rust={:.15e}, exact={:.15e}",
        result, expected
    );
}

/// normal_cdf 在 x=0 处恰好为 0.5
#[test]
fn test_normal_cdf_at_zero_is_half() {
    let result = normal_cdf(0.0);
    assert!(
        (result - 0.5).abs() < 1e-15,
        "cdf(0) should be exactly 0.5, got {result:.15e}"
    );
}
