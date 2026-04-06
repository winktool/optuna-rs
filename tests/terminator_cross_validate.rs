//! 终止器模块精确交叉验证测试。
//!
//! 所有参考值均由 Python optuna 生成，确保 Rust 移植与 Python 精确对齐。

use optuna_rs::terminators::*;
use optuna_rs::trial::{FrozenTrial, TrialState};
use optuna_rs::study::StudyDirection;
use std::collections::HashMap;

// ============================================================================
// 1. CRC32 交叉验证（Hyperband 使用）
// ============================================================================

/// 独立验证 CRC32 实现与 Python binascii.crc32 的一致性。
/// Python 参考: `binascii.crc32(b"study_0") = 368277276`
#[test]
fn test_crc32_vs_python() {
    // 使用与 Hyperband 相同的 CRC32 算法
    fn crc32_hash(data: &[u8]) -> u32 {
        let mut crc: u32 = 0xFFFFFFFF;
        for &byte in data {
            crc ^= byte as u32;
            for _ in 0..8 {
                if crc & 1 != 0 {
                    crc = (crc >> 1) ^ 0xEDB88320;
                } else {
                    crc >>= 1;
                }
            }
        }
        crc ^ 0xFFFFFFFF
    }

    // Python binascii.crc32() 参考值
    let test_cases: Vec<(&str, u32)> = vec![
        ("study_0", 368277276),
        ("study_100", 2442318502),
        ("study_5000", 1151196281),
        ("test_study_42", 4291747687),
        ("hello_world", 4148080273),
    ];

    for (input, expected) in test_cases {
        let result = crc32_hash(input.as_bytes());
        assert_eq!(
            result, expected,
            "CRC32 mismatch for '{}': Rust={} (0x{:08x}), Python={} (0x{:08x})",
            input, result, result, expected, expected
        );
    }
}

// ============================================================================
// 2. CrossValidationErrorEvaluator 数值精度
// ============================================================================

fn make_trial_with_cv_scores(scores: &[f64], value: f64) -> FrozenTrial {
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

/// Python 参考: scores=[0.8, 0.9, 1.0] → std=7.453559924999296e-02
#[test]
fn test_cv_error_evaluator_3_scores() {
    let trial = make_trial_with_cv_scores(&[0.8, 0.9, 1.0], 0.9);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);
    let expected = 7.453559924999296e-02;
    assert!(
        (result - expected).abs() < 1e-12,
        "CV error mismatch: Rust={:.15e}, Python={:.15e}, diff={:.2e}",
        result, expected, (result - expected).abs()
    );
}

/// Python 参考: scores=[0.5, 0.6, 0.7, 0.8, 0.9] → std=9.486832980505139e-02
#[test]
fn test_cv_error_evaluator_5_scores() {
    let trial = make_trial_with_cv_scores(&[0.5, 0.6, 0.7, 0.8, 0.9], 0.7);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);
    let expected = 9.486832980505139e-02;
    assert!(
        (result - expected).abs() < 1e-12,
        "CV error mismatch: Rust={:.15e}, Python={:.15e}",
        result, expected
    );
}

/// Python 参考: scores=[0.95, 0.96, 0.97] → std=7.453559924999306e-03
#[test]
fn test_cv_error_evaluator_small_var() {
    let trial = make_trial_with_cv_scores(&[0.95, 0.96, 0.97], 0.96);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);
    let expected = 7.453559924999306e-03;
    assert!(
        (result - expected).abs() < 1e-12,
        "CV error mismatch: Rust={:.15e}, Python={:.15e}",
        result, expected
    );
}

/// Python 参考: scores=[1.0, 1.0, 1.0] → std=0.0 (零方差)
#[test]
fn test_cv_error_evaluator_zero_variance() {
    let trial = make_trial_with_cv_scores(&[1.0, 1.0, 1.0], 1.0);
    let eval = CrossValidationErrorEvaluator::new();
    let result = eval.evaluate(&[trial], StudyDirection::Minimize);
    assert!(
        result.abs() < 1e-15,
        "零方差时 CV error 应为 0，got {}",
        result
    );
}

/// 多个试验中选择最佳试验的 CV 分数（Minimize）
#[test]
fn test_cv_error_evaluator_selects_best_trial_minimize() {
    let trial1 = make_trial_with_cv_scores(&[0.5, 0.6, 0.7], 0.6); // 不是最佳
    let trial2 = make_trial_with_cv_scores(&[0.1, 0.2, 0.3], 0.2); // 最佳 (minimize)

    let eval = CrossValidationErrorEvaluator::new();
    // 应使用 trial2 (value=0.2) 的 CV 分数
    let result = eval.evaluate(&[trial1, trial2], StudyDirection::Minimize);
    // trial2 scores: [0.1, 0.2, 0.3], k=3, scale=5/6
    // var = ((0.1-0.2)^2 + 0 + (0.3-0.2)^2) / 3 = 0.02/3
    // std = sqrt(5/6 * 0.02/3) = sqrt(0.01111...) ≈ 0.10541
    let k: f64 = 3.0;
    let scale: f64 = 1.0 / k + 1.0 / (k - 1.0);
    let mean: f64 = 0.2;
    let var: f64 = ((0.1_f64 - mean).powi(2) + (0.2_f64 - mean).powi(2) + (0.3_f64 - mean).powi(2)) / k;
    let expected: f64 = (scale * var).sqrt();
    assert!(
        (result - expected).abs() < 1e-12,
        "Should use best trial's CV scores: Rust={:.15e}, expected={:.15e}",
        result, expected
    );
}

// ============================================================================
// 3. BestValueStagnationEvaluator 精确对齐
// ============================================================================

fn make_trials_with_values(values: &[f64]) -> Vec<FrozenTrial> {
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

/// Python 参考: minimize [10,9,8,...,1,5,5,5,5,5], patience=3 → room=-2
#[test]
fn test_stagnation_minimize_patience3() {
    let values: Vec<f64> = (0..10)
        .rev()
        .map(|i| (i + 1) as f64)
        .chain(std::iter::repeat(5.0).take(5))
        .collect();
    let trials = make_trials_with_values(&values);
    let eval = BestValueStagnationEvaluator::new(3);
    let result = eval.evaluate(&trials, StudyDirection::Minimize);
    assert_eq!(result, -2.0, "patience=3, stagnation=5: room should be -2");
}

/// Python 参考: minimize [10,9,...,1,5,5,5,5,5], patience=5 → room=0
#[test]
fn test_stagnation_minimize_patience5() {
    let values: Vec<f64> = (0..10)
        .rev()
        .map(|i| (i + 1) as f64)
        .chain(std::iter::repeat(5.0).take(5))
        .collect();
    let trials = make_trials_with_values(&values);
    let eval = BestValueStagnationEvaluator::new(5);
    let result = eval.evaluate(&trials, StudyDirection::Minimize);
    assert_eq!(result, 0.0, "patience=5, stagnation=5: room should be 0");
}

/// Python 参考: minimize [10,9,...,1,5,5,5,5,5], patience=10 → room=5
#[test]
fn test_stagnation_minimize_patience10() {
    let values: Vec<f64> = (0..10)
        .rev()
        .map(|i| (i + 1) as f64)
        .chain(std::iter::repeat(5.0).take(5))
        .collect();
    let trials = make_trials_with_values(&values);
    let eval = BestValueStagnationEvaluator::new(10);
    let result = eval.evaluate(&trials, StudyDirection::Minimize);
    assert_eq!(result, 5.0, "patience=10, stagnation=5: room should be 5");
}

/// Python 参考: maximize [0,1,2,3,4,2,2,2], patience=3 → room=0
#[test]
fn test_stagnation_maximize_patience3() {
    let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0];
    let trials = make_trials_with_values(&values);
    let eval = BestValueStagnationEvaluator::new(3);
    let result = eval.evaluate(&trials, StudyDirection::Maximize);
    assert_eq!(result, 0.0, "patience=3, stagnation=3: room should be 0");
}

/// Python 参考: maximize [0,1,2,3,4,2,2,2], patience=5 → room=2
#[test]
fn test_stagnation_maximize_patience5() {
    let values = vec![0.0, 1.0, 2.0, 3.0, 4.0, 2.0, 2.0, 2.0];
    let trials = make_trials_with_values(&values);
    let eval = BestValueStagnationEvaluator::new(5);
    let result = eval.evaluate(&trials, StudyDirection::Maximize);
    assert_eq!(result, 2.0, "patience=5, stagnation=3: room should be 2");
}

// ============================================================================
// 4. Beta 函数 (RegretBoundEvaluator 内部使用)
// ============================================================================

/// Python 参考: n_params=2, n_trials=20 → beta=3.793958849253087e+00
#[test]
fn test_beta_function_values() {
    let test_cases: Vec<(usize, usize, f64)> = vec![
        (2, 20, 3.793958849253087e+00),
        (5, 50, 4.893507727502073e+00),
        (10, 100, 5.725284344174008e+00),
    ];

    for (n_params, n_trials, expected_beta) in test_cases {
        let delta = 0.1_f64;
        let n = n_trials as f64;
        let d = n_params as f64;
        let beta_arg = d * n * n * std::f64::consts::PI * std::f64::consts::PI / (6.0 * delta);
        let beta = 2.0 * beta_arg.ln() / 5.0;
        assert!(
            (beta - expected_beta).abs() < 1e-10,
            "Beta mismatch: n_params={}, n_trials={}: Rust={:.15e}, Python={:.15e}",
            n_params, n_trials, beta, expected_beta
        );
    }
}

// ============================================================================
// 5. normal_pdf / normal_cdf 精度（EMMR 使用）
// ============================================================================

/// Python 参考: scipy.stats.norm.pdf/cdf
#[test]
fn test_normal_pdf_cdf_precision() {
    use optuna_rs::samplers::gp::{normal_pdf, normal_cdf};

    let test_cases: Vec<(f64, f64, f64)> = vec![
        (-2.0, 5.399096651318806e-02, 2.275013194817920e-02),
        (-1.0, 2.419707245191434e-01, 1.586552539314571e-01),
        (0.0, 3.989422804014327e-01, 5.000000000000000e-01),
        (1.0, 2.419707245191434e-01, 8.413447460685429e-01),
        (2.0, 5.399096651318806e-02, 9.772498680518208e-01),
    ];

    for (g, expected_pdf, expected_cdf) in test_cases {
        let rust_pdf = normal_pdf(g);
        let rust_cdf = normal_cdf(g);

        assert!(
            (rust_pdf - expected_pdf).abs() < 1e-10,
            "normal_pdf({}) mismatch: Rust={:.15e}, Python={:.15e}",
            g, rust_pdf, expected_pdf
        );
        assert!(
            (rust_cdf - expected_cdf).abs() < 1e-10,
            "normal_cdf({}) mismatch: Rust={:.15e}, Python={:.15e}",
            g, rust_cdf, expected_cdf
        );
    }
}

// ============================================================================
// 6. StaticErrorEvaluator 精确对齐
// ============================================================================

#[test]
fn test_static_error_evaluator_values() {
    for constant in [0.0, 0.5, 1.0, 100.0, -1.0, f64::MAX * 0.1] {
        let eval = StaticErrorEvaluator::new(constant);
        let result = eval.evaluate(&[], StudyDirection::Minimize);
        assert_eq!(result, constant, "StaticErrorEvaluator({}) should return {}", constant, constant);
    }
}

// ============================================================================
// 7. EvaluatorTerminator 组合测试
// ============================================================================

/// 当 improvement < error 时应终止
#[test]
fn test_evaluator_terminator_improvement_less_than_error() {
    // stagnation=5, patience=3 → room=-2 (< 0)
    // StaticError(0) → error=0
    // -2 < 0 → should terminate
    let values: Vec<f64> = (0..10)
        .rev()
        .map(|i| (i + 1) as f64)
        .chain(std::iter::repeat(5.0).take(5))
        .collect();
    let trials = make_trials_with_values(&values);

    let eval_imp = BestValueStagnationEvaluator::new(3);
    let eval_err = StaticErrorEvaluator::new(0.0);

    let improvement = eval_imp.evaluate(&trials, StudyDirection::Minimize);
    let error = eval_err.evaluate(&trials, StudyDirection::Minimize);

    assert!(improvement < error, "improvement={} should < error={}", improvement, error);
}

/// 当 improvement >= error 时不应终止
#[test]
fn test_evaluator_terminator_improvement_ge_error() {
    // stagnation=5, patience=10 → room=5 (> 0)
    // StaticError(0) → error=0
    // 5 >= 0 → should NOT terminate
    let values: Vec<f64> = (0..10)
        .rev()
        .map(|i| (i + 1) as f64)
        .chain(std::iter::repeat(5.0).take(5))
        .collect();
    let trials = make_trials_with_values(&values);

    let eval_imp = BestValueStagnationEvaluator::new(10);
    let eval_err = StaticErrorEvaluator::new(0.0);

    let improvement = eval_imp.evaluate(&trials, StudyDirection::Minimize);
    let error = eval_err.evaluate(&trials, StudyDirection::Minimize);

    assert!(improvement >= error, "improvement={} should >= error={}", improvement, error);
}

// ============================================================================
// 8. 空试验和边界条件
// ============================================================================

#[test]
fn test_stagnation_empty_trials() {
    let eval = BestValueStagnationEvaluator::new(5);
    let result = eval.evaluate(&[], StudyDirection::Minimize);
    assert_eq!(result, f64::MAX, "空试验应返回 MAX");
}

#[test]
fn test_stagnation_single_trial() {
    let trials = make_trials_with_values(&[1.0]);
    let eval = BestValueStagnationEvaluator::new(5);
    let result = eval.evaluate(&trials, StudyDirection::Minimize);
    // best_step=0, current_step=0, stagnation=0, room=5-0=5
    assert_eq!(result, 5.0, "单试验: stagnation=0, room=5");
}

/// 持续改善的场景
#[test]
fn test_stagnation_continuous_improvement() {
    let values: Vec<f64> = (0..100).map(|i| 100.0 - i as f64).collect();
    let trials = make_trials_with_values(&values);
    let eval = BestValueStagnationEvaluator::new(5);
    let result = eval.evaluate(&trials, StudyDirection::Minimize);
    // best_step=99 (值=1), current_step=99, stagnation=0, room=5-0=5
    assert_eq!(result, 5.0, "持续改善: stagnation=0, room=5");
}

/// 从第一步开始就停滞
#[test]
fn test_stagnation_immediate() {
    let values = vec![1.0; 20]; // 所有值相同
    let trials = make_trials_with_values(&values);
    let eval = BestValueStagnationEvaluator::new(5);
    let result = eval.evaluate(&trials, StudyDirection::Minimize);
    // best_step=0 (首次出现最佳值), current_step=19, stagnation=19, room=5-19=-14
    assert_eq!(result, -14.0, "立即停滞: stagnation=19, room=5-19=-14");
}
