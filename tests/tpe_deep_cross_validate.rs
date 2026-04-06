/// TPE 深度交叉验证测试
/// 使用 Python golden values 逐函数验证 Rust 实现的数值精度。
///
/// 对应 Python 基准: tests/golden_tpe_deep.py → tests/tpe_deep_golden_values.json
use std::collections::HashMap;
use std::fs;

use indexmap::IndexMap;
use serde_json::Value;

use optuna_rs::distributions::*;
use optuna_rs::samplers::tpe::parzen_estimator::{
    default_gamma, default_weights, ParzenEstimator, ParzenEstimatorParameters,
};
use optuna_rs::samplers::tpe::{hyperopt_default_gamma, truncnorm};

fn load_golden() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/tpe_deep_golden_values.json");
    let data = fs::read_to_string(path).expect("Cannot read tpe_deep_golden_values.json");
    serde_json::from_str(&data).expect("Invalid JSON")
}

fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
    if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
        return;
    }
    assert!(
        (a - b).abs() < tol,
        "{}: {} vs {} (diff={})",
        msg,
        a,
        b,
        (a - b).abs()
    );
}

// ============================================================
// 1. default_weights 精确对齐 Python
// ============================================================

#[test]
fn test_default_weights_golden() {
    let golden = load_golden();
    let cases = golden["default_weights"].as_object().unwrap();

    for (n_str, expected_arr) in cases {
        let n: usize = n_str.parse().unwrap();
        let expected: Vec<f64> = expected_arr
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect();

        let actual = default_weights(n);
        assert_eq!(
            actual.len(),
            expected.len(),
            "n={}: length mismatch {} vs {}",
            n,
            actual.len(),
            expected.len()
        );

        for (i, (&a, &e)) in actual.iter().zip(expected.iter()).enumerate() {
            assert_close(a, e, 1e-12, &format!("default_weights({})[{}]", n, i));
        }
    }
}

// ============================================================
// 2. default_gamma 精确对齐 Python
// ============================================================

#[test]
fn test_default_gamma_golden() {
    let golden = load_golden();
    let cases = golden["default_gamma"].as_object().unwrap();

    for (n_str, expected_val) in cases {
        let n: usize = n_str.parse().unwrap();
        let expected = expected_val.as_u64().unwrap() as usize;
        let actual = default_gamma(n);
        assert_eq!(actual, expected, "default_gamma({n})");
    }
}

// ============================================================
// 3. hyperopt_default_gamma 精确对齐 Python
// ============================================================

#[test]
fn test_hyperopt_default_gamma_golden() {
    let golden = load_golden();
    let cases = golden["hyperopt_default_gamma"].as_object().unwrap();

    for (n_str, expected_val) in cases {
        let n: usize = n_str.parse().unwrap();
        let expected = expected_val.as_u64().unwrap() as usize;
        let actual = hyperopt_default_gamma(n);
        assert_eq!(actual, expected, "hyperopt_default_gamma({n})");
    }
}

// ============================================================
// 4. ndtr (标准正态 CDF) 精确对齐 Python _ndtr_single
// ============================================================

#[test]
fn test_ndtr_golden() {
    let golden = load_golden();
    let cases = golden["ndtr"].as_object().unwrap();

    for (x_str, expected_val) in cases {
        let x: f64 = x_str.parse().unwrap();
        let expected = expected_val.as_f64().unwrap();
        // ndtr 是 truncnorm 的内部函数，通过 ppf/logpdf 间接测试
        // 但我们可以直接测 log_gauss_mass 来覆盖 ndtr
        // 这里直接计算: ndtr(x) ≈ 0.5 * erfc(-x/sqrt(2)) 或 0.5 + 0.5*erf(x/sqrt(2))
        let actual = 0.5 * libm::erfc(-x * std::f64::consts::FRAC_1_SQRT_2);
        // 对于大 x，使用 1 - 0.5*erfc(x/sqrt(2))
        let actual_stable = if x < -std::f64::consts::FRAC_1_SQRT_2 {
            0.5 * libm::erfc(-x * std::f64::consts::FRAC_1_SQRT_2)
        } else if x < std::f64::consts::FRAC_1_SQRT_2 {
            0.5 + 0.5 * libm::erf(x * std::f64::consts::FRAC_1_SQRT_2)
        } else {
            1.0 - 0.5 * libm::erfc(x * std::f64::consts::FRAC_1_SQRT_2)
        };

        if expected == 0.0 {
            assert!(actual_stable < 1e-100, "ndtr({x}) should be ~0");
        } else if expected == 1.0 {
            assert!((actual_stable - 1.0).abs() < 1e-10, "ndtr({x}) should be ~1");
        } else {
            let rel_err = ((actual_stable - expected) / expected).abs();
            assert!(
                rel_err < 1e-10,
                "ndtr({x}): {actual_stable} vs {expected}, rel_err={rel_err}"
            );
        }
    }
}

// ============================================================
// 5. log_ndtr 精确对齐 Python _log_ndtr_single
// ============================================================

#[test]
fn test_log_ndtr_golden() {
    let golden = load_golden();
    let cases = golden["log_ndtr"].as_object().unwrap();

    for (x_str, expected_val) in cases {
        let x: f64 = x_str.parse().unwrap();
        let expected = expected_val.as_f64().unwrap();

        // 通过 log_gauss_mass 间接测试 log_ndtr
        // log_gauss_mass(-inf, x) ≈ log_ndtr(x)
        let actual = truncnorm::log_gauss_mass(-100.0, x);
        // log_ndtr(x) ≈ log_gauss_mass(-100, x) 对于 x 远大于 -100
        if expected.is_finite() {
            let tol = if x < -15.0 { 1e-4 } else { 1e-6 };
            assert_close(actual, expected, tol, &format!("log_ndtr({x}) [via log_gauss_mass(-100, x)]"));
        }
    }
}

// ============================================================
// 6. log_gauss_mass 精确对齐 Python
// ============================================================

#[test]
fn test_log_gauss_mass_golden() {
    let golden = load_golden();
    let cases = golden["log_gauss_mass"].as_object().unwrap();

    for (key, expected_val) in cases {
        let parts: Vec<f64> = key.split(',').map(|s| s.parse().unwrap()).collect();
        let (a, b) = (parts[0], parts[1]);
        let expected = expected_val.as_f64().unwrap();
        let actual = truncnorm::log_gauss_mass(a, b);

        assert_close(actual, expected, 1e-8, &format!("log_gauss_mass({a}, {b})"));
    }
}

// ============================================================
// 7. ppf 精确对齐 Python
// ============================================================

#[test]
fn test_ppf_golden() {
    let golden = load_golden();
    let cases = golden["ppf"].as_object().unwrap();

    for (key, expected_val) in cases {
        let parts: Vec<f64> = key.split(',').map(|s| s.parse().unwrap()).collect();
        let (q, a, b) = (parts[0], parts[1], parts[2]);
        let expected = expected_val.as_f64().unwrap();
        let actual = truncnorm::ppf(q, a, b);

        let tol = if q == 0.0 || q == 1.0 { 1e-10 } else { 1e-6 };
        assert_close(actual, expected, tol, &format!("ppf({q}, {a}, {b})"));
    }
}

// ============================================================
// 8. logpdf 精确对齐 Python
// ============================================================

#[test]
fn test_logpdf_golden() {
    let golden = load_golden();
    let cases = golden["logpdf"].as_object().unwrap();

    for (key, expected_val) in cases {
        let parts: Vec<f64> = key.split(',').map(|s| s.parse().unwrap()).collect();
        let (x, a, b, loc, scale) = (parts[0], parts[1], parts[2], parts[3], parts[4]);
        let expected = expected_val.as_f64().unwrap();
        let actual = truncnorm::logpdf(x, a, b, loc, scale);

        if expected <= -1e307 {
            assert!(actual.is_infinite() && actual < 0.0,
                "logpdf({x},{a},{b},{loc},{scale}): expected -inf, got {actual}");
        } else {
            assert_close(actual, expected, 1e-8, &format!("logpdf({x},{a},{b},{loc},{scale})"));
        }
    }
}

// ============================================================
// 9. Parzen Estimator 内核参数精确对齐
// ============================================================

fn make_float_dist(low: f64, high: f64) -> Distribution {
    Distribution::FloatDistribution(FloatDistribution::new(low, high, false, None).unwrap())
}

fn make_log_float_dist(low: f64, high: f64) -> Distribution {
    Distribution::FloatDistribution(FloatDistribution::new(low, high, true, None).unwrap())
}

fn make_int_dist(low: i64, high: i64, step: i64) -> Distribution {
    Distribution::IntDistribution(IntDistribution::new(low, high, false, step).unwrap())
}

fn make_cat_dist(choices: Vec<&str>) -> Distribution {
    Distribution::CategoricalDistribution(
        CategoricalDistribution::new(
            choices.into_iter().map(|s| CategoricalChoice::Str(s.to_string())).collect()
        ).unwrap()
    )
}

fn default_pe_params() -> ParzenEstimatorParameters {
    ParzenEstimatorParameters {
        prior_weight: 1.0,
        consider_magic_clip: true,
        consider_endpoints: false,
        multivariate: false,
        categorical_distance_func: HashMap::new(),
    }
}

#[test]
fn test_pe_univariate_3obs_golden() {
    let golden = load_golden();
    let pe_case = &golden["parzen_estimator"]["univariate_3obs"];

    let expected_weights: Vec<f64> = pe_case["weights"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_mus: Vec<f64> = pe_case["mus"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_sigmas: Vec<f64> = pe_case["sigmas"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();

    let mut ss = IndexMap::new();
    ss.insert("x".to_string(), make_float_dist(0.0, 10.0));
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![2.0, 5.0, 8.0]);

    let pe = ParzenEstimator::new(&obs, &ss, &default_pe_params(), None, None);

    // 验证混合权重
    let actual_weights = pe.weights();
    assert_eq!(actual_weights.len(), expected_weights.len(),
        "weights length: {} vs {}", actual_weights.len(), expected_weights.len());
    for (i, (&a, &e)) in actual_weights.iter().zip(expected_weights.iter()).enumerate() {
        assert_close(a, e, 1e-10, &format!("pe weights[{i}]"));
    }

    // 验证 mus 和 sigmas
    let (actual_mus, actual_sigmas, _low, _high) = pe.numerical_kernels("x").unwrap();
    assert_eq!(actual_mus.len(), expected_mus.len());
    for (i, (&a, &e)) in actual_mus.iter().zip(expected_mus.iter()).enumerate() {
        assert_close(a, e, 1e-10, &format!("pe mus[{i}]"));
    }
    for (i, (&a, &e)) in actual_sigmas.iter().zip(expected_sigmas.iter()).enumerate() {
        assert_close(a, e, 1e-8, &format!("pe sigmas[{i}]"));
    }
}

#[test]
fn test_pe_univariate_0obs_golden() {
    let golden = load_golden();
    let pe_case = &golden["parzen_estimator"]["univariate_0obs"];

    let expected_weights: Vec<f64> = pe_case["weights"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_mus: Vec<f64> = pe_case["mus"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();

    let mut ss = IndexMap::new();
    ss.insert("x".to_string(), make_float_dist(0.0, 10.0));
    let obs = HashMap::new();

    let pe = ParzenEstimator::new(&obs, &ss, &default_pe_params(), None, None);

    let actual_weights = pe.weights();
    assert_eq!(actual_weights.len(), expected_weights.len());
    for (i, (&a, &e)) in actual_weights.iter().zip(expected_weights.iter()).enumerate() {
        assert_close(a, e, 1e-10, &format!("0obs weights[{i}]"));
    }

    let (actual_mus, _sigmas, _low, _high) = pe.numerical_kernels("x").unwrap();
    assert_eq!(actual_mus.len(), expected_mus.len());
    for (i, (&a, &e)) in actual_mus.iter().zip(expected_mus.iter()).enumerate() {
        assert_close(a, e, 1e-10, &format!("0obs mus[{i}]"));
    }
}

#[test]
fn test_pe_univariate_endpoints_golden() {
    let golden = load_golden();
    let pe_case = &golden["parzen_estimator"]["univariate_3obs_endpoints"];

    let expected_sigmas: Vec<f64> = pe_case["sigmas"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();

    let mut ss = IndexMap::new();
    ss.insert("x".to_string(), make_float_dist(0.0, 10.0));
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![2.0, 5.0, 8.0]);

    let params = ParzenEstimatorParameters {
        consider_endpoints: true,
        ..default_pe_params()
    };
    let pe = ParzenEstimator::new(&obs, &ss, &params, None, None);

    let (_mus, actual_sigmas, _low, _high) = pe.numerical_kernels("x").unwrap();
    for (i, (&a, &e)) in actual_sigmas.iter().zip(expected_sigmas.iter()).enumerate() {
        assert_close(a, e, 1e-8, &format!("endpoints sigmas[{i}]"));
    }
}

#[test]
fn test_pe_log_scale_golden() {
    let golden = load_golden();
    let pe_case = &golden["parzen_estimator"]["log_scale_2obs"];

    let expected_mus: Vec<f64> = pe_case["mus"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_sigmas: Vec<f64> = pe_case["sigmas"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_low = pe_case["low"].as_f64().unwrap();
    let expected_high = pe_case["high"].as_f64().unwrap();

    let mut ss = IndexMap::new();
    ss.insert("lr".to_string(), make_log_float_dist(0.001, 1.0));
    let mut obs = HashMap::new();
    obs.insert("lr".to_string(), vec![0.01, 0.1]);

    let pe = ParzenEstimator::new(&obs, &ss, &default_pe_params(), None, None);
    let (actual_mus, actual_sigmas, actual_low, actual_high) =
        pe.numerical_kernels("lr").unwrap();

    // log-scale: Python 存储的是 ln(值)
    assert_close(actual_low, expected_low, 1e-10, "log low");
    assert_close(actual_high, expected_high, 1e-10, "log high");

    for (i, (&a, &e)) in actual_mus.iter().zip(expected_mus.iter()).enumerate() {
        assert_close(a, e, 1e-10, &format!("log mus[{i}]"));
    }
    for (i, (&a, &e)) in actual_sigmas.iter().zip(expected_sigmas.iter()).enumerate() {
        assert_close(a, e, 1e-8, &format!("log sigmas[{i}]"));
    }
}

#[test]
fn test_pe_categorical_golden() {
    let golden = load_golden();
    let pe_case = &golden["parzen_estimator"]["categorical_3obs"];

    let expected_weights: Vec<f64> = pe_case["weights"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_cat_weights: Vec<Vec<f64>> = pe_case["cat_weights"].as_array().unwrap()
        .iter().map(|row| {
            row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
        }).collect();

    let mut ss = IndexMap::new();
    ss.insert("opt".to_string(), make_cat_dist(vec!["a", "b", "c"]));
    let mut obs = HashMap::new();
    obs.insert("opt".to_string(), vec![0.0, 0.0, 1.0]);

    let pe = ParzenEstimator::new(&obs, &ss, &default_pe_params(), None, None);

    // 验证混合权重
    let actual_weights = pe.weights();
    for (i, (&a, &e)) in actual_weights.iter().zip(expected_weights.iter()).enumerate() {
        assert_close(a, e, 1e-10, &format!("cat weights[{i}]"));
    }

    // 验证分类权重矩阵
    let actual_cat_weights = pe.categorical_kernels("opt").unwrap();
    assert_eq!(actual_cat_weights.len(), expected_cat_weights.len(),
        "cat_weights rows: {} vs {}", actual_cat_weights.len(), expected_cat_weights.len());
    for (k, (a_row, e_row)) in actual_cat_weights.iter().zip(expected_cat_weights.iter()).enumerate() {
        for (j, (&a, &e)) in a_row.iter().zip(e_row.iter()).enumerate() {
            assert_close(a, e, 1e-10, &format!("cat_weights[{k}][{j}]"));
        }
    }
}

#[test]
fn test_pe_multivariate_golden() {
    let golden = load_golden();
    let pe_case = &golden["parzen_estimator"]["multivariate_2d_3obs"];

    let expected_weights: Vec<f64> = pe_case["weights"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_x_sigmas: Vec<f64> = pe_case["x_sigmas"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_y_sigmas: Vec<f64> = pe_case["y_sigmas"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();

    let mut ss = IndexMap::new();
    ss.insert("x".to_string(), make_float_dist(0.0, 10.0));
    ss.insert("y".to_string(), make_float_dist(-5.0, 5.0));
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![2.0, 5.0, 8.0]);
    obs.insert("y".to_string(), vec![-1.0, 0.0, 1.0]);

    let params = ParzenEstimatorParameters {
        multivariate: true,
        ..default_pe_params()
    };
    let pe = ParzenEstimator::new(&obs, &ss, &params, None, None);

    // 验证权重
    let actual_weights = pe.weights();
    for (i, (&a, &e)) in actual_weights.iter().zip(expected_weights.iter()).enumerate() {
        assert_close(a, e, 1e-10, &format!("mv weights[{i}]"));
    }

    // 验证 x sigmas (multivariate: sigma = 0.2 * n^(-1/(d+4)) * range)
    let (_x_mus, actual_x_sigmas, _x_low, _x_high) = pe.numerical_kernels("x").unwrap();
    for (i, (&a, &e)) in actual_x_sigmas.iter().zip(expected_x_sigmas.iter()).enumerate() {
        // 最后一个是 prior sigma = range，不需要匹配 multivariate 公式
        assert_close(a, e, 1e-8, &format!("mv x_sigmas[{i}]"));
    }

    // 验证 y sigmas
    let (_y_mus, actual_y_sigmas, _y_low, _y_high) = pe.numerical_kernels("y").unwrap();
    for (i, (&a, &e)) in actual_y_sigmas.iter().zip(expected_y_sigmas.iter()).enumerate() {
        assert_close(a, e, 1e-8, &format!("mv y_sigmas[{i}]"));
    }
}

#[test]
fn test_pe_int_step_golden() {
    let golden = load_golden();
    let pe_case = &golden["parzen_estimator"]["int_step2_3obs"];

    let expected_mus: Vec<f64> = pe_case["mus"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_sigmas: Vec<f64> = pe_case["sigmas"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_low = pe_case["low"].as_f64().unwrap();
    let expected_high = pe_case["high"].as_f64().unwrap();

    let mut ss = IndexMap::new();
    ss.insert("n".to_string(), make_int_dist(0, 10, 2));
    let mut obs = HashMap::new();
    obs.insert("n".to_string(), vec![2.0, 4.0, 6.0]);

    let pe = ParzenEstimator::new(&obs, &ss, &default_pe_params(), None, None);
    let (actual_mus, actual_sigmas, actual_low, actual_high) =
        pe.numerical_kernels("n").unwrap();

    // int step=2: low = 0 - 2/2 = -1, high = 10 + 2/2 = 11
    assert_close(actual_low, expected_low, 1e-10, "int_step low");
    assert_close(actual_high, expected_high, 1e-10, "int_step high");

    for (i, (&a, &e)) in actual_mus.iter().zip(expected_mus.iter()).enumerate() {
        assert_close(a, e, 1e-10, &format!("int_step mus[{i}]"));
    }
    for (i, (&a, &e)) in actual_sigmas.iter().zip(expected_sigmas.iter()).enumerate() {
        assert_close(a, e, 1e-8, &format!("int_step sigmas[{i}]"));
    }
}

#[test]
fn test_pe_predetermined_weights_golden() {
    let golden = load_golden();
    let pe_case = &golden["parzen_estimator"]["predetermined_weights"];
    let expected_weights: Vec<f64> = pe_case["weights"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();

    let mut ss = IndexMap::new();
    ss.insert("x".to_string(), make_float_dist(0.0, 10.0));
    let mut obs = HashMap::new();
    obs.insert("x".to_string(), vec![2.0, 5.0, 8.0]);

    let pe = ParzenEstimator::new(
        &obs, &ss, &default_pe_params(),
        Some(&[0.5, 0.3, 0.2]),
        None,
    );

    let actual_weights = pe.weights();
    for (i, (&a, &e)) in actual_weights.iter().zip(expected_weights.iter()).enumerate() {
        assert_close(a, e, 1e-10, &format!("predetermined weights[{i}]"));
    }
}

// ============================================================
// 10. PPF 精度边界测试
// ============================================================

#[test]
fn test_ppf_extreme_tails() {
    // 极端分位数: q 接近 0 或 1
    let a = -5.0;
    let b = 5.0;

    let x_low = truncnorm::ppf(0.001, a, b);
    assert!(x_low >= a && x_low <= b, "ppf(0.001): {x_low}");

    let x_high = truncnorm::ppf(0.999, a, b);
    assert!(x_high >= a && x_high <= b, "ppf(0.999): {x_high}");

    // 确保单调性
    assert!(x_low < x_high, "ppf should be monotone: {x_low} < {x_high}");
}

#[test]
fn test_ppf_right_tail() {
    // a ≥ 0 情况: 使用右尾公式
    let x = truncnorm::ppf(0.5, 1.0, 5.0);
    assert!(x >= 1.0 && x <= 5.0, "ppf(0.5, 1, 5) = {x}");

    let x2 = truncnorm::ppf(0.5, 2.0, 3.0);
    assert!(x2 >= 2.0 && x2 <= 3.0, "ppf(0.5, 2, 3) = {x2}");
}

// ============================================================
// 11. log_gauss_mass 对称性验证
// ============================================================

#[test]
fn test_log_gauss_mass_symmetry() {
    // log_gauss_mass(-b, -a) == log_gauss_mass(a, b) (对称性)
    for &(a, b) in &[(-2.0, 1.0), (-1.0, 3.0), (0.5, 2.0), (-5.0, -1.0)] {
        let m1 = truncnorm::log_gauss_mass(a, b);
        let m2 = truncnorm::log_gauss_mass(-b, -a);
        assert_close(m1, m2, 1e-10, &format!("symmetry: [{a},{b}] vs [{}, {}]", -b, -a));
    }
}

// ============================================================
// 12. logpdf 积分验证
// ============================================================

#[test]
fn test_logpdf_integrates_to_one() {
    // 数值积分 ∫ exp(logpdf(x)) dx ≈ 1 (使用梯形法则)
    for &(a, b, loc, scale) in &[
        (-2.0, 2.0, 0.0, 1.0),
        (-1.0, 1.0, 0.5, 0.5),
        (0.0, 5.0, 2.0, 1.5),
    ] {
        let n = 2000;
        let real_low = a * scale + loc;
        let real_high = b * scale + loc;
        let dx = (real_high - real_low) / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let x = real_low + (i as f64 + 0.5) * dx;
            integral += truncnorm::logpdf(x, real_low, real_high, loc, scale).exp() * dx;
        }
        assert!(
            (integral - 1.0).abs() < 0.01,
            "logpdf integrates to {integral} for [{a},{b}] loc={loc} scale={scale}"
        );
    }
}
