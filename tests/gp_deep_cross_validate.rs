/// GP 深度交叉验证测试
/// 使用 Python golden values 逐函数验证 Rust GP 实现的数值精度。
///
/// 对应 Python 基准: tests/golden_gp_deep.py → tests/gp_deep_golden_values.json
use std::fs;

use serde_json::Value;

use optuna_rs::distributions::*;
use optuna_rs::samplers::gp::{
    cholesky, default_log_prior, erfcx, log_ei, log_ndtr, matern52, normalize_param,
    solve_lower, solve_upper, unnormalize_param, GPRegressor,
};

fn load_golden() -> Value {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/gp_deep_golden_values.json");
    let data = fs::read_to_string(path).expect("Cannot read gp_deep_golden_values.json");
    serde_json::from_str(&data).expect("Invalid JSON")
}

fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
    if a.is_infinite() && b.is_infinite() && a.signum() == b.signum() {
        return;
    }
    assert!(
        (a - b).abs() < tol,
        "{}: {} vs {} (diff={})",
        msg, a, b, (a - b).abs()
    );
}

// ============================================================
// 1. Matern 5/2 核函数精确对齐
// ============================================================

#[test]
fn test_matern52_deep_golden() {
    let golden = load_golden();
    let cases = golden["matern52"].as_object().unwrap();

    for (d2_str, expected_val) in cases {
        let d2: f64 = d2_str.parse().unwrap();
        let expected = expected_val.as_f64().unwrap();
        let actual = matern52(d2);
        assert_close(actual, expected, 1e-12, &format!("matern52({d2})"));
    }
}

// ============================================================
// 2. 归一化/反归一化精确对齐
// ============================================================

#[test]
fn test_normalize_float_linear_golden() {
    let golden = load_golden();
    let cases = golden["normalize"].as_object().unwrap();

    for (key, val) in cases {
        if !key.starts_with("float_linear_") { continue; }
        let expected_norm = val["normalized"].as_f64().unwrap();
        let expected_unnorm = val["unnormalized"].as_f64().unwrap();

        // 解析 key: float_linear_{val}_{low}_{high}
        let parts: Vec<&str> = key.strip_prefix("float_linear_").unwrap().splitn(3, '_').collect();
        let v: f64 = parts[0].parse().unwrap();
        let low: f64 = parts[1].parse().unwrap();
        let high: f64 = parts[2].parse().unwrap();

        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(low, high, false, None).unwrap()
        );

        let actual_norm = normalize_param(v, &dist);
        assert_close(actual_norm, expected_norm, 1e-12,
            &format!("normalize({key})"));

        let actual_unnorm = unnormalize_param(actual_norm, &dist);
        assert_close(actual_unnorm, expected_unnorm, 1e-10,
            &format!("unnormalize({key})"));
    }
}

#[test]
fn test_normalize_float_log_golden() {
    let golden = load_golden();
    let cases = golden["normalize"].as_object().unwrap();

    for (key, val) in cases {
        if !key.starts_with("float_log_") { continue; }
        let expected_norm = val["normalized"].as_f64().unwrap();

        let parts: Vec<&str> = key.strip_prefix("float_log_").unwrap().splitn(3, '_').collect();
        let v: f64 = parts[0].parse().unwrap();
        let low: f64 = parts[1].parse().unwrap();
        let high: f64 = parts[2].parse().unwrap();

        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(low, high, true, None).unwrap()
        );

        let actual_norm = normalize_param(v, &dist);
        assert_close(actual_norm, expected_norm, 1e-10,
            &format!("normalize_log({key})"));

        // 往返验证
        let roundtrip = unnormalize_param(actual_norm, &dist);
        assert_close(roundtrip, v, 1e-8,
            &format!("roundtrip({key})"));
    }
}

#[test]
fn test_normalize_float_step_golden() {
    let golden = load_golden();
    let cases = golden["normalize"].as_object().unwrap();

    for (key, val) in cases {
        if !key.starts_with("float_step_") { continue; }
        let expected_norm = val["normalized"].as_f64().unwrap();

        let parts: Vec<&str> = key.strip_prefix("float_step_").unwrap().splitn(4, '_').collect();
        let v: f64 = parts[0].parse().unwrap();
        let low: f64 = parts[1].parse().unwrap();
        let high: f64 = parts[2].parse().unwrap();
        let step: f64 = parts[3].parse().unwrap();

        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(low, high, false, Some(step)).unwrap()
        );

        let actual_norm = normalize_param(v, &dist);
        assert_close(actual_norm, expected_norm, 1e-12,
            &format!("normalize_step({key})"));
    }
}

#[test]
fn test_normalize_int_golden() {
    let golden = load_golden();
    let cases = golden["normalize"].as_object().unwrap();

    for (key, val) in cases {
        if !key.starts_with("int_") { continue; }
        let expected_norm = val["normalized"].as_f64().unwrap();
        let expected_unnorm = val["unnormalized"].as_f64().unwrap();

        let parts: Vec<&str> = key.strip_prefix("int_").unwrap().splitn(4, '_').collect();
        let v: f64 = parts[0].parse::<i64>().unwrap() as f64;
        let low: i64 = parts[1].parse().unwrap();
        let high: i64 = parts[2].parse().unwrap();
        let step: i64 = parts[3].parse().unwrap();

        let dist = Distribution::IntDistribution(
            IntDistribution::new(low, high, false, step).unwrap()
        );

        let actual_norm = normalize_param(v, &dist);
        assert_close(actual_norm, expected_norm, 1e-12,
            &format!("normalize_int({key})"));

        let actual_unnorm = unnormalize_param(actual_norm, &dist);
        assert_close(actual_unnorm, expected_unnorm, 1e-8,
            &format!("unnormalize_int({key})"));
    }
}

// ============================================================
// 3. erfcx 精确对齐
// ============================================================

#[test]
fn test_erfcx_deep_golden() {
    let golden = load_golden();
    let cases = golden["erfcx"].as_object().unwrap();

    for (x_str, expected_val) in cases {
        let x: f64 = x_str.parse().unwrap();
        let expected = expected_val.as_f64().unwrap();
        let actual = erfcx(x);

        // erfcx 对极大值可能有大绝对值，使用相对误差
        if expected.abs() > 1.0 {
            let rel_err = ((actual - expected) / expected).abs();
            assert!(rel_err < 1e-6,
                "erfcx({x}): {actual} vs {expected}, rel_err={rel_err}");
        } else {
            assert_close(actual, expected, 1e-8, &format!("erfcx({x})"));
        }
    }
}

// ============================================================
// 4. erfinv 精度验证 (通过 erf 往返)
// ============================================================

#[test]
fn test_erfinv_roundtrip() {
    let golden = load_golden();
    let cases = golden["erfinv"].as_object().unwrap();

    for (x_str, expected_val) in cases {
        let x: f64 = x_str.parse().unwrap();
        let expected = expected_val.as_f64().unwrap();

        // 验证: erf(erfinv(x)) ≈ x
        let erf_val = libm::erf(expected);
        assert_close(erf_val, x, 1e-12, &format!("erf(erfinv({x}))"));
    }
}

// ============================================================
// 5. log_ndtr 精确对齐
// ============================================================

#[test]
fn test_log_ndtr_deep_golden() {
    let golden = load_golden();
    let cases = golden["log_ndtr"].as_object().unwrap();

    for (z_str, expected_val) in cases {
        let z: f64 = z_str.parse().unwrap();
        let expected = expected_val.as_f64().unwrap();
        let actual = log_ndtr(z);

        // 对极端尾部使用更宽松的容差
        let tol = if z < -30.0 { 1e-3 } else if z < -10.0 { 1e-6 } else { 1e-8 };
        assert_close(actual, expected, tol, &format!("log_ndtr({z})"));
    }
}

// ============================================================
// 6. standard_logei 精确对齐
// ============================================================

#[test]
fn test_standard_logei_deep_golden() {
    let golden = load_golden();
    let cases = golden["standard_logei"].as_object().unwrap();

    for (z_str, expected_val) in cases {
        let z: f64 = z_str.parse().unwrap();
        let expected = expected_val.as_f64().unwrap();

        // log_ei(z+0, 1.0, 0.0) = standard_logei(z) + log(1) = standard_logei(z)
        let actual = log_ei(z, 1.0, 0.0);

        if expected <= -1e307 {
            assert!(actual.is_infinite() && actual < 0.0 || actual < -100.0,
                "standard_logei({z}): expected very negative, got {actual}");
        } else {
            let tol = if z < -20.0 { 1e-3 } else { 1e-6 };
            assert_close(actual, expected, tol, &format!("standard_logei({z})"));
        }
    }
}

// ============================================================
// 7. logEI(mean, var, f0) 精确对齐
// ============================================================

#[test]
fn test_logei_full_golden() {
    let golden = load_golden();
    let cases = golden["logei"].as_object().unwrap();

    for (key, expected_val) in cases {
        let parts: Vec<f64> = key.split(',').map(|s| s.parse().unwrap()).collect();
        let (mean, var, f0) = (parts[0], parts[1], parts[2]);
        let expected = expected_val.as_f64().unwrap();

        // Rust log_ei 内部加 stabilizing_noise，所以用 var 直接传
        let actual = log_ei(mean, var, f0);

        assert_close(actual, expected, 1e-6, &format!("logei({mean},{var},{f0})"));
    }
}

// ============================================================
// 8. default_log_prior 精确对齐
// ============================================================

#[test]
fn test_log_prior_golden() {
    let golden = load_golden();
    let cases = golden["log_prior"].as_object().unwrap();

    for (_key, val) in cases {
        let expected_total = val["total"].as_f64().unwrap();

        // 解析参数（from key format: "[1.0, 1.0]|1.0|1e-06"）
        let key_str = _key.to_string();
        let parts: Vec<&str> = key_str.split('|').collect();

        // 解析 inv_sq_ls
        let ls_str = parts[0].trim_start_matches('[').trim_end_matches(']');
        let inv_sq_ls: Vec<f64> = ls_str.split(", ").map(|s| s.parse().unwrap()).collect();
        let ks: f64 = parts[1].parse().unwrap();
        let nv: f64 = parts[2].parse().unwrap();

        let actual = default_log_prior(&inv_sq_ls, ks, nv);
        assert_close(actual, expected_total, 1e-10,
            &format!("log_prior({_key})"));
    }
}

// ============================================================
// 9. GP LML 精确对齐
// ============================================================

#[test]
fn test_gp_lml_1d_deep_golden() {
    let golden = load_golden();
    let case = &golden["gp_lml"]["1d_3pts"];

    let x_train: Vec<Vec<f64>> = case["X"].as_array().unwrap().iter()
        .map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let y_train: Vec<f64> = case["y"].as_array().unwrap().iter()
        .map(|v| v.as_f64().unwrap()).collect();
    let inv_sq_ls: Vec<f64> = case["inv_sq_ls"].as_array().unwrap().iter()
        .map(|v| v.as_f64().unwrap()).collect();
    let kernel_scale = case["kernel_scale"].as_f64().unwrap();
    let noise_var = case["noise_var"].as_f64().unwrap();
    let expected_lml = case["lml"].as_f64().unwrap();

    let gpr = GPRegressor::new(x_train, y_train, vec![false], inv_sq_ls, kernel_scale, noise_var);
    let actual_lml = gpr.log_marginal_likelihood();

    assert_close(actual_lml, expected_lml, 1e-8, "GP LML 1D");
}

#[test]
fn test_gp_lml_2d_deep_golden() {
    let golden = load_golden();
    let case = &golden["gp_lml"]["2d_4pts"];

    let x_train: Vec<Vec<f64>> = case["X"].as_array().unwrap().iter()
        .map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let y_train: Vec<f64> = case["y"].as_array().unwrap().iter()
        .map(|v| v.as_f64().unwrap()).collect();
    let inv_sq_ls: Vec<f64> = case["inv_sq_ls"].as_array().unwrap().iter()
        .map(|v| v.as_f64().unwrap()).collect();
    let kernel_scale = case["kernel_scale"].as_f64().unwrap();
    let noise_var = case["noise_var"].as_f64().unwrap();
    let expected_lml = case["lml"].as_f64().unwrap();

    let gpr = GPRegressor::new(x_train, y_train, vec![false, false], inv_sq_ls, kernel_scale, noise_var);
    let actual_lml = gpr.log_marginal_likelihood();

    assert_close(actual_lml, expected_lml, 1e-8, "GP LML 2D");
}

// ============================================================
// 10. GP 后验预测精确对齐
// ============================================================

#[test]
fn test_gp_posterior_deep_golden() {
    let golden = load_golden();
    let lml_case = &golden["gp_lml"]["2d_4pts"];
    let post_case = &golden["gp_posterior"]["2d_at_025_075"];

    let x_train: Vec<Vec<f64>> = lml_case["X"].as_array().unwrap().iter()
        .map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let y_train: Vec<f64> = lml_case["y"].as_array().unwrap().iter()
        .map(|v| v.as_f64().unwrap()).collect();
    let inv_sq_ls: Vec<f64> = lml_case["inv_sq_ls"].as_array().unwrap().iter()
        .map(|v| v.as_f64().unwrap()).collect();
    let kernel_scale = lml_case["kernel_scale"].as_f64().unwrap();
    let noise_var = lml_case["noise_var"].as_f64().unwrap();

    let gpr = GPRegressor::new(x_train, y_train, vec![false, false], inv_sq_ls, kernel_scale, noise_var);

    let x_pred: Vec<f64> = post_case["x"].as_array().unwrap().iter()
        .map(|v| v.as_f64().unwrap()).collect();
    let expected_mean = post_case["mean"].as_f64().unwrap();
    let expected_var = post_case["var"].as_f64().unwrap();

    let (actual_mean, actual_var) = gpr.posterior(&x_pred);

    assert_close(actual_mean, expected_mean, 1e-8, "GP posterior mean");
    assert_close(actual_var, expected_var, 1e-6, "GP posterior var");
}

// ============================================================
// 11. Cholesky + solve 精确对齐
// ============================================================

#[test]
fn test_cholesky_solve_deep() {
    // 3×3 正定矩阵
    let a = vec![
        vec![4.0, 2.0, 1.0],
        vec![2.0, 5.0, 3.0],
        vec![1.0, 3.0, 6.0],
    ];
    let b = vec![1.0, 2.0, 3.0];

    let l = cholesky(&a).expect("cholesky failed");

    // 验证 L * L^T ≈ A
    for i in 0..3 {
        for j in 0..3 {
            let mut sum = 0.0;
            for k in 0..3 {
                sum += l[i][k] * l[j][k];
            }
            assert_close(sum, a[i][j], 1e-12, &format!("L*L^T[{i}][{j}]"));
        }
    }

    // 验证 solve_lower + solve_upper
    let u = solve_lower(&l, &b);
    let x = solve_upper(&l, &u);

    // 验证 A * x ≈ b
    for i in 0..3 {
        let mut ax_i = 0.0;
        for j in 0..3 {
            ax_i += a[i][j] * x[j];
        }
        assert_close(ax_i, b[i], 1e-10, &format!("Ax[{i}]"));
    }
}

// ============================================================
// 12. Constant Liar 效果验证 (通过后验间接测试)
// ============================================================

#[test]
fn test_gp_posterior_variance_decreases_with_more_data() {
    // 验证: 增加训练数据后，后验方差应整体下降
    let x_few = vec![vec![0.0], vec![1.0]];
    let y_few = vec![0.0, 1.0];

    let x_more = vec![vec![0.0], vec![0.5], vec![1.0]];
    let y_more = vec![0.0, 0.5, 1.0];

    let gpr_few = GPRegressor::new(x_few, y_few, vec![false], vec![1.0], 1.0, 0.01);
    let gpr_more = GPRegressor::new(x_more, y_more, vec![false], vec![1.0], 1.0, 0.01);

    let (_, var_few) = gpr_few.posterior(&[0.5]);
    let (_, var_more) = gpr_more.posterior(&[0.5]);

    assert!(var_more < var_few,
        "more data should reduce variance at interpolation point: {var_more} < {var_few}");
}

#[test]
fn test_gp_posterior_mean_interpolates() {
    // 验证: 后验 mean 应接近训练点的 y 值（低噪声时）
    let x_train = vec![vec![0.0], vec![0.5], vec![1.0]];
    let y_train = vec![0.0, 1.0, 0.0];

    let gpr = GPRegressor::new(x_train, y_train.clone(), vec![false], vec![1.0], 1.0, 0.001);

    for (x, y) in [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)] {
        let (mean, _var) = gpr.posterior(&[x]);
        assert_close(mean, y, 0.1, &format!("posterior mean at training point x={x}"));
    }
}

// ============================================================
// 13. 归一化往返精度验证
// ============================================================

#[test]
fn test_normalize_roundtrip_all_types() {
    // Float linear
    let dist_fl = Distribution::FloatDistribution(
        FloatDistribution::new(-5.0, 15.0, false, None).unwrap()
    );
    for v in &[-5.0, 0.0, 7.5, 15.0] {
        let norm = normalize_param(*v, &dist_fl);
        let back = unnormalize_param(norm, &dist_fl);
        assert_close(back, *v, 1e-10, &format!("roundtrip float linear {v}"));
    }

    // Float log
    let dist_log = Distribution::FloatDistribution(
        FloatDistribution::new(0.01, 100.0, true, None).unwrap()
    );
    for v in &[0.01, 1.0, 10.0, 100.0] {
        let norm = normalize_param(*v, &dist_log);
        let back = unnormalize_param(norm, &dist_log);
        assert_close(back, *v, 1e-6, &format!("roundtrip float log {v}"));
    }

    // Int
    let dist_int = Distribution::IntDistribution(
        IntDistribution::new(0, 10, false, 1).unwrap()
    );
    for v in &[0.0, 5.0, 10.0] {
        let norm = normalize_param(*v, &dist_int);
        let back = unnormalize_param(norm, &dist_int);
        assert_close(back, *v, 1e-8, &format!("roundtrip int {v}"));
    }
}
