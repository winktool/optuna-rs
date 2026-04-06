//! GP 归一化/反归一化交叉验证测试
//!
//! 使用 Python golden_gp_normalize.py 金标准值验证 GP search space 归一化精度。
//! 特别关注 IntDistribution 的银行家舍入对齐。

use optuna_rs::distributions::{Distribution, FloatDistribution, IntDistribution};
use optuna_rs::samplers::gp::{normalize_param, unnormalize_param};

const TOL: f64 = 1e-12;

fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
    let diff = (a - b).abs();
    assert!(
        diff < tol,
        "{}: expected {:.15e}, got {:.15e}, diff={:.2e}",
        msg, b, a, diff
    );
}

fn float_dist(low: f64, high: f64) -> Distribution {
    Distribution::FloatDistribution(FloatDistribution { low, high, log: false, step: None })
}

fn float_dist_step(low: f64, high: f64, step: f64) -> Distribution {
    Distribution::FloatDistribution(FloatDistribution { low, high, log: false, step: Some(step) })
}

fn float_dist_log(low: f64, high: f64) -> Distribution {
    Distribution::FloatDistribution(FloatDistribution { low, high, log: true, step: None })
}

fn int_dist(low: i64, high: i64) -> Distribution {
    Distribution::IntDistribution(IntDistribution { low, high, log: false, step: 1 })
}

// ========== normalize_param 测试 ==========

/// Python 金标准: Float[0,10] 归一化
#[test]
fn test_gp_normalize_float_python() {
    let d = float_dist(0.0, 10.0);
    assert_close(normalize_param(0.0, &d), 0.0, TOL, "v=0");
    assert_close(normalize_param(5.0, &d), 0.5, TOL, "v=5");
    assert_close(normalize_param(10.0, &d), 1.0, TOL, "v=10");
    assert_close(normalize_param(3.7, &d), 0.37, TOL, "v=3.7");
}

/// Python 金标准: Float[0,10,step=2] 归一化
#[test]
fn test_gp_normalize_float_step_python() {
    let d = float_dist_step(0.0, 10.0, 2.0);
    let expected = [
        (0.0, 8.333333333333333e-02),
        (2.0, 2.500000000000000e-01),
        (4.0, 4.166666666666667e-01),
        (6.0, 5.833333333333334e-01),
        (8.0, 7.500000000000000e-01),
        (10.0, 9.166666666666666e-01),
    ];
    for (v, exp) in expected {
        assert_close(normalize_param(v, &d), exp, TOL, &format!("v={}", v));
    }
}

/// Python 金标准: Float log[1e-3, 1.0] 归一化
#[test]
fn test_gp_normalize_float_log_python() {
    let d = float_dist_log(1e-3, 1.0);
    assert_close(normalize_param(1e-3, &d), 0.0, TOL, "v=1e-3");
    assert_close(normalize_param(0.01, &d), 1.0 / 3.0, 1e-10, "v=0.01");
    assert_close(normalize_param(0.1, &d), 2.0 / 3.0, 1e-10, "v=0.1");
    assert_close(normalize_param(1.0, &d), 1.0, TOL, "v=1.0");
}

/// Python 金标准: Int[1,10] 归一化 (step=1 → bounds ±0.5)
#[test]
fn test_gp_normalize_int_python() {
    let d = int_dist(1, 10);
    assert_close(normalize_param(1.0, &d), 0.05, TOL, "v=1");
    assert_close(normalize_param(5.0, &d), 0.45, TOL, "v=5");
    assert_close(normalize_param(10.0, &d), 0.95, TOL, "v=10");
}

// ========== unnormalize_param 测试 ==========

/// Python 金标准: Float[0,10] 反归一化
#[test]
fn test_gp_unnormalize_float_python() {
    let d = float_dist(0.0, 10.0);
    assert_close(unnormalize_param(0.0, &d), 0.0, TOL, "n=0");
    assert_close(unnormalize_param(0.5, &d), 5.0, TOL, "n=0.5");
    assert_close(unnormalize_param(1.0, &d), 10.0, TOL, "n=1");
    assert_close(unnormalize_param(0.37, &d), 3.7, TOL, "n=0.37");
}

/// Python 金标准: Int[1,10] 反归一化 + 银行家舍入 ← 关键对齐测试
#[test]
fn test_gp_unnormalize_int_banker_rounding_python() {
    let d = int_dist(1, 10);

    // n=0.0: raw=0.5 → round=0 → clamp=1
    assert_close(unnormalize_param(0.0, &d), 1.0, TOL, "n=0.0→clamp=1");

    // n=0.25: raw=3.0 → round=3
    assert_close(unnormalize_param(0.25, &d), 3.0, TOL, "n=0.25→3");

    // n=0.5: raw=5.5 → Python round(5.5)=6 (even)
    assert_close(unnormalize_param(0.5, &d), 6.0, TOL, "n=0.5→6 (bank round)");

    // n=0.75: raw=8.0 → round=8
    assert_close(unnormalize_param(0.75, &d), 8.0, TOL, "n=0.75→8");

    // n=1.0: raw=10.5 → round=10 → clamp=10
    assert_close(unnormalize_param(1.0, &d), 10.0, TOL, "n=1.0→clamp=10");
}

/// Python 金标准: Int[1,10] 银行家舍入边界 (4.5→4, 2.5→2, 6.5→6, 8.5→8)
/// 这些是 .round() 与 bank round 结果不同的关键边界！
#[test]
fn test_gp_unnormalize_int_banker_critical_boundaries_python() {
    let d = int_dist(1, 10);

    // n=0.2: raw=2.5 → Python round(2.5)=2 (NOT 3)
    assert_close(unnormalize_param(0.2, &d), 2.0, TOL, "n=0.2→2 (bank round 2.5→2)");

    // n=0.4: raw=4.5 → Python round(4.5)=4 (NOT 5)
    assert_close(unnormalize_param(0.4, &d), 4.0, TOL, "n=0.4→4 (bank round 4.5→4)");

    // n=0.6: raw=6.5 → Python round(6.5)=6 (NOT 7)
    assert_close(unnormalize_param(0.6, &d), 6.0, TOL, "n=0.6→6 (bank round 6.5→6)");

    // n=0.8: raw=8.5 → Python round(8.5)=8 (NOT 9)
    assert_close(unnormalize_param(0.8, &d), 8.0, TOL, "n=0.8→8 (bank round 8.5→8)");
}

// ========== 往返测试 ==========

/// normalize → unnormalize 应恢复原值 (Float)
#[test]
fn test_gp_roundtrip_float_python() {
    let d = float_dist(0.0, 10.0);
    for v in [0.0, 2.5, 5.0, 7.5, 10.0] {
        let n = normalize_param(v, &d);
        let u = unnormalize_param(n, &d);
        assert_close(u, v, TOL, &format!("roundtrip v={}", v));
    }
}

/// normalize → unnormalize 应恢复原值 (Float log)
#[test]
fn test_gp_roundtrip_float_log_python() {
    let d = float_dist_log(1e-3, 1.0);
    for v in [1e-3, 0.01, 0.1, 0.5, 1.0] {
        let n = normalize_param(v, &d);
        let u = unnormalize_param(n, &d);
        assert_close(u, v, 1e-10, &format!("roundtrip v={}", v));
    }
}

/// normalize → unnormalize 应恢复原值 (Int)
#[test]
fn test_gp_roundtrip_int_python() {
    let d = int_dist(1, 10);
    for v in 1..=10 {
        let n = normalize_param(v as f64, &d);
        let u = unnormalize_param(n, &d);
        assert_close(u, v as f64, TOL, &format!("roundtrip v={}", v));
    }
}

/// GP 先验计算验证
#[test]
fn test_gp_log_prior_python() {
    use optuna_rs::samplers::gp::default_log_prior;

    // inv_sq_ls = [1.0], kernel_scale = 1.0, noise_var = 0.01
    let lp = default_log_prior(&[1.0], 1.0, 0.01);
    // ls_prior = -(0.1/1 + 0.1*1) = -0.2
    // ks_prior = ln(1) - 1 = -1
    // nv_prior = 0.1*ln(0.01) - 30*0.01 = 0.1*(-4.60517) - 0.3 = -0.760517
    let expected = -0.2 + (-1.0) + (0.1 * 0.01_f64.ln() - 30.0 * 0.01);
    assert_close(lp, expected, 1e-10, "log_prior");
}
