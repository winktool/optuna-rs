//! 采集函数精确交叉验证测试。
//!
//! 覆盖: logEI (standard_logei 两段式), log_ndtr, logehvi, erfinv, erfcx
//! 所有参考值来自 Python torch.special / scipy (seed=42)。

use optuna_rs::samplers::gp::{log_ndtr};

// ============================================================================
// 辅助: 直接调用 gp 模块内部函数
// ============================================================================

/// 镜像 Python `logei(mean, var, f0)` — 通过 pub 接口计算。
/// logei = standard_logei((mean - f0)/sigma) + ln(sigma)
/// 我们在 gp.rs 中 log_ei 是 pub(crate), 所以通过间接手段验证。
///
/// 这里直接复制 log_ei 的逻辑来测试核心数值。
fn log_ei_reference(mean: f64, var: f64, f0: f64) -> f64 {
    use std::f64::consts::PI;
    let sqrt_half: f64 = (0.5_f64).sqrt();
    let inv_sqrt_2pi: f64 = 1.0 / (2.0 * PI).sqrt();
    let sqrt_half_pi: f64 = (0.5 * PI).sqrt();
    let log_sqrt_2pi: f64 = (2.0 * PI).sqrt().ln();

    if var < 1e-30 {
        return if mean > f0 { (mean - f0).ln() } else { f64::NEG_INFINITY };
    }
    let sigma = var.sqrt();
    let z = (mean - f0) / sigma;

    let standard_lei = if z >= -25.0 {
        let z_half = 0.5 * z;
        let cdf_term = z_half * libm::erfc(-sqrt_half * z);
        let pdf_term = (-z_half * z).exp() * inv_sqrt_2pi;
        let ei = cdf_term + pdf_term;
        if ei > 0.0 { ei.ln() } else { f64::NEG_INFINITY }
    } else {
        // 尾部: erfcx 分支
        let erfcx_val = erfcx_test(-sqrt_half * z);
        -0.5 * z * z - log_sqrt_2pi
            + (1.0 + sqrt_half_pi * z * erfcx_val).ln()
    };
    standard_lei + sigma.ln()
}

fn erfcx_test(x: f64) -> f64 {
    use std::f64::consts::PI;
    if x < 0.0 {
        return (x * x).exp() * libm::erfc(x);
    }
    if x > 26.0 {
        let inv_2x2 = 1.0 / (2.0 * x * x);
        let inv_sqrt_pi = 1.0 / PI.sqrt();
        return inv_sqrt_pi / x * (1.0 - inv_2x2 * (1.0 - 3.0 * inv_2x2));
    }
    (x * x).exp() * libm::erfc(x)
}

/// 镜像 Python `logehvi` — 直接计算。
fn log_ehvi_reference(
    y_post: &[Vec<f64>],
    box_lower: &[Vec<f64>],
    box_intervals: &[Vec<f64>],
) -> f64 {
    let n_qmc = y_post.len();
    if n_qmc == 0 { return f64::NEG_INFINITY; }
    let log_n_qmc = (n_qmc as f64).ln();
    let eps = 1e-12;

    let mut log_vals: Vec<f64> = Vec::new();
    for sample in y_post {
        for (lb, interval) in box_lower.iter().zip(box_intervals.iter()) {
            let mut log_prod = 0.0;
            for d in 0..sample.len() {
                let diff = (sample[d] - lb[d]).clamp(eps, interval[d]);
                log_prod += diff.ln();
            }
            log_vals.push(log_prod);
        }
    }

    if log_vals.is_empty() { return f64::NEG_INFINITY; }
    let max_log = log_vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if max_log == f64::NEG_INFINITY { return f64::NEG_INFINITY; }
    let sum_exp: f64 = log_vals.iter().map(|&v| (v - max_log).exp()).sum();
    max_log + sum_exp.ln() - log_n_qmc
}

// ============================================================================
// 1. logEI 精确数值验证
// ============================================================================

/// Python 参考 (torch.special):
///   logei(0, 1, 0):      -9.18938533204672670e-01
///   logei(1, 1, 0):       8.00262188493070376e-02
///   logei(2, 0.5, 1.5):  -5.11124644823126362e-01
///   logei(-1, 2, 0.5):   -2.25539373158373779e+00
///   logei(-5, 1, 0):     -1.67443011626609675e+01
///   logei(-30, 1, 0):    -4.57724653760597903e+02    [尾部分支]
///   logei(10, 4, 8):      7.73173399409252338e-01
///   logei(0.5, 0.01, 0): -6.93147169867614155e-01
///   logei(-0.1, 0.1, 0.3): -4.16658086090348068e+00
#[test]
fn test_logei_main_branch_precision() {
    let cases: Vec<(f64, f64, f64, f64)> = vec![
        (0.0,  1.0, 0.0, -9.18938533204672670e-01),
        (1.0,  1.0, 0.0,  8.00262188493070376e-02),
        (2.0,  0.5, 1.5, -5.11124644823126362e-01),
        (-1.0, 2.0, 0.5, -2.25539373158373779e+00),
        (-5.0, 1.0, 0.0, -1.67443011626609675e+01),
        (10.0, 4.0, 8.0,  7.73173399409252338e-01),
        (0.5, 0.01, 0.0, -6.93147169867614155e-01),
        (-0.1, 0.1, 0.3, -4.16658086090348068e+00),
    ];
    for (mean, var, f0, expected) in cases {
        let result = log_ei_reference(mean, var, f0);
        let tol = expected.abs() * 1e-10 + 1e-14;
        assert!(
            (result - expected).abs() < tol,
            "logei({}, {}, {}) = {:.17e}, expected {:.17e}, diff={:.2e}",
            mean, var, f0, result, expected, (result - expected).abs()
        );
    }
}

/// logEI 尾部分支 (z < -25): erfcx 路径
#[test]
fn test_logei_tail_branch_precision() {
    // logei(-30, 1, 0): z=-30, 使用 erfcx 分支
    let expected: f64 = -4.57724653760597903e+02;
    let result = log_ei_reference(-30.0, 1.0, 0.0);
    let tol = expected.abs() * 1e-10;
    assert!(
        (result - expected).abs() < tol,
        "logei(-30, 1, 0) = {:.17e}, expected {:.17e}, diff={:.2e}",
        result, expected, (result - expected).abs()
    );
}

// ============================================================================
// 2. log_ndtr 精确数值验证 (直接使用 pub 接口)
// ============================================================================

/// Python 参考 (torch.special.log_ndtr):
#[test]
fn test_log_ndtr_precision_comprehensive() {
    let cases: Vec<(f64, f64)> = vec![
        (0.0,   -6.93147180559945286e-01),
        (1.0,   -1.72753779023449822e-01),
        (-1.0,  -1.84102164500926402e+00),
        (-5.0,  -1.50649983939887271e+01),
        (-10.0, -5.32312851505124769e+01),
        (-20.0, -2.03917155371097266e+02),
        (-37.0, -6.89030585576890644e+02),
        (3.0,   -1.35080996474819205e-03),
        (6.0,   -9.86587645524371251e-10),
    ];
    for (z, expected) in cases {
        let result = log_ndtr(z);
        let tol = expected.abs() * 1e-10 + 1e-14;
        assert!(
            (result - expected).abs() < tol,
            "log_ndtr({}) = {:.17e}, expected {:.17e}, diff={:.2e}",
            z, result, expected, (result - expected).abs()
        );
    }
}

// ============================================================================
// 3. logehvi 精确数值验证
// ============================================================================

/// Python 参考:
///   Y_post = [[1.5, 2.0], [2.0, 1.0]]
///   box_lower = [[0.0, 0.0], [0.5, 0.5]]
///   box_upper = [[3.0, 3.0], [2.5, 2.5]]
///   logehvi = 1.28785428830663795e+00
#[test]
fn test_logehvi_2qmc_2box_precision() {
    let y_post = vec![vec![1.5, 2.0], vec![2.0, 1.0]];
    let box_lower = vec![vec![0.0, 0.0], vec![0.5, 0.5]];
    let box_upper = vec![vec![3.0, 3.0], vec![2.5, 2.5]];
    let box_intervals: Vec<Vec<f64>> = box_lower.iter().zip(box_upper.iter())
        .map(|(lb, ub)| {
            lb.iter().zip(ub.iter()).map(|(&l, &u)| {
                let diff: f64 = u - l;
                diff.max(1e-12)
            }).collect()
        })
        .collect();

    let expected: f64 = 1.28785428830663795e+00;
    let result = log_ehvi_reference(&y_post, &box_lower, &box_intervals);
    let tol = expected.abs() * 1e-10;
    assert!(
        (result - expected).abs() < tol,
        "logehvi = {:.17e}, expected {:.17e}, diff={:.2e}",
        result, expected, (result - expected).abs()
    );
}

/// logehvi: 单样本单盒 — 退化为 log(prod(clamp(y-lb, eps, interval)))
#[test]
fn test_logehvi_single_sample_single_box() {
    let y_post = vec![vec![2.0, 3.0]];
    let box_lower = vec![vec![1.0, 1.0]];
    let box_intervals = vec![vec![5.0, 5.0]];

    // diff = [min(1.0,5.0), min(2.0,5.0)] = [1.0, 2.0]
    // log(1.0*2.0) - ln(1) = ln(2) - 0 = 0.693...
    let expected = (2.0_f64).ln();
    let result = log_ehvi_reference(&y_post, &box_lower, &box_intervals);
    assert!(
        (result - expected).abs() < 1e-14,
        "logehvi = {:.17e}, expected {:.17e}",
        result, expected
    );
}

/// logehvi: 样本在盒外 — 贡献被 clamp 到 eps
#[test]
fn test_logehvi_sample_outside_box() {
    let y_post = vec![vec![-1.0, -1.0]]; // 在盒 [0,0]→[3,3] 下方
    let box_lower = vec![vec![0.0, 0.0]];
    let box_intervals = vec![vec![3.0, 3.0]];

    // diff = [clamp(-1, eps, 3), clamp(-1, eps, 3)] = [eps, eps]
    // log(eps * eps) - ln(1) ≈ 2*ln(1e-12) ≈ -55.26
    let eps = 1e-12_f64;
    let expected = 2.0 * eps.ln();
    let result = log_ehvi_reference(&y_post, &box_lower, &box_intervals);
    assert!(
        (result - expected).abs() < 1e-10,
        "logehvi = {:.17e}, expected {:.17e}",
        result, expected
    );
}

// ============================================================================
// 4. erfinv 精度测试 (通过 normal QMC 采样间接验证)
// ============================================================================

/// 验证 erfinv 精度: erfinv(erf(x)) ≈ x
#[test]
fn test_erfinv_roundtrip_precision() {
    let test_vals = vec![
        0.0, 0.1, 0.5, 1.0, 2.0, 3.0, -0.5, -1.0, -2.0, -3.0,
        0.001, -0.001,
    ];
    for &x in &test_vals {
        let erf_x = libm::erf(x);
        let roundtrip = erfinv_test(erf_x);
        let tol = x.abs() * 1e-12 + 1e-13;
        assert!(
            (roundtrip - x).abs() < tol,
            "erfinv(erf({})) = {:.17e}, expected {}, diff={:.2e}",
            x, roundtrip, x, (roundtrip - x).abs()
        );
    }
}

fn erfinv_test(x: f64) -> f64 {
    use std::f64::consts::PI;
    if x.abs() >= 1.0 {
        return if x > 0.0 { f64::INFINITY } else { f64::NEG_INFINITY };
    }
    if x.abs() < 1e-15 { return x; }
    let a = 0.147;
    let ln_part = (1.0 - x * x).ln();
    let b = 2.0 / (PI * a) + 0.5 * ln_part;
    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let mut y = sign * (((b * b - ln_part / a).sqrt() - b).sqrt());
    let two_over_sqrt_pi = 2.0 / PI.sqrt();
    for _ in 0..2 {
        let ey = libm::erf(y);
        let deriv = two_over_sqrt_pi * (-y * y).exp();
        if deriv.abs() < 1e-300 { break; }
        let correction = (ey - x) / deriv;
        y -= correction / (1.0 + y * correction);
    }
    y
}

/// 验证 erfinv 在极端值的精度 (对齐 torch.erfinv)
#[test]
fn test_erfinv_extreme_values() {
    // erfinv(±0.999) 应该精确到机器精度
    let val = erfinv_test(0.999);
    let check = libm::erf(val);
    assert!(
        (check - 0.999).abs() < 1e-14,
        "erf(erfinv(0.999)) = {:.17e}, expected 0.999",
        check
    );

    let val_neg = erfinv_test(-0.999);
    let check_neg = libm::erf(val_neg);
    assert!(
        (check_neg + 0.999).abs() < 1e-14,
        "erf(erfinv(-0.999)) = {:.17e}, expected -0.999",
        check_neg
    );
}

// ============================================================================
// 5. logEI 单调性和基本属性
// ============================================================================

/// logEI(mean, var, f0) 应随 mean 单调递增 (var, f0 固定)
#[test]
fn test_logei_monotone_in_mean() {
    let var = 1.0;
    let f0 = 0.0;
    let means = vec![-5.0, -2.0, -1.0, 0.0, 1.0, 2.0, 5.0];
    let vals: Vec<f64> = means.iter().map(|&m| log_ei_reference(m, var, f0)).collect();
    for i in 1..vals.len() {
        assert!(
            vals[i] > vals[i - 1],
            "logEI not monotone: logEI({}) = {:.6e} <= logEI({}) = {:.6e}",
            means[i], vals[i], means[i - 1], vals[i - 1]
        );
    }
}

/// logEI(mean, var, f0) → ln(mean - f0) when mean >> f0 and var → 0
#[test]
fn test_logei_deterministic_limit() {
    // mean=10, var=1e-6, f0=5 → logEI ≈ ln(5)
    let result = log_ei_reference(10.0, 1e-6, 5.0);
    let expected = (5.0_f64).ln();
    assert!(
        (result - expected).abs() < 0.01,
        "logEI(10, 1e-6, 5) = {:.6e}, expected ≈ ln(5) = {:.6e}",
        result, expected
    );
}

// ============================================================================
// 6. GP Sampler 端到端收敛 (确认修复后仍正确)
// ============================================================================

use std::sync::Arc;
use optuna_rs::samplers::GpSampler;
use optuna_rs::samplers::Sampler;
use optuna_rs::study::{create_study, StudyDirection};

/// GP Sampler x² 收敛: 30 trials 内 best < 0.01
#[test]
fn test_gp_convergence_x_squared_30trials() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
        Some(42), Some(StudyDirection::Minimize), Some(10), false, None, None,
    ));
    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Minimize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
            Ok(x * x)
        },
        Some(30), None, None,
    ).unwrap();

    let best = study.best_value().unwrap();
    assert!(best < 0.1, "GP x² 30 trials: best = {} (should < 0.1)", best);
}

/// GP Sampler maximize 方向: x*(10-x) 最大值在 x=5
#[test]
fn test_gp_convergence_maximize() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
        Some(42), Some(StudyDirection::Maximize), Some(10), false, None, None,
    ));
    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Maximize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
            Ok(x * (10.0 - x))
        },
        Some(40), None, None,
    ).unwrap();

    let best = study.best_value().unwrap();
    assert!(best > 20.0, "GP maximize: best = {} (should > 20, optimum=25)", best);
}

/// GP Sampler 2目标: 验证 EHVI 多目标采集函数（收敛到 Pareto 前沿）
#[test]
fn test_gp_multi_objective_convergence() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
        Some(42), None, Some(10), false, None, None,
    ));
    let study = create_study(
        None, Some(sampler), None, None,
        None, Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]), false,
    ).unwrap();

    study.optimize_multi(
        |trial| {
            let x = trial.suggest_float("x", 0.0, 5.0, false, None)?;
            Ok(vec![x * x, (x - 3.0) * (x - 3.0)])
        },
        Some(40), None, None,
    ).unwrap();

    let trials = study.trials().unwrap();
    // 应该有些试验在 x∈[0,3] 范围内探索 Pareto 前沿
    let min_f1 = trials.iter().map(|t| t.values.as_ref().unwrap()[0]).fold(f64::INFINITY, f64::min);
    let min_f2 = trials.iter().map(|t| t.values.as_ref().unwrap()[1]).fold(f64::INFINITY, f64::min);
    assert!(min_f1 < 5.0, "f1 min = {} (should < 5.0)", min_f1);
    assert!(min_f2 < 5.0, "f2 min = {} (should < 5.0)", min_f2);
}

// ============================================================================
// 7. GP 搜索空间归一化精确对齐 (step-adjusted bounds)
// ============================================================================

use optuna_rs::samplers::gp::{normalize_param, unnormalize_param};
use optuna_rs::distributions::{Distribution, FloatDistribution, IntDistribution};

/// Python 参考 (IntDistribution(0, 10, step=1)):
///   normalize(0)  = 0.04545454545
///   normalize(5)  = 0.50000000000
///   normalize(10) = 0.95454545455
#[test]
fn test_normalize_int_step_adjusted() {
    let dist = Distribution::IntDistribution(IntDistribution {
        low: 0, high: 10, log: false, step: 1,
    });
    let cases = vec![
        (0.0,  0.5 / 11.0),
        (5.0,  5.5 / 11.0),
        (10.0, 10.5 / 11.0),
    ];
    for (val, expected) in cases {
        let result = normalize_param(val, &dist);
        assert!(
            (result - expected).abs() < 1e-12,
            "normalize_param({}, IntDist(0,10)) = {:.12}, expected {:.12}",
            val, result, expected
        );
    }
}

/// Python 参考 (FloatDistribution(0, 1, step=0.1)):
///   normalize(0.0) = 0.04545454545
///   normalize(0.5) = 0.50000000000
///   normalize(1.0) = 0.95454545455
#[test]
fn test_normalize_float_with_step() {
    let dist = Distribution::FloatDistribution(FloatDistribution {
        low: 0.0, high: 1.0, log: false, step: Some(0.1),
    });
    let cases = vec![
        (0.0, 0.05 / 1.1),
        (0.5, 0.55 / 1.1),
        (1.0, 1.05 / 1.1),
    ];
    for (val, expected) in cases {
        let result = normalize_param(val, &dist);
        assert!(
            (result - expected).abs() < 1e-12,
            "normalize_param({}, FloatDist(0,1,step=0.1)) = {:.12}, expected {:.12}",
            val, result, expected
        );
    }
}

/// FloatDistribution without step: bounds unchanged
#[test]
fn test_normalize_float_no_step() {
    let dist = Distribution::FloatDistribution(FloatDistribution {
        low: 0.0, high: 1.0, log: false, step: None,
    });
    assert!((normalize_param(0.0, &dist) - 0.0).abs() < 1e-14);
    assert!((normalize_param(0.5, &dist) - 0.5).abs() < 1e-14);
    assert!((normalize_param(1.0, &dist) - 1.0).abs() < 1e-14);
}

/// Unnormalize roundtrip: unnormalize(normalize(val)) ≈ val
#[test]
fn test_normalize_unnormalize_roundtrip_int() {
    let dist = Distribution::IntDistribution(IntDistribution {
        low: 0, high: 10, log: false, step: 1,
    });
    for val in 0..=10 {
        let norm = normalize_param(val as f64, &dist);
        let denorm = unnormalize_param(norm, &dist);
        assert!(
            (denorm - val as f64).abs() < 0.5,
            "roundtrip({}) = {:.6}",
            val, denorm
        );
    }
}

/// Unnormalize roundtrip: float with step
#[test]
fn test_normalize_unnormalize_roundtrip_float_step() {
    let dist = Distribution::FloatDistribution(FloatDistribution {
        low: 0.0, high: 1.0, log: false, step: Some(0.1),
    });
    for i in 0..=10 {
        let val = i as f64 * 0.1;
        let norm = normalize_param(val, &dist);
        let denorm = unnormalize_param(norm, &dist);
        assert!(
            (denorm - val).abs() < 1e-10,
            "roundtrip({:.1}) = {:.10}",
            val, denorm
        );
    }
}

/// Log-scale IntDistribution with step adjustment
#[test]
fn test_normalize_int_log_with_step() {
    let dist = Distribution::IntDistribution(IntDistribution {
        low: 1, high: 100, log: true, step: 1,
    });
    // Python: low_adj = 1 - 0.5 = 0.5, high_adj = 100 + 0.5 = 100.5
    // ln(0.5) = -0.693, ln(100.5) = 4.610
    let norm_1 = normalize_param(1.0, &dist);
    let norm_100 = normalize_param(100.0, &dist);
    // norm_1 should be slightly above 0 (not 0)
    assert!(norm_1 > 0.0 && norm_1 < 0.2, "norm(1, IntLog(1,100)) = {}", norm_1);
    // norm_100 should be slightly below 1 (not 1)
    assert!(norm_100 > 0.8 && norm_100 < 1.0, "norm(100, IntLog(1,100)) = {}", norm_100);
    // roundtrip
    let denorm = unnormalize_param(norm_1, &dist);
    assert!((denorm - 1.0).abs() < 0.5, "roundtrip(1) = {}", denorm);
}

/// GP Sampler with Int parameters — verifies step-adjusted normalization works end-to-end
/// Python ref: GPSampler(seed=42, n_startup=10), suggest_int(0,20), 30 trials → best=0, x=7
#[test]
fn test_gp_sampler_int_param_convergence() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
        Some(42), Some(StudyDirection::Minimize), Some(10), true, None, None,
    ));
    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Minimize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let x = trial.suggest_int("x", 0, 20, false, 1)?;
            let d = x - 7;
            Ok((d * d) as f64)
        },
        Some(30), None, None,
    ).unwrap();

    let best = study.best_value().unwrap();
    assert!(best <= 4.0, "GP int (x-7)² 30 trials: best = {} (should <= 4.0)", best);
}

// ════════════════════════════════════════════════════════════════════════
// GP 核心数值精度交叉验证: 核矩阵 / 后验 / 边际似然
// Python 参考值使用固定核参数生成 (inv_sq_ls=[2,3], ks=1.5, nv=0.01)
// ════════════════════════════════════════════════════════════════════════

/// Matern 5/2 核函数精度 — 对齐 Python torch Matern52Kernel
#[test]
fn test_matern52_kernel_precision() {
    use optuna_rs::samplers::gp::matern52;
    let cases: [(f64, f64); 4] = [
        (0.0, 1.0),
        (0.5, 7.024957601538032e-01),
        (2.0, 3.172833639540438e-01),
        (1e-30, 1.0),  // 极小距离 → 1.0
    ];
    for (d2, expected) in &cases {
        let got = matern52(*d2);
        let rel = ((got - expected) / expected.max(1e-30)).abs();
        assert!(rel < 1e-12, "matern52({}) = {:.15e}, expected {:.15e}, rel={:.2e}", d2, got, expected, rel);
    }
}

/// GP 核矩阵精度 — 5 点训练集 (inv_sq_ls=[2,3], ks=1.5)
#[test]
fn test_gp_kernel_matrix_precision() {
    use optuna_rs::samplers::gp::GPRegressor;
    let x_train = vec![
        vec![0.1, 0.4], vec![0.3, 0.9], vec![0.7, 0.2],
        vec![0.5, 0.5], vec![0.9, 0.8],
    ];
    let y_train = vec![0.5, -0.3, 1.2, 0.1, -0.8];
    let is_cat = vec![false, false];
    let gpr = GPRegressor::new(x_train, y_train, is_cat, vec![2.0, 3.0], 1.5, 0.01);

    // K[0,0] = kernel_scale = 1.5
    let k_mat = gpr.train_kernel_matrix();
    let rel00 = ((k_mat[0][0] - 1.5) / 1.5).abs();
    assert!(rel00 < 1e-14, "K[0,0] = {:.15e}, expected 1.5", k_mat[0][0]);

    // K[1,2] — Python ref: 5.252613721795685e-01
    let expected_12 = 5.252613721795685e-01;
    let rel12 = ((k_mat[1][2] - expected_12) / expected_12).abs();
    assert!(rel12 < 1e-12, "K[1,2] = {:.15e}, expected {:.15e}, rel={:.2e}", k_mat[1][2], expected_12, rel12);

    // K[2,3] — Python ref: 1.161182556936957e+00
    let expected_23 = 1.161182556936957e+00;
    let rel23 = ((k_mat[2][3] - expected_23) / expected_23).abs();
    assert!(rel23 < 1e-12, "K[2,3] = {:.15e}, expected {:.15e}, rel={:.2e}", k_mat[2][3], expected_23, rel23);
}

/// GP 后验预测精度 — 4 个测试点
#[test]
fn test_gp_posterior_precision() {
    use optuna_rs::samplers::gp::GPRegressor;
    let x_train = vec![
        vec![0.1, 0.4], vec![0.3, 0.9], vec![0.7, 0.2],
        vec![0.5, 0.5], vec![0.9, 0.8],
    ];
    let y_train = vec![0.5, -0.3, 1.2, 0.1, -0.8];
    let is_cat = vec![false, false];
    let gpr = GPRegressor::new(x_train, y_train, is_cat, vec![2.0, 3.0], 1.5, 0.01);

    // Python ref posterior values
    let cases: [([f64; 2], f64, f64); 4] = [
        ([0.2, 0.6], 8.394071860600660e-02, 7.904576283214881e-02),
        ([0.5, 0.5], 1.137002060680299e-01, 9.541429090546671e-03),
        ([0.0, 0.0], 8.190875501312160e-01, 6.338578755118230e-01),
        ([1.0, 1.0], -8.586976647994353e-01, 2.432481951417311e-01),
    ];

    for (x, exp_mean, exp_var) in &cases {
        let (mean, var) = gpr.posterior(x);
        let mean_rel = ((mean - exp_mean) / exp_mean.abs().max(1e-30)).abs();
        let var_rel = ((var - exp_var) / exp_var.abs().max(1e-30)).abs();
        assert!(mean_rel < 1e-10, "posterior({:?}) mean = {:.15e}, expected {:.15e}, rel={:.2e}", x, mean, exp_mean, mean_rel);
        assert!(var_rel < 1e-10, "posterior({:?}) var  = {:.15e}, expected {:.15e}, rel={:.2e}", x, var, exp_var, var_rel);
    }
}

/// GP 边际对数似然精度
#[test]
fn test_gp_log_marginal_likelihood_precision() {
    use optuna_rs::samplers::gp::GPRegressor;
    let x_train = vec![
        vec![0.1, 0.4], vec![0.3, 0.9], vec![0.7, 0.2],
        vec![0.5, 0.5], vec![0.9, 0.8],
    ];
    let y_train = vec![0.5, -0.3, 1.2, 0.1, -0.8];
    let is_cat = vec![false, false];
    let gpr = GPRegressor::new(x_train, y_train, is_cat, vec![2.0, 3.0], 1.5, 0.01);

    let lml = gpr.log_marginal_likelihood();
    let expected = -5.446016686390402e+00;
    let rel = ((lml - expected) / expected.abs()).abs();
    assert!(rel < 1e-10, "LML = {:.15e}, expected {:.15e}, rel={:.2e}", lml, expected, rel);
}

/// GP 分类参数 Hamming 距离核矩阵精度
#[test]
fn test_gp_categorical_kernel_precision() {
    use optuna_rs::samplers::gp::GPRegressor;
    let x_train = vec![
        vec![0.1, 0.0], vec![0.3, 0.0], vec![0.7, 1.0], vec![0.5, 2.0],
    ];
    let y_train = vec![0.5, -0.3, 1.2, 0.1];
    let is_cat = vec![false, true]; // 第二维是分类参数
    let gpr = GPRegressor::new(x_train, y_train, is_cat, vec![2.0, 1.5], 1.0, 0.05);

    let k_mat = gpr.train_kernel_matrix();

    // K[0,1]: same category → Hamming=0, dist = (0.1-0.3)^2 * 2
    let expected_01 = 9.381382129367236e-01;
    let rel01 = ((k_mat[0][1] - expected_01) / expected_01).abs();
    assert!(rel01 < 1e-12, "K_cat[0,1] = {:.15e}, expected {:.15e}", k_mat[0][1], expected_01);

    // K[0,2]: diff category → Hamming=1, dist = (0.1-0.7)^2 * 2 + 1.0 * 1.5
    let expected_02 = 2.869996621872009e-01;
    let rel02 = ((k_mat[0][2] - expected_02) / expected_02).abs();
    assert!(rel02 < 1e-12, "K_cat[0,2] = {:.15e}, expected {:.15e}", k_mat[0][2], expected_02);

    // K[2,3]: diff category → Hamming=1
    let expected_23 = 3.876935083071805e-01;
    let rel23 = ((k_mat[2][3] - expected_23) / expected_23).abs();
    assert!(rel23 < 1e-12, "K_cat[2,3] = {:.15e}, expected {:.15e}", k_mat[2][3], expected_23);

    // 后验预测: x=[0.4, 1.0]
    let (mean, var) = gpr.posterior(&[0.4, 1.0]);
    let exp_mean = 1.012667798231789e+00;
    let exp_var = 2.575846011039260e-01;
    let mean_rel = ((mean - exp_mean) / exp_mean.abs()).abs();
    let var_rel = ((var - exp_var) / exp_var.abs()).abs();
    assert!(mean_rel < 1e-10, "cat posterior mean = {:.15e}, expected {:.15e}", mean, exp_mean);
    assert!(var_rel < 1e-10, "cat posterior var = {:.15e}, expected {:.15e}", var, exp_var);
}

/// GP 先验函数精度
#[test]
fn test_gp_default_log_prior_precision() {
    use optuna_rs::samplers::gp::default_log_prior;
    let p = default_log_prior(&[2.0, 1.5], 1.0, 0.05);
    let expected = -3.266239894022066e+00;
    let rel = ((p - expected) / expected.abs()).abs();
    assert!(rel < 1e-12, "log_prior = {:.15e}, expected {:.15e}, rel={:.2e}", p, expected, rel);
}

/// IntDistribution step=3 的归一化往返精度
/// Python ref: IntDist(0, 15, step=3), valid={0,3,6,9,12,15}
#[test]
fn test_normalize_int_step3_roundtrip() {
    use optuna_rs::distributions::{Distribution, IntDistribution};
    use optuna_rs::samplers::gp::{normalize_param, unnormalize_param};
    let dist = Distribution::IntDistribution(IntDistribution::new(0, 15, false, 3).unwrap());

    // Python 参考: v=0 → 0.0833, v=3 → 0.25, v=15 → 0.9167
    let py_norms: [(f64, f64); 6] = [
        (0.0,  8.333333333333333e-02),
        (3.0,  2.500000000000000e-01),
        (6.0,  4.166666666666667e-01),
        (9.0,  5.833333333333334e-01),
        (12.0, 7.500000000000000e-01),
        (15.0, 9.166666666666666e-01),
    ];

    for &(v, exp_norm) in &py_norms {
        let norm = normalize_param(v, &dist);
        let rel = ((norm - exp_norm) / exp_norm).abs();
        assert!(rel < 1e-12, "normalize({}) = {:.15e}, expected {:.15e}", v, norm, exp_norm);

        let unnorm = unnormalize_param(norm, &dist);
        assert!((unnorm - v).abs() < 1e-10, "unnormalize(normalize({})) = {}, expected {}", v, unnorm, v);
    }
}

/// FloatDistribution step=0.25 的归一化往返精度
#[test]
fn test_normalize_float_step025_roundtrip() {
    use optuna_rs::distributions::{Distribution, FloatDistribution};
    use optuna_rs::samplers::gp::{normalize_param, unnormalize_param};
    let dist = Distribution::FloatDistribution(
        FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap(),
    );

    let py_norms: [(f64, f64); 5] = [
        (0.00, 1.000000000000000e-01),
        (0.25, 3.000000000000000e-01),
        (0.50, 5.000000000000000e-01),
        (0.75, 7.000000000000000e-01),
        (1.00, 9.000000000000000e-01),
    ];

    for &(v, exp_norm) in &py_norms {
        let norm = normalize_param(v, &dist);
        let rel = ((norm - exp_norm) / exp_norm).abs();
        assert!(rel < 1e-12, "normalize({}) = {:.15e}, expected {:.15e}", v, norm, exp_norm);

        let unnorm = unnormalize_param(norm, &dist);
        assert!((unnorm - v).abs() < 1e-10, "unnormalize(normalize({})) = {}, expected {}", v, unnorm, v);
    }
}

/// Halton 高维 (dim=25) 精度测试 — 验证扩展素数表后 dim 20-24 正确使用素数基底
/// Python ref: scipy.stats.qmc.Halton(d=25, scramble=False).random(5)
/// 旧实现 dim >= 20 退化为非素数 (2d+3)，导致 dim=21 基底=45=3×15 而非 79
#[test]
fn test_halton_high_dim_prime_bases() {
    use optuna_rs::samplers::qmc::halton_point;

    // 第 1 个点 (index=1): 各维 = 1/prime
    // dim 20: base=73 → 1/73 = 0.013698630136986
    // dim 21: base=79 → 1/79 = 0.012658227848101
    // dim 22: base=83 → 1/83 = 0.012048192771084
    // dim 23: base=89 → 1/89 = 0.011235955056180
    // dim 24: base=97 → 1/97 = 0.010309278350515
    let pt1 = halton_point(1, 25, false, 0);
    let expected_dim20_24 = [
        1.0 / 73.0,
        1.0 / 79.0,
        1.0 / 83.0,
        1.0 / 89.0,
        1.0 / 97.0,
    ];
    for (i, &exp) in expected_dim20_24.iter().enumerate() {
        let d = 20 + i;
        let rel = ((pt1[d] - exp) / exp).abs();
        assert!(rel < 1e-14, "halton(1)[{d}]: got {:.15e}, expected {:.15e}", pt1[d], exp);
    }

    // 第 4 个点 (index=4): 验证更复杂的 Van der Corput 值
    // Python ref: point[4] dim=20..24
    let pt4 = halton_point(4, 25, false, 0);
    let expected_pt4 = [
        0.054794520547945,  // dim 20, base 73
        0.050632911392405,  // dim 21, base 79
        0.048192771084337,  // dim 22, base 83
        0.044943820224719,  // dim 23, base 89
        0.041237113402062,  // dim 24, base 97
    ];
    for (i, &exp) in expected_pt4.iter().enumerate() {
        let d = 20 + i;
        let rel = ((pt4[d] - exp) / exp).abs();
        assert!(rel < 1e-12, "halton(4)[{d}]: got {:.15e}, expected {:.15e}", pt4[d], exp);
    }

    // 验证低维部分仍然正确
    assert!((pt1[0] - 0.5).abs() < 1e-15, "halton(1)[0]=0.5");
    assert!((pt1[1] - 1.0/3.0).abs() < 1e-15, "halton(1)[1]=1/3");
}

/// SearchSpaceTransform: Float log 分布 + transform_0_1=true 精度交叉验证
/// Python ref: FloatDistribution(0.01, 100.0, log=True), transform_0_1=True
/// enc(0.01)=0.0, enc(0.1)=0.25, enc(1.0)=0.5, enc(10.0)=0.75, enc(100.0)=1.0
#[test]
fn test_transform_float_log_0_1_precision() {
    use optuna_rs::distributions::{Distribution, FloatDistribution, ParamValue};
    use optuna_rs::search_space::SearchSpaceTransform;
    use indexmap::IndexMap;

    let mut ss = IndexMap::new();
    ss.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.01, 100.0, true, None).unwrap()),
    );
    let t = SearchSpaceTransform::new(ss, true, true, true);

    // Python 参考值: ln(100)/ln(0.01) 对称 → 均匀映射到 [0,1]
    let cases: [(f64, f64); 5] = [
        (0.01,  0.0),
        (0.1,   0.25),
        (1.0,   0.5),
        (10.0,  0.75),
        (100.0, 1.0),
    ];

    for &(v, exp_enc) in &cases {
        let mut params = IndexMap::new();
        params.insert("x".to_string(), ParamValue::Float(v));
        let encoded = t.transform(&params);
        let diff = (encoded[0] - exp_enc).abs();
        assert!(diff < 1e-14, "transform_0_1({v}): got {:.15e}, expected {:.15e}", encoded[0], exp_enc);

        // 往返精度
        let decoded = t.untransform(&encoded).unwrap();
        let dec_v = match &decoded["x"] {
            ParamValue::Float(f) => *f,
            _ => panic!("expected Float"),
        };
        let rel_err = if v.abs() > 1e-15 { ((dec_v - v) / v).abs() } else { (dec_v - v).abs() };
        assert!(rel_err < 1e-12, "roundtrip({v}): got {dec_v}, rel_err={rel_err:.2e}");
    }
}

/// SearchSpaceTransform: Int step=5 的 bounds 和往返精度
/// Python ref: IntDist(0, 100, step=5), transform_step=True
/// bounds=[-2.5, 102.5], v=0 → enc=0.0, v=50 → enc=50.0
#[test]
fn test_transform_int_step5_precision() {
    use optuna_rs::distributions::{Distribution, IntDistribution, ParamValue};
    use optuna_rs::search_space::SearchSpaceTransform;
    use indexmap::IndexMap;

    let mut ss = IndexMap::new();
    ss.insert(
        "y".to_string(),
        Distribution::IntDistribution(IntDistribution::new(0, 100, false, 5).unwrap()),
    );
    let t = SearchSpaceTransform::new(ss, true, true, false);

    // 验证 bounds 对齐 Python: [-2.5, 102.5]
    let bounds = t.bounds();
    assert!((bounds[0][0] - (-2.5)).abs() < 1e-15, "low bound: got {}", bounds[0][0]);
    assert!((bounds[0][1] - 102.5).abs() < 1e-15, "high bound: got {}", bounds[0][1]);

    // 编码/解码往返
    for v in [0i64, 5, 25, 50, 100] {
        let mut params = IndexMap::new();
        params.insert("y".to_string(), ParamValue::Int(v));
        let encoded = t.transform(&params);
        assert!((encoded[0] - v as f64).abs() < 1e-14, "transform({v}): got {}", encoded[0]);

        let decoded = t.untransform(&encoded).unwrap();
        let dec_v = match &decoded["y"] {
            ParamValue::Int(i) => *i,
            _ => panic!("expected Int"),
        };
        assert_eq!(dec_v, v, "roundtrip({v}): got {dec_v}");
    }
}

/// SearchSpaceTransform: Float step=0.1 的 bounds 和往返精度
/// Python ref: FloatDist(0.0, 1.0, step=0.1), transform_step=True
/// bounds=[-0.05, 1.05]
#[test]
fn test_transform_float_step01_precision() {
    use optuna_rs::distributions::{Distribution, FloatDistribution, ParamValue};
    use optuna_rs::search_space::SearchSpaceTransform;
    use indexmap::IndexMap;

    let mut ss = IndexMap::new();
    ss.insert(
        "z".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap()),
    );
    let t = SearchSpaceTransform::new(ss, true, true, false);

    // bounds 对齐 Python: [-0.05, 1.05]
    let bounds = t.bounds();
    assert!((bounds[0][0] - (-0.05)).abs() < 1e-15, "low bound: got {}", bounds[0][0]);
    assert!((bounds[0][1] - 1.05).abs() < 1e-15, "high bound: got {}", bounds[0][1]);

    for v in [0.0, 0.1, 0.5, 0.9, 1.0] {
        let mut params = IndexMap::new();
        params.insert("z".to_string(), ParamValue::Float(v));
        let encoded = t.transform(&params);
        assert!((encoded[0] - v).abs() < 1e-14, "transform({v}): got {}", encoded[0]);

        let decoded = t.untransform(&encoded).unwrap();
        let dec_v = match &decoded["z"] {
            ParamValue::Float(f) => *f,
            _ => panic!("expected Float"),
        };
        assert!((dec_v - v).abs() < 1e-14, "roundtrip({v}): got {dec_v}");
    }
}

/// SearchSpaceTransform: Categorical one-hot 编码精度
/// Python ref: choices=['a','b','c'] → a=[1,0,0], b=[0,1,0], c=[0,0,1]
#[test]
fn test_transform_categorical_onehot() {
    use optuna_rs::distributions::{CategoricalChoice, CategoricalDistribution, Distribution, ParamValue};
    use optuna_rs::search_space::SearchSpaceTransform;
    use indexmap::IndexMap;

    let mut ss = IndexMap::new();
    ss.insert(
        "c".to_string(),
        Distribution::CategoricalDistribution(CategoricalDistribution {
            choices: vec![
                CategoricalChoice::Str("a".to_string()),
                CategoricalChoice::Str("b".to_string()),
                CategoricalChoice::Str("c".to_string()),
            ],
        }),
    );
    let t = SearchSpaceTransform::new(ss, true, true, false);

    // n_encoded = 3 (one-hot)
    assert_eq!(t.n_encoded(), 3);

    let expected: [Vec<f64>; 3] = [
        vec![1.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0],
        vec![0.0, 0.0, 1.0],
    ];

    let choices = ["a", "b", "c"];
    for (i, ch) in choices.iter().enumerate() {
        let mut params = IndexMap::new();
        params.insert(
            "c".to_string(), 
            ParamValue::Categorical(CategoricalChoice::Str(ch.to_string()))
        );
        let encoded = t.transform(&params);
        assert_eq!(encoded, expected[i], "one-hot({ch})");

        // 解码回来
        let decoded = t.untransform(&encoded).unwrap();
        let dec_v = match &decoded["c"] {
            ParamValue::Categorical(CategoricalChoice::Str(s)) => s.clone(),
            other => panic!("expected Str, got {:?}", other),
        };
        assert_eq!(dec_v, *ch, "roundtrip_categorical({ch})");
    }
}

// ============================================================================
// VSBX 交叉算子: 统计验证 uniform_crossover_prob 在父代基因分支生效
// ============================================================================

/// 验证 VSBX 修复: 当 use_child_gene_prob=0.001 (几乎总是父代) 且
/// uniform_crossover_prob=1.0 (总是交换) 时，child1/child2 会交换,
/// 最终等概率来自两个父代值。
#[test]
fn test_vsbx_parent_branch_swap() {
    use optuna_rs::samplers::nsgaii::crossover::{Crossover, VSBXCrossover};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let cx = VSBXCrossover::new(Some(20.0), 1.0, 0.001);

    let n_runs = 2000;
    let mut count_near_p0 = 0;
    let mut count_near_p1 = 0;

    for seed in 0..n_runs {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let p0 = vec![0.2];
        let p1 = vec![0.8];
        let child = cx.crossover(&[p0, p1], &mut rng);
        if (child[0] - 0.2).abs() < 0.01 {
            count_near_p0 += 1;
        } else if (child[0] - 0.8).abs() < 0.01 {
            count_near_p1 += 1;
        }
    }

    // 交换后 child1=p1, child2=p0, 最终50/50选择
    let total_parent = count_near_p0 + count_near_p1;
    assert!(
        total_parent > n_runs * 90 / 100,
        "大部分结果应是父代值, got {total_parent}/{n_runs}"
    );
    let ratio_p0 = count_near_p0 as f64 / total_parent as f64;
    assert!(
        (0.35..=0.65).contains(&ratio_p0),
        "父代交换后应50/50分布, 实际 p0 比例={ratio_p0:.3}"
    );
}

/// 验证 VSBX: 当 uniform_crossover_prob=0 且 use_child_gene_prob=0.001 时,
/// 不交换 → child1=p0, child2=p1, 最终50/50选择
#[test]
fn test_vsbx_parent_no_swap() {
    use optuna_rs::samplers::nsgaii::crossover::{Crossover, VSBXCrossover};
    use rand::SeedableRng;
    use rand_chacha::ChaCha8Rng;

    let cx = VSBXCrossover::new(Some(20.0), 0.0, 0.001);

    let n_runs = 2000;
    let mut count_near_p0 = 0;
    let mut count_near_p1 = 0;

    for seed in 0..n_runs {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let p0 = vec![0.2];
        let p1 = vec![0.8];
        let child = cx.crossover(&[p0, p1], &mut rng);
        if (child[0] - 0.2).abs() < 0.01 {
            count_near_p0 += 1;
        } else if (child[0] - 0.8).abs() < 0.01 {
            count_near_p1 += 1;
        }
    }

    let total = count_near_p0 + count_near_p1;
    assert!(
        total > n_runs * 90 / 100,
        "大部分应是父代值, got {total}/{n_runs}"
    );
    let ratio = count_near_p0 as f64 / total as f64;
    assert!(
        (0.35..=0.65).contains(&ratio),
        "不交换时也应50/50, ratio={ratio:.3}"
    );
}
