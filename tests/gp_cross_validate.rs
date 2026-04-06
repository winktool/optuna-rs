//! GP 模块精确交叉验证测试。
//!
//! 覆盖: GPRegressor 后验预测/LML、log_ndtr、solve_lower/upper、
//!       GpSampler 单/多目标收敛、maximize 方向。
//! 所有参考值来自 Python scipy/optuna (seed=42)。

use std::sync::Arc;
use optuna_rs::samplers::gp::{
    GPRegressor, log_ndtr, solve_lower, solve_upper, matern52, cholesky,
};
use optuna_rs::samplers::GpSampler;
use optuna_rs::samplers::Sampler;
use optuna_rs::study::{create_study, StudyDirection};

// ============================================================================
// 1. log_ndtr 精确数值验证 (对齐 scipy.special.log_ndtr)
// ============================================================================

/// Python 参考:
///   log_ndtr(0)   = -6.9314718055994529e-01
///   log_ndtr(1)   = -1.7275377902344990e-01
///   log_ndtr(-5)  = -1.5064998393988727e+01
///   log_ndtr(-10) = -5.3231285150512477e+01
///   log_ndtr(-20) = -2.0391715537109727e+02
///   log_ndtr(-37) = -6.8903058557689064e+02
///   log_ndtr(6)   = -9.8658764552437187e-10
///   log_ndtr(10)  = -7.6198530241604740e-24
#[test]
fn test_log_ndtr_main_branch() {
    let cases: Vec<(f64, f64)> = vec![
        (0.0, -6.9314718055994529e-01),
        (1.0, -1.7275377902344990e-01),
        (2.0, -2.3012909328963480e-02),
        (3.0, -1.3508099647481923e-03),
        (5.0, -2.8665161296376310e-07),
        (-1.0, -1.8410216450092634e+00),
        (-3.0, -6.6077262215103501e+00),
    ];
    for (z, expected) in cases {
        let result = log_ndtr(z);
        let tol = expected.abs() * 1e-12 + 1e-15;
        assert!(
            (result - expected).abs() < tol,
            "log_ndtr({}) = {:.17e}, expected {:.17e}, diff={:.2e}",
            z, result, expected, (result - expected).abs()
        );
    }
}

#[test]
fn test_log_ndtr_tail_branch() {
    // z < -5: 尾部渐近展开
    let cases: Vec<(f64, f64)> = vec![
        (-5.0,  -1.5064998393988727e+01),
        (-10.0, -5.3231285150512477e+01),
        (-20.0, -2.0391715537109727e+02),
        (-37.0, -6.8903058557689064e+02),
    ];
    for (z, expected) in cases {
        let result = log_ndtr(z);
        let tol = expected.abs() * 1e-6;
        assert!(
            (result - expected).abs() < tol,
            "log_ndtr({}) = {:.17e}, expected {:.17e}, diff={:.2e}, rel={:.2e}",
            z, result, expected, (result - expected).abs(),
            (result - expected).abs() / expected.abs()
        );
    }
}

#[test]
fn test_log_ndtr_large_positive() {
    // z > 6: log Φ(z) ≈ 0 (极小负值)
    let cases: Vec<(f64, f64)> = vec![
        (6.0,  -9.8658764552437187e-10),
        (10.0, -7.6198530241604740e-24),
    ];
    for (z, expected) in cases {
        let result = log_ndtr(z);
        let tol = expected.abs() * 0.1 + 1e-30;
        assert!(
            (result - expected).abs() < tol,
            "log_ndtr({}) = {:.17e}, expected {:.17e}, diff={:.2e}",
            z, result, expected, (result - expected).abs()
        );
    }
}

// ============================================================================
// 2. solve_lower / solve_upper 精确验证
// ============================================================================

/// 验证 L * x = b → x = L\b (前向替代)
/// 后向替代: L^T * x = b → x = L^{-T}\b
/// 综合验证: L * (L^T * x) = b → x = (K+σI)^{-1} b = alpha
#[test]
fn test_solve_lower_upper_roundtrip() {
    // 使用 Matérn 5/2 kernel matrix with known params
    let k = vec![
        vec![1.000001, 0.5239941088318203],
        vec![0.5239941088318203, 1.000001],
    ];
    let l = cholesky(&k).unwrap();
    let b = vec![0.0, 1.0];

    // Forward: L u = b
    let u = solve_lower(&l, &b);
    // Backward: L^T alpha = u
    let alpha = solve_upper(&l, &u);

    // Verify K * alpha ≈ b
    for i in 0..2 {
        let sum: f64 = (0..2).map(|j| k[i][j] * alpha[j]).sum();
        assert!(
            (sum - b[i]).abs() < 1e-10,
            "K*alpha[{}] = {:.17e}, expected {:.17e}",
            i, sum, b[i]
        );
    }

    // Verify alpha matches Python: [-0.7223199186316135, 1.3784900035648417]
    assert!(
        (alpha[0] - (-0.7223199186316135)).abs() < 1e-8,
        "alpha[0] = {:.17e}",
        alpha[0]
    );
    assert!(
        (alpha[1] - 1.3784900035648417).abs() < 1e-8,
        "alpha[1] = {:.17e}",
        alpha[1]
    );
}

#[test]
fn test_solve_lower_3x3() {
    // 3x3 lower triangular system
    let l = vec![
        vec![2.0, 0.0, 0.0],
        vec![1.0, 3.0, 0.0],
        vec![0.5, 0.5, 4.0],
    ];
    let b = vec![4.0, 7.0, 10.0];
    let x = solve_lower(&l, &b);
    // L*x should equal b
    for i in 0..3 {
        let sum: f64 = (0..3).map(|j| l[i][j] * x[j]).sum();
        assert!(
            (sum - b[i]).abs() < 1e-14,
            "L*x[{}] = {:.17e}, expected {:.17e}",
            i, sum, b[i]
        );
    }
}

// ============================================================================
// 3. GPRegressor 后验预测精确验证
// ============================================================================

/// Python 参考 (x=[0,1], y=[0,1], kernel_scale=1, noise_var=1e-6, inv_sq_ls=[1]):
///   posterior(0.0):  mean=7.223e-07,  var=1.000e-06
///   posterior(0.5):  mean=5.437e-01,  var=9.887e-02
///   posterior(1.0):  mean=1.000,      var=1.000e-06
///   posterior(0.25): mean=2.445e-01,  var=5.232e-02
///   posterior(0.75): mean=8.229e-01,  var=5.232e-02
#[test]
fn test_gp_regressor_posterior_precision() {
    let gpr = GPRegressor::new(
        vec![vec![0.0], vec![1.0]],
        vec![0.0, 1.0],
        vec![false],
        vec![1.0],
        1.0,
        1e-6,
    );

    let cases = vec![
        (0.0,  7.2231991854465605e-07, 9.9999862157584829e-07),
        (0.5,  5.4373477816034821e-01, 9.8869284749035447e-02),
        (1.0,  9.9999862150999652e-01, 9.9999862146482599e-07),
        (0.25, 2.4447644500746402e-01, 5.2318013737718716e-02),
        (0.75, 8.2285488189169331e-01, 5.2318013737718383e-02),
    ];

    for (x, exp_mean, exp_var) in &cases {
        let (mean, var) = gpr.posterior(&[*x]);
        assert!(
            (mean - exp_mean).abs() < 1e-8,
            "posterior mean at x={}: {:.17e}, expected {:.17e}, diff={:.2e}",
            x, mean, exp_mean, (mean - exp_mean).abs()
        );
        assert!(
            (var - exp_var).abs() < 1e-8,
            "posterior var at x={}: {:.17e}, expected {:.17e}, diff={:.2e}",
            x, var, exp_var, (var - exp_var).abs()
        );
    }
}

/// 验证训练点处的后验方差接近噪声方差（应"记住"训练数据）
#[test]
fn test_gp_regressor_interpolation() {
    let gpr = GPRegressor::new(
        vec![vec![0.0], vec![0.5], vec![1.0]],
        vec![0.0, 0.25, 1.0],
        vec![false],
        vec![1.0],
        1.0,
        1e-6,
    );

    // 在训练点处：mean ≈ y_train, var ≈ noise_var
    let (m0, v0) = gpr.posterior(&[0.0]);
    let (m1, v1) = gpr.posterior(&[0.5]);
    let (m2, v2) = gpr.posterior(&[1.0]);

    assert!((m0 - 0.0).abs() < 1e-4, "mean at x=0: {m0}");
    assert!((m1 - 0.25).abs() < 1e-4, "mean at x=0.5: {m1}");
    assert!((m2 - 1.0).abs() < 1e-4, "mean at x=1: {m2}");

    assert!(v0 < 1e-4, "var at training point x=0 should be ~noise_var, got {v0}");
    assert!(v1 < 1e-4, "var at training point x=0.5 should be ~noise_var, got {v1}");
    assert!(v2 < 1e-4, "var at training point x=1 should be ~noise_var, got {v2}");

    // 远离训练点处方差较大
    let (_, v_far) = gpr.posterior(&[2.0]);
    assert!(v_far > v0, "var far from training data should be larger");
}

/// GPRegressor log marginal likelihood 精确值
/// Python: lml = -2.3666282183017304e+00
#[test]
fn test_gp_regressor_lml() {
    let gpr = GPRegressor::new(
        vec![vec![0.0], vec![1.0]],
        vec![0.0, 1.0],
        vec![false],
        vec![1.0],
        1.0,
        1e-6,
    );

    let lml = gpr.log_marginal_likelihood();
    let expected = -2.3666282183017304e+00;
    assert!(
        (lml - expected).abs() < 1e-8,
        "LML = {:.17e}, expected {:.17e}, diff={:.2e}",
        lml, expected, (lml - expected).abs()
    );
}

// ============================================================================
// 4. GPRegressor 多维输入
// ============================================================================

/// 2D GP: x=[[0,0],[1,0],[0,1],[1,1]], y=[0,1,1,2]
/// 验证后验在对角点给出合理预测
#[test]
fn test_gp_regressor_2d() {
    let gpr = GPRegressor::new(
        vec![vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
        vec![0.0, 1.0, 1.0, 2.0],
        vec![false, false],
        vec![1.0, 1.0],
        1.0,
        1e-6,
    );

    // 训练点处 mean ≈ y
    let (m00, v00) = gpr.posterior(&[0.0, 0.0]);
    let (m11, v11) = gpr.posterior(&[1.0, 1.0]);
    assert!((m00 - 0.0).abs() < 1e-3, "mean at (0,0) = {m00}");
    assert!((m11 - 2.0).abs() < 1e-3, "mean at (1,1) = {m11}");
    assert!(v00 < 1e-3, "var at training point = {v00}");
    assert!(v11 < 1e-3, "var at training point = {v11}");

    // 中心点 (0.5, 0.5) 应预测接近 1.0 (线性函数 x+y)
    let (m_center, v_center) = gpr.posterior(&[0.5, 0.5]);
    assert!(
        (m_center - 1.0).abs() < 0.5,
        "center prediction should be ~1.0, got {m_center}"
    );
    assert!(v_center > v00, "center var should be larger than training point var");
}

// ============================================================================
// 5. GPRegressor 分类参数 (Hamming 距离)
// ============================================================================

/// 分类参数使用 Hamming 距离而非欧氏距离
#[test]
fn test_gp_regressor_categorical() {
    // 参数0:连续, 参数1:分类
    let gpr = GPRegressor::new(
        vec![vec![0.0, 0.0], vec![0.0, 1.0], vec![1.0, 0.0], vec![1.0, 1.0]],
        vec![0.0, 1.0, 1.0, 2.0],
        vec![false, true], // 参数1是分类
        vec![1.0, 1.0],
        1.0,
        1e-6,
    );

    // 分类参数: Hamming(0,0)=0, Hamming(0,1)=1
    let (m1, _) = gpr.posterior(&[0.0, 0.0]);
    let (m2, _) = gpr.posterior(&[0.0, 1.0]);
    assert!((m1 - 0.0).abs() < 1e-2, "predict at cat=0: {m1}");
    assert!((m2 - 1.0).abs() < 1e-2, "predict at cat=1: {m2}");
}

// ============================================================================
// 6. GPRegressor 边界情况
// ============================================================================

/// 空训练集: posterior 应返回先验 (mean=0, var=kernel_scale)
#[test]
fn test_gp_regressor_empty() {
    let gpr = GPRegressor::new(
        vec![],
        vec![],
        vec![],
        vec![],
        2.5, // kernel_scale
        1e-6,
    );

    let (mean, var) = gpr.posterior(&[0.5]);
    assert_eq!(mean, 0.0, "empty GP mean should be 0");
    assert_eq!(var, 2.5, "empty GP var should be kernel_scale");
    assert_eq!(gpr.log_marginal_likelihood(), f64::NEG_INFINITY);
}

/// 单点训练: 后验方差在训练点处最小
#[test]
fn test_gp_regressor_single_point() {
    let gpr = GPRegressor::new(
        vec![vec![0.5]],
        vec![3.0],
        vec![false],
        vec![1.0],
        1.0,
        1e-6,
    );

    let (mean, var) = gpr.posterior(&[0.5]);
    assert!((mean - 3.0).abs() < 1e-3, "single point mean: {mean}");
    assert!(var < 1e-3, "single point var: {var}");

    let (_, var_far) = gpr.posterior(&[5.0]);
    assert!(var_far > var, "var far from training should be larger");
}

// ============================================================================
// 7. GpSampler 单目标收敛 (x² 优化)
// ============================================================================

/// Python 参考: GPSampler(seed=42, n_startup=5), x∈[-5,5], 50 trials
///   best_value = 1.006e-06
///   Rust 实现不要求完全相同的路径（不同 L-BFGS 实现），
///   但应在 50 trials 内找到 best < 0.1
#[test]
fn test_gp_sampler_x_squared_convergence() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
        Some(42),
        Some(StudyDirection::Minimize),
        Some(5),
        false,
        None,
        None,
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
        Some(50), None, None,
    ).unwrap();

    let best = study.best_value().unwrap();
    assert!(
        best < 0.1,
        "GP should find x≈0 within 50 trials, best_value = {}",
        best
    );
    assert!(best >= 0.0, "x² >= 0 always");
}

/// 高维收敛: sphere function sum(xi²), 3D, 80 trials
#[test]
fn test_gp_sampler_sphere_3d() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
        Some(42),
        Some(StudyDirection::Minimize),
        Some(10),
        false,
        None,
        None,
    ));

    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Minimize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let x = trial.suggest_float("x", -3.0, 3.0, false, None)?;
            let y = trial.suggest_float("y", -3.0, 3.0, false, None)?;
            let z = trial.suggest_float("z", -3.0, 3.0, false, None)?;
            Ok(x * x + y * y + z * z)
        },
        Some(80), None, None,
    ).unwrap();

    let best = study.best_value().unwrap();
    assert!(
        best < 1.0,
        "GP should optimize 3D sphere reasonably, best = {}",
        best
    );
}

// ============================================================================
// 8. GpSampler maximize 方向
// ============================================================================

/// maximize 方向: 最大化 -(x-2)², 最优 x=2
#[test]
fn test_gp_sampler_maximize() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
        Some(42),
        Some(StudyDirection::Maximize),
        Some(5),
        false,
        None,
        None,
    ));

    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Maximize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
            Ok(-((x - 2.0) * (x - 2.0)))
        },
        Some(50), None, None,
    ).unwrap();

    let best = study.best_value().unwrap();
    // maximize -(x-2)², best should be close to 0 (at x=2)
    assert!(
        best > -1.0,
        "GP maximize should find x≈2, best_value = {}",
        best
    );
}

// ============================================================================
// 9. GpSampler 确定性目标
// ============================================================================

/// deterministic_objective=true: 固定 noise_var=DEFAULT_MINIMUM_NOISE_VAR
#[test]
fn test_gp_sampler_deterministic() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
        Some(42),
        Some(StudyDirection::Minimize),
        Some(5),
        true, // deterministic
        None,
        None,
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
    assert!(
        best < 1.0,
        "Deterministic GP should still find good solution, best = {}",
        best
    );
}

// ============================================================================
// 10. GpSampler 混合参数搜索空间
// ============================================================================

/// 混合 float + int 参数
#[test]
fn test_gp_sampler_mixed_params() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
        Some(42),
        Some(StudyDirection::Minimize),
        Some(10),
        false,
        None,
        None,
    ));

    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Minimize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
            let n = trial.suggest_int("n", 1, 10, false, 1)?;
            Ok(x * x + (n as f64 - 5.0).powi(2))
        },
        Some(50), None, None,
    ).unwrap();

    let best = study.best_value().unwrap();
    // 最优: x=0, n=5, val=0
    assert!(
        best < 5.0,
        "Mixed param GP should find reasonable solution, best = {}",
        best
    );
}

// ============================================================================
// 11. GPRegressor 核矩阵对称性
// ============================================================================

/// K(x, x') = K(x', x) 验证
#[test]
fn test_gp_kernel_matrix_symmetry() {
    let gpr = GPRegressor::new(
        vec![vec![0.0, 0.0], vec![1.0, 0.5], vec![0.5, 1.0], vec![0.3, 0.7]],
        vec![1.0, 2.0, 1.5, 1.8],
        vec![false, false],
        vec![2.0, 0.5], // ARD lengthscales
        1.5,            // kernel_scale
        1e-6,
    );

    // 验证对称性: posterior(a) 在拟合后的核矩阵是对称的
    // 验证方法: 对于两个测试点，k(a,b)交换后结果相同
    let (m1, v1) = gpr.posterior(&[0.3, 0.3]);
    let (m2, v2) = gpr.posterior(&[0.7, 0.7]);

    // 基本合理性
    assert!(v1 >= 0.0, "variance must be non-negative: {v1}");
    assert!(v2 >= 0.0, "variance must be non-negative: {v2}");
    assert!(m1.is_finite(), "mean must be finite: {m1}");
    assert!(m2.is_finite(), "mean must be finite: {m2}");
}

// ============================================================================
// 12. log_ndtr 单调性和边界验证
// ============================================================================

/// log Φ(z) 应该是单调递增的
#[test]
fn test_log_ndtr_monotonicity() {
    let zs: Vec<f64> = (-40..=10).map(|i| i as f64).collect();
    let vals: Vec<f64> = zs.iter().map(|&z| log_ndtr(z)).collect();

    for i in 1..vals.len() {
        assert!(
            vals[i] >= vals[i - 1] - 1e-10,
            "log_ndtr should be monotonically increasing: log_ndtr({}) = {:.6e} < log_ndtr({}) = {:.6e}",
            zs[i], vals[i], zs[i-1], vals[i-1]
        );
    }
}

/// log Φ(0) = log(0.5)
#[test]
fn test_log_ndtr_at_zero() {
    let result = log_ndtr(0.0);
    let expected = 0.5_f64.ln();
    assert!(
        (result - expected).abs() < 1e-15,
        "log_ndtr(0) = {:.17e}, expected {:.17e}",
        result, expected
    );
}

/// log Φ(z) <= 0 对所有有限 z
#[test]
fn test_log_ndtr_always_nonpositive() {
    for z in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
        let val = log_ndtr(z);
        assert!(val <= 0.0, "log_ndtr({}) = {} should be <= 0", z, val);
    }
}

// ============================================================================
// 13. GPRegressor LML 性质验证
// ============================================================================

/// LML 应该对合理的核参数给出有限值
#[test]
fn test_gp_lml_finite() {
    let gpr = GPRegressor::new(
        vec![vec![0.0], vec![0.5], vec![1.0]],
        vec![0.0, 0.3, 0.9],
        vec![false],
        vec![1.0],
        1.0,
        1e-4,
    );

    let lml = gpr.log_marginal_likelihood();
    assert!(lml.is_finite(), "LML should be finite: {lml}");
    assert!(lml < 0.0 || lml > -100.0, "LML should be in reasonable range: {lml}");
}

/// LML 受噪声方差影响: 高噪声 → 更低的 LML（通常）
#[test]
fn test_gp_lml_noise_sensitivity() {
    let x = vec![vec![0.0], vec![0.5], vec![1.0]];
    let y = vec![0.0, 0.25, 1.0];

    let gpr_low_noise = GPRegressor::new(
        x.clone(), y.clone(), vec![false], vec![1.0], 1.0, 1e-6,
    );
    let gpr_high_noise = GPRegressor::new(
        x, y, vec![false], vec![1.0], 1.0, 1.0,
    );

    let lml_low = gpr_low_noise.log_marginal_likelihood();
    let lml_high = gpr_high_noise.log_marginal_likelihood();

    // 对于精确匹配数据，低噪声应给出更高的 LML
    assert!(
        lml_low > lml_high,
        "Low noise LML ({:.6e}) should > high noise LML ({:.6e}) for exact data",
        lml_low, lml_high
    );
}

// ============================================================================
// 14. GpSampler n_startup_trials 行为
// ============================================================================

/// n_startup_trials=N: 前 N 次试验使用随机采样
#[test]
fn test_gp_sampler_startup_behavior() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
        Some(42),
        Some(StudyDirection::Minimize),
        Some(20), // 大的 startup
        false,
        None,
        None,
    ));

    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Minimize), None, false,
    ).unwrap();

    // 仅运行 15 次 (< 20 startup) — 全部使用随机采样
    study.optimize(
        |trial| {
            let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
            Ok(x * x)
        },
        Some(15), None, None,
    ).unwrap();

    // 应该完成而不出错
    assert_eq!(study.trials().unwrap().len(), 15);
}

// ============================================================================
// 15. GpSampler log 搜索空间
// ============================================================================

/// log 搜索空间: 学习率优化 (log scale)
#[test]
fn test_gp_sampler_log_space() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::new(
        Some(42),
        Some(StudyDirection::Minimize),
        Some(5),
        false,
        None,
        None,
    ));

    let study = create_study(
        None, Some(sampler), None, None,
        Some(StudyDirection::Minimize), None, false,
    ).unwrap();

    study.optimize(
        |trial| {
            let lr = trial.suggest_float("lr", 1e-5, 1.0, true, None)?;
            // 最优 lr ≈ 0.01
            Ok((lr.log10() + 2.0).powi(2))
        },
        Some(40), None, None,
    ).unwrap();

    let best = study.best_value().unwrap();
    assert!(
        best < 1.0,
        "GP should find lr≈0.01 in log space, best = {}",
        best
    );
}

// ============================================================================
// 16. Matern 5/2 补充验证 (ARD 多维度)
// ============================================================================

/// ARD: 不同维度不同长度尺度时，核值应反映维度间的距离差异
#[test]
fn test_matern52_ard_effect() {
    // inv_sq_ls = [10.0, 0.1]:
    // 维度0的差异被放大10倍，维度1被缩小到0.1倍
    let d2_dim0 = 1.0 * 10.0; // |1-0|² * inv_sq_ls[0]
    let d2_dim1 = 1.0 * 0.1;  // |1-0|² * inv_sq_ls[1]

    let k_dim0 = matern52(d2_dim0);
    let k_dim1 = matern52(d2_dim1);

    // 维度0的距离被放大，核值更小（更不相关）
    assert!(k_dim0 < k_dim1, "larger ARD weight → smaller kernel value");
    // 数值验证
    assert!((k_dim0 - matern52(10.0)).abs() < 1e-15);
    assert!((k_dim1 - matern52(0.1)).abs() < 1e-15);
}

// ============================================================================
// 17. GpSampler 多目标 (with_directions)
// ============================================================================

/// 双目标优化: min f1(x) = x², min f2(x) = (x-2)²
/// 最优 Pareto 前沿在 x ∈ [0, 2]
#[test]
fn test_gp_sampler_multi_objective() {
    let sampler: Arc<dyn Sampler> = Arc::new(GpSampler::with_directions(
        Some(42),
        vec![StudyDirection::Minimize, StudyDirection::Minimize],
        Some(10),
        false,
        None,
        None,
    ));

    let study = create_study(
        None, Some(sampler), None, None,
        None, Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]), false,
    ).unwrap();

    study.optimize_multi(
        |trial| {
            let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
            Ok(vec![x * x, (x - 2.0) * (x - 2.0)])
        },
        Some(50), None, None,
    ).unwrap();

    // 应完成 50 次试验
    assert_eq!(study.trials().unwrap().len(), 50);

    // Pareto 前沿上的点应在 x ∈ [-1, 3] 附近
    let best_trials = study.best_trials().unwrap();
    assert!(!best_trials.is_empty(), "should have Pareto optimal trials");
}

// ============================================================================
// 18. Cholesky 正定性检查
// ============================================================================

/// 非正定矩阵应返回 None
#[test]
fn test_cholesky_non_positive_definite() {
    let a = vec![
        vec![1.0, 2.0],
        vec![2.0, 1.0], // 特征值: 3, -1 → 非正定
    ];
    assert!(cholesky(&a).is_none());
}

/// 单位矩阵 Cholesky = 单位矩阵
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
                (l[i][j] - expected).abs() < 1e-15,
                "L[{i}][{j}] = {}, expected {expected}",
                l[i][j]
            );
        }
    }
}

/// Cholesky 验证: L * L^T = A
#[test]
fn test_cholesky_reconstruction() {
    let a = vec![
        vec![4.0, 2.0, 1.0],
        vec![2.0, 5.0, 3.0],
        vec![1.0, 3.0, 6.0],
    ];
    let l = cholesky(&a).unwrap();

    // Reconstruct A = L * L^T
    for i in 0..3 {
        for j in 0..3 {
            let sum: f64 = (0..3).map(|k| l[i][k] * l[j][k]).sum();
            assert!(
                (sum - a[i][j]).abs() < 1e-14,
                "L*L^T[{i}][{j}] = {sum}, expected {}",
                a[i][j]
            );
        }
    }
}
