/// GP 端到端精确交叉验证测试
/// Golden values 来源: Python numpy (tests/golden_gp_values.py)
///
/// 覆盖:
///   - Matern52 核函数 (9 距离点)
///   - 1D GP 后验 (mean, variance, 7 测试点)
///   - 2D GP 后验 (mean, variance, 4 测试点)
///   - GP 边际对数似然 (1D + 2D)
///   - default_log_prior (4 参数组合)
///   - 多变量 PE sigma 公式 (5 case)
///   - GP with categorical 混合参数
///   - log_ndtr (12 z 值)
///   - erfcx (8 x 值)
///   - logEI (6 案例)

use optuna_rs::samplers::gp::{
    matern52, cholesky, solve_lower, solve_upper,
    GPRegressor, default_log_prior, log_ndtr, log_ei, erfcx,
};

// ═══════════════════════════════════════════════════════════════
// Matern52 核函数精确测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_matern52_golden_values() {
    // d², expected (from Python)
    let cases: Vec<(f64, f64)> = vec![
        (0.0,   1.0000000000000000e+00),
        (0.01,  9.9175923617117756e-01),
        (0.1,   9.2389902190413087e-01),
        (0.5,   7.0249576015380322e-01),
        (1.0,   5.2399410883182029e-01),
        (2.0,   3.1728336395404377e-01),
        (5.0,   9.6577240320225036e-02),
        (10.0,  2.1010393769135008e-02),
        (100.0, 3.6956962220528686e-08),
    ];
    for (d2, expected) in cases {
        let actual = matern52(d2);
        let rel = if expected.abs() > 1e-15 {
            ((actual - expected) / expected).abs()
        } else {
            (actual - expected).abs()
        };
        assert!(
            rel < 1e-12,
            "matern52({}) = {}, expected {}, rel err = {}",
            d2, actual, expected, rel
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// 1D GP 后验精确测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_gp_posterior_1d_golden() {
    let x_train = vec![vec![0.2], vec![0.5], vec![0.8]];
    let y_train = vec![1.0, -0.5, 0.3];
    let inv_sq_ls = vec![1.0];
    let kernel_scale = 1.0;
    let noise_var = 0.01;
    let is_categorical = vec![false];

    let gpr = GPRegressor::new(
        x_train, y_train, is_categorical,
        inv_sq_ls, kernel_scale, noise_var,
    );

    // (x, expected_mean, expected_var) from Python
    let golden: Vec<(f64, f64, f64)> = vec![
        (0.00, 1.5611050283800978e+00, 4.5654550348492751e-02),
        (0.20, 8.0848108382485329e-01, 8.9958827032083910e-03),
        (0.35, 2.0508859056404383e-01, 6.8819228614136474e-03),
        (0.50, -1.6875719516501597e-01, 7.1935790153017098e-03),
        (0.65, -1.5749889738123812e-01, 6.8819228614136474e-03),
        (0.80, 1.3752589651988245e-01, 8.9958827032087241e-03),
        (1.00, 6.2141880303344976e-01, 4.5654550348493639e-02),
    ];

    for (x, exp_mean, exp_var) in golden {
        let (mean, var) = gpr.posterior(&[x]);
        let rel_mean = if exp_mean.abs() > 1e-10 {
            ((mean - exp_mean) / exp_mean).abs()
        } else {
            (mean - exp_mean).abs()
        };
        let rel_var = ((var - exp_var) / exp_var).abs();
        assert!(
            rel_mean < 1e-10,
            "posterior mean at x={}: got {}, expected {}, rel={}",
            x, mean, exp_mean, rel_mean
        );
        assert!(
            rel_var < 1e-10,
            "posterior var at x={}: got {}, expected {}, rel={}",
            x, var, exp_var, rel_var
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// 1D GP 边际对数似然精确测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_gp_lml_1d_golden() {
    let gpr = GPRegressor::new(
        vec![vec![0.2], vec![0.5], vec![0.8]],
        vec![1.0, -0.5, 0.3],
        vec![false],
        vec![1.0], 1.0, 0.01,
    );
    let lml = gpr.log_marginal_likelihood();
    let expected = -2.0960254425500437e+01;
    let rel = ((lml - expected) / expected).abs();
    assert!(
        rel < 1e-12,
        "LML: got {}, expected {}, rel={}",
        lml, expected, rel
    );
}

// ═══════════════════════════════════════════════════════════════
// 2D GP 后验精确测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_gp_posterior_2d_golden() {
    let x_train = vec![
        vec![0.1, 0.3],
        vec![0.4, 0.6],
        vec![0.7, 0.2],
        vec![0.9, 0.8],
    ];
    let y_train = vec![0.5, -1.0, 0.8, -0.3];
    let inv_sq_ls = vec![2.0, 0.5];
    let kernel_scale = 1.5;
    let noise_var = 0.02;
    let is_categorical = vec![false, false];

    let gpr = GPRegressor::new(
        x_train, y_train, is_categorical,
        inv_sq_ls, kernel_scale, noise_var,
    );

    let golden: Vec<(Vec<f64>, f64, f64)> = vec![
        (vec![0.0, 0.0], 1.0874924161585806e+00, 1.2326108765582355e-01),
        (vec![0.5, 0.5], -5.2658994891958777e-01, 2.4449152529324003e-02),
        (vec![0.25, 0.45], -2.8466699189962874e-01, 2.2594043616262605e-02),
        (vec![1.0, 1.0], -4.1081032758964220e-01, 8.0957529375710990e-02),
    ];

    for (x, exp_mean, exp_var) in golden {
        let (mean, var) = gpr.posterior(&x);
        let rel_mean = if exp_mean.abs() > 1e-10 {
            ((mean - exp_mean) / exp_mean).abs()
        } else {
            (mean - exp_mean).abs()
        };
        let rel_var = ((var - exp_var) / exp_var).abs();
        assert!(
            rel_mean < 1e-10,
            "2D posterior mean at x={:?}: got {}, expected {}, rel={}",
            x, mean, exp_mean, rel_mean
        );
        assert!(
            rel_var < 1e-10,
            "2D posterior var at x={:?}: got {}, expected {}, rel={}",
            x, var, exp_var, rel_var
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// 2D GP LML 精确测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_gp_lml_2d_golden() {
    let gpr = GPRegressor::new(
        vec![
            vec![0.1, 0.3], vec![0.4, 0.6],
            vec![0.7, 0.2], vec![0.9, 0.8],
        ],
        vec![0.5, -1.0, 0.8, -0.3],
        vec![false, false],
        vec![2.0, 0.5], 1.5, 0.02,
    );
    let lml = gpr.log_marginal_likelihood();
    let expected = -8.8160202231802707e+00;
    let rel = ((lml - expected) / expected).abs();
    assert!(
        rel < 1e-12,
        "2D LML: got {}, expected {}, rel={}",
        lml, expected, rel
    );
}

// ═══════════════════════════════════════════════════════════════
// default_log_prior 精确测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_default_log_prior_golden() {
    let cases = vec![
        (vec![1.0],           1.0, 0.01,  -1.9605170185988090e+00),
        (vec![0.5, 2.0],      1.5, 0.001, -2.3153104197900491e+00),
        (vec![10.0],          0.1, 0.1,   -6.6428436022934498e+00),
        (vec![0.1, 0.1, 0.1], 5.0, 0.05,  -8.2201353149212988e+00),
    ];

    for (ils, ks, nv, expected) in cases {
        let actual = default_log_prior(&ils, ks, nv);
        let rel = ((actual - expected) / expected).abs();
        assert!(
            rel < 1e-12,
            "log_prior(ils={:?}, ks={}, nv={}) = {}, expected {}, rel={}",
            ils, ks, nv, actual, expected, rel
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// GP with categorical (mixed) 精确测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_gp_posterior_categorical_golden() {
    let x_train = vec![
        vec![0.2, 0.0],
        vec![0.5, 1.0],
        vec![0.8, 0.0],
    ];
    let y_train = vec![1.0, -0.5, 0.3];
    let inv_sq_ls = vec![1.0, 1.0];
    let kernel_scale = 1.0;
    let noise_var = 0.01;
    let is_categorical = vec![false, true]; // 第2维为 categorical

    let gpr = GPRegressor::new(
        x_train, y_train, is_categorical,
        inv_sq_ls, kernel_scale, noise_var,
    );

    // x=[0.3, 0.0] (same cat as obs 0,2)
    let (m1, v1) = gpr.posterior(&[0.3, 0.0]);
    let exp_m1 = 8.9952192933278718e-01;
    let exp_v1 = 1.3404750186127634e-02;
    assert!( ((m1 - exp_m1)/exp_m1).abs() < 1e-10,
        "cat mean at [0.3,0.0]: {} vs {}", m1, exp_m1);
    assert!( ((v1 - exp_v1)/exp_v1).abs() < 1e-10,
        "cat var at [0.3,0.0]: {} vs {}", v1, exp_v1);

    // x=[0.3, 1.0] (same cat as obs 1)
    let (m2, v2) = gpr.posterior(&[0.3, 1.0]);
    let exp_m2 = -3.7187678389406620e-01;
    let exp_v2 = 6.3886929183215035e-02;
    assert!( ((m2 - exp_m2)/exp_m2).abs() < 1e-10,
        "cat mean at [0.3,1.0]: {} vs {}", m2, exp_m2);
    assert!( ((v2 - exp_v2)/exp_v2).abs() < 1e-10,
        "cat var at [0.3,1.0]: {} vs {}", v2, exp_v2);
}

// ═══════════════════════════════════════════════════════════════
// Multivariate PE sigma 公式精确测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_multivariate_pe_sigma_golden() {
    let sigma0_mag: f64 = 0.2;
    let cases: Vec<(usize, usize, f64, f64, f64)> = vec![
        // (n_mus, n_params, low, high, expected)
        (3,   2, 0.0, 10.0, 1.6653663553112086e+00),
        (5,   1, 0.0, 1.0,  1.4495593273553911e-01),
        (10,  3, -5.0, 5.0, 1.4393713460023041e+00),
        (1,   1, 0.0, 1.0,  2.0000000000000001e-01),
        (100, 4, 0.0, 1.0,  1.1246826503806982e-01),
    ];
    for (n_mus, n_params, low, high, expected) in cases {
        let sigma = sigma0_mag
            * (n_mus.max(1) as f64).powf(-1.0 / (n_params as f64 + 4.0))
            * (high - low);
        let rel = ((sigma - expected) / expected).abs();
        assert!(
            rel < 1e-14,
            "mv sigma(n_mus={}, n_params={}, [{},{}]) = {}, expected {}, rel={}",
            n_mus, n_params, low, high, sigma, expected, rel
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// Cholesky + solve 精确测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_cholesky_solve_golden() {
    // A = [[4, 2, 1], [2, 5, 3], [1, 3, 6]]
    let a = vec![
        vec![4.0, 2.0, 1.0],
        vec![2.0, 5.0, 3.0],
        vec![1.0, 3.0, 6.0],
    ];
    let l = cholesky(&a).expect("cholesky should succeed");
    let b = vec![1.0, 2.0, 3.0];

    // L x = b (forward solve)
    let x_lower = solve_lower(&l, &b);
    // L^T y = x_lower (backward solve) => y = A^{-1} b
    let y = solve_upper(&l, &x_lower);

    // Verify: A * y ≈ b
    for i in 0..3 {
        let sum: f64 = (0..3).map(|j| a[i][j] * y[j]).sum();
        assert!(
            (sum - b[i]).abs() < 1e-12,
            "A*y[{}] = {}, expected {}", i, sum, b[i]
        );
    }

    // Verify L * L^T ≈ A
    for i in 0..3 {
        for j in 0..3 {
            let llt: f64 = (0..3).map(|k| l[i][k] * l[j][k]).sum();
            assert!(
                (llt - a[i][j]).abs() < 1e-12,
                "LLT[{},{}] = {}, expected {}", i, j, llt, a[i][j]
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════
// GP 不变量测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_gp_posterior_variance_nonneg_invariant() {
    let gpr = GPRegressor::new(
        vec![vec![0.1], vec![0.3], vec![0.5], vec![0.7], vec![0.9]],
        vec![0.5, -1.0, 0.0, 0.8, -0.3],
        vec![false],
        vec![1.0], 1.0, 0.01,
    );
    // 检查 101 个点的方差都非负
    for i in 0..=100 {
        let x = i as f64 / 100.0;
        let (_, var) = gpr.posterior(&[x]);
        assert!(
            var >= 0.0,
            "variance at x={} was negative: {}", x, var
        );
    }
}

#[test]
fn test_gp_posterior_interpolation_invariant() {
    // 在训练点上，mean 应该接近 y_train，variance 应该接近 noise_var
    let gpr = GPRegressor::new(
        vec![vec![0.2], vec![0.5], vec![0.8]],
        vec![1.0, -0.5, 0.3],
        vec![false],
        vec![1.0], 1.0, 0.001, // very low noise
    );
    for (&x, &y) in [0.2, 0.5, 0.8].iter().zip([1.0, -0.5, 0.3].iter()) {
        let (mean, var) = gpr.posterior(&[x]);
        assert!(
            (mean - y).abs() < 0.1,
            "at x={}, mean={} should be near y={}", x, mean, y
        );
        assert!(
            var < 0.01,
            "at x={}, var={} should be near 0 (near training point)", x, var
        );
    }
}

#[test]
fn test_gp_symmetry_invariant() {
    // 对称数据 => 对称预测
    let gpr = GPRegressor::new(
        vec![vec![0.2], vec![0.8]],
        vec![1.0, 1.0], // symmetric y
        vec![false],
        vec![1.0], 1.0, 0.01,
    );
    let (m1, v1) = gpr.posterior(&[0.3]);
    let (m2, v2) = gpr.posterior(&[0.7]);
    assert!(
        (m1 - m2).abs() < 1e-10,
        "symmetric means: {} vs {}", m1, m2
    );
    assert!(
        (v1 - v2).abs() < 1e-10,
        "symmetric vars: {} vs {}", v1, v2
    );
}

// ═══════════════════════════════════════════════════════════════
// log_ndtr 精确测试 (对齐 torch.special.log_ndtr)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_log_ndtr_golden_full_range() {
    // (z, expected) from Python torch.special.log_ndtr
    let cases: Vec<(f64, f64)> = vec![
        (-50.0, -1.2548313611394201e+03),
        (-30.0, -4.5432124395634321e+02),
        (-26.0, -3.4217850892992794e+02),
        (-10.0, -5.3231285150512477e+01),
        ( -5.0, -1.5064998393988727e+01),
        ( -3.0, -6.6077262215103501e+00),
        ( -1.0, -1.8410216450092640e+00),
        (  0.0, -6.9314718055994529e-01),
        (  1.0, -1.7275377902344982e-01),
        (  3.0, -1.3508099647481920e-03),
        (  5.0, -2.8665161296376331e-07),
        ( 10.0, -7.6198530241604961e-24),
    ];
    for (z, expected) in cases {
        let actual = log_ndtr(z);
        // 对极端尾部（绝对值极小）使用宽松容差
        let tol = if expected.abs() < 1e-15 { 0.02 } else { 1e-6 };
        let rel = if expected.abs() > 1e-30 {
            ((actual - expected) / expected).abs()
        } else {
            (actual - expected).abs()
        };
        assert!(
            rel < tol,
            "log_ndtr({}) = {:.16e}, expected {:.16e}, rel={}",
            z, actual, expected, rel
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// erfcx 精确测试 (对齐 torch.special.erfcx)
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_erfcx_golden() {
    let cases: Vec<(f64, f64)> = vec![
        ( 0.0, 1.0000000000000000e+00),
        ( 0.5, 6.1569034419292579e-01),
        ( 1.0, 4.2758357615580700e-01),
        ( 5.0, 1.1070463773306861e-01),
        (10.0, 5.6140992743822588e-02),
        (25.0, 2.2549572432641357e-02),
        (30.0, 1.8795888861416754e-02),
        (50.0, 1.1281536265323772e-02),
    ];
    for (x, expected) in cases {
        let actual = erfcx(x);
        let rel = ((actual - expected) / expected).abs();
        assert!(
            rel < 1e-6,
            "erfcx({}) = {:.16e}, expected {:.16e}, rel={}",
            x, actual, expected, rel
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// logEI (acquisition function) 精确测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_log_ei_golden() {
    // (mean, var, f0, expected_logEI) from Python optuna._gp.acqf.standard_logei + log(sigma)
    let cases: Vec<(f64, f64, f64, f64)> = vec![
        ( 1.00, 0.250, 0.5, -6.1312096171063823e-01),
        ( 0.50, 1.000, 0.5, -9.1893853320467267e-01),
        (-1.00, 0.500, 0.0, -3.6838015353932638e+00),
        ( 0.00, 0.010, 5.0, -1.2610467679614544e+03),
        (10.00, 4.000, 5.0,  1.6102392461521098e+00),
        ( 0.10, 0.001, 0.1, -4.3728161726957406e+00),
    ];
    for (mean, var, f0, expected) in cases {
        let actual = log_ei(mean, var, f0);
        let rel = if expected.abs() > 1e-10 {
            ((actual - expected) / expected).abs()
        } else {
            (actual - expected).abs()
        };
        assert!(
            rel < 1e-4,
            "log_ei(mean={}, var={}, f0={}) = {:.16e}, expected {:.16e}, rel={}",
            mean, var, f0, actual, expected, rel
        );
    }
}

// ═══════════════════════════════════════════════════════════════
// logEI 不变量测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn test_log_ei_monotone_in_mean() {
    // 固定 var 和 f0, logEI 应随 mean 单调递增
    let var = 1.0;
    let f0 = 0.0;
    let means: Vec<f64> = (-50..=50).map(|i| i as f64 * 0.1).collect();
    let mut prev = f64::NEG_INFINITY;
    for mean in means {
        let lei = log_ei(mean, var, f0);
        assert!(
            lei >= prev - 1e-10,
            "logEI not monotone: mean={}, logEI={}, prev={}",
            mean, lei, prev
        );
        prev = lei;
    }
}

#[test]
fn test_log_ei_zero_var() {
    // var=0: logEI = log(max(mean-f0, 0))
    let lei_pos = log_ei(3.0, 0.0, 1.0);
    assert!(
        (lei_pos - (2.0_f64).ln()).abs() < 1e-10,
        "logEI(3,0,1) = {}, expected ln(2)={}", lei_pos, 2.0_f64.ln()
    );
    let lei_neg = log_ei(1.0, 0.0, 3.0);
    assert!(
        lei_neg == f64::NEG_INFINITY,
        "logEI(1,0,3) should be -inf, got {}", lei_neg
    );
}
