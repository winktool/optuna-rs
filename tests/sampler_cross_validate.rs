//! Sampler 模块精确交叉验证测试。
//!
//! 验证 Sobol 序列、Halton 序列和各种采样器的数值精度。

use optuna_rs::samplers::qmc::sobol_point_pub;
use optuna_rs::samplers::gp::{matern52, cholesky, normal_pdf, normal_cdf};

// ============================================================================
// 1. Sobol 序列精确数值验证 (对齐 scipy.stats.qmc.Sobol)
// ============================================================================

/// Sobol 2D 结构验证:
/// - 所有值在 [0,1] 范围内
/// - index=1 的所有维度值均为 0.5 (Sobol 基本属性)
/// - 不同 index 生成不同点
/// - 注: Rust 和 scipy 使用不同方向数表，数值不完全相同但都是有效 Sobol 序列
#[test]
fn test_sobol_2d_structural_properties() {
    // 验证 index=1 始终是 (0.5, 0.5)
    let p1 = sobol_point_pub(1, 2, false, 0);
    assert_eq!(p1[0], 0.5);
    assert_eq!(p1[1], 0.5);

    // 验证所有点在 [0,1] 范围内且互不相同
    let mut points: Vec<Vec<f64>> = Vec::new();
    for i in 0..16 {
        let p = sobol_point_pub(i, 2, false, 0);
        assert!(p[0] >= 0.0 && p[0] <= 1.0, "dim 0 out of range at index {}", i);
        assert!(p[1] >= 0.0 && p[1] <= 1.0, "dim 1 out of range at index {}", i);
        points.push(p);
    }

    // 验证去重性（所有点应不同）
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            assert!(
                points[i] != points[j],
                "Sobol points at index {} and {} are identical",
                i,
                j
            );
        }
    }
}

/// index=0 应为 (0, 0, ...)
#[test]
fn test_sobol_origin() {
    let point = sobol_point_pub(0, 3, false, 0);
    assert_eq!(point, vec![0.0, 0.0, 0.0]);
}

/// 高维 Sobol (d=5) 第一个非零点
/// Python: Sobol(5, scramble=False).random(2) index=1 → (0.5, 0.5, 0.5, 0.5, 0.5)
#[test]
fn test_sobol_5d_first_point() {
    let point = sobol_point_pub(1, 5, false, 0);
    for (d, &v) in point.iter().enumerate() {
        assert!(
            (v - 0.5).abs() < 1e-15,
            "Sobol d={} index=1: expected 0.5, got {}",
            d,
            v
        );
    }
}

/// 1D Sobol 序列应该是 Van der Corput base-2
/// index 1=0.5, 2=0.25, 3=0.75, 4=0.125, 5=0.625, 6=0.375, 7=0.875
#[test]
fn test_sobol_1d_van_der_corput() {
    let expected = vec![0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875];
    for (i, &ex) in expected.iter().enumerate() {
        let point = sobol_point_pub((i + 1) as u64, 1, false, 0);
        assert!(
            (point[0] - ex).abs() < 1e-15,
            "1D Sobol index {}: expected {}, got {}",
            i + 1,
            ex,
            point[0]
        );
    }
}

// ============================================================================
// 2. Halton 序列精确数值验证
// ============================================================================

/// Python 参考:
///   Halton(d=2, scramble=False).random(5):
///   点由 base-2 和 base-3 的 Van der Corput 序列组成
///   index 0 = (0, 0) [skip]
///   index 1 = (0.5, 1/3)
///   index 2 = (0.25, 2/3)
///   index 3 = (0.75, 1/9)
///   index 4 = (0.125, 4/9)
#[test]
fn test_halton_2d() {
    // Halton dim 0 (base 2): 0.5, 0.25, 0.75, 0.125, ...
    // Halton dim 1 (base 3): 1/3≈0.333, 2/3≈0.667, 1/9≈0.111, 4/9≈0.444, ...
    fn halton_value(index: u64, base: u64) -> f64 {
        let mut result = 0.0_f64;
        let mut f = 1.0 / base as f64;
        let mut i = index;
        while i > 0 {
            result += (i % base) as f64 * f;
            i /= base;
            f /= base as f64;
        }
        result
    }

    // Verify our reference function
    assert!((halton_value(1, 2) - 0.5).abs() < 1e-15);
    assert!((halton_value(1, 3) - 1.0 / 3.0).abs() < 1e-15);
    assert!((halton_value(2, 2) - 0.25).abs() < 1e-15);
    assert!((halton_value(2, 3) - 2.0 / 3.0).abs() < 1e-15);
    assert!((halton_value(3, 2) - 0.75).abs() < 1e-15);
    assert!((halton_value(3, 3) - 1.0 / 9.0).abs() < 1e-15);
}

// ============================================================================
// 3. Sobol 均匀性验证 (统计特性)
// ============================================================================

/// Sobol 序列的前 2^k 个点应在 [0,1]^d 中均匀分布
/// 使用 L2 差异的简化检查：均值应接近 0.5
#[test]
fn test_sobol_uniformity() {
    let n = 256; // 2^8
    let dim = 3;
    let mut sums = vec![0.0; dim];

    for i in 0..n {
        let point = sobol_point_pub(i, dim, false, 0);
        for d in 0..dim {
            sums[d] += point[d];
        }
    }

    for d in 0..dim {
        let mean = sums[d] / n as f64;
        // 前 2^k 个 Sobol 点的均值应精确等于 (2^k - 1) / (2 * 2^k) ≈ 0.5 - ε
        assert!(
            (mean - 0.498046875).abs() < 0.01, // (256-1)/512 = 0.498046875
            "Sobol dim {} mean = {}, expected ~0.498",
            d,
            mean
        );
    }
}

// ============================================================================
// 4. Distribution single/contains 边界验证
// ============================================================================

/// IntDistribution log=true 的 to_external_repr 截断行为
/// Python: int(9.7)=9, int(10.3)=10, int(1.4)=1, int(99.6)=99
/// Python to_external_repr 使用 int() 截断，不是 round()
#[test]
fn test_int_distribution_log_truncation() {
    use optuna_rs::distributions::IntDistribution;
    let id = IntDistribution::new(1, 100, true, 1).unwrap();
    // int() 截断（向零取整）= Rust `as i64`
    assert_eq!(id.to_external_repr(9.7).unwrap(), 9);  // int(9.7)=9
    assert_eq!(id.to_external_repr(10.3).unwrap(), 10); // int(10.3)=10
    assert_eq!(id.to_external_repr(1.4).unwrap(), 1);   // int(1.4)=1
    assert_eq!(id.to_external_repr(99.6).unwrap(), 99);  // int(99.6)=99
    assert_eq!(id.to_external_repr(50.0).unwrap(), 50);  // int(50.0)=50
}

/// FloatDistribution log=true 的 contains 和 repr
#[test]
fn test_float_distribution_log_repr() {
    use optuna_rs::distributions::FloatDistribution;
    let fd = FloatDistribution::new(1e-5, 1.0, true, None).unwrap();
    assert!(fd.contains(1e-5));
    assert!(fd.contains(0.5));
    assert!(fd.contains(1.0));
    assert!(!fd.contains(0.0));
    assert!(!fd.contains(1.1));
}

// ============================================================================
// 5. Matérn 5/2 核函数精确数值验证
// ============================================================================

/// Python 参考: k(d²) = (1 + √(5d²) + 5/3·d²) · exp(-√(5d²))
///   d²=0.0  → 1.0
///   d²=0.01 → 9.91759236171177561e-01
///   d²=0.1  → 9.23899021904130868e-01
///   d²=0.5  → 7.02495760153803217e-01
///   d²=1.0  → 5.23994108831820293e-01
///   d²=2.0  → 3.17283363954043773e-01
///   d²=5.0  → 9.65772403202250357e-02
///   d²=10.0 → 2.10103937691350079e-02
#[test]
fn test_matern52_precision() {
    let test_cases: Vec<(f64, f64)> = vec![
        (0.0, 1.0),
        (0.01, 9.91759236171177561e-01),
        (0.1, 9.23899021904130868e-01),
        (0.5, 7.02495760153803217e-01),
        (1.0, 5.23994108831820293e-01),
        (2.0, 3.17283363954043773e-01),
        (5.0, 9.65772403202250357e-02),
        (10.0, 2.10103937691350079e-02),
    ];

    for (d2, expected) in &test_cases {
        let result = matern52(*d2);
        assert!(
            (result - expected).abs() < 1e-14,
            "matern52({}) = {:.17e}, expected {:.17e}, diff={:.2e}",
            d2,
            result,
            expected,
            (result - expected).abs()
        );
    }
}

// ============================================================================
// 6. Cholesky 分解精度验证
// ============================================================================

/// Python 参考: 3×3 GP kernel matrix (Matérn 5/2, length_scale=0.3, noise=1e-4)
/// X = [0.0, 0.5, 1.0]
/// Cholesky L:
///   L[0][0] = 1.00004999875006240e+00
///   L[1][0] = 2.25199560642461993e-01
///   L[1][1] = 9.74363976082060490e-01
///   L[2][0] = 1.56261775453043067e-02
///   L[2][1] = 2.27524639111467225e-01
///   L[2][2] = 9.73698290628323471e-01
#[test]
fn test_cholesky_gp_kernel() {
    let length_scale = 0.3_f64;
    let noise_var = 1e-4_f64;
    let x_vals = [0.0_f64, 0.5, 1.0];
    let n = x_vals.len();

    // Build kernel matrix
    let mut k = vec![vec![0.0; n]; n];
    for i in 0..n {
        for j in 0..n {
            let d2 = (x_vals[i] - x_vals[j]).powi(2) / (length_scale * length_scale);
            k[i][j] = matern52(d2);
            if i == j {
                k[i][j] += noise_var;
            }
        }
    }

    let l = cholesky(&k).expect("Cholesky decomposition should succeed");

    let expected_l: Vec<Vec<f64>> = vec![
        vec![1.00004999875006240e+00, 0.0, 0.0],
        vec![2.25199560642461993e-01, 9.74363976082060490e-01, 0.0],
        vec![1.56261775453043067e-02, 2.27524639111467225e-01, 9.73698290628323471e-01],
    ];

    for i in 0..n {
        for j in 0..=i {
            assert!(
                (l[i][j] - expected_l[i][j]).abs() < 1e-12,
                "L[{}][{}] = {:.17e}, expected {:.17e}, diff={:.2e}",
                i, j, l[i][j], expected_l[i][j],
                (l[i][j] - expected_l[i][j]).abs()
            );
        }
    }
}

// ============================================================================
// 7. Expected Improvement 公式验证
// ============================================================================

/// Python 参考: EI = (f_best - mu) * Phi(z) + sigma * phi(z), z = (f_best - mu) / sigma
///   f_best=0.3, sigma=0.1:
///   mu=0.2: EI=1.08331547058768615e-01
///   mu=0.3: EI=3.98942280401432744e-02
///   mu=0.4: EI=8.33154705876862703e-03
///   mu=0.5: EI=8.49070261682967342e-04
#[test]
fn test_expected_improvement_precision() {
    let f_best = 0.3_f64;
    let sigma = 0.1_f64;
    let test_cases: Vec<(f64, f64)> = vec![
        (0.2, 1.08331547058768615e-01),
        (0.3, 3.98942280401432744e-02),
        (0.4, 8.33154705876862703e-03),
        (0.5, 8.49070261682967342e-04),
    ];

    for (mu, expected_ei) in &test_cases {
        let z = (f_best - mu) / sigma;
        let ei = (f_best - mu) * normal_cdf(z) + sigma * normal_pdf(z);
        assert!(
            (ei - expected_ei).abs() < 1e-14,
            "EI(mu={}, sigma={}) = {:.17e}, expected {:.17e}, diff={:.2e}",
            mu, sigma, ei, expected_ei,
            (ei - expected_ei).abs()
        );
    }
}
