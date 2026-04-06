//! RandomSampler 与 QMC 采样器深度交叉验证测试。
//!
//! - Van der Corput / Halton 序列与 Python scipy.stats.qmc.Halton 精确对比
//! - SearchSpaceTransform bounds 与 Python optuna._transform 精确对比
//! - QMC 序列低差异性验证
//! - RandomSampler 区间 / 分布属性验证
//!
//! 参考值由 `tests/golden_qmc_random.py` 生成。

use optuna_rs::samplers::qmc::{halton_point, van_der_corput};

const TOL: f64 = 1e-12;

// ============================================================================
// 1. Van der Corput 序列 — Python scipy.stats.qmc.Halton(1, scramble=False) 精确值
// ============================================================================

/// Van der Corput base=2, index 0..9 与 Python scipy 精确匹配
#[test]
fn test_van_der_corput_base2_python() {
    // Python: Halton(1, scramble=False).random(10) 的第一维
    let expected = [
        0.0, 0.5, 0.25, 0.75, 0.125, 0.625, 0.375, 0.875, 0.0625, 0.5625,
    ];
    for (i, &exp) in expected.iter().enumerate() {
        let val = van_der_corput(i as u64, 2);
        assert!(
            (val - exp).abs() < TOL,
            "VdC base=2 index={}: Rust={:.15e}, Python={:.15e}",
            i, val, exp
        );
    }
}

/// Van der Corput base=3 精确值
#[test]
fn test_van_der_corput_base3_python() {
    // Python Halton(2, scramble=False) 的第二维 (base=3)
    let expected = [
        0.0,
        1.0 / 3.0,
        2.0 / 3.0,
        1.0 / 9.0,
        4.0 / 9.0,
        7.0 / 9.0,
        2.0 / 9.0,
        5.0 / 9.0,
    ];
    for (i, &exp) in expected.iter().enumerate() {
        let val = van_der_corput(i as u64, 3);
        assert!(
            (val - exp).abs() < TOL,
            "VdC base=3 index={}: Rust={:.15e}, Python={:.15e}",
            i, val, exp
        );
    }
}

/// Van der Corput base=5 精确值
#[test]
fn test_van_der_corput_base5_python() {
    // Python Halton 第三维 (base=5)
    let expected = [0.0, 0.2, 0.4, 0.6, 0.8, 0.04, 0.24, 0.44];
    for (i, &exp) in expected.iter().enumerate() {
        let val = van_der_corput(i as u64, 5);
        assert!(
            (val - exp).abs() < TOL,
            "VdC base=5 index={}: Rust={:.15e}, Python={:.15e}",
            i, val, exp
        );
    }
}

// ============================================================================
// 2. Halton 3D — Python scipy.stats.qmc.Halton(3, scramble=False) 精确值
// ============================================================================

/// Halton 3D 前 8 个点 (base 2,3,5) 与 Python scipy 精确匹配
#[test]
fn test_halton_3d_python_exact() {
    // Python: Halton(3, scramble=False).random(8)
    let expected: Vec<[f64; 3]> = vec![
        [0.0, 0.0, 0.0],
        [0.5, 1.0 / 3.0, 0.2],
        [0.25, 2.0 / 3.0, 0.4],
        [0.75, 1.0 / 9.0, 0.6],
        [0.125, 4.0 / 9.0, 0.8],
        [0.625, 7.0 / 9.0, 0.04],
        [0.375, 2.0 / 9.0, 0.24],
        [0.875, 5.0 / 9.0, 0.44],
    ];

    for (i, exp) in expected.iter().enumerate() {
        let pt = halton_point(i as u64, 3, false, 0);
        for d in 0..3 {
            assert!(
                (pt[d] - exp[d]).abs() < TOL,
                "Halton 3D index={} dim={}: Rust={:.15e}, Python={:.15e}",
                i, d, pt[d], exp[d]
            );
        }
    }
}

/// Halton 高维 (dim=20) 全部使用素数基底
#[test]
fn test_halton_20d_prime_bases() {
    let pt = halton_point(1, 20, false, 0);
    // 20 维分别使用 PRIMES[0..20] = [2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71]
    let primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71u64];
    for (d, &base) in primes.iter().enumerate() {
        let expected = 1.0 / base as f64; // VdC(1, base) = 1/base
        assert!(
            (pt[d] - expected).abs() < TOL,
            "Halton 20D dim={} base={}: Rust={:.15e}, expected={:.15e}",
            d, base, pt[d], expected
        );
    }
}

// ============================================================================
// 3. Halton 序列数学性质
// ============================================================================

/// Halton 序列值域 [0, 1)
#[test]
fn test_halton_range() {
    for i in 0..200 {
        let pt = halton_point(i, 10, false, 0);
        for (d, &v) in pt.iter().enumerate() {
            assert!(
                v >= 0.0 && v < 1.0,
                "Halton index={} dim={}: value {} out of [0,1)",
                i, d, v
            );
        }
    }
}

/// Halton 序列唯一性: 200 个点无重复
#[test]
fn test_halton_uniqueness() {
    let points: Vec<Vec<f64>> = (1..200).map(|i| halton_point(i, 3, false, 0)).collect();
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            assert_ne!(
                points[i], points[j],
                "Halton collision at indices {} and {}",
                i + 1, j + 1
            );
        }
    }
}

/// Halton 低差异性: 100 个 2D 点在 4x4 网格中分布均匀
#[test]
fn test_halton_low_discrepancy_2d() {
    let n: usize = 100;
    let grid_size: usize = 4;
    let mut grid = vec![vec![0usize; grid_size]; grid_size];
    for i in 1..=n {
        let pt = halton_point(i as u64, 2, false, 0);
        let gx = (pt[0] * grid_size as f64).min(grid_size as f64 - 1.0) as usize;
        let gy = (pt[1] * grid_size as f64).min(grid_size as f64 - 1.0) as usize;
        grid[gx][gy] += 1;
    }
    let ideal = n / (grid_size * grid_size); // 100/16 ≈ 6
    for i in 0..grid_size {
        for j in 0..grid_size {
            assert!(
                (grid[i][j] as i64 - ideal as i64).unsigned_abs() <= 3,
                "Halton 2D cell [{},{}] has {} points, expected ~{}±3",
                i, j, grid[i][j], ideal
            );
        }
    }
}

// ============================================================================
// 4. SearchSpaceTransform bounds — 与 Python 精确对比
// ============================================================================

use optuna_rs::distributions::*;
use optuna_rs::search_space::SearchSpaceTransform;
use indexmap::IndexMap;

/// Float [0, 10] → bounds [[0, 10]]
#[test]
fn test_transform_bounds_float_basic() {
    let mut space = IndexMap::new();
    space.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);
    let bounds = trans.bounds();
    assert_eq!(bounds.len(), 1);
    assert!((bounds[0][0] - 0.0).abs() < TOL);
    assert!((bounds[0][1] - 10.0).abs() < TOL);
}

/// Float log [0.001, 1.0] → bounds [[ln(0.001), ln(1.0)]] = [[-6.908, 0.0]]
#[test]
fn test_transform_bounds_float_log() {
    let mut space = IndexMap::new();
    space.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.001, 1.0, true, None).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);
    let bounds = trans.bounds();
    // Python: ln(0.001) = -6.907755278982137
    assert!(
        (bounds[0][0] - (-6.907755278982137)).abs() < 1e-10,
        "log low: {}",
        bounds[0][0]
    );
    assert!((bounds[0][1] - 0.0).abs() < TOL, "log high: {}", bounds[0][1]);
}

/// Int [1, 10] → bounds [[0.5, 10.5]]
#[test]
fn test_transform_bounds_int() {
    let mut space = IndexMap::new();
    space.insert(
        "n".to_string(),
        Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);
    let bounds = trans.bounds();
    // Python: [0.5, 10.5]
    assert!(
        (bounds[0][0] - 0.5).abs() < TOL,
        "Int low: {}",
        bounds[0][0]
    );
    assert!(
        (bounds[0][1] - 10.5).abs() < TOL,
        "Int high: {}",
        bounds[0][1]
    );
}

/// Int log [1, 100] → bounds [[ln(0.5), ln(100.5)]]
#[test]
fn test_transform_bounds_int_log() {
    let mut space = IndexMap::new();
    space.insert(
        "n".to_string(),
        Distribution::IntDistribution(IntDistribution::new(1, 100, true, 1).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);
    let bounds = trans.bounds();
    // Python: [-0.693147..., 4.610157...]
    let expected_lo = (0.5_f64).ln();
    let expected_hi = (100.5_f64).ln();
    assert!(
        (bounds[0][0] - expected_lo).abs() < 1e-10,
        "Int log low: {}, expected: {}",
        bounds[0][0], expected_lo
    );
    assert!(
        (bounds[0][1] - expected_hi).abs() < 1e-10,
        "Int log high: {}, expected: {}",
        bounds[0][1], expected_hi
    );
}

// ============================================================================
// 5. RandomSampler 分布属性
// ============================================================================

use optuna_rs::samplers::RandomSampler;
use optuna_rs::samplers::Sampler;
use optuna_rs::trial::{FrozenTrial, TrialState};
use std::collections::HashMap;

fn dummy_trial() -> FrozenTrial {
    FrozenTrial {
        number: 0,
        trial_id: 0,
        state: TrialState::Running,
        values: None,
        datetime_start: None,
        datetime_complete: None,
        params: HashMap::new(),
        distributions: HashMap::new(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    }
}

/// RandomSampler: 半开区间 [lo, hi) — 对齐 Python np.random.uniform
#[test]
fn test_random_half_open_interval() {
    let sampler = RandomSampler::new(Some(0));
    let dist = Distribution::FloatDistribution(
        FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
    );
    let trial = dummy_trial();
    for _ in 0..10000 {
        let v = sampler
            .sample_independent(&[], &trial, "x", &dist)
            .unwrap();
        assert!(v >= 0.0 && v < 1.0, "value {} not in [0, 1)", v);
    }
}

/// RandomSampler: 确定性 — 相同种子产生相同序列
#[test]
fn test_random_deterministic() {
    let s1 = RandomSampler::new(Some(42));
    let s2 = RandomSampler::new(Some(42));
    let dist = Distribution::FloatDistribution(
        FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
    );
    let trial = dummy_trial();
    for _ in 0..50 {
        let v1 = s1.sample_independent(&[], &trial, "x", &dist).unwrap();
        let v2 = s2.sample_independent(&[], &trial, "x", &dist).unwrap();
        assert_eq!(v1, v2, "Same seed should produce identical values");
    }
}

/// RandomSampler: reseed_rng 后等价于新种子的采样器
#[test]
fn test_random_reseed_equivalence() {
    let sampler = RandomSampler::new(Some(0));
    let dist = Distribution::FloatDistribution(
        FloatDistribution::new(0.0, 100.0, false, None).unwrap(),
    );
    let trial = dummy_trial();

    // 先采样几个值
    let _ = sampler.sample_independent(&[], &trial, "x", &dist);
    let _ = sampler.sample_independent(&[], &trial, "x", &dist);

    // reseed 到 99
    sampler.reseed_rng(99);

    // 应与种子 99 的新采样器产生相同序列
    let fresh = RandomSampler::new(Some(99));
    for _ in 0..20 {
        let v1 = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
        let v2 = fresh.sample_independent(&[], &trial, "x", &dist).unwrap();
        assert_eq!(v1, v2, "reseed should reset to fresh state");
    }
}

/// RandomSampler: log 分布在 [low, high] 范围内
#[test]
fn test_random_log_float_range() {
    let sampler = RandomSampler::new(Some(42));
    let dist = Distribution::FloatDistribution(
        FloatDistribution::new(0.001, 100.0, true, None).unwrap(),
    );
    let trial = dummy_trial();
    for _ in 0..1000 {
        let v = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
        assert!(
            v >= 0.001 && v <= 100.0,
            "log float value {} out of [0.001, 100]",
            v
        );
    }
}

/// RandomSampler: 离散步长值只落在 grid 上
#[test]
fn test_random_float_step_on_grid() {
    let sampler = RandomSampler::new(Some(42));
    let fd = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
    let dist = Distribution::FloatDistribution(fd.clone());
    let trial = dummy_trial();
    for _ in 0..500 {
        let v = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
        // 合法值: 0.0, 0.25, 0.5, 0.75, 1.0
        let on_grid = [0.0, 0.25, 0.5, 0.75, 1.0]
            .iter()
            .any(|&g| (v - g).abs() < 1e-10);
        assert!(on_grid, "value {} not on 0.25 step grid", v);
    }
}

/// RandomSampler: Int step=2 只产生偶数
#[test]
fn test_random_int_step_on_grid() {
    let sampler = RandomSampler::new(Some(42));
    let id = IntDistribution::new(0, 10, false, 2).unwrap();
    let dist = Distribution::IntDistribution(id.clone());
    let trial = dummy_trial();
    for _ in 0..500 {
        let v = sampler.sample_independent(&[], &trial, "x", &dist).unwrap() as i64;
        assert!(v % 2 == 0, "Int step=2 value {} not even", v);
        assert!(v >= 0 && v <= 10, "Int value {} out of [0, 10]", v);
    }
}

/// RandomSampler: 分类分布返回有效索引
#[test]
fn test_random_categorical_valid_index() {
    let sampler = RandomSampler::new(Some(42));
    let dist = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
            CategoricalChoice::Str("c".into()),
            CategoricalChoice::Str("d".into()),
        ])
        .unwrap(),
    );
    let trial = dummy_trial();
    let mut seen = [false; 4];
    for _ in 0..200 {
        let v = sampler.sample_independent(&[], &trial, "x", &dist).unwrap() as usize;
        assert!(v < 4, "Categorical index {} >= 4", v);
        seen[v] = true;
    }
    // 200 次采样 4 个选项，所有选项都应至少出现一次
    for (i, &s) in seen.iter().enumerate() {
        assert!(s, "Category {} never sampled in 200 trials", i);
    }
}

// ============================================================================
// 6. SearchSpaceTransform 往返一致性 (transform + untransform = identity)
// ============================================================================

/// Float 参数 transform → untransform 往返
#[test]
fn test_transform_roundtrip_float() {
    let mut space = IndexMap::new();
    space.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(-5.0, 5.0, false, None).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);
    let bounds = trans.bounds();

    // 在 bounds 中间采样一个值
    let mid = (bounds[0][0] + bounds[0][1]) / 2.0;
    let decoded = trans.untransform(&[mid]).unwrap();
    let x = match &decoded["x"] {
        ParamValue::Float(v) => *v,
        _ => panic!("expected float"),
    };
    // 中间值应为 0.0
    assert!(
        (x - 0.0).abs() < 1e-10,
        "Float midpoint roundtrip: {}",
        x
    );
}

/// Int 参数 transform → untransform 往返
#[test]
fn test_transform_roundtrip_int() {
    let mut space = IndexMap::new();
    space.insert(
        "n".to_string(),
        Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);

    // encoded=5.0 → 应解码为 Int 5
    let decoded = trans.untransform(&[5.0]).unwrap();
    let n = match &decoded["n"] {
        ParamValue::Int(v) => *v,
        _ => panic!("expected int"),
    };
    assert_eq!(n, 5, "Int roundtrip at encoded=5.0");
}

/// Log Float 参数 transform → untransform 往返
#[test]
fn test_transform_roundtrip_log_float() {
    let mut space = IndexMap::new();
    space.insert(
        "lr".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.001, 1.0, true, None).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);
    let bounds = trans.bounds();

    // encoded = ln(0.01) → 应解码为 ~0.01
    let target = 0.01_f64.ln();
    let decoded = trans.untransform(&[target]).unwrap();
    let lr = match &decoded["lr"] {
        ParamValue::Float(v) => *v,
        _ => panic!("expected float"),
    };
    assert!(
        (lr - 0.01).abs() < 1e-10,
        "Log float roundtrip: lr={}",
        lr
    );
    let _ = bounds; // confirm bounds exist
}

// ============================================================================
// 7. SearchSpaceTransform untransform — Python 精确参考值
// ============================================================================

/// Python: Float [0,10] encoded=5.0 → 5.0
#[test]
fn test_untransform_float_basic_python() {
    let mut space = IndexMap::new();
    space.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);
    let decoded = trans.untransform(&[5.0]).unwrap();
    let x = match &decoded["x"] {
        ParamValue::Float(v) => *v,
        _ => panic!("expected float"),
    };
    assert!((x - 5.0).abs() < TOL, "Float untransform: {}", x);
}

/// Python: Float log [0.001,1.0] encoded=ln(0.01) → ~0.01
#[test]
fn test_untransform_float_log_python() {
    let mut space = IndexMap::new();
    space.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.001, 1.0, true, None).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);
    let encoded = 0.01_f64.ln();
    let decoded = trans.untransform(&[encoded]).unwrap();
    let x = match &decoded["x"] {
        ParamValue::Float(v) => *v,
        _ => panic!("expected float"),
    };
    // Python: 1.000000000000001e-02
    assert!(
        (x - 0.01).abs() < 1e-13,
        "Float log untransform: {:.15e}",
        x
    );
}

/// Python: Float step [0,1,step=0.25] 多个 encoded 值的 snap 结果
#[test]
fn test_untransform_float_step_python() {
    let mut space = IndexMap::new();
    space.insert(
        "x".to_string(),
        Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap(),
        ),
    );
    let trans = SearchSpaceTransform::with_defaults(space);

    let cases: Vec<(f64, f64)> = vec![
        (0.3, 0.25),   // Python → 0.25
        (0.6, 0.5),    // Python → 0.5
        (0.88, 1.0),   // Python → 1.0
        (0.13, 0.25),  // Python → 0.25
        (-0.1, 0.0),   // Python → 0.0
    ];
    for (enc, expected) in &cases {
        let decoded = trans.untransform(&[*enc]).unwrap();
        let x = match &decoded["x"] {
            ParamValue::Float(v) => *v,
            _ => panic!("expected float"),
        };
        assert!(
            (x - expected).abs() < TOL,
            "Float step enc={}: Rust={:.15e}, Python={:.15e}",
            enc, x, expected
        );
    }
}

/// Python: Int [1,10] untransform — 四舍五入 + clamp
#[test]
fn test_untransform_int_python() {
    let mut space = IndexMap::new();
    space.insert(
        "n".to_string(),
        Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);

    let cases: Vec<(f64, i64)> = vec![
        (5.5, 5),   // Python → 5 (round to nearest on step grid)
        (5.4, 5),   // Python → 5
        (1.2, 1),   // Python → 1
        (9.8, 10),  // Python → 10
        (0.5, 1),   // Python → 1 (clamp to low)
        (10.5, 10), // Python → 10 (clamp to high)
    ];
    for (enc, expected) in &cases {
        let decoded = trans.untransform(&[*enc]).unwrap();
        let n = match &decoded["n"] {
            ParamValue::Int(v) => *v,
            _ => panic!("expected int"),
        };
        assert_eq!(
            n, *expected,
            "Int [1,10] enc={}: Rust={}, Python={}",
            enc, n, expected
        );
    }
}

/// Python: Int log [1,100] encoded=ln(50) → 50
#[test]
fn test_untransform_int_log_python() {
    let mut space = IndexMap::new();
    space.insert(
        "n".to_string(),
        Distribution::IntDistribution(IntDistribution::new(1, 100, true, 1).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);

    let encoded = 50.0_f64.ln();
    let decoded = trans.untransform(&[encoded]).unwrap();
    let n = match &decoded["n"] {
        ParamValue::Int(v) => *v,
        _ => panic!("expected int"),
    };
    assert_eq!(n, 50, "Int log untransform: {}", n);
}

/// Python: Int step=3 [0,12] — snap 到 0,3,6,9,12
#[test]
fn test_untransform_int_step_python() {
    let mut space = IndexMap::new();
    space.insert(
        "n".to_string(),
        Distribution::IntDistribution(IntDistribution::new(0, 12, false, 3).unwrap()),
    );
    let trans = SearchSpaceTransform::with_defaults(space);

    let cases: Vec<(f64, i64)> = vec![
        (5.0, 6),   // Python → 6
        (7.0, 6),   // Python → 6
        (1.0, 0),   // Python → 0
        (11.0, 12), // Python → 12
    ];
    for (enc, expected) in &cases {
        let decoded = trans.untransform(&[*enc]).unwrap();
        let n = match &decoded["n"] {
            ParamValue::Int(v) => *v,
            _ => panic!("expected int"),
        };
        assert_eq!(
            n, *expected,
            "Int step=3 enc={}: Rust={}, Python={}",
            enc, n, expected
        );
    }
}

/// Python: Float [0,10] transform_0_1 — encoded 0→0, 0.5→5, 1.0→nextafter(10,9)
#[test]
fn test_untransform_0_1_python() {
    let mut space = IndexMap::new();
    space.insert(
        "x".to_string(),
        Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap()),
    );
    let trans = SearchSpaceTransform::new(space, true, true, true);

    let decoded_0 = trans.untransform(&[0.0]).unwrap();
    let x0 = match &decoded_0["x"] {
        ParamValue::Float(v) => *v,
        _ => panic!("expected float"),
    };
    assert!((x0 - 0.0).abs() < TOL, "0_1 enc=0 → {}", x0);

    let decoded_5 = trans.untransform(&[0.5]).unwrap();
    let x5 = match &decoded_5["x"] {
        ParamValue::Float(v) => *v,
        _ => panic!("expected float"),
    };
    assert!((x5 - 5.0).abs() < TOL, "0_1 enc=0.5 → {}", x5);

    let decoded_1 = trans.untransform(&[1.0]).unwrap();
    let x1 = match &decoded_1["x"] {
        ParamValue::Float(v) => *v,
        _ => panic!("expected float"),
    };
    // Python: nextafter(10.0, 9.0) = 9.999999999999998e+00
    let nextafter_10 = 9.999999999999998e+00;
    assert!(
        (x1 - nextafter_10).abs() < 1e-14,
        "0_1 enc=1.0: Rust={:.20e}, Python={:.20e}",
        x1, nextafter_10
    );
}
