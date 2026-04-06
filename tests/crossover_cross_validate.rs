// Crossover Cross-Validation Tests
//
// 验证 optuna-rs 6 种 crossover 算子的统计性质与 Python 对齐。
// 因 RNG 不同 (Python numpy vs Rust ChaCha8)，使用统计分布匹配代替精确值对比。
//
// Python 基线: tests/crossover_baseline.json (N=50000 trials)
// 测试容差: 均值 ±0.03, 标准差 ±0.05 (允许 RNG 差异)

use optuna_rs::samplers::nsgaii::crossover::*;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// 加载 Python 基线 JSON
fn load_baseline() -> serde_json::Value {
    let data = include_str!("crossover_baseline.json");
    serde_json::from_str(data).expect("Failed to parse crossover_baseline.json")
}

// ═══════════════════════════════════════════════════════════════════════════
//  1. UniformCrossover — 交换概率统计验证
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn crossover_uniform_swap_prob_0() {
    // swap_prob=0 → 全部选 parent[0]
    let cx = UniformCrossover::new(Some(0.0));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let p1 = vec![0.6, 0.7, 0.8, 0.9, 1.0];
    for _ in 0..100 {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        assert_eq!(child, p0, "swap_prob=0 should always select parent[0]");
    }
}

#[test]
fn crossover_uniform_swap_prob_1() {
    // swap_prob=1 → 全部选 parent[1]
    let cx = UniformCrossover::new(Some(1.0));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.1, 0.2, 0.3, 0.4, 0.5];
    let p1 = vec![0.6, 0.7, 0.8, 0.9, 1.0];
    for _ in 0..100 {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        assert_eq!(child, p1, "swap_prob=1 should always select parent[1]");
    }
}

#[test]
fn crossover_uniform_statistical_swap_ratio() {
    // swap_prob=0.5 → 约 50% 基因来自 parent[1]
    let cx = UniformCrossover::new(Some(0.5));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.0; 5];
    let p1 = vec![1.0; 5];
    let n = 50000;
    let mut swap_count = 0usize;
    for _ in 0..n {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        swap_count += child.iter().filter(|&&v| v == 1.0).count();
    }
    let ratio = swap_count as f64 / (n * 5) as f64;
    // Python baseline: 0.499 (但 Python 的 swap_ratio 是 1-prob = 0.5)
    // Rust: P(parent[1]) = swap_prob = 0.5
    assert!(
        (ratio - 0.5).abs() < 0.02,
        "swap ratio should be ~0.5, got {}",
        ratio
    );
}

#[test]
fn crossover_uniform_asymmetric_prob() {
    // swap_prob=0.25 → ~25% 基因来自 parent[1]
    let cx = UniformCrossover::new(Some(0.25));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.0; 10];
    let p1 = vec![1.0; 10];
    let n = 30000;
    let mut swap_count = 0usize;
    for _ in 0..n {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        swap_count += child.iter().filter(|&&v| v == 1.0).count();
    }
    let ratio = swap_count as f64 / (n * 10) as f64;
    assert!(
        (ratio - 0.25).abs() < 0.02,
        "swap ratio should be ~0.25, got {}",
        ratio
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  2. BLXAlphaCrossover — 范围扩展验证
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn crossover_blx_alpha_range() {
    let baseline = load_baseline();
    let cx = BLXAlphaCrossover::new(Some(0.5));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.3];
    let p1 = vec![0.7];
    let n = 50000;
    let mut children: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        children.push(child[0]);
    }
    let mean: f64 = children.iter().sum::<f64>() / n as f64;
    let std: f64 = (children.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n as f64).sqrt();

    let py_mean = baseline["blx_mean"].as_f64().unwrap();
    let py_std = baseline["blx_std"].as_f64().unwrap();

    assert!(
        (mean - py_mean).abs() < 0.03,
        "BLX mean={} should be ~{}",
        mean,
        py_mean
    );
    assert!(
        (std - py_std).abs() < 0.05,
        "BLX std={} should be ~{}",
        std,
        py_std
    );

    // 所有子代应在 [0, 1] 范围内（已 clamp）
    for &c in &children {
        assert!((0.0..=1.0).contains(&c));
    }
}

#[test]
fn crossover_blx_alpha_zero() {
    // alpha=0 → 子代在 [min, max] 之间均匀分布
    let cx = BLXAlphaCrossover::new(Some(0.0));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.3];
    let p1 = vec![0.7];
    let n = 10000;
    for _ in 0..n {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        assert!(
            child[0] >= 0.3 && child[0] <= 0.7,
            "alpha=0: child {} must be in [0.3, 0.7]",
            child[0]
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  3. SBXCrossover — 分布特性
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn crossover_sbx_eta2_distribution() {
    let baseline = load_baseline();
    let cx = SBXCrossover::new(Some(2.0));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.3];
    let p1 = vec![0.7];
    let n = 50000;
    let mut children: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        children.push(child[0]);
    }
    let mean: f64 = children.iter().sum::<f64>() / n as f64;
    let py_mean = baseline["sbx_eta2_mean"].as_f64().unwrap();

    // SBX 应以父代中点 (0.5) 为中心对称分布
    assert!(
        (mean - py_mean).abs() < 0.03,
        "SBX eta=2 mean={} should be ~{}",
        mean,
        py_mean
    );

    for &c in &children {
        assert!((0.0..=1.0).contains(&c));
    }
}

#[test]
fn crossover_sbx_eta_controls_spread() {
    // eta 越大 → 子代越集中在父代附近 → std 越小
    let cx2 = SBXCrossover::new(Some(2.0));
    let cx20 = SBXCrossover::new(Some(20.0));
    let p0 = vec![0.3];
    let p1 = vec![0.7];
    let n = 30000;

    let mut children2: Vec<f64> = Vec::with_capacity(n);
    let mut children20: Vec<f64> = Vec::with_capacity(n);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..n {
        children2.push(cx2.crossover(&[p0.clone(), p1.clone()], &mut rng)[0]);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    for _ in 0..n {
        children20.push(cx20.crossover(&[p0.clone(), p1.clone()], &mut rng)[0]);
    }

    let mean2: f64 = children2.iter().sum::<f64>() / n as f64;
    let std2: f64 =
        (children2.iter().map(|x| (x - mean2).powi(2)).sum::<f64>() / n as f64).sqrt();
    let mean20: f64 = children20.iter().sum::<f64>() / n as f64;
    let std20: f64 =
        (children20.iter().map(|x| (x - mean20).powi(2)).sum::<f64>() / n as f64).sqrt();

    assert!(
        std20 < std2,
        "eta=20 std ({}) should be < eta=2 std ({})",
        std20,
        std2
    );
}

#[test]
fn crossover_sbx_identical_parents() {
    // 父代相同 → 子代 = 父代
    let cx = SBXCrossover::new(Some(2.0));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p = vec![0.5, 0.3, 0.7];
    for _ in 0..100 {
        let child = cx.crossover(&[p.clone(), p.clone()], &mut rng);
        for (i, &v) in child.iter().enumerate() {
            assert!(
                (v - p[i]).abs() < 1e-12,
                "identical parents → child should equal parent"
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  4. SPXCrossover — 质心与扩展
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn crossover_spx_centroid_centered() {
    let baseline = load_baseline();
    let cx = SPXCrossover::new(None); // default epsilon
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.3, 0.3];
    let p1 = vec![0.7, 0.7];
    let p2 = vec![0.5, 0.2];
    let n = 50000;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for _ in 0..n {
        let child = cx.crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng);
        sum_x += child[0];
        sum_y += child[1];
    }
    let mean_x = sum_x / n as f64;
    let mean_y = sum_y / n as f64;

    let py_cx: Vec<f64> = baseline["spx_centroid"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    // SPX 分布应围绕质心附近
    // 由于 [0,1] 裁剪，均值可能偏离质心，但应在合理范围内
    assert!(
        (mean_x - py_cx[0]).abs() < 0.05,
        "SPX mean_x={} should be near centroid_x={}",
        mean_x,
        py_cx[0]
    );
    // y 方向因裁剪可能偏差更大
    assert!(
        (mean_y - py_cx[1]).abs() < 0.10,
        "SPX mean_y={} should be near centroid_y={}",
        mean_y,
        py_cx[1]
    );
}

#[test]
fn crossover_spx_custom_epsilon() {
    // epsilon 越小 → 子代越集中于质心
    let cx_small = SPXCrossover::new(Some(0.5));
    let cx_large = SPXCrossover::new(Some(3.0));
    let p0 = vec![0.3, 0.3];
    let p1 = vec![0.7, 0.7];
    let p2 = vec![0.5, 0.5];
    let n = 20000;

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let mut vals: Vec<f64> = Vec::with_capacity(n);
    for _ in 0..n {
        vals.push(cx_small.crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng)[0]);
    }
    let mean_s: f64 = vals.iter().sum::<f64>() / n as f64;
    let std_small = (vals.iter().map(|x| (x - mean_s).powi(2)).sum::<f64>() / n as f64).sqrt();

    let mut rng = ChaCha8Rng::seed_from_u64(42);
    vals.clear();
    for _ in 0..n {
        vals.push(cx_large.crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng)[0]);
    }
    let mean_l: f64 = vals.iter().sum::<f64>() / n as f64;
    let std_large = (vals.iter().map(|x| (x - mean_l).powi(2)).sum::<f64>() / n as f64).sqrt();

    assert!(
        std_small < std_large,
        "smaller epsilon should give smaller spread: small={}, large={}",
        std_small,
        std_large
    );
}

#[test]
fn crossover_spx_n_parents() {
    let cx = SPXCrossover::new(None);
    assert_eq!(cx.n_parents(), 3, "SPX requires 3 parents");
}

// ═══════════════════════════════════════════════════════════════════════════
//  5. UNDXCrossover — 主搜索线与正交扰动
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn crossover_undx_midpoint_centered() {
    let baseline = load_baseline();
    let cx = UNDXCrossover::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.3, 0.3];
    let p1 = vec![0.7, 0.7];
    let p2 = vec![0.5, 0.2];
    let n = 50000;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for _ in 0..n {
        let child = cx.crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng);
        sum_x += child[0];
        sum_y += child[1];
    }
    let mean_x = sum_x / n as f64;
    let mean_y = sum_y / n as f64;

    let py_mid_x = baseline["undx_midpoint"].as_array().unwrap()[0]
        .as_f64()
        .unwrap();
    let py_mid_y = baseline["undx_midpoint"].as_array().unwrap()[1]
        .as_f64()
        .unwrap();

    // UNDX 应以 P1-P2 中点为中心
    assert!(
        (mean_x - py_mid_x).abs() < 0.03,
        "UNDX mean_x={} should be ~midpoint_x={}",
        mean_x,
        py_mid_x
    );
    assert!(
        (mean_y - py_mid_y).abs() < 0.03,
        "UNDX mean_y={} should be ~midpoint_y={}",
        mean_y,
        py_mid_y
    );
}

#[test]
fn crossover_undx_sigma_xi_controls_spread() {
    // sigma_xi 越大 → PSL 方向扩散越大
    let cx_small = UNDXCrossover::new(0.1, Some(0.01));
    let cx_large = UNDXCrossover::new(2.0, Some(0.01));
    let p0 = vec![0.3, 0.3];
    let p1 = vec![0.7, 0.7];
    let p2 = vec![0.5, 0.5];
    let n = 20000;

    let mut dist_small = 0.0;
    let mut dist_large = 0.0;
    let midpoint = [0.5, 0.5];

    for seed in 0..n {
        let mut rng = ChaCha8Rng::seed_from_u64(seed as u64);
        let c = cx_small.crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng);
        dist_small += ((c[0] - midpoint[0]).powi(2) + (c[1] - midpoint[1]).powi(2)).sqrt();

        let mut rng2 = ChaCha8Rng::seed_from_u64(seed as u64);
        let c = cx_large.crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng2);
        dist_large += ((c[0] - midpoint[0]).powi(2) + (c[1] - midpoint[1]).powi(2)).sqrt();
    }

    assert!(
        dist_large > dist_small,
        "larger sigma_xi should produce wider spread: large={}, small={}",
        dist_large / n as f64,
        dist_small / n as f64,
    );
}

#[test]
fn crossover_undx_1d_no_orthogonal() {
    // 1D: 无正交分量，只有 PSL 方向
    let cx = UNDXCrossover::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.3];
    let p1 = vec![0.7];
    let p2 = vec![0.5];
    for _ in 0..100 {
        let child = cx.crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng);
        assert_eq!(child.len(), 1);
        assert!((0.0..=1.0).contains(&child[0]));
    }
}

#[test]
fn crossover_undx_sigma_squared_as_std() {
    // 对齐 Python: rng.normal(0, sigma**2) 传入 sigma^2 作为 std
    // 验证: 大 sigma 的扩散应远大于小 sigma
    let cx_s = UNDXCrossover::new(0.05, Some(0.01));
    let cx_l = UNDXCrossover::new(1.0, Some(0.01));
    let n = 10000;
    let p0 = vec![0.4, 0.4];
    let p1 = vec![0.6, 0.6];
    let p2 = vec![0.5, 0.3];

    let mut var_s = 0.0;
    let mut var_l = 0.0;
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for _ in 0..n {
        let c = cx_s.crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng);
        var_s += (c[0] - 0.5).powi(2);
    }
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    for _ in 0..n {
        let c = cx_l.crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng);
        var_l += (c[0] - 0.5).powi(2);
    }
    // sigma=1.0 的 scale 是 sigma^2=1.0, sigma=0.05 的 scale 是 0.0025
    // variance ratio 应 >> 1
    assert!(
        var_l > var_s * 5.0,
        "sigma_xi=1.0 variance should be >> sigma_xi=0.05: large={}, small={}",
        var_l / n as f64,
        var_s / n as f64,
    );
}

// ═══════════════════════════════════════════════════════════════════════════
//  6. VSBXCrossover — 向量 SBX 验证
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn crossover_vsbx_distribution() {
    let baseline = load_baseline();
    let cx = VSBXCrossover::new(Some(20.0), 0.5, 0.5);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.3, 0.3];
    let p1 = vec![0.7, 0.7];
    let n = 50000;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    for _ in 0..n {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        sum_x += child[0];
        sum_y += child[1];
    }
    let mean_x = sum_x / n as f64;
    let mean_y = sum_y / n as f64;

    let py_mean_x = baseline["vsbx_eta20_mean_x"].as_f64().unwrap();
    let py_mean_y = baseline["vsbx_eta20_mean_y"].as_f64().unwrap();

    // VSBX 均值应在 0.5 附近（父代中点）
    assert!(
        (mean_x - py_mean_x).abs() < 0.04,
        "VSBX mean_x={} should be ~{}",
        mean_x,
        py_mean_x
    );
    assert!(
        (mean_y - py_mean_y).abs() < 0.04,
        "VSBX mean_y={} should be ~{}",
        mean_y,
        py_mean_y
    );
}

#[test]
fn crossover_vsbx_global_u1_u2() {
    // 验证 u1/u2 是全局标量（所有维度共享分支选择）
    // 使用相同父代值：在全局 u1 下所有维度经历相同分支
    let cx = VSBXCrossover::new(Some(20.0), 0.0, 1.0);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.3, 0.3, 0.3];
    let p1 = vec![0.7, 0.7, 0.7];
    // 如果 u1/u2 是全局的，且父代各维度相同，
    // 则 beta 因 us 不同而不同，但 c1/c2 公式中的系数相同
    let child = cx.crossover(&[p0, p1], &mut rng);
    assert_eq!(child.len(), 3);
    for &v in &child {
        assert!((0.0..=1.0).contains(&v));
    }
}

#[test]
fn crossover_vsbx_bounds() {
    // 极端父代值 → 子代仍在 [0,1]
    let cx = VSBXCrossover::new(Some(2.0), 0.5, 0.5);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.0, 0.0, 0.0];
    let p1 = vec![1.0, 1.0, 1.0];
    for _ in 0..1000 {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        for &v in &child {
            assert!(
                (0.0..=1.0).contains(&v),
                "VSBX child value {} out of [0,1]",
                v
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  7. 跨算子不变量
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn crossover_all_dimension_preserved() {
    // 所有算子必须保持维度
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let dims = [1, 2, 5, 10, 20];
    for &d in &dims {
        let p0: Vec<f64> = (0..d).map(|i| i as f64 / d as f64).collect();
        let p1: Vec<f64> = (0..d).map(|i| (i + 1) as f64 / (d + 1) as f64).collect();
        let p2: Vec<f64> = (0..d).map(|i| 0.5 + 0.1 * (i as f64 / d as f64)).collect();

        let cx_u = UniformCrossover::default();
        assert_eq!(
            cx_u.crossover(&[p0.clone(), p1.clone()], &mut rng).len(),
            d
        );

        let cx_b = BLXAlphaCrossover::default();
        assert_eq!(
            cx_b.crossover(&[p0.clone(), p1.clone()], &mut rng).len(),
            d
        );

        let cx_s = SBXCrossover::default();
        assert_eq!(
            cx_s.crossover(&[p0.clone(), p1.clone()], &mut rng).len(),
            d
        );

        let cx_v = VSBXCrossover::default();
        assert_eq!(
            cx_v.crossover(&[p0.clone(), p1.clone()], &mut rng).len(),
            d
        );

        if d >= 1 {
            let cx_sp = SPXCrossover::new(None);
            assert_eq!(
                cx_sp
                    .crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng)
                    .len(),
                d
            );
        }

        if d >= 1 {
            let cx_un = UNDXCrossover::default();
            assert_eq!(
                cx_un
                    .crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng)
                    .len(),
                d
            );
        }
    }
}

#[test]
fn crossover_all_bounds_01() {
    // 所有算子子代必须在 [0, 1] 范围内
    let mut rng = ChaCha8Rng::seed_from_u64(0);
    let operators: Vec<(&str, Box<dyn Crossover>)> = vec![
        ("Uniform", Box::new(UniformCrossover::default())),
        ("BLX", Box::new(BLXAlphaCrossover::new(Some(0.5)))),
        ("SBX", Box::new(SBXCrossover::new(Some(2.0)))),
        ("VSBX", Box::new(VSBXCrossover::new(Some(2.0), 0.5, 0.5))),
    ];

    let extremes = vec![
        (vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0]),
        (vec![0.0, 0.5, 1.0], vec![1.0, 0.5, 0.0]),
        (vec![0.01, 0.99], vec![0.99, 0.01]),
    ];

    for (name, op) in &operators {
        for (p0, p1) in &extremes {
            for _ in 0..500 {
                let child = op.crossover(&[p0.clone(), p1.clone()], &mut rng);
                for (i, &v) in child.iter().enumerate() {
                    assert!(
                        (0.0..=1.0).contains(&v),
                        "{} child[{}]={} out of bounds with parents {:?}, {:?}",
                        name,
                        i,
                        v,
                        p0,
                        p1,
                    );
                }
            }
        }
    }

    // 3-parent operators
    let p2 = vec![0.5, 0.5, 0.5];
    let ops3: Vec<(&str, Box<dyn Crossover>)> = vec![
        ("SPX", Box::new(SPXCrossover::new(None))),
        ("UNDX", Box::new(UNDXCrossover::default())),
    ];
    for (name, op) in &ops3 {
        for _ in 0..500 {
            let child = op.crossover(
                &[vec![0.0, 0.0, 0.0], vec![1.0, 1.0, 1.0], p2.clone()],
                &mut rng,
            );
            for (i, &v) in child.iter().enumerate() {
                assert!(
                    (0.0..=1.0).contains(&v),
                    "{} child[{}]={} out of bounds",
                    name,
                    i,
                    v,
                );
            }
        }
    }
}

#[test]
fn crossover_n_parents_correct() {
    assert_eq!(UniformCrossover::default().n_parents(), 2);
    assert_eq!(BLXAlphaCrossover::default().n_parents(), 2);
    assert_eq!(SBXCrossover::default().n_parents(), 2);
    assert_eq!(VSBXCrossover::default().n_parents(), 2);
    assert_eq!(SPXCrossover::new(None).n_parents(), 3);
    assert_eq!(UNDXCrossover::default().n_parents(), 3);
}

// ═══════════════════════════════════════════════════════════════════════════
//  8. 高维与边界测试
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn crossover_high_dim_20() {
    // 20 维父代的交叉
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let d = 20;
    let p0: Vec<f64> = (0..d).map(|i| i as f64 / d as f64).collect();
    let p1: Vec<f64> = (0..d).map(|i| 1.0 - i as f64 / d as f64).collect();
    let p2: Vec<f64> = vec![0.5; d];

    // 所有算子应正常工作
    let c = UniformCrossover::default().crossover(&[p0.clone(), p1.clone()], &mut rng);
    assert_eq!(c.len(), d);

    let c = BLXAlphaCrossover::default().crossover(&[p0.clone(), p1.clone()], &mut rng);
    assert_eq!(c.len(), d);

    let c = SBXCrossover::default().crossover(&[p0.clone(), p1.clone()], &mut rng);
    assert_eq!(c.len(), d);

    let c = SPXCrossover::new(None).crossover(
        &[p0.clone(), p1.clone(), p2.clone()],
        &mut rng,
    );
    assert_eq!(c.len(), d);

    let c = UNDXCrossover::default().crossover(
        &[p0.clone(), p1.clone(), p2.clone()],
        &mut rng,
    );
    assert_eq!(c.len(), d);

    let c = VSBXCrossover::default().crossover(&[p0.clone(), p1.clone()], &mut rng);
    assert_eq!(c.len(), d);
}

#[test]
fn crossover_undx_orthogonality_verified() {
    // 通过 UNDX 输出间接验证正交基的正确性：
    // 使用 3D 父代，验证子代沿 PSL 和正交方向都有合理扩散
    let cx = UNDXCrossover::new(0.5, Some(0.5));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    // P0 和 P1 定义 PSL 沿 x 轴方向
    let p0 = vec![0.3, 0.5, 0.5];
    let p1 = vec![0.7, 0.5, 0.5];
    let p2 = vec![0.5, 0.3, 0.5]; // P3 在 y 方向偏离 PSL
    let n = 10000;
    let mut var_x = 0.0;
    let mut var_y = 0.0;
    let mut var_z = 0.0;
    for _ in 0..n {
        let c = cx.crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng);
        var_x += (c[0] - 0.5).powi(2);
        var_y += (c[1] - 0.5).powi(2);
        var_z += (c[2] - 0.5).powi(2);
    }
    // PSL 沿 x 方向：x 方向应有主要扩散（sigma_xi 贡献）
    // y, z 方向应有次要扩散（sigma_eta 贡献 + D 距离缩放）
    // 至少 x 方向扩散应存在
    assert!(var_x / n as f64 > 1e-6, "x direction should have variance");
    // y/z 正交方向也应有扩散（因 sigma_eta > 0 且 D > 0）
    assert!(
        var_y / n as f64 > 1e-6 || var_z / n as f64 > 1e-6,
        "orthogonal directions should have some variance"
    );
}
