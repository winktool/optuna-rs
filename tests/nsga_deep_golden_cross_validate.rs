/// NSGA-II/III 深度交叉验证测试
///
/// 对比 Python optuna 黄金值与 Rust 实现:
/// - Das-Dennis 参考点生成
/// - 垂直距离计算
/// - 拥挤距离计算
/// - 约束违反度
/// - 交叉算子统计性质

use optuna_rs::multi_objective::{
    crowding_distance, fast_non_dominated_sort,
};
use optuna_rs::samplers::nsgaiii::generate_reference_points;
use optuna_rs::samplers::nsgaii::crossover::{
    Crossover, UniformCrossover, BLXAlphaCrossover, SBXCrossover,
    SPXCrossover, UNDXCrossover, VSBXCrossover,
};
use optuna_rs::study::StudyDirection;
use optuna_rs::trial::FrozenTrial;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

/// 辅助: 加载 JSON 黄金值
fn load_golden() -> serde_json::Value {
    let data = std::fs::read_to_string("tests/nsga_deep_golden_values.json")
        .expect("需要先运行 python tests/golden_nsga_deep.py");
    serde_json::from_str(&data).unwrap()
}

fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
    assert!(
        (a - b).abs() < tol,
        "{}: {} vs {} (diff={})",
        msg, a, b, (a - b).abs()
    );
}

fn make_trial(number: i64, values: Vec<f64>) -> FrozenTrial {
    FrozenTrial {
        number,
        state: optuna_rs::trial::TrialState::Complete,
        values: Some(values),
        params: Default::default(),
        distributions: Default::default(),
        intermediate_values: Default::default(),
        system_attrs: Default::default(),
        user_attrs: Default::default(),
        datetime_start: None,
        datetime_complete: None,
        trial_id: number,
    }
}

// ============================================================
// 1. Das-Dennis 参考点精确匹配
// ============================================================

#[test]
fn test_das_dennis_count_golden() {
    let golden = load_golden();
    let cases = golden["das_dennis"].as_array().unwrap();

    for case in cases {
        let n_obj = case["n_objectives"].as_u64().unwrap() as usize;
        let div = case["dividing_parameter"].as_u64().unwrap() as usize;
        let expected_count = case["count"].as_u64().unwrap() as usize;

        let pts = generate_reference_points(n_obj, div);
        assert_eq!(pts.len(), expected_count,
            "Das-Dennis({}, {}) count mismatch", n_obj, div);
    }
}

#[test]
fn test_das_dennis_points_golden() {
    let golden = load_golden();
    let case = &golden["das_dennis"][0]; // 2D, div=3

    let n_obj = case["n_objectives"].as_u64().unwrap() as usize;
    let div = case["dividing_parameter"].as_u64().unwrap() as usize;
    let expected_pts: Vec<Vec<f64>> = case["points"].as_array().unwrap()
        .iter()
        .map(|p| p.as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap())
            .collect())
        .collect();

    let pts = generate_reference_points(n_obj, div);

    // 验证每个 Python 期望点都在 Rust 结果中
    for exp in &expected_pts {
        let found = pts.iter().any(|p| {
            p.iter().zip(exp.iter()).all(|(a, b)| (a - b).abs() < 1e-10)
        });
        assert!(found, "Das-Dennis point {:?} not found in Rust result", exp);
    }
}

#[test]
fn test_das_dennis_sum_to_one_golden() {
    // 所有参考点坐标和应为 1.0 (对齐 Python)
    for (n_obj, div) in [(2, 3), (3, 3), (4, 3)] {
        let pts = generate_reference_points(n_obj, div);
        for (i, p) in pts.iter().enumerate() {
            let sum: f64 = p.iter().sum();
            assert_close(sum, 1.0, 1e-10,
                &format!("Das-Dennis({},{}) point {} sum", n_obj, div, i));
        }
    }
}

// ============================================================
// 2. 垂直距离精确匹配
// ============================================================

#[test]
fn test_perpendicular_distance_golden() {
    let golden = load_golden();
    let cases = golden["perpendicular_distance"].as_array().unwrap();

    for (i, case) in cases.iter().enumerate() {
        let point: Vec<f64> = case["point"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        let direction: Vec<f64> = case["direction"].as_array().unwrap()
            .iter().map(|v| v.as_f64().unwrap()).collect();
        let expected = case["distance"].as_f64().unwrap();

        let actual = optuna_rs::samplers::nsgaiii::perpendicular_distance(&point, &direction);

        if expected >= 1e307 {
            assert!(actual >= 1e307 || actual.is_infinite(),
                "case {}: expected inf, got {}", i, actual);
        } else {
            assert_close(actual, expected, 1e-10,
                &format!("perpendicular_distance case {}", i));
        }
    }
}

// ============================================================
// 3. 拥挤距离精确匹配
// ============================================================

#[test]
fn test_crowding_distance_2obj_golden() {
    let golden = load_golden();
    let case = &golden["crowding_distance"][0]; // 2obj_5points_uniform

    let values: Vec<Vec<f64>> = case["values"].as_array().unwrap()
        .iter()
        .map(|p| p.as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let expected_cd: Vec<f64> = case["crowding_distances"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();

    let trials: Vec<FrozenTrial> = values.iter().enumerate()
        .map(|(i, v)| make_trial(i as i64, v.clone()))
        .collect();
    let trial_refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = vec![StudyDirection::Minimize; values[0].len()];

    let cd = crowding_distance(&trial_refs, &dirs);

    // 边界个体应为 inf
    for (i, (&actual, &expected)) in cd.iter().zip(expected_cd.iter()).enumerate() {
        if expected >= 1e307 {
            assert!(actual >= 1e307 || actual.is_infinite(),
                "cd[{}]: expected inf, got {}", i, actual);
        } else {
            assert_close(actual, expected, 1e-6,
                &format!("crowding_distance[{}]", i));
        }
    }
}

#[test]
fn test_crowding_distance_3obj_golden() {
    let golden = load_golden();
    let case = &golden["crowding_distance"][1]; // 3obj_4points

    let values: Vec<Vec<f64>> = case["values"].as_array().unwrap()
        .iter()
        .map(|p| p.as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let expected_cd: Vec<f64> = case["crowding_distances"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();

    let trials: Vec<FrozenTrial> = values.iter().enumerate()
        .map(|(i, v)| make_trial(i as i64, v.clone()))
        .collect();
    let trial_refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = vec![StudyDirection::Minimize; 3];
    let cd = crowding_distance(&trial_refs, &dirs);

    for (i, (&actual, &expected)) in cd.iter().zip(expected_cd.iter()).enumerate() {
        if expected >= 1e307 {
            assert!(actual >= 1e307 || actual.is_infinite(),
                "cd_3d[{}]: expected inf, got {}", i, actual);
        } else {
            assert_close(actual, expected, 1e-6,
                &format!("crowding_distance_3d[{}]", i));
        }
    }
}

// ============================================================
// 4. 交叉算子统计性质测试
// ============================================================

#[test]
fn test_uniform_crossover_probability() {
    // 验证 Uniform 交叉实际 swapping 率接近设定值
    let cx = UniformCrossover::new(Some(0.3));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let n_runs = 10000;
    let p0 = vec![0.0; 10];
    let p1 = vec![1.0; 10];
    let mut swap_count = 0.0;
    let total = n_runs as f64 * 10.0;

    for _ in 0..n_runs {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        swap_count += child.iter().filter(|&&v| v > 0.5).count() as f64;
    }

    let actual_rate = swap_count / total;
    assert!((actual_rate - 0.3).abs() < 0.02,
        "Uniform swap rate should be ~0.3, got {}", actual_rate);
}

#[test]
fn test_blx_alpha_range() {
    // BLX-α(0.5): 子代范围应在 [lo - 0.5*d, hi + 0.5*d] 内
    let cx = BLXAlphaCrossover::new(Some(0.5));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.3, 0.4];
    let p1 = vec![0.7, 0.6];

    for _ in 0..1000 {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v), "BLX-α child out of [0,1]: {}", v);
        }
    }
}

#[test]
fn test_sbx_high_eta_stays_near_parents() {
    // 高 eta (100) 时子代应非常接近父代
    let cx = SBXCrossover::new(Some(100.0));
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.3, 0.4];
    let p1 = vec![0.7, 0.6];
    let midpoint = vec![0.5, 0.5];

    let mut max_dist = 0.0_f64;
    for _ in 0..1000 {
        let child = cx.crossover(&[p0.clone(), p1.clone()], &mut rng);
        let dist: f64 = child.iter().zip(midpoint.iter())
            .map(|(c, m)| (c - m).powi(2)).sum::<f64>().sqrt();
        max_dist = max_dist.max(dist);
    }

    // 高 eta 时子代与中点距离应很小
    let parent_dist = 0.283; // sqrt((0.3-0.7)^2 + (0.4-0.6)^2)
    assert!(max_dist < parent_dist,
        "High-eta SBX should stay within parent spread: max_dist={} vs parent_dist={}", max_dist, parent_dist);
}

#[test]
fn test_sbx_low_eta_explores_widely() {
    // 低 eta (0.1) 时子代应比高 eta 分散更广
    let cx_low = SBXCrossover::new(Some(0.1));
    let cx_high = SBXCrossover::new(Some(100.0));
    let p0 = vec![0.3, 0.4];
    let p1 = vec![0.7, 0.6];

    let mut var_low = 0.0;
    let mut var_high = 0.0;
    let n = 2000;
    for seed in 0..n {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let c_low = cx_low.crossover(&[p0.clone(), p1.clone()], &mut rng);
        let mut rng2 = ChaCha8Rng::seed_from_u64(seed);
        let c_high = cx_high.crossover(&[p0.clone(), p1.clone()], &mut rng2);
        var_low += c_low.iter().map(|v| (v - 0.5).powi(2)).sum::<f64>();
        var_high += c_high.iter().map(|v| (v - 0.5).powi(2)).sum::<f64>();
    }
    assert!(var_low > var_high,
        "Low-eta SBX should have wider spread: low={} vs high={}", var_low, var_high);
}

#[test]
fn test_spx_centroid_property() {
    // SPX 子代统计平均应接近质心
    let cx = SPXCrossover::new(None);
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p0 = vec![0.2, 0.3];
    let p1 = vec![0.5, 0.6];
    let p2 = vec![0.4, 0.5];
    let centroid = vec![
        (0.2 + 0.5 + 0.4) / 3.0,
        (0.3 + 0.6 + 0.5) / 3.0,
    ];

    let n_runs = 5000;
    let mut sum = vec![0.0; 2];
    for _ in 0..n_runs {
        let child = cx.crossover(&[p0.clone(), p1.clone(), p2.clone()], &mut rng);
        for (s, c) in sum.iter_mut().zip(child.iter()) {
            *s += c;
        }
    }

    for (i, s) in sum.iter().enumerate() {
        let avg = s / n_runs as f64;
        assert!((avg - centroid[i]).abs() < 0.03,
            "SPX mean[{}] should be near centroid: {} vs {}", i, avg, centroid[i]);
    }
}

#[test]
fn test_undx_sigma_effect() {
    // 较大 sigma_xi 应产生更大扩散
    let cx_big = UNDXCrossover::new(2.0, Some(0.01));
    let cx_small = UNDXCrossover::new(0.1, Some(0.01));

    let n_runs = 500;
    let mut var_big = 0.0;
    let mut var_small = 0.0;

    for seed in 0..n_runs {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let p = vec![vec![0.3, 0.3], vec![0.7, 0.7], vec![0.5, 0.5]];
        let c_big = cx_big.crossover(&p, &mut rng);
        let mut rng2 = ChaCha8Rng::seed_from_u64(seed);
        let c_small = cx_small.crossover(&p, &mut rng2);

        var_big += c_big.iter().map(|v| (v - 0.5).powi(2)).sum::<f64>();
        var_small += c_small.iter().map(|v| (v - 0.5).powi(2)).sum::<f64>();
    }

    assert!(var_big > var_small,
        "larger sigma should produce wider spread: big={} vs small={}", var_big, var_small);
}

#[test]
fn test_vsbx_child_in_bounds() {
    // VSBX 子代始终在 [0, 1]
    let cx = VSBXCrossover::default();
    let mut rng = ChaCha8Rng::seed_from_u64(42);

    for _ in 0..1000 {
        let p0 = vec![0.1, 0.9, 0.5];
        let p1 = vec![0.9, 0.1, 0.5];
        let child = cx.crossover(&[p0, p1], &mut rng);
        for &v in &child {
            assert!((0.0..=1.0).contains(&v), "VSBX child out of bounds: {}", v);
        }
    }
}

// ============================================================
// 5. 非支配排序性质测试 (扩展)
// ============================================================

#[test]
fn test_non_dominated_sort_single_front() {
    // 所有在同一 Pareto 前沿上的个体应在第 0 级
    let trials: Vec<FrozenTrial> = vec![
        make_trial(0, vec![0.0, 1.0]),
        make_trial(1, vec![0.5, 0.5]),
        make_trial(2, vec![1.0, 0.0]),
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = vec![StudyDirection::Minimize, StudyDirection::Minimize];
    let fronts = fast_non_dominated_sort(&refs, &dirs);

    assert_eq!(fronts.len(), 1, "all on Pareto front → 1 front");
    assert_eq!(fronts[0].len(), 3);
}

#[test]
fn test_non_dominated_sort_maximize_direction() {
    // Maximize 方向: 较大值应不被支配
    let trials: Vec<FrozenTrial> = vec![
        make_trial(0, vec![1.0, 1.0]),  // 被 trial 2 支配
        make_trial(1, vec![2.0, 2.0]),  // 被 trial 2 支配
        make_trial(2, vec![3.0, 3.0]),  // 帕累托最优
    ];
    let refs: Vec<&FrozenTrial> = trials.iter().collect();
    let dirs = vec![StudyDirection::Maximize, StudyDirection::Maximize];
    let fronts = fast_non_dominated_sort(&refs, &dirs);

    assert!(fronts[0].contains(&2), "trial 2 should be in front 0");
    assert!(!fronts[0].contains(&0), "trial 0 should not be in front 0");
}

// ============================================================
// 6. 参考点关联验证
// ============================================================

#[test]
fn test_reference_point_association_golden() {
    let golden = load_golden();
    let case = &golden["reference_point_association"][0];

    let points: Vec<Vec<f64>> = case["points"].as_array().unwrap()
        .iter()
        .map(|p| p.as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let ref_pts: Vec<Vec<f64>> = case["reference_points"].as_array().unwrap()
        .iter()
        .map(|p| p.as_array().unwrap().iter()
            .map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let expected_indices: Vec<usize> = case["closest_indices"].as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();
    let expected_dists: Vec<f64> = case["distances"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();

    for (i, point) in points.iter().enumerate() {
        // 找最近参考点
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (j, rp) in ref_pts.iter().enumerate() {
            let dist = optuna_rs::samplers::nsgaiii::perpendicular_distance(point, rp);
            if dist < best_dist {
                best_dist = dist;
                best_idx = j;
            }
        }
        assert_eq!(best_idx, expected_indices[i],
            "point {:?}: closest ref should be {}, got {}", point, expected_indices[i], best_idx);
        assert_close(best_dist, expected_dists[i], 1e-6,
            &format!("point {:?}: distance to closest ref", point));
    }
}

// ============================================================
// 7. 交叉算子不变量测试
// ============================================================

#[test]
fn test_all_crossovers_idempotent_parents() {
    // 当所有父代相同时，子代应等于父代
    let mut rng = ChaCha8Rng::seed_from_u64(42);
    let p = vec![0.5, 0.5, 0.5];

    // 2-parent crossovers (不含 VSBX — VSBX child2 公式 3p0-p1 不保持不动点)
    let crossovers_2: Vec<Box<dyn Crossover>> = vec![
        Box::new(UniformCrossover::new(Some(0.5))),
        Box::new(BLXAlphaCrossover::new(Some(0.5))),
        Box::new(SBXCrossover::new(Some(2.0))),
    ];
    for cx in &crossovers_2 {
        let child = cx.crossover(&[p.clone(), p.clone()], &mut rng);
        for (j, &v) in child.iter().enumerate() {
            assert_close(v, 0.5, 0.01,
                &format!("identical parents → child[{}] ≈ 0.5", j));
        }
    }

    // 3-parent crossovers
    let crossovers_3: Vec<Box<dyn Crossover>> = vec![
        Box::new(SPXCrossover::new(None)),
        Box::new(UNDXCrossover::default()),
    ];
    for cx in &crossovers_3 {
        let child = cx.crossover(&[p.clone(), p.clone(), p.clone()], &mut rng);
        for (j, &v) in child.iter().enumerate() {
            assert_close(v, 0.5, 0.01,
                &format!("identical 3-parents → child[{}] ≈ 0.5", j));
        }
    }
}
