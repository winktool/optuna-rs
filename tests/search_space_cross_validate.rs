//! 搜索空间模块交叉验证测试
//!
//! 使用 Python golden_search_space.py 生成的金标准值，
//! 验证 IntersectionSearchSpace 和 GroupDecomposedSearchSpace 的精确行为。

use std::collections::HashMap;
use indexmap::IndexMap;
use optuna_rs::distributions::{Distribution, FloatDistribution, IntDistribution, CategoricalDistribution, CategoricalChoice};
use optuna_rs::search_space::{SearchSpaceGroup, GroupDecomposedSearchSpace, IntersectionSearchSpace};
use optuna_rs::trial::{FrozenTrial, TrialState};

// ========== 辅助函数 ==========

fn float_dist(low: f64, high: f64) -> Distribution {
    Distribution::FloatDistribution(FloatDistribution { low, high, log: false, step: None })
}

fn int_dist(low: i64, high: i64) -> Distribution {
    Distribution::IntDistribution(IntDistribution { low, high, log: false, step: 1 })
}

fn cat_dist(choices: Vec<&str>) -> Distribution {
    Distribution::CategoricalDistribution(CategoricalDistribution {
        choices: choices.into_iter().map(|s| CategoricalChoice::Str(s.to_string())).collect(),
    })
}

fn make_trial(number: i64, state: TrialState, dists: HashMap<String, Distribution>) -> FrozenTrial {
    FrozenTrial {
        number,
        trial_id: number,
        state,
        values: if state == TrialState::Complete { Some(vec![number as f64]) } else { None },
        datetime_start: None,
        datetime_complete: None,
        params: HashMap::new(),
        distributions: dists,
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    }
}

fn sorted_keys(map: &IndexMap<String, Distribution>) -> Vec<String> {
    let mut keys: Vec<String> = map.keys().cloned().collect();
    keys.sort();
    keys
}

fn sorted_group_keys(group: &SearchSpaceGroup) -> Vec<Vec<String>> {
    let mut result: Vec<Vec<String>> = group.search_spaces().iter()
        .map(|s: &HashMap<String, Distribution>| {
            let mut keys: Vec<String> = s.keys().cloned().collect();
            keys.sort();
            keys
        })
        .collect();
    result.sort();
    result
}

// ========== SearchSpaceGroup 分裂测试 ==========

/// Python 金标准: {x,y,z} + {x,y} → 2 groups: {x,y}, {z}
#[test]
fn test_group_split_xyz_then_xy_python() {
    let mut group = SearchSpaceGroup::new();
    let mut d1 = HashMap::new();
    d1.insert("x".into(), float_dist(0.0, 1.0));
    d1.insert("y".into(), float_dist(0.0, 1.0));
    d1.insert("z".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d1);
    assert_eq!(group.search_spaces().len(), 1);

    let mut d2 = HashMap::new();
    d2.insert("x".into(), float_dist(0.0, 1.0));
    d2.insert("y".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d2);

    // Python: 2 groups, keys=[['x', 'y'], ['z']]
    assert_eq!(group.search_spaces().len(), 2);
    let keys = sorted_group_keys(&group);
    assert_eq!(keys, vec![vec!["x", "y"], vec!["z"]]);
}

/// Python 金标准: {a,b,c} + {b,c,d} → 3 groups: {a}, {b,c}, {d}
#[test]
fn test_group_split_abc_then_bcd_python() {
    let mut group = SearchSpaceGroup::new();
    let mut d1 = HashMap::new();
    d1.insert("a".into(), float_dist(0.0, 1.0));
    d1.insert("b".into(), float_dist(0.0, 1.0));
    d1.insert("c".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d1);

    let mut d2 = HashMap::new();
    d2.insert("b".into(), float_dist(0.0, 1.0));
    d2.insert("c".into(), float_dist(0.0, 1.0));
    d2.insert("d".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d2);

    // Python: 3 groups, keys=[['a'], ['b', 'c'], ['d']]
    assert_eq!(group.search_spaces().len(), 3);
    let keys = sorted_group_keys(&group);
    assert_eq!(keys, vec![vec!["a"], vec!["b", "c"], vec!["d"]]);
}

/// Python 金标准: {a,b,c} + {b,c,d} + {a,d} → 3 groups (不变)
#[test]
fn test_group_split_abc_bcd_ad_python() {
    let mut group = SearchSpaceGroup::new();
    let mut d1 = HashMap::new();
    d1.insert("a".into(), float_dist(0.0, 1.0));
    d1.insert("b".into(), float_dist(0.0, 1.0));
    d1.insert("c".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d1);

    let mut d2 = HashMap::new();
    d2.insert("b".into(), float_dist(0.0, 1.0));
    d2.insert("c".into(), float_dist(0.0, 1.0));
    d2.insert("d".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d2);

    let mut d3 = HashMap::new();
    d3.insert("a".into(), float_dist(0.0, 1.0));
    d3.insert("d".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d3);

    // Python: 3 groups, keys=[['a'], ['b', 'c'], ['d']]
    assert_eq!(group.search_spaces().len(), 3);
    let keys = sorted_group_keys(&group);
    assert_eq!(keys, vec![vec!["a"], vec!["b", "c"], vec!["d"]]);
}

/// Python 金标准: 渐进式分裂 5 步
#[test]
fn test_group_progressive_split_python() {
    let mut group = SearchSpaceGroup::new();

    // Step 1: {x,y} → 1 group
    let mut d1 = HashMap::new();
    d1.insert("x".into(), float_dist(0.0, 1.0));
    d1.insert("y".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d1);
    assert_eq!(group.search_spaces().len(), 1);

    // Step 2: {x,y} → 1 group (不变)
    group.add_distributions(&d1);
    assert_eq!(group.search_spaces().len(), 1);

    // Step 3: {x} → 2 groups: {x}, {y}
    let mut d2 = HashMap::new();
    d2.insert("x".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d2);
    assert_eq!(group.search_spaces().len(), 2);
    let keys = sorted_group_keys(&group);
    assert_eq!(keys, vec![vec!["x"], vec!["y"]]);

    // Step 4: {z} → 3 groups: {x}, {y}, {z}
    let mut d3 = HashMap::new();
    d3.insert("z".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d3);
    assert_eq!(group.search_spaces().len(), 3);
    let keys = sorted_group_keys(&group);
    assert_eq!(keys, vec![vec!["x"], vec!["y"], vec!["z"]]);

    // Step 5: {x,z} → 3 groups (不合并)
    let mut d4 = HashMap::new();
    d4.insert("x".into(), float_dist(0.0, 1.0));
    d4.insert("z".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d4);
    assert_eq!(group.search_spaces().len(), 3);
    let keys = sorted_group_keys(&group);
    assert_eq!(keys, vec![vec!["x"], vec!["y"], vec!["z"]]);
}

/// SearchSpaceGroup: 空分布不影响已有组
#[test]
fn test_group_empty_distributions_python() {
    let mut group = SearchSpaceGroup::new();
    group.add_distributions(&HashMap::new());
    assert!(group.search_spaces().is_empty());

    let mut d1 = HashMap::new();
    d1.insert("x".into(), float_dist(0.0, 1.0));
    group.add_distributions(&d1);
    assert_eq!(group.search_spaces().len(), 1);

    group.add_distributions(&HashMap::new());
    assert_eq!(group.search_spaces().len(), 1);
}

// ========== IntersectionSearchSpace 交集测试 ==========

/// Python 金标准: trial(x,y,z) → {x,y,z}
#[test]
fn test_intersection_single_trial_python() {
    let mut iss = IntersectionSearchSpace::new(false);
    let trials = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), float_dist(0.0, 1.0));
            d.insert("z".into(), float_dist(0.0, 1.0));
            d
        }),
    ];
    let result = iss.calculate(&trials);
    assert_eq!(sorted_keys(&result), vec!["x", "y", "z"]);
}

/// Python 金标准: trial(x,y,z) + trial(x,y) → {x,y}
#[test]
fn test_intersection_shrinks_python() {
    let mut iss = IntersectionSearchSpace::new(false);
    let trials = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), float_dist(0.0, 1.0));
            d.insert("z".into(), float_dist(0.0, 1.0));
            d
        }),
        make_trial(1, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), float_dist(0.0, 1.0));
            d
        }),
    ];
    let result = iss.calculate(&trials);
    // Python: ['x', 'y']
    assert_eq!(sorted_keys(&result), vec!["x", "y"]);
}

/// Python 金标准: trial(x,y,z) + trial(x,y) + trial(x,w) → {x}
#[test]
fn test_intersection_continues_shrinking_python() {
    let mut iss = IntersectionSearchSpace::new(false);
    let trials = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), float_dist(0.0, 1.0));
            d.insert("z".into(), float_dist(0.0, 1.0));
            d
        }),
        make_trial(1, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), float_dist(0.0, 1.0));
            d
        }),
        make_trial(2, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("w".into(), float_dist(0.0, 1.0));
            d
        }),
    ];
    let result = iss.calculate(&trials);
    // Python: ['x']
    assert_eq!(sorted_keys(&result), vec!["x"]);
}

/// Python 金标准: 同名不同范围分布 → 冲突 → 空
#[test]
fn test_intersection_distribution_conflict_python() {
    let mut iss = IntersectionSearchSpace::new(false);
    let trials = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d
        }),
        make_trial(1, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 10.0)); // 不同范围
            d
        }),
    ];
    let result = iss.calculate(&trials);
    // Python: [] (空)
    assert!(result.is_empty(), "Distribution conflict should yield empty intersection");
}

/// IntersectionSearchSpace: Running 状态被忽略
#[test]
fn test_intersection_ignores_running_python() {
    let mut iss = IntersectionSearchSpace::new(false);
    let trials = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), float_dist(0.0, 1.0));
            d
        }),
        make_trial(1, TrialState::Running, {
            let mut d = HashMap::new();
            d.insert("z".into(), float_dist(0.0, 1.0));
            d
        }),
    ];
    let result = iss.calculate(&trials);
    assert_eq!(sorted_keys(&result), vec!["x", "y"]);
}

/// IntersectionSearchSpace: include_pruned=true 纳入 Pruned 试验
#[test]
fn test_intersection_include_pruned_python() {
    let mut iss = IntersectionSearchSpace::new(true);
    let trials = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), float_dist(0.0, 1.0));
            d
        }),
        make_trial(1, TrialState::Pruned, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            // 缺少 y → 交集仅 {x}
            d
        }),
    ];
    let result = iss.calculate(&trials);
    assert_eq!(sorted_keys(&result), vec!["x"]);
}

// ========== GroupDecomposedSearchSpace 测试 ==========

/// Python 金标准: trial(x,y) + trial(x,z) → 3 groups: {x}, {y}, {z}
#[test]
fn test_group_decomposed_xy_xz_python() {
    let mut gdss = GroupDecomposedSearchSpace::new(false);
    let trials = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), int_dist(0, 10));
            d
        }),
        make_trial(1, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("z".into(), cat_dist(vec!["a", "b", "c"]));
            d
        }),
    ];
    let result = gdss.calculate(1, &trials);
    // Python: 3 groups, keys=[['x'], ['y'], ['z']]
    assert_eq!(result.search_spaces().len(), 3);
    let keys = sorted_group_keys(&result);
    assert_eq!(keys, vec![vec!["x"], vec!["y"], vec!["z"]]);
}

/// Python 金标准: trial(x,y) + trial(x,z) + trial(y,z) → 3 groups (不变)
#[test]
fn test_group_decomposed_xy_xz_yz_python() {
    let mut gdss = GroupDecomposedSearchSpace::new(false);
    let trials = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), int_dist(0, 10));
            d
        }),
        make_trial(1, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("z".into(), cat_dist(vec!["a", "b", "c"]));
            d
        }),
        make_trial(2, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("y".into(), int_dist(0, 10));
            d.insert("z".into(), cat_dist(vec!["a", "b", "c"]));
            d
        }),
    ];
    let result = gdss.calculate(1, &trials);
    // Python: 3 groups, keys=[['x'], ['y'], ['z']]
    assert_eq!(result.search_spaces().len(), 3);
    let keys = sorted_group_keys(&result);
    assert_eq!(keys, vec![vec!["x"], vec!["y"], vec!["z"]]);
}

/// GroupDecomposed: Pruned 默认被忽略
#[test]
fn test_group_decomposed_ignores_pruned_python() {
    let mut gdss = GroupDecomposedSearchSpace::new(false);
    let trials = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), float_dist(0.0, 1.0));
            d
        }),
        make_trial(1, TrialState::Pruned, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d
        }),
    ];
    let result = gdss.calculate(1, &trials);
    // 仅 Complete 试验: {x,y} → 1 group
    assert_eq!(result.search_spaces().len(), 1);
    assert_eq!(sorted_group_keys(&result), vec![vec!["x", "y"]]);
}

/// GroupDecomposed: include_pruned=true
#[test]
fn test_group_decomposed_includes_pruned_python() {
    let mut gdss = GroupDecomposedSearchSpace::new(true);
    let trials = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), float_dist(0.0, 1.0));
            d
        }),
        make_trial(1, TrialState::Pruned, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d
        }),
    ];
    let result = gdss.calculate(1, &trials);
    // 纳入 Pruned: {x,y} + {x} → 分裂为 {x}, {y}
    assert_eq!(result.search_spaces().len(), 2);
    let keys = sorted_group_keys(&result);
    assert_eq!(keys, vec![vec!["x"], vec!["y"]]);
}

/// GroupDecomposed: 混合类型分布
#[test]
fn test_group_decomposed_mixed_types_python() {
    let mut gdss = GroupDecomposedSearchSpace::new(false);
    let trials = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("lr".into(), float_dist(1e-5, 1e-1));
            d.insert("n_layers".into(), int_dist(1, 10));
            d.insert("optimizer".into(), cat_dist(vec!["adam", "sgd"]));
            d
        }),
        make_trial(1, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("lr".into(), float_dist(1e-5, 1e-1));
            d.insert("n_layers".into(), int_dist(1, 10));
            d.insert("dropout".into(), float_dist(0.0, 0.5));
            d
        }),
    ];
    let result = gdss.calculate(1, &trials);
    // lr 和 n_layers 共同出现; optimizer 和 dropout 各独立
    assert_eq!(result.search_spaces().len(), 3);
    let keys = sorted_group_keys(&result);
    assert_eq!(keys, vec![vec!["dropout"], vec!["lr", "n_layers"], vec!["optimizer"]]);
}

/// 交叉验证: Intersection + GroupDecomposed 一致性
/// 当所有试验参数相同时，Intersection 应返回全部，GroupDecomposed 应只有 1 组
#[test]
fn test_intersection_group_consistency_python() {
    let mut dists = HashMap::new();
    dists.insert("x".into(), float_dist(0.0, 1.0));
    dists.insert("y".into(), int_dist(0, 10));
    dists.insert("z".into(), cat_dist(vec!["a", "b"]));

    let trials = vec![
        make_trial(0, TrialState::Complete, dists.clone()),
        make_trial(1, TrialState::Complete, dists.clone()),
        make_trial(2, TrialState::Complete, dists.clone()),
    ];

    let mut iss = IntersectionSearchSpace::new(false);
    let intersection = iss.calculate(&trials);
    assert_eq!(sorted_keys(&intersection), vec!["x", "y", "z"]);

    let mut gdss = GroupDecomposedSearchSpace::new(false);
    let group = gdss.calculate(1, &trials);
    assert_eq!(group.search_spaces().len(), 1);
    assert_eq!(sorted_group_keys(&group), vec![vec!["x", "y", "z"]]);
}

/// IntersectionSearchSpace: 增量缓存正确性
#[test]
fn test_intersection_incremental_cache_python() {
    let mut iss = IntersectionSearchSpace::new(false);

    // 第 1 次调用: 1 个试验
    let trials1 = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), float_dist(0.0, 1.0));
            d
        }),
    ];
    let r1 = iss.calculate(&trials1);
    assert_eq!(sorted_keys(&r1), vec!["x", "y"]);

    // 第 2 次调用: 2 个试验 (缓存应跳过第 0 个)
    let trials2 = vec![
        make_trial(0, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d.insert("y".into(), float_dist(0.0, 1.0));
            d
        }),
        make_trial(1, TrialState::Complete, {
            let mut d = HashMap::new();
            d.insert("x".into(), float_dist(0.0, 1.0));
            d
        }),
    ];
    let r2 = iss.calculate(&trials2);
    assert_eq!(sorted_keys(&r2), vec!["x"]);
}

/// IntersectionSearchSpace: 空试验列表
#[test]
fn test_intersection_empty_trials_python() {
    let mut iss = IntersectionSearchSpace::new(false);
    let result = iss.calculate(&[]);
    assert!(result.is_empty());
}
