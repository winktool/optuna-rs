//! 分布、Study、SearchSpace 模块精确交叉验证测试。
//!
//! 所有参考值均由 Python optuna 生成，确保 Rust 移植与 Python 精确对齐。

use optuna_rs::distributions::{
    CategoricalChoice, CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
};
use optuna_rs::search_space::IntersectionSearchSpace;
use optuna_rs::study::{create_study, StudyDirection};
use optuna_rs::trial::{FrozenTrial, TrialState};
use std::collections::HashMap;

// ============================================================================
// 1. FloatDistribution to_internal_repr / to_external_repr
// ============================================================================

/// Python 参考:
///   FloatDistribution(0, 10): to_internal_repr(3.14) = 3.14
///   FloatDistribution(0, 10): to_external_repr(3.14) = 3.14
#[test]
fn test_float_distribution_identity_repr() {
    let fd = FloatDistribution::new(0.0, 10.0, false, None).unwrap();
    assert_eq!(fd.to_internal_repr(3.14).unwrap(), 3.14);
    assert_eq!(fd.to_external_repr(3.14), 3.14);
}

/// Python 参考:
///   FloatDistribution(0, 1, step=0.1): to_internal_repr(0.35) = 0.35
///   FloatDistribution(0, 1, step=0.1): to_external_repr(0.35) = 0.35
#[test]
fn test_float_distribution_step_repr() {
    let fd = FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap();
    assert_eq!(fd.to_internal_repr(0.35).unwrap(), 0.35);
    assert_eq!(fd.to_external_repr(0.35), 0.35);
}

/// 验证 FloatDistribution contains 边界行为
#[test]
fn test_float_distribution_contains() {
    let fd = FloatDistribution::new(0.0, 10.0, false, None).unwrap();
    assert!(fd.contains(0.0));
    assert!(fd.contains(5.0));
    assert!(fd.contains(10.0));
    assert!(!fd.contains(-0.001));
    assert!(!fd.contains(10.001));
}

/// 验证 FloatDistribution single
#[test]
fn test_float_distribution_single() {
    let fd_single = FloatDistribution::new(5.0, 5.0, false, None).unwrap();
    assert!(fd_single.single());

    let fd_range = FloatDistribution::new(0.0, 10.0, false, None).unwrap();
    assert!(!fd_range.single());
}

// ============================================================================
// 2. IntDistribution to_internal_repr / to_external_repr
// ============================================================================

/// Python 参考:
///   IntDistribution(0, 10): to_internal_repr(5) = 5.0
///   IntDistribution(0, 10): to_external_repr(5.0) = 5
#[test]
fn test_int_distribution_basic_repr() {
    let id = IntDistribution::new(0, 10, false, 1).unwrap();
    assert_eq!(id.to_internal_repr(5).unwrap(), 5.0);
    assert_eq!(id.to_external_repr(5.0).unwrap(), 5);
}

/// Python 参考:
///   IntDistribution(0, 10, step=2): to_internal_repr(4) = 4.0
///   IntDistribution(0, 10, step=2): to_external_repr(4.0) = 4
#[test]
fn test_int_distribution_step_repr() {
    let id = IntDistribution::new(0, 10, false, 2).unwrap();
    assert_eq!(id.to_internal_repr(4).unwrap(), 4.0);
    assert_eq!(id.to_external_repr(4.0).unwrap(), 4);
}

/// Python 参考:
///   IntDistribution(1, 100, log=True): to_internal_repr(10) = 10.0
///   IntDistribution(1, 100, log=True): to_external_repr(10.0) = 10
#[test]
fn test_int_distribution_log_repr() {
    let id = IntDistribution::new(1, 100, true, 1).unwrap();
    assert_eq!(id.to_internal_repr(10).unwrap(), 10.0);
    assert_eq!(id.to_external_repr(10.0).unwrap(), 10);
}

/// 验证 IntDistribution contains
#[test]
fn test_int_distribution_contains() {
    let id = IntDistribution::new(0, 10, false, 1).unwrap();
    assert!(id.contains(0.0));
    assert!(id.contains(5.0));
    assert!(id.contains(10.0));
    assert!(!id.contains(-1.0));
    assert!(!id.contains(11.0));
}

/// 验证 IntDistribution step contains
#[test]
fn test_int_distribution_step_contains() {
    let id = IntDistribution::new(0, 10, false, 2).unwrap();
    assert!(id.contains(0.0));
    assert!(id.contains(2.0));
    assert!(id.contains(4.0));
    assert!(id.contains(10.0));
    // step=2 时，奇数值不在分布内
    assert!(!id.contains(1.0));
    assert!(!id.contains(3.0));
}

// ============================================================================
// 3. CategoricalDistribution to_internal_repr / to_external_repr
// ============================================================================

/// Python 参考:
///   CategoricalDistribution(['a','b','c']):
///     to_internal_repr('a') = 0, to_internal_repr('b') = 1, to_internal_repr('c') = 2
///     to_external_repr(0.0) = 'a', to_external_repr(1.0) = 'b', to_external_repr(2.0) = 'c'
#[test]
fn test_categorical_string_repr() {
    let cd = CategoricalDistribution::new(vec![
        CategoricalChoice::Str("a".into()),
        CategoricalChoice::Str("b".into()),
        CategoricalChoice::Str("c".into()),
    ])
    .unwrap();

    assert_eq!(
        cd.to_internal_repr(&CategoricalChoice::Str("a".into()))
            .unwrap(),
        0.0
    );
    assert_eq!(
        cd.to_internal_repr(&CategoricalChoice::Str("b".into()))
            .unwrap(),
        1.0
    );
    assert_eq!(
        cd.to_internal_repr(&CategoricalChoice::Str("c".into()))
            .unwrap(),
        2.0
    );

    assert_eq!(
        cd.to_external_repr(0.0).unwrap(),
        CategoricalChoice::Str("a".into())
    );
    assert_eq!(
        cd.to_external_repr(1.0).unwrap(),
        CategoricalChoice::Str("b".into())
    );
    assert_eq!(
        cd.to_external_repr(2.0).unwrap(),
        CategoricalChoice::Str("c".into())
    );
}

/// Python 参考:
///   CategoricalDistribution([None, 1, 2.5, 'hello']):
///     to_internal_repr(None)=0, to_internal_repr(1)=1,
///     to_internal_repr(2.5)=2, to_internal_repr('hello')=3
#[test]
fn test_categorical_mixed_types_repr() {
    let cd = CategoricalDistribution::new(vec![
        CategoricalChoice::None,
        CategoricalChoice::Int(1),
        CategoricalChoice::Float(2.5),
        CategoricalChoice::Str("hello".into()),
    ])
    .unwrap();

    assert_eq!(
        cd.to_internal_repr(&CategoricalChoice::None).unwrap(),
        0.0
    );
    assert_eq!(
        cd.to_internal_repr(&CategoricalChoice::Int(1)).unwrap(),
        1.0
    );
    assert_eq!(
        cd.to_internal_repr(&CategoricalChoice::Float(2.5)).unwrap(),
        2.0
    );
    assert_eq!(
        cd.to_internal_repr(&CategoricalChoice::Str("hello".into()))
            .unwrap(),
        3.0
    );
}

/// Python 参考: CategoricalDistribution([NaN, 1.0, 2.0]):
///   to_internal_repr(NaN) = 0
///   to_external_repr(0.0) = NaN (is_nan=True)
#[test]
fn test_categorical_nan_repr() {
    let cd = CategoricalDistribution::new(vec![
        CategoricalChoice::Float(f64::NAN),
        CategoricalChoice::Float(1.0),
        CategoricalChoice::Float(2.0),
    ])
    .unwrap();

    // NaN → index 0
    assert_eq!(
        cd.to_internal_repr(&CategoricalChoice::Float(f64::NAN))
            .unwrap(),
        0.0
    );

    // index 0 → NaN
    let ext = cd.to_external_repr(0.0).unwrap();
    match ext {
        CategoricalChoice::Float(v) => assert!(v.is_nan(), "Expected NaN, got {}", v),
        _ => panic!("Expected Float(NaN), got {:?}", ext),
    }
}

/// 验证 Distribution enum 层面的 to_internal_repr / to_external_repr 联动
#[test]
fn test_distribution_enum_repr() {
    use optuna_rs::distributions::ParamValue;

    // Float
    let dist = Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap());
    assert_eq!(
        dist.to_internal_repr(&ParamValue::Float(3.14)).unwrap(),
        3.14
    );
    assert_eq!(
        dist.to_external_repr(3.14).unwrap(),
        ParamValue::Float(3.14)
    );

    // Int
    let dist =
        Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());
    assert_eq!(
        dist.to_internal_repr(&ParamValue::Int(5)).unwrap(),
        5.0
    );
    assert_eq!(
        dist.to_external_repr(5.0).unwrap(),
        ParamValue::Int(5)
    );

    // Categorical
    let dist = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
        ])
        .unwrap(),
    );
    assert_eq!(
        dist.to_internal_repr(&ParamValue::Categorical(CategoricalChoice::Str("b".into())))
            .unwrap(),
        1.0
    );
    assert_eq!(
        dist.to_external_repr(0.0).unwrap(),
        ParamValue::Categorical(CategoricalChoice::Str("a".into()))
    );
}

/// Distribution enum 跨类型转换：ParamValue::Int → CategoricalDistribution
/// 对齐 Python: 当分类选项为 Int 时，传入 ParamValue::Int 应匹配
#[test]
fn test_distribution_cross_type_categorical() {
    use optuna_rs::distributions::ParamValue;

    let dist = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::None,
            CategoricalChoice::Int(1),
            CategoricalChoice::Int(2),
        ])
        .unwrap(),
    );

    // ParamValue::Int(1) → should match CategoricalChoice::Int(1) → index 1
    assert_eq!(
        dist.to_internal_repr(&ParamValue::Int(1)).unwrap(),
        1.0
    );
    assert_eq!(
        dist.to_internal_repr(&ParamValue::Int(2)).unwrap(),
        2.0
    );
}

// ============================================================================
// 4. Study best_trial / best_value 行为验证
// ============================================================================

fn make_completed_trial(
    number: i64,
    value: f64,
    params: HashMap<String, f64>,
) -> FrozenTrial {
    let mut p = HashMap::new();
    let mut d = HashMap::new();
    for (name, val) in &params {
        p.insert(
            name.clone(),
            optuna_rs::distributions::ParamValue::Float(*val),
        );
        d.insert(
            name.clone(),
            Distribution::FloatDistribution(
                FloatDistribution::new(0.0, 10.0, false, None).unwrap(),
            ),
        );
    }
    FrozenTrial {
        number,
        trial_id: number,
        state: TrialState::Complete,
        values: Some(vec![value]),
        datetime_start: None,
        datetime_complete: None,
        params: p,
        distributions: d,
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    }
}

/// Python 参考:
///   minimize: 4 trials with values [5.0, 2.0, 8.0, 2.0]
///   best_value = 2.0, best_trial.number = 1 (先出现的那个)
#[test]
fn test_study_best_trial_minimize() {
    let study = create_study(None, None, None, None, Some(StudyDirection::Minimize), None, false)
        .unwrap();
    let storage = study.storage();

    let values = [5.0, 2.0, 8.0, 2.0];
    for (i, val) in values.iter().enumerate() {
        let template = make_completed_trial(i as i64, *val, [("x".into(), i as f64)].into());
        storage.create_new_trial(study.study_id(), Some(&template)).unwrap();
    }

    let best = study.best_trial().unwrap();
    assert_eq!(best.value().unwrap().unwrap(), 2.0);
    // Python: 相同值时返回最先出现的 → number=1
    assert_eq!(best.number, 1);
}

/// Python 参考:
///   maximize: 3 trials with values [5.0, 8.0, 2.0]
///   best_value = 8.0, best_trial.number = 1
#[test]
fn test_study_best_trial_maximize() {
    let study = create_study(None, None, None, None, Some(StudyDirection::Maximize), None, false)
        .unwrap();
    let storage = study.storage();

    let values = [5.0, 8.0, 2.0];
    for (i, val) in values.iter().enumerate() {
        let template = make_completed_trial(i as i64, *val, [("x".into(), i as f64)].into());
        storage.create_new_trial(study.study_id(), Some(&template)).unwrap();
    }

    let best = study.best_trial().unwrap();
    assert_eq!(best.value().unwrap().unwrap(), 8.0);
    assert_eq!(best.number, 1);
}

/// 没有已完成试验时应返回错误
#[test]
fn test_study_best_trial_no_completed() {
    let study = create_study(None, None, None, None, Some(StudyDirection::Minimize), None, false)
        .unwrap();
    assert!(study.best_trial().is_err());
}

// ============================================================================
// 5. IntersectionSearchSpace 行为验证
// ============================================================================

/// Python 参考:
///   Trial 0: x(Float[0,1]), y(Int[0,10]), z(Cat['a','b','c'])
///   Trial 1: x(Float[0,1]), y(Int[0,10])  (no z)
///   Trial 2: x(Float[0,1]), y(Int[0,10]), z(Cat['a','b','c']), w(Float[-1,1])
///   交集 = {x, y}
#[test]
fn test_intersection_search_space_basic() {
    let dist_x = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    let dist_y = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());
    let dist_z = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Str("b".into()),
            CategoricalChoice::Str("c".into()),
        ])
        .unwrap(),
    );
    let dist_w = Distribution::FloatDistribution(FloatDistribution::new(-1.0, 1.0, false, None).unwrap());

    use optuna_rs::distributions::ParamValue;

    // Trial 0: x, y, z
    let trial0 = FrozenTrial {
        number: 0,
        trial_id: 0,
        state: TrialState::Complete,
        values: Some(vec![0.5]),
        datetime_start: None,
        datetime_complete: None,
        params: [
            ("x".into(), ParamValue::Float(0.5)),
            ("y".into(), ParamValue::Int(5)),
            ("z".into(), ParamValue::Categorical(CategoricalChoice::Str("a".into()))),
        ]
        .into(),
        distributions: [
            ("x".into(), dist_x.clone()),
            ("y".into(), dist_y.clone()),
            ("z".into(), dist_z.clone()),
        ]
        .into(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    };

    // Trial 1: x, y (no z)
    let trial1 = FrozenTrial {
        number: 1,
        trial_id: 1,
        state: TrialState::Complete,
        values: Some(vec![0.3]),
        datetime_start: None,
        datetime_complete: None,
        params: [
            ("x".into(), ParamValue::Float(0.3)),
            ("y".into(), ParamValue::Int(3)),
        ]
        .into(),
        distributions: [("x".into(), dist_x.clone()), ("y".into(), dist_y.clone())].into(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    };

    // Trial 2: x, y, z, w
    let trial2 = FrozenTrial {
        number: 2,
        trial_id: 2,
        state: TrialState::Complete,
        values: Some(vec![0.1]),
        datetime_start: None,
        datetime_complete: None,
        params: [
            ("x".into(), ParamValue::Float(0.1)),
            ("y".into(), ParamValue::Int(1)),
            ("z".into(), ParamValue::Categorical(CategoricalChoice::Str("b".into()))),
            ("w".into(), ParamValue::Float(-0.5)),
        ]
        .into(),
        distributions: [
            ("x".into(), dist_x.clone()),
            ("y".into(), dist_y.clone()),
            ("z".into(), dist_z.clone()),
            ("w".into(), dist_w.clone()),
        ]
        .into(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    };

    let trials = vec![trial0, trial1, trial2];
    let mut iss = IntersectionSearchSpace::new(false);
    let result = iss.calculate(&trials);

    // 交集应为 {x, y}（z 在 trial1 中缺失，w 只在 trial2 中）
    let keys: Vec<&String> = result.keys().collect();
    assert_eq!(keys, vec!["x", "y"]);

    // 验证分布类型正确
    assert_eq!(result["x"], dist_x);
    assert_eq!(result["y"], dist_y);
}

/// 如果分布定义不同，参数应被排除
#[test]
fn test_intersection_search_space_different_distribution() {
    let dist_x1 = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    let dist_x2 = Distribution::FloatDistribution(FloatDistribution::new(0.0, 2.0, false, None).unwrap()); // 不同 high
    let dist_y = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());

    use optuna_rs::distributions::ParamValue;

    let trial0 = FrozenTrial {
        number: 0,
        trial_id: 0,
        state: TrialState::Complete,
        values: Some(vec![0.5]),
        datetime_start: None,
        datetime_complete: None,
        params: [("x".into(), ParamValue::Float(0.5)), ("y".into(), ParamValue::Int(5))].into(),
        distributions: [("x".into(), dist_x1), ("y".into(), dist_y.clone())].into(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    };

    let trial1 = FrozenTrial {
        number: 1,
        trial_id: 1,
        state: TrialState::Complete,
        values: Some(vec![0.3]),
        datetime_start: None,
        datetime_complete: None,
        params: [("x".into(), ParamValue::Float(0.3)), ("y".into(), ParamValue::Int(3))].into(),
        distributions: [("x".into(), dist_x2), ("y".into(), dist_y.clone())].into(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    };

    let trials = vec![trial0, trial1];
    let mut iss = IntersectionSearchSpace::new(false);
    let result = iss.calculate(&trials);

    // x 的分布不同，应被排除；只有 y 在交集中
    let keys: Vec<&String> = result.keys().collect();
    assert_eq!(keys, vec!["y"]);
}

/// 只有一个试验时，交集=该试验的所有参数
#[test]
fn test_intersection_search_space_single_trial() {
    let dist_x = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    let dist_y = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());

    use optuna_rs::distributions::ParamValue;

    let trial0 = FrozenTrial {
        number: 0,
        trial_id: 0,
        state: TrialState::Complete,
        values: Some(vec![0.5]),
        datetime_start: None,
        datetime_complete: None,
        params: [("x".into(), ParamValue::Float(0.5)), ("y".into(), ParamValue::Int(5))].into(),
        distributions: [("x".into(), dist_x.clone()), ("y".into(), dist_y.clone())].into(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    };

    let trials = vec![trial0];
    let mut iss = IntersectionSearchSpace::new(false);
    let result = iss.calculate(&trials);

    assert_eq!(result.len(), 2);
    assert_eq!(result["x"], dist_x);
    assert_eq!(result["y"], dist_y);
}

/// 空试验列表 → 空交集
#[test]
fn test_intersection_search_space_empty() {
    let trials: Vec<FrozenTrial> = vec![];
    let mut iss = IntersectionSearchSpace::new(false);
    let result = iss.calculate(&trials);
    assert!(result.is_empty());
}

/// 验证 Pruned 试验在 include_pruned=false 时被忽略
#[test]
fn test_intersection_search_space_pruned_excluded() {
    let dist_x = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    let dist_y = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());

    use optuna_rs::distributions::ParamValue;

    let trial0 = FrozenTrial {
        number: 0,
        trial_id: 0,
        state: TrialState::Complete,
        values: Some(vec![0.5]),
        datetime_start: None,
        datetime_complete: None,
        params: [("x".into(), ParamValue::Float(0.5)), ("y".into(), ParamValue::Int(5))].into(),
        distributions: [("x".into(), dist_x.clone()), ("y".into(), dist_y.clone())].into(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    };

    // Pruned 试验只有 x（没有 y）
    let trial1 = FrozenTrial {
        number: 1,
        trial_id: 1,
        state: TrialState::Pruned,
        values: None,
        datetime_start: None,
        datetime_complete: None,
        params: [("x".into(), ParamValue::Float(0.3))].into(),
        distributions: [("x".into(), dist_x.clone())].into(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    };

    let trials = vec![trial0, trial1];

    // include_pruned=false: Pruned 试验被忽略，所以交集 = trial0 的全部
    let mut iss = IntersectionSearchSpace::new(false);
    let result = iss.calculate(&trials);
    assert_eq!(result.len(), 2);

    // include_pruned=true: Pruned 试验参与，交集 = {x}
    let mut iss2 = IntersectionSearchSpace::new(true);
    let result2 = iss2.calculate(&trials);
    assert_eq!(result2.len(), 1);
    assert!(result2.contains_key("x"));
}

// ============================================================================
// 6. Distribution JSON 序列化 / 反序列化 (对齐 Python 格式)
// ============================================================================

/// 验证 Distribution JSON 序列化格式与 Python 兼容
#[test]
fn test_distribution_json_roundtrip() {
    use optuna_rs::distributions::{distribution_to_json, json_to_distribution};

    // Float
    let fd = Distribution::FloatDistribution(FloatDistribution::new(0.0, 10.0, false, None).unwrap());
    let json = distribution_to_json(&fd).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(fd, back);

    // Float log
    let fd_log = Distribution::FloatDistribution(FloatDistribution::new(1e-5, 1.0, true, None).unwrap());
    let json = distribution_to_json(&fd_log).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(fd_log, back);

    // Float step
    let fd_step = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap());
    let json = distribution_to_json(&fd_step).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(fd_step, back);

    // Int
    let id = Distribution::IntDistribution(IntDistribution::new(0, 100, false, 1).unwrap());
    let json = distribution_to_json(&id).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(id, back);

    // Int log
    let id_log = Distribution::IntDistribution(IntDistribution::new(1, 1000, true, 1).unwrap());
    let json = distribution_to_json(&id_log).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(id_log, back);

    // Int step
    let id_step = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 3).unwrap());
    let json = distribution_to_json(&id_step).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(id_step, back);

    // Categorical
    let cd = Distribution::CategoricalDistribution(
        CategoricalDistribution::new(vec![
            CategoricalChoice::Str("a".into()),
            CategoricalChoice::Int(42),
            CategoricalChoice::Float(3.14),
            CategoricalChoice::None,
        ])
        .unwrap(),
    );
    let json = distribution_to_json(&cd).unwrap();
    let back = json_to_distribution(&json).unwrap();
    assert_eq!(cd, back);
}

/// 验证 Python 旧版分布格式（deprecated）反序列化
#[test]
fn test_distribution_legacy_json_compat() {
    use optuna_rs::distributions::json_to_distribution;

    // UniformDistribution → FloatDistribution
    let json = r#"{"name":"UniformDistribution","attributes":{"low":0.0,"high":1.0}}"#;
    let dist = json_to_distribution(json).unwrap();
    match dist {
        Distribution::FloatDistribution(fd) => {
            assert_eq!(fd.low, 0.0);
            assert_eq!(fd.high, 1.0);
            assert!(!fd.log);
        }
        _ => panic!("Expected FloatDistribution"),
    }

    // LogUniformDistribution → FloatDistribution(log=true)
    let json = r#"{"name":"LogUniformDistribution","attributes":{"low":1e-7,"high":1.0}}"#;
    let dist = json_to_distribution(json).unwrap();
    match dist {
        Distribution::FloatDistribution(fd) => {
            assert!(fd.log);
        }
        _ => panic!("Expected FloatDistribution with log=true"),
    }

    // IntUniformDistribution → IntDistribution
    let json = r#"{"name":"IntUniformDistribution","attributes":{"low":0,"high":10,"step":1}}"#;
    let dist = json_to_distribution(json).unwrap();
    match dist {
        Distribution::IntDistribution(id) => {
            assert_eq!(id.low, 0);
            assert_eq!(id.high, 10);
        }
        _ => panic!("Expected IntDistribution"),
    }
}

// ============================================================================
// 7. Distribution equality 语义
// ============================================================================

/// 验证 Float/Int 分布相等性语义
#[test]
fn test_distribution_equality() {
    let a = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
    let b = FloatDistribution::new(0.0, 1.0, false, None).unwrap();
    assert_eq!(a, b);

    let c = FloatDistribution::new(0.0, 2.0, false, None).unwrap();
    assert_ne!(a, c);

    let d = FloatDistribution::new(0.001, 1.0, true, None).unwrap();
    let d2 = FloatDistribution::new(0.001, 1.0, true, None).unwrap();
    assert_eq!(d, d2);
    assert_ne!(a, d);

    let e = FloatDistribution::new(0.0, 1.0, false, Some(0.1)).unwrap();
    assert_ne!(a, e);

    let ia = IntDistribution::new(0, 10, false, 1).unwrap();
    let ib = IntDistribution::new(0, 10, false, 1).unwrap();
    assert_eq!(ia, ib);

    let ic = IntDistribution::new(0, 10, false, 2).unwrap();
    assert_ne!(ia, ic);
}

/// 验证 CategoricalDistribution 相等性（NaN 处理）
#[test]
fn test_categorical_equality_with_nan() {
    let a = CategoricalDistribution::new(vec![
        CategoricalChoice::Float(f64::NAN),
        CategoricalChoice::Float(1.0),
    ])
    .unwrap();
    let b = CategoricalDistribution::new(vec![
        CategoricalChoice::Float(f64::NAN),
        CategoricalChoice::Float(1.0),
    ])
    .unwrap();
    // 对齐 Python: CategoricalDistribution.__eq__ 认为 NaN==NaN
    assert_eq!(a, b);
}

// ============================================================================
// 8. IntersectionSearchSpace 增量计算缓存验证
// ============================================================================

/// 验证增量计算与一次性计算结果一致
#[test]
fn test_intersection_search_space_incremental() {
    let dist_x = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    let dist_y = Distribution::IntDistribution(IntDistribution::new(0, 10, false, 1).unwrap());

    use optuna_rs::distributions::ParamValue;

    // 逐步添加试验
    let trial0 = FrozenTrial {
        number: 0,
        trial_id: 0,
        state: TrialState::Complete,
        values: Some(vec![0.5]),
        datetime_start: None,
        datetime_complete: None,
        params: [("x".into(), ParamValue::Float(0.5)), ("y".into(), ParamValue::Int(5))].into(),
        distributions: [("x".into(), dist_x.clone()), ("y".into(), dist_y.clone())].into(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    };

    let trial1 = FrozenTrial {
        number: 1,
        trial_id: 1,
        state: TrialState::Complete,
        values: Some(vec![0.3]),
        datetime_start: None,
        datetime_complete: None,
        params: [("x".into(), ParamValue::Float(0.3))].into(),
        distributions: [("x".into(), dist_x.clone())].into(),
        user_attrs: HashMap::new(),
        system_attrs: HashMap::new(),
        intermediate_values: HashMap::new(),
    };

    let mut iss = IntersectionSearchSpace::new(false);

    // 第一次计算：只有 trial0
    let result1 = iss.calculate(&[trial0.clone()]);
    assert_eq!(result1.len(), 2);

    // 第二次计算：加入 trial1，y 应被移除
    let result2 = iss.calculate(&[trial0, trial1]);
    assert_eq!(result2.len(), 1);
    assert!(result2.contains_key("x"));
}
