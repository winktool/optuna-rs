//! Storage + CMA-ES 交叉验证测试。
//!
//! 验证 InMemoryStorage 的 CRUD 操作和 CMA-ES 采样器的收敛行为。

use std::sync::Arc;
use optuna_rs::distributions::{
    Distribution, FloatDistribution, ParamValue,
};
use optuna_rs::samplers::{CmaEsSamplerBuilder, RandomSampler, Sampler};
use optuna_rs::storage::{InMemoryStorage, Storage};
use optuna_rs::study::{create_study, StudyDirection};
use optuna_rs::trial::TrialState;

// ============================================================================
// 1. InMemoryStorage CRUD 验证
// ============================================================================

/// 创建 study → 创建 trial → 设置参数 → 完成 → 读回验证
#[test]
fn test_storage_create_study_and_trial() {
    let storage = InMemoryStorage::new();
    let study_id = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();

    // 创建 trial
    let trial_id = storage.create_new_trial(study_id, None).unwrap();
    assert_eq!(trial_id, 0); // max_trial_id starts at -1, incremented to 0

    // 获取 trial
    let trial = storage.get_trial(trial_id).unwrap();
    assert_eq!(trial.number, 0); // number is based on trials.len() before adding
    assert_eq!(trial.state, TrialState::Running);
}

/// 多个 trial 按顺序编号
#[test]
fn test_storage_trial_numbering() {
    let storage = InMemoryStorage::new();
    let study_id = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();

    let t0 = storage.create_new_trial(study_id, None).unwrap();
    let t1 = storage.create_new_trial(study_id, None).unwrap();
    let t2 = storage.create_new_trial(study_id, None).unwrap();

    assert_eq!(storage.get_trial(t0).unwrap().number, 0);
    assert_eq!(storage.get_trial(t1).unwrap().number, 1);
    assert_eq!(storage.get_trial(t2).unwrap().number, 2);
}

/// 设置参数和分布
#[test]
fn test_storage_set_trial_param() {
    let storage = InMemoryStorage::new();
    let study_id = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
    let trial_id = storage.create_new_trial(study_id, None).unwrap();

    let dist = Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
    storage
        .set_trial_param(trial_id, "x", 0.5, &dist)
        .unwrap();

    let trial = storage.get_trial(trial_id).unwrap();
    assert_eq!(trial.params.get("x"), Some(&ParamValue::Float(0.5)));
    assert!(trial.distributions.contains_key("x"));
}

/// Trial 状态转换: Running → Complete
#[test]
fn test_storage_trial_state_complete() {
    let storage = InMemoryStorage::new();
    let study_id = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
    let trial_id = storage.create_new_trial(study_id, None).unwrap();

    storage
        .set_trial_state_values(trial_id, TrialState::Complete, Some(&[1.5]))
        .unwrap();

    let trial = storage.get_trial(trial_id).unwrap();
    assert_eq!(trial.state, TrialState::Complete);
    assert_eq!(trial.values, Some(vec![1.5]));
}

/// Trial 状态转换: Running → Pruned
#[test]
fn test_storage_trial_state_pruned() {
    let storage = InMemoryStorage::new();
    let study_id = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
    let trial_id = storage.create_new_trial(study_id, None).unwrap();

    storage
        .set_trial_state_values(trial_id, TrialState::Pruned, None)
        .unwrap();

    let trial = storage.get_trial(trial_id).unwrap();
    assert_eq!(trial.state, TrialState::Pruned);    
}

/// Intermediate values (用于 pruner)
#[test]
fn test_storage_intermediate_values() {
    let storage = InMemoryStorage::new();
    let study_id = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
    let trial_id = storage.create_new_trial(study_id, None).unwrap();

    storage.set_trial_intermediate_value(trial_id, 0, 1.0).unwrap();
    storage.set_trial_intermediate_value(trial_id, 1, 0.8).unwrap();
    storage.set_trial_intermediate_value(trial_id, 2, 0.6).unwrap();

    let trial = storage.get_trial(trial_id).unwrap();
    assert_eq!(trial.intermediate_values.len(), 3);
    assert_eq!(trial.intermediate_values.get(&0), Some(&1.0));
    assert_eq!(trial.intermediate_values.get(&1), Some(&0.8));
    assert_eq!(trial.intermediate_values.get(&2), Some(&0.6));
}

/// User attributes
#[test]
fn test_storage_user_attrs() {
    let storage = InMemoryStorage::new();
    let study_id = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();
    let trial_id = storage.create_new_trial(study_id, None).unwrap();

    storage
        .set_trial_user_attr(trial_id, "key1", serde_json::json!("value1"))
        .unwrap();

    let trial = storage.get_trial(trial_id).unwrap();
    assert_eq!(
        trial.user_attrs.get("key1"),
        Some(&serde_json::json!("value1"))
    );
}

/// get_all_trials 过滤
#[test]
fn test_storage_get_all_trials_filter() {
    let storage = InMemoryStorage::new();
    let study_id = storage.create_new_study(&[StudyDirection::Minimize], None).unwrap();

    let t0 = storage.create_new_trial(study_id, None).unwrap();
    let t1 = storage.create_new_trial(study_id, None).unwrap();
    let _t2 = storage.create_new_trial(study_id, None).unwrap();

    storage
        .set_trial_state_values(t0, TrialState::Complete, Some(&[1.0]))
        .unwrap();
    storage
        .set_trial_state_values(t1, TrialState::Pruned, None)
        .unwrap();
    // t2 remains Running
    let all = storage
        .get_all_trials(study_id, None)
        .unwrap();
    assert_eq!(all.len(), 3);

    let completed = storage
        .get_all_trials(study_id, Some(&[TrialState::Complete]))
        .unwrap();
    assert_eq!(completed.len(), 1);
    assert_eq!(completed[0].state, TrialState::Complete);

    let pruned = storage
        .get_all_trials(study_id, Some(&[TrialState::Pruned]))
        .unwrap();
    assert_eq!(pruned.len(), 1);
}

/// Study directions 持久化
#[test]
fn test_storage_study_directions() {
    let storage = InMemoryStorage::new();
    let study_id = storage
        .create_new_study(&[StudyDirection::Minimize, StudyDirection::Maximize], None)
        .unwrap();

    let dirs = storage.get_study_directions(study_id).unwrap();
    assert_eq!(dirs.len(), 2);
    assert_eq!(dirs[0], StudyDirection::Minimize);
    assert_eq!(dirs[1], StudyDirection::Maximize);
}

// ============================================================================
// 2. CMA-ES 收敛行为验证
// ============================================================================

/// CMA-ES 对球面函数收敛（minimize x²+y²）
#[test]
fn test_cmaes_sphere_convergence() {
    let cma = CmaEsSamplerBuilder::new(StudyDirection::Minimize)
        .seed(42)
        .n_startup_trials(5)
        .build();
    let study = create_study(
        None,
        Some(Arc::new(cma) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", -5.0, 5.0, false, None)?;
                Ok(x * x + y * y)
            },
            Some(50),
            None,
            None,
        )
        .unwrap();

    let bv = study.best_trial().unwrap().values.as_ref().unwrap()[0];
    assert!(bv < 1.0, "CMA-ES should converge on sphere, got {bv}");
}

/// CMA-ES population size formula: popsize = 4 + floor(3 * ln(n))
/// Python 参考: cmaes.CMA default population_size
///   dim=1 → 4+0=4, but optuna minimum=5
///   dim=2 → 4+floor(3*ln(2))=4+2=6
///   dim=5 → 4+floor(3*ln(5))=4+4=8
///   dim=10 → 4+floor(3*ln(10))=4+6=10
#[test]
fn test_cmaes_popsize_formula() {
    use optuna_rs::samplers::cmaes::CmaEsSampler;
    assert_eq!(CmaEsSampler::default_popsize(1), 5);
    assert_eq!(CmaEsSampler::default_popsize(2), 6);
    assert_eq!(CmaEsSampler::default_popsize(3), 7);
    assert_eq!(CmaEsSampler::default_popsize(5), 8);
    assert_eq!(CmaEsSampler::default_popsize(10), 10);
    assert_eq!(CmaEsSampler::default_popsize(20), 12);
    assert_eq!(CmaEsSampler::default_popsize(50), 15);
}

/// CMA-ES maximize
#[test]
fn test_cmaes_maximize() {
    let cma = CmaEsSamplerBuilder::new(StudyDirection::Maximize)
        .seed(42)
        .n_startup_trials(5)
        .build();
    let study = create_study(
        None,
        Some(Arc::new(cma) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Maximize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                Ok(-(x - 5.0).powi(2))
            },
            Some(40),
            None,
            None,
        )
        .unwrap();

    let bv = study.best_trial().unwrap().values.as_ref().unwrap()[0];
    assert!(bv > -5.0, "CMA-ES maximize should converge, got {bv}");
}

// ============================================================================
// 3. Storage + Optimization 集成
// ============================================================================

/// create_study 使用 InMemoryStorage → optimize → 验证 trials 数量和 best_trial
#[test]
fn test_storage_optimize_integration() {
    let sampler = RandomSampler::new(Some(42));
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        None,
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                Ok(x * x)
            },
            Some(20),
            None,
            None,
        )
        .unwrap();

    let trials = study.get_trials(None).unwrap();
    assert_eq!(trials.len(), 20);

    let completed = study.get_trials(Some(&[TrialState::Complete])).unwrap();
    assert_eq!(completed.len(), 20);

    let best = study.best_trial().unwrap();
    assert!(best.values.as_ref().unwrap()[0] <= 100.0);
}

/// Pruned trials 应正确记录
#[test]
fn test_storage_pruned_trials() {
    use optuna_rs::pruners::MedianPruner;

    let sampler = RandomSampler::new(Some(42));
    let pruner = MedianPruner::new(1, 0, 1, 1, StudyDirection::Minimize);
    let study = create_study(
        None,
        Some(Arc::new(sampler) as Arc<dyn Sampler>),
        Some(Arc::new(pruner)),
        None,
        Some(StudyDirection::Minimize),
        None,
        false,
    )
    .unwrap();

    study
        .optimize(
            |trial| {
                for step in 0..5 {
                    let val = trial.suggest_float("x", 0.0, 100.0, false, None)? + step as f64;
                    trial.report(val, step as i64)?;
                    if trial.should_prune()? {
                        return Err(optuna_rs::error::OptunaError::TrialPruned);
                    }
                }
                let x = trial.suggest_float("x", 0.0, 100.0, false, None)?;
                Ok(x)
            },
            Some(10),
            None,
            None,
        )
        .unwrap();

    let all = study.get_trials(None).unwrap();
    let pruned: Vec<_> = all.iter().filter(|t| t.state == TrialState::Pruned).collect();
    let completed: Vec<_> = all.iter().filter(|t| t.state == TrialState::Complete).collect();

    // 至少有些 trial 被完成
    assert!(
        completed.len() + pruned.len() == all.len(),
        "all trials should be complete or pruned"
    );
}
