//! 测试工具模块。
//!
//! 对应 Python `optuna.testing`。
//! 提供确定性采样器、剪枝器和测试辅助函数，方便编写单元测试。

use std::collections::HashMap;

use chrono::Utc;

use crate::distributions::{Distribution, ParamValue};
use crate::error::{OptunaError, Result};
use crate::pruners::Pruner;
use crate::samplers::Sampler;
use crate::study::{Study, StudyDirection, create_study};
use crate::trial::{FrozenTrial, Trial, TrialState};

// ============================================================================
// DeterministicSampler
// ============================================================================

/// 确定性采样器 — 返回预设参数值。
///
/// 对应 Python `optuna.testing.samplers.DeterministicSampler`。
/// 用于需要精确控制参数值的测试场景。
pub struct DeterministicSampler {
    params: HashMap<String, f64>,
}

impl DeterministicSampler {
    /// 创建确定性采样器。
    ///
    /// # 参数
    /// * `params` - 参数名到内部值的映射
    pub fn new(params: HashMap<String, f64>) -> Self {
        Self { params }
    }
}

impl Sampler for DeterministicSampler {
    fn infer_relative_search_space(
        &self,
        _trials: &[FrozenTrial],
    ) -> HashMap<String, Distribution> {
        HashMap::new()
    }

    fn sample_relative(
        &self,
        _trials: &[FrozenTrial],
        _search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        Ok(HashMap::new())
    }

    fn sample_independent(
        &self,
        _trials: &[FrozenTrial],
        _trial: &FrozenTrial,
        param_name: &str,
        _distribution: &Distribution,
    ) -> Result<f64> {
        self.params.get(param_name).copied().ok_or_else(|| {
            OptunaError::ValueError(format!(
                "DeterministicSampler: 未预设参数 '{param_name}'"
            ))
        })
    }
}

// ============================================================================
// DeterministicPruner
// ============================================================================

/// 确定性剪枝器 — 始终返回预设的剪枝决策。
///
/// 对应 Python `optuna.testing.pruners.DeterministicPruner`。
pub struct DeterministicPruner {
    is_pruning: bool,
}

impl DeterministicPruner {
    /// 创建确定性剪枝器。
    pub fn new(is_pruning: bool) -> Self {
        Self { is_pruning }
    }
}

impl Pruner for DeterministicPruner {
    fn prune(&self, _study_trials: &[FrozenTrial], _trial: &FrozenTrial, _storage: Option<&dyn crate::storage::Storage>) -> Result<bool> {
        Ok(self.is_pruning)
    }
}

// ============================================================================
// 测试目标函数
// ============================================================================

/// 始终失败的目标函数。
///
/// 对应 Python `optuna.testing.objectives.fail_objective`。
pub fn fail_objective(_trial: &mut Trial) -> Result<f64> {
    Err(OptunaError::ValueError("intentional failure".to_string()))
}

/// 始终被剪枝的目标函数。
///
/// 对应 Python `optuna.testing.objectives.pruned_objective`。
pub fn pruned_objective(trial: &mut Trial) -> Result<f64> {
    trial.report(1.0, 0)?;
    Err(OptunaError::TrialPruned)
}

// ============================================================================
// FrozenTrial 构建器
// ============================================================================

/// 创建测试用 FrozenTrial 的辅助函数。
///
/// 对应 Python `optuna.testing.trials._create_frozen_trial`。
///
/// # 参数
/// * `number` - 试验编号（默认 0）
/// * `values` - 目标值
/// * `constraints` - 约束值
/// * `params` - 参数
/// * `distributions` - 参数分布
/// * `state` - 试验状态（默认 Complete）
pub fn create_frozen_trial(
    number: i64,
    values: Option<Vec<f64>>,
    constraints: Option<Vec<f64>>,
    params: Option<HashMap<String, ParamValue>>,
    distributions: Option<HashMap<String, Distribution>>,
    state: Option<TrialState>,
) -> FrozenTrial {
    let state = state.unwrap_or(TrialState::Complete);
    let mut system_attrs = HashMap::new();

    if let Some(c) = constraints {
        system_attrs.insert(
            crate::multi_objective::CONSTRAINTS_KEY.to_string(),
            serde_json::to_value(c).unwrap(),
        );
    }

    FrozenTrial {
        number,
        state,
        values,
        datetime_start: Some(Utc::now()),
        datetime_complete: if state.is_finished() {
            Some(Utc::now())
        } else {
            None
        },
        params: params.unwrap_or_default(),
        distributions: distributions.unwrap_or_default(),
        user_attrs: HashMap::new(),
        system_attrs,
        intermediate_values: HashMap::new(),
        trial_id: number,
    }
}

// ============================================================================
// 预填充 Study 构造器
// ============================================================================

/// 创建包含预填充试验的 Study。
///
/// 对应 Python `optuna.testing.visualization.prepare_study_with_trials`。
/// 返回一个包含 3 个已完成试验的 Study，用于可视化测试。
///
/// # 试验设置 (与 Python 一致)
/// | # | param_a | param_b | param_c | param_d | value |
/// |---|---------|---------|---------|---------|-------|
/// | 0 | 1.0     | 2.0     | 3.0     | 4.0     | `value_for_first` |
/// | 1 | —       | 0.0     | —       | 4.0     | 2.0 |
/// | 2 | 2.5     | 1.0     | 4.5     | 2.0     | 1.0 |
///
/// Trial 1 intentionally has missing params (param_a, param_c) to test
/// visualization robustness with incomplete data.
pub fn prepare_study_with_trials(
    n_objectives: usize,
    direction: StudyDirection,
    value_for_first_trial: f64,
) -> Result<Study> {
    use crate::distributions::{CategoricalChoice, FloatDistribution};

    let directions: Vec<StudyDirection> = vec![direction; n_objectives];
    let study = create_study(
        None,
        None,
        None,
        None,
        if n_objectives == 1 {
            Some(direction)
        } else {
            None
        },
        if n_objectives > 1 {
            Some(directions)
        } else {
            None
        },
        false,
    )?;

    let dist_a = Distribution::FloatDistribution(FloatDistribution::new(0.0, 3.0, false, None)?);
    let dist_b = Distribution::FloatDistribution(FloatDistribution::new(0.0, 3.0, false, None)?);
    let dist_c = Distribution::FloatDistribution(FloatDistribution::new(2.0, 5.0, false, None)?);
    let dist_d = Distribution::FloatDistribution(FloatDistribution::new(0.0, 5.0, false, None)?);

    // Trial 0: all 4 params, value = value_for_first_trial
    {
        let values = vec![value_for_first_trial; n_objectives];
        let mut params = HashMap::new();
        params.insert("param_a".to_string(), ParamValue::Float(1.0));
        params.insert("param_b".to_string(), ParamValue::Float(2.0));
        params.insert("param_c".to_string(), ParamValue::Float(3.0));
        params.insert("param_d".to_string(), ParamValue::Float(4.0));
        let mut dists = HashMap::new();
        dists.insert("param_a".to_string(), dist_a.clone());
        dists.insert("param_b".to_string(), dist_b.clone());
        dists.insert("param_c".to_string(), dist_c.clone());
        dists.insert("param_d".to_string(), dist_d.clone());
        let trial = create_frozen_trial(0, Some(values), None, Some(params), Some(dists), None);
        study.add_trial(&trial)?;
    }

    // Trial 1: only param_b and param_d (missing param_a, param_c), value = 2.0
    {
        let values = vec![2.0; n_objectives];
        let mut params = HashMap::new();
        params.insert("param_b".to_string(), ParamValue::Float(0.0));
        params.insert("param_d".to_string(), ParamValue::Float(4.0));
        let mut dists = HashMap::new();
        dists.insert("param_b".to_string(), dist_b.clone());
        dists.insert("param_d".to_string(), dist_d.clone());
        let trial = create_frozen_trial(1, Some(values), None, Some(params), Some(dists), None);
        study.add_trial(&trial)?;
    }

    // Trial 2: all 4 params, value = 1.0
    {
        let values = vec![1.0; n_objectives];
        let mut params = HashMap::new();
        params.insert("param_a".to_string(), ParamValue::Float(2.5));
        params.insert("param_b".to_string(), ParamValue::Float(1.0));
        params.insert("param_c".to_string(), ParamValue::Float(4.5));
        params.insert("param_d".to_string(), ParamValue::Float(2.0));
        let mut dists = HashMap::new();
        dists.insert("param_a".to_string(), dist_a.clone());
        dists.insert("param_b".to_string(), dist_b.clone());
        dists.insert("param_c".to_string(), dist_c.clone());
        dists.insert("param_d".to_string(), dist_d.clone());
        let trial = create_frozen_trial(2, Some(values), None, Some(params), Some(dists), None);
        study.add_trial(&trial)?;
    }

    Ok(study)
}

// ============================================================================
// STORAGE_MODES 常量
// ============================================================================

/// 存储模式列表，用于参数化测试。
///
/// 对应 Python `optuna.testing.storages.STORAGE_MODES`。
pub const STORAGE_MODES: &[&str] = &[
    "inmemory",
    #[cfg(feature = "rdb")]
    "sqlite",
    #[cfg(feature = "rdb")]
    "cached_sqlite",
    "journal",
    #[cfg(feature = "redis-storage")]
    "journal_redis",
    #[cfg(feature = "grpc")]
    "grpc_rdb",
];

// ============================================================================
// StorageSupplier — 按名称创建存储实例
// ============================================================================

/// 根据存储模式名称创建存储后端实例。
///
/// 对应 Python `optuna.testing.storages.StorageSupplier`。
/// 返回 `(Arc<dyn Storage>, Option<tempfile::TempDir>)`:
/// - 存储实例
/// - 临时目录句柄（需要的话保持存活以保留临时文件）
///
/// 支持的模式:
/// - `"inmemory"` → `InMemoryStorage`
/// - `"journal"` → `JournalFileStorage`（使用临时文件）
///
/// 需要 feature flag 的模式:
/// - `"sqlite"` → `RdbStorage` (feature `rdb`)
/// - `"cached_sqlite"` → `CachedStorage` 包装 `RdbStorage` (feature `rdb`)
///
/// # 示例
///
/// ```ignore
/// let (storage, _tmpdir) = create_storage("inmemory").unwrap();
/// // 使用 storage...
/// ```
pub fn create_storage(mode: &str) -> Result<(std::sync::Arc<dyn crate::storage::Storage>, Option<tempfile::TempDir>)> {
    use std::sync::Arc;
    use crate::storage::Storage;

    match mode {
        "inmemory" => {
            let s: Arc<dyn Storage> = Arc::new(crate::storage::InMemoryStorage::new());
            Ok((s, None))
        }
        "journal" => {
            let tmpdir = tempfile::tempdir().map_err(|e| {
                crate::error::OptunaError::StorageInternalError(format!("tempdir: {e}"))
            })?;
            let path = tmpdir.path().join("journal.log");
            let s: Arc<dyn Storage> = Arc::new(
                crate::storage::JournalFileStorage::new(path.to_str().unwrap())?
            );
            Ok((s, Some(tmpdir)))
        }
        #[cfg(feature = "rdb")]
        "sqlite" => {
            let tmpdir = tempfile::tempdir().map_err(|e| {
                crate::error::OptunaError::StorageInternalError(format!("tempdir: {e}"))
            })?;
            let path = tmpdir.path().join("test.db");
            let url = format!("sqlite:///{}", path.display());
            let s: Arc<dyn Storage> = Arc::new(
                crate::storage::RdbStorage::new(&url)?
            );
            Ok((s, Some(tmpdir)))
        }
        #[cfg(feature = "rdb")]
        "cached_sqlite" => {
            let tmpdir = tempfile::tempdir().map_err(|e| {
                crate::error::OptunaError::StorageInternalError(format!("tempdir: {e}"))
            })?;
            let path = tmpdir.path().join("test.db");
            let url = format!("sqlite:///{}", path.display());
            let rdb = crate::storage::RdbStorage::new(&url)?;
            let s: Arc<dyn Storage> = Arc::new(
                crate::storage::CachedStorage::new(Arc::new(rdb))
            );
            Ok((s, Some(tmpdir)))
        }
        _ => Err(crate::error::OptunaError::ValueError(
            format!("unsupported storage mode: '{mode}'")
        )),
    }
}

// ============================================================================
// 采样器合规测试套件
// ============================================================================

/// 运行采样器基本合规测试。
///
/// 对应 Python `optuna.testing.pytest_samplers.BasicSamplerTestCase`。
/// 验证采样器能正确处理 float、int、categorical 分布。
///
/// # 参数
/// * `sampler` - 待测试的采样器实例
/// * `n_trials` - 每个测试运行的试验数（默认 10）
pub fn test_sampler_basic(
    sampler: std::sync::Arc<dyn crate::samplers::Sampler>,
    n_trials: Option<usize>,
) -> Result<()> {
    use crate::distributions::{CategoricalChoice, FloatDistribution, IntDistribution,
                                CategoricalDistribution};

    let n = n_trials.unwrap_or(10);

    // 测试 1: float 采样
    {
        let study = create_study(
            None, Some(sampler.clone()), None, None,
            Some(StudyDirection::Minimize), None, false,
        )?;
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
                Ok(x * x)
            },
            Some(n),
            None,
            None,
        )?;
        let trials = study.get_trials(Some(&[TrialState::Complete]))?;
        assert!(!trials.is_empty(), "sampler should produce complete trials for float");
    }

    // 测试 2: int 采样
    {
        let study = create_study(
            None, Some(sampler.clone()), None, None,
            Some(StudyDirection::Minimize), None, false,
        )?;
        study.optimize(
            |trial| {
                let x = trial.suggest_int("x", 0, 10, false, 1)?;
                Ok(x as f64)
            },
            Some(n),
            None,
            None,
        )?;
        let trials = study.get_trials(Some(&[TrialState::Complete]))?;
        assert!(!trials.is_empty(), "sampler should produce complete trials for int");
    }

    // 测试 3: categorical 采样
    {
        let study = create_study(
            None, Some(sampler.clone()), None, None,
            Some(StudyDirection::Minimize), None, false,
        )?;
        study.optimize(
            |trial| {
                let choices = vec![
                    CategoricalChoice::Str("a".to_string()),
                    CategoricalChoice::Str("b".to_string()),
                    CategoricalChoice::Str("c".to_string()),
                ];
                let _x = trial.suggest_categorical("x", choices)?;
                Ok(0.0)
            },
            Some(n),
            None,
            None,
        )?;
        let trials = study.get_trials(Some(&[TrialState::Complete]))?;
        assert!(!trials.is_empty(), "sampler should produce complete trials for categorical");
    }

    Ok(())
}

/// 运行采样器多目标合规测试。
///
/// 对应 Python `optuna.testing.pytest_samplers.MultiObjectiveSamplerTestCase`。
///
/// # 参数
/// * `sampler` - 待测试的采样器实例
/// * `directions` - 优化方向
/// * `n_trials` - 试验数（默认 20）
pub fn test_sampler_multi_objective(
    sampler: std::sync::Arc<dyn crate::samplers::Sampler>,
    directions: Vec<StudyDirection>,
    n_trials: Option<usize>,
) -> Result<()> {
    let n = n_trials.unwrap_or(20);
    let n_obj = directions.len();

    let study = create_study(
        None, Some(sampler), None, None, None, Some(directions), false,
    )?;
    study.optimize_multi(
        |trial| {
            let x = trial.suggest_float("x", 0.0, 10.0, false, None)?;
            let y = trial.suggest_float("y", 0.0, 10.0, false, None)?;
            let mut values = Vec::with_capacity(n_obj);
            values.push(x);
            if n_obj > 1 { values.push(y); }
            for _ in 2..n_obj { values.push(x + y); }
            Ok(values)
        },
        Some(n),
        None,
        None,
    )?;

    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    assert!(!trials.is_empty(), "multi-objective sampler should produce complete trials");

    // 验证所有试验的 values 维度正确
    for t in &trials {
        if let Some(vals) = &t.values {
            assert_eq!(vals.len(), n_obj, "trial values should match n_objectives");
        }
    }

    Ok(())
}

// ============================================================================
// 存储合规测试套件
// ============================================================================

/// 运行存储后端合规测试。
///
/// 对应 Python `optuna.testing.pytest_storages.StorageTestCase`。
/// 验证存储后端实现的 CRUD 操作是否正确。
///
/// # 参数
/// * `storage` - 待测试的存储实例
pub fn test_storage_crud(storage: std::sync::Arc<dyn crate::storage::Storage>) -> Result<()> {
    // 1. Create study
    let study_id = storage.create_new_study(
        &[StudyDirection::Minimize],
        Some("test_study"),
    )?;
    assert!(study_id >= 0);

    // 2. Get study name/id round-trip
    let name = storage.get_study_name_from_id(study_id)?;
    assert_eq!(name, "test_study");
    let id2 = storage.get_study_id_from_name("test_study")?;
    assert_eq!(id2, study_id);

    // 3. Get directions
    let dirs = storage.get_study_directions(study_id)?;
    assert_eq!(dirs, vec![StudyDirection::Minimize]);

    // 4. Study user attrs
    storage.set_study_user_attr(
        study_id, "key1", serde_json::json!("value1"),
    )?;
    let attrs = storage.get_study_user_attrs(study_id)?;
    assert_eq!(attrs.get("key1"), Some(&serde_json::json!("value1")));

    // 5. Study system attrs
    storage.set_study_system_attr(
        study_id, "sys_key", serde_json::json!(42),
    )?;
    let sys_attrs = storage.get_study_system_attrs(study_id)?;
    assert_eq!(sys_attrs.get("sys_key"), Some(&serde_json::json!(42)));

    // 6. Create trial
    let trial_id = storage.create_new_trial(study_id, None)?;
    assert!(trial_id >= 0);

    // 7. Get trial
    let trial = storage.get_trial(trial_id)?;
    assert_eq!(trial.state, TrialState::Running);

    // 8. Set trial param
    let dist = Distribution::FloatDistribution(
        crate::distributions::FloatDistribution::new(0.0, 10.0, false, None)?
    );
    storage.set_trial_param(trial_id, "x", 5.0, &dist)?;

    // 9. Set trial intermediate value
    storage.set_trial_intermediate_value(trial_id, 0, 0.5)?;

    // 10. Set trial user attr
    storage.set_trial_user_attr(trial_id, "note", serde_json::json!("test"))?;

    // 11. Set trial system attr
    storage.set_trial_system_attr(trial_id, "sys", serde_json::json!(true))?;

    // 12. Complete trial
    let ok = storage.set_trial_state_values(trial_id, TrialState::Complete, Some(&[1.0]))?;
    assert!(ok);

    // 13. Verify completed trial
    let trial = storage.get_trial(trial_id)?;
    assert_eq!(trial.state, TrialState::Complete);
    assert_eq!(trial.values, Some(vec![1.0]));
    assert!(trial.params.contains_key("x"));
    assert!(trial.intermediate_values.contains_key(&0));
    assert_eq!(trial.user_attrs.get("note"), Some(&serde_json::json!("test")));

    // 14. Get all trials
    let all_trials = storage.get_all_trials(study_id, None)?;
    assert_eq!(all_trials.len(), 1);

    // 15. Get trials by state filter
    let complete = storage.get_all_trials(study_id, Some(&[TrialState::Complete]))?;
    assert_eq!(complete.len(), 1);
    let running = storage.get_all_trials(study_id, Some(&[TrialState::Running]))?;
    assert_eq!(running.len(), 0);

    // 16. Create second trial, fail it
    let trial_id2 = storage.create_new_trial(study_id, None)?;
    let ok = storage.set_trial_state_values(trial_id2, TrialState::Fail, None)?;
    assert!(ok);
    let failed = storage.get_all_trials(study_id, Some(&[TrialState::Fail]))?;
    assert_eq!(failed.len(), 1);

    // 17. Get all studies
    let all_studies = storage.get_all_studies()?;
    assert!(!all_studies.is_empty());

    // 18. Delete study
    storage.delete_study(study_id)?;

    // 19. Verify deletion — get_study_name_from_id should fail
    assert!(storage.get_study_name_from_id(study_id).is_err());

    Ok(())
}

/// 测试存储后端的并发安全性。
///
/// 对应 Python `optuna.testing.pytest_storages` 中的并发测试。
/// 多线程同时创建试验，验证最终试验数正确。
///
/// # 参数
/// * `storage` - 待测试的存储实例
/// * `n_threads` - 并发线程数（默认 4）
/// * `n_trials_per_thread` - 每线程创建的试验数（默认 5）
pub fn test_storage_concurrent(
    storage: std::sync::Arc<dyn crate::storage::Storage>,
    n_threads: Option<usize>,
    n_trials_per_thread: Option<usize>,
) -> Result<()> {
    let n_threads = n_threads.unwrap_or(4);
    let n_per = n_trials_per_thread.unwrap_or(5);

    let study_id = storage.create_new_study(
        &[StudyDirection::Minimize],
        Some("concurrent_test"),
    )?;

    let mut handles = Vec::new();
    for _ in 0..n_threads {
        let s = storage.clone();
        let handle = std::thread::spawn(move || {
            for i in 0..n_per {
                let tid = s.create_new_trial(study_id, None).unwrap();
                s.set_trial_state_values(tid, TrialState::Complete, Some(&[i as f64]))
                    .unwrap();
            }
        });
        handles.push(handle);
    }

    for h in handles {
        h.join().expect("thread should not panic");
    }

    let trials = storage.get_all_trials(study_id, Some(&[TrialState::Complete]))?;
    assert_eq!(trials.len(), n_threads * n_per);

    Ok(())
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_deterministic_sampler() {
        let mut params = HashMap::new();
        params.insert("x".to_string(), 0.5);
        let sampler = DeterministicSampler::new(params);

        let dist = Distribution::FloatDistribution(crate::distributions::FloatDistribution {
            low: 0.0,
            high: 1.0,
            step: None,
            log: false,
        });
        let trial = create_frozen_trial(0, Some(vec![0.0]), None, None, None, None);
        let val = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
        assert!((val - 0.5).abs() < 1e-10);

        // 未知参数应返回错误
        assert!(sampler.sample_independent(&[], &trial, "y", &dist).is_err());
    }

    #[test]
    fn test_deterministic_pruner() {
        let pruner_yes = DeterministicPruner::new(true);
        let pruner_no = DeterministicPruner::new(false);

        let study = create_study(None, None, None, None, Some(StudyDirection::Minimize), None, false).unwrap();
        let trial = create_frozen_trial(0, Some(vec![0.0]), None, None, None, None);
        let trials = study.get_trials(None).unwrap();

        assert!(pruner_yes.prune(&trials, &trial, None).unwrap());
        assert!(!pruner_no.prune(&trials, &trial, None).unwrap());
    }

    #[test]
    fn test_create_frozen_trial_defaults() {
        let trial = create_frozen_trial(0, None, None, None, None, None);
        assert_eq!(trial.number, 0);
        assert_eq!(trial.state, TrialState::Complete);
        assert!(trial.params.is_empty());
    }

    #[test]
    fn test_create_frozen_trial_with_constraints() {
        let trial = create_frozen_trial(1, Some(vec![1.0]), Some(vec![0.0, -1.0]), None, None, None);
        assert!(trial.system_attrs.contains_key(crate::multi_objective::CONSTRAINTS_KEY));
    }

    #[test]
    fn test_fail_objective() {
        let study = create_study(None, None, None, None, Some(StudyDirection::Minimize), None, false).unwrap();
        // 默认 optimize 不设 catch，ValueError 会向上传播（对齐 Python 行为）
        let result = study.optimize(fail_objective, Some(3), None, None);
        assert!(result.is_err());
        // 使用 optimize_with_options + catch=["*"] 时应静默捕获
        let study2 = create_study(None, None, None, None, Some(StudyDirection::Minimize), None, false).unwrap();
        let result2 = study2.optimize_with_options(fail_objective, Some(3), None, 1, &["*"], None, false);
        assert!(result2.is_ok());
        let trials = study2.get_trials(Some(&[TrialState::Fail])).unwrap();
        assert_eq!(trials.len(), 3);
    }

    #[test]
    fn test_pruned_objective() {
        let study = create_study(None, None, None, None, Some(StudyDirection::Minimize), None, false).unwrap();
        let result = study.optimize(pruned_objective, Some(3), None, None);
        assert!(result.is_ok());
        let trials = study.get_trials(Some(&[TrialState::Pruned])).unwrap();
        assert_eq!(trials.len(), 3);
    }

    #[test]
    fn test_create_storage_inmemory() {
        let (storage, tmpdir) = create_storage("inmemory").unwrap();
        assert!(tmpdir.is_none());
        // 验证可以创建 study
        let id = storage.create_new_study(
            &[StudyDirection::Minimize],
            Some("test"),
        ).unwrap();
        assert!(id >= 0);
    }

    #[test]
    fn test_create_storage_journal() {
        let (storage, tmpdir) = create_storage("journal").unwrap();
        assert!(tmpdir.is_some());
        let id = storage.create_new_study(
            &[StudyDirection::Minimize],
            Some("test_journal"),
        ).unwrap();
        assert!(id >= 0);
    }

    #[test]
    fn test_sampler_basic_random() {
        let sampler: std::sync::Arc<dyn crate::samplers::Sampler> =
            std::sync::Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        test_sampler_basic(sampler, Some(5)).unwrap();
    }

    #[test]
    fn test_sampler_multi_objective_random() {
        let sampler: std::sync::Arc<dyn crate::samplers::Sampler> =
            std::sync::Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        test_sampler_multi_objective(
            sampler,
            vec![StudyDirection::Minimize, StudyDirection::Maximize],
            Some(5),
        ).unwrap();
    }

    #[test]
    fn test_storage_crud_inmemory() {
        let (storage, _) = create_storage("inmemory").unwrap();
        test_storage_crud(storage).unwrap();
    }

    #[test]
    fn test_storage_crud_journal() {
        let (storage, _tmpdir) = create_storage("journal").unwrap();
        test_storage_crud(storage).unwrap();
    }

    #[test]
    fn test_storage_concurrent_inmemory() {
        let (storage, _) = create_storage("inmemory").unwrap();
        test_storage_concurrent(storage, Some(4), Some(5)).unwrap();
    }
}
