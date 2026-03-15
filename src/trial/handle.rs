use std::collections::HashMap;
use std::sync::Arc;

use crate::distributions::{
    CategoricalChoice, CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
    ParamValue,
};
use crate::error::{OptunaError, Result};
use crate::pruners::Pruner;
use crate::samplers::Sampler;
use crate::storage::Storage;
use crate::study::StudyDirection;

/// 运行中试验的可变句柄。
///
/// 对应 Python `optuna.trial.Trial`。由 `Study::ask()` 创建。
/// 提供 `suggest_*` 方法记录采样参数到存储，
/// `report` 方法记录中间值，`should_prune` 方法查询剪枝器。
pub struct Trial {
    trial_id: i64,
    study_id: i64,
    storage: Arc<dyn Storage>,
    sampler: Arc<dyn Sampler>,
    pruner: Arc<dyn Pruner>,
    number: i64,
    /// 研究的优化方向列表（用于多目标检查）。
    directions: Vec<StudyDirection>,
    /// Relative param values pre-sampled by the sampler (internal repr).
    relative_params: HashMap<String, f64>,
    /// Relative search space inferred by sampler.
    relative_search_space: HashMap<String, Distribution>,
    /// Fixed params injected by enqueue_trial (external repr).
    fixed_params: HashMap<String, ParamValue>,
}

impl Trial {
    /// Create a new `Trial`. Called internally by `Study::ask()`.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        trial_id: i64,
        study_id: i64,
        number: i64,
        storage: Arc<dyn Storage>,
        sampler: Arc<dyn Sampler>,
        pruner: Arc<dyn Pruner>,
        directions: Vec<StudyDirection>,
        relative_params: HashMap<String, f64>,
        relative_search_space: HashMap<String, Distribution>,
        fixed_params: HashMap<String, ParamValue>,
    ) -> Self {
        Self {
            trial_id,
            study_id,
            storage,
            sampler,
            pruner,
            number,
            directions,
            relative_params,
            relative_search_space,
            fixed_params,
        }
    }

    /// The trial's unique id.
    pub fn trial_id(&self) -> i64 {
        self.trial_id
    }

    /// The trial's number within the study (0-indexed).
    pub fn number(&self) -> i64 {
        self.number
    }

    /// Suggest a floating-point parameter.
    pub fn suggest_float(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        log: bool,
        step: Option<f64>,
    ) -> Result<f64> {
        let dist = Distribution::FloatDistribution(FloatDistribution::new(low, high, log, step)?);
        let internal = self.suggest(name, &dist)?;
        dist.to_external_repr(internal).map(|v| match v {
            crate::distributions::ParamValue::Float(f) => f,
            _ => unreachable!(),
        })
    }

    /// 对齐 Python `Trial.suggest_float(name, low, high, step=None, log=False)`。
    pub fn suggest_float_py(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        step: Option<f64>,
        log: bool,
    ) -> Result<f64> {
        self.suggest_float(name, low, high, log, step)
    }

    /// 对齐 Python 默认参数形式：`suggest_float(name, low, high)`。
    pub fn suggest_float_default(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest_float_py(name, low, high, None, false)
    }

    /// 对齐 Python `suggest_float(..., step=...)` 的便捷入口。
    pub fn suggest_float_step(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        step: f64,
    ) -> Result<f64> {
        self.suggest_float_py(name, low, high, Some(step), false)
    }

    /// 对齐 Python `suggest_float(..., log=True)` 的便捷入口。
    pub fn suggest_float_log(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest_float_py(name, low, high, None, true)
    }

    /// 对齐 Python 已弃用别名 `suggest_uniform()`。
    pub fn suggest_uniform(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest_float_default(name, low, high)
    }

    /// 对齐 Python 已弃用别名 `suggest_loguniform()`。
    pub fn suggest_loguniform(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest_float_log(name, low, high)
    }

    /// 对齐 Python 已弃用别名 `suggest_discrete_uniform()`。
    pub fn suggest_discrete_uniform(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        q: f64,
    ) -> Result<f64> {
        self.suggest_float_step(name, low, high, q)
    }

    /// Suggest an integer parameter.
    pub fn suggest_int(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        log: bool,
        step: i64,
    ) -> Result<i64> {
        let dist = Distribution::IntDistribution(IntDistribution::new(low, high, log, step)?);
        let internal = self.suggest(name, &dist)?;
        dist.to_external_repr(internal).map(|v| match v {
            crate::distributions::ParamValue::Int(i) => i,
            _ => unreachable!(),
        })
    }

    /// 对齐 Python `Trial.suggest_int(name, low, high, step=1, log=False)`。
    pub fn suggest_int_py(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        step: i64,
        log: bool,
    ) -> Result<i64> {
        self.suggest_int(name, low, high, log, step)
    }

    /// 对齐 Python 默认参数形式：`suggest_int(name, low, high)`。
    pub fn suggest_int_default(&mut self, name: &str, low: i64, high: i64) -> Result<i64> {
        self.suggest_int_py(name, low, high, 1, false)
    }

    /// 对齐 Python `suggest_int(..., step=...)` 的便捷入口。
    pub fn suggest_int_step(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        step: i64,
    ) -> Result<i64> {
        self.suggest_int_py(name, low, high, step, false)
    }

    /// 对齐 Python `suggest_int(..., log=True)` 的便捷入口。
    pub fn suggest_int_log(&mut self, name: &str, low: i64, high: i64) -> Result<i64> {
        self.suggest_int_py(name, low, high, 1, true)
    }

    /// Suggest a categorical parameter.
    pub fn suggest_categorical(
        &mut self,
        name: &str,
        choices: Vec<CategoricalChoice>,
    ) -> Result<CategoricalChoice> {
        let dist =
            Distribution::CategoricalDistribution(CategoricalDistribution::new(choices)?);
        let internal = self.suggest(name, &dist)?;
        dist.to_external_repr(internal).map(|v| match v {
            crate::distributions::ParamValue::Categorical(c) => c,
            _ => unreachable!(),
        })
    }

    /// Core suggest logic: check if already suggested or in relative params,
    /// otherwise fall back to independent sampling.
    ///
    /// 对应 Python `Trial._suggest()`。
    /// 使用 check_distribution_compatibility() 进行兼容性检查，
    /// 允许同类型但不同 range 的分布（对齐 Python 行为）。
    fn suggest(&mut self, name: &str, dist: &Distribution) -> Result<f64> {
        // Check if this param was already set (re-suggest returns same value)
        let existing = self.storage.get_trial(self.trial_id)?;
        if let Some(existing_dist) = existing.distributions.get(name) {
            // 对齐 Python: 同名参数若分布不一致, 仅发出警告并沿用首次分布/取值。
            if existing_dist != dist {
                crate::optuna_warn!(
                    "Inconsistent parameter values for distribution with name \"{}\"! This might be a configuration mistake. Optuna allows to call the same distribution with the same name more than once in a trial. When the parameter values are inconsistent optuna only uses the values of the first call and ignores all following. Using these values: {:?}",
                    name,
                    existing_dist
                );
            }
            // 兼容性检查：允许同类型不同 range（对齐 Python）
            crate::distributions::check_distribution_compatibility(existing_dist, dist)?;
            let val = existing.params.get(name).unwrap();
            return dist.to_internal_repr(val);
        }

        // Python `_is_fixed_param` 对齐：固定参数优先，越界仅告警。
        let internal = if let Some(value) = self.fixed_params.get(name) {
            let internal = dist.to_internal_repr(value)?;
            if !dist.contains(internal) {
                crate::optuna_warn!(
                    "Fixed parameter {} with value {:?} is out of range for distribution {:?}.",
                    name,
                    value,
                    dist
                );
            }
            internal
        // Python `_is_relative_param` 对齐：检查 relative search space 与分布兼容性。
        } else if let Some(&v) = self.relative_params.get(name) {
            if !self.relative_search_space.contains_key(name) {
                return Err(OptunaError::ValueError(format!(
                    "The parameter {name} was sampled by `sample_relative` method but it is not contained in the relative search space."
                )));
            }

            let relative_distribution = self.relative_search_space.get(name).unwrap();
            crate::distributions::check_distribution_compatibility(relative_distribution, dist)?;

            // Python: relative param 若不在请求分布范围内，则回退独立采样。
            if dist.contains(v) {
                v
            } else {
                let all_trials = self.storage.get_all_trials(self.study_id, None)?;
                self.sampler.sample_independent(&all_trials, &existing, name, dist)?
            }
        } else {
            // Fall back to independent sampling
            // 获取所有历史试验供采样器参考（对齐 Python study._get_trials）
            let all_trials = self.storage.get_all_trials(self.study_id, None)?;
            self.sampler.sample_independent(&all_trials, &existing, name, dist)?
        };

        // Record the param in storage
        self.storage
            .set_trial_param(self.trial_id, name, internal, dist)?;
        Ok(internal)
    }

    /// Report an intermediate objective value at a given step.
    ///
    /// 对应 Python `Trial.report()`。
    /// step 必须 >= 0，多目标优化时不可调用 report。
    /// 同一 step 重复 report 时忽略并发出警告（对齐 Python）。
    pub fn report(&self, value: f64, step: i64) -> Result<()> {
        // 多目标优化时禁止调用 report（对齐 Python NotImplementedError）
        if self.directions.len() > 1 {
            return Err(OptunaError::NotImplemented(
                "Trial.report is not supported for multi-objective optimization.".into(),
            ));
        }
        if step < 0 {
            return Err(OptunaError::ValueError(format!(
                "The `step` argument is {step} but cannot be negative."
            )));
        }
        // 检查是否已经 report 过该 step（对齐 Python：重复 report 忽略并警告）
        let trial = self.storage.get_trial(self.trial_id)?;
        if trial.intermediate_values.contains_key(&step) {
            crate::optuna_warn!(
                "The reported value is ignored because this `step` {} is already reported.",
                step
            );
            return Ok(());
        }
        self.storage
            .set_trial_intermediate_value(self.trial_id, step, value)
    }

    /// Check if the trial should be pruned.
    ///
    /// 对应 Python `Trial.should_prune()`。
    /// 多目标优化时禁止调用（对齐 Python NotImplementedError）。
    pub fn should_prune(&self) -> Result<bool> {
        if self.directions.len() > 1 {
            return Err(OptunaError::NotImplemented(
                "Trial.should_prune is not supported for multi-objective optimization.".into(),
            ));
        }
        // 对齐 Python: 传入所有状态的试验（SuccessiveHalving 需要 Running 试验的信息）
        let all_trials = self
            .storage
            .get_all_trials(self.study_id, None)?;
        let trial = self.storage.get_trial(self.trial_id)?;
        self.pruner.prune(&all_trials, &trial, Some(self.storage.as_ref()))
    }

    /// Set a user attribute on the trial.
    pub fn set_user_attr(&self, key: &str, value: serde_json::Value) -> Result<()> {
        self.storage.set_trial_user_attr(self.trial_id, key, value)
    }

    /// Set a system attribute on the trial.
    pub fn set_system_attr(&self, key: &str, value: serde_json::Value) -> Result<()> {
        self.storage
            .set_trial_system_attr(self.trial_id, key, value)
    }

    // ── 属性访问器 (对应 Python Trial 的 property) ──

    /// 返回当前已建议的参数字典。
    /// 对应 Python `Trial.params`.
    pub fn params(
        &self,
    ) -> Result<HashMap<String, crate::distributions::ParamValue>> {
        let ft = self.storage.get_trial(self.trial_id)?;
        Ok(ft.params)
    }

    /// 返回当前已建议的参数分布字典。
    /// 对应 Python `Trial.distributions`.
    pub fn distributions(&self) -> Result<HashMap<String, Distribution>> {
        let ft = self.storage.get_trial(self.trial_id)?;
        Ok(ft.distributions)
    }

    /// 返回用户属性字典。
    /// 对应 Python `Trial.user_attrs`.
    pub fn user_attrs(&self) -> Result<HashMap<String, serde_json::Value>> {
        let ft = self.storage.get_trial(self.trial_id)?;
        Ok(ft.user_attrs)
    }

    /// 返回系统属性字典。
    /// 对应 Python `Trial.system_attrs` (deprecated in Python 3.1.0).
    pub fn system_attrs(&self) -> Result<HashMap<String, serde_json::Value>> {
        let ft = self.storage.get_trial(self.trial_id)?;
        Ok(ft.system_attrs)
    }

    /// 返回试验开始时间。
    /// 对应 Python `Trial.datetime_start`.
    pub fn datetime_start(&self) -> Result<Option<chrono::DateTime<chrono::Utc>>> {
        let ft = self.storage.get_trial(self.trial_id)?;
        Ok(ft.datetime_start)
    }

    /// 返回相对采样参数（内部表示）。
    /// 对应 Python `Trial.relative_params` 的只读语义。
    pub fn relative_params_internal(&self) -> HashMap<String, f64> {
        self.relative_params.clone()
    }

    /// 返回 fixed 参数（外部表示）。
    /// 对应 Python 从 trial system attrs 读取 fixed_params 的可见语义。
    pub fn fixed_params(&self) -> HashMap<String, ParamValue> {
        self.fixed_params.clone()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::distributions::Distribution;
    use crate::error::Result;
    use crate::pruners::Pruner;
    use crate::samplers::Sampler;
    use crate::study::{StudyDirection, create_study};
    use crate::trial::FrozenTrial;

    #[test]
    fn test_trial_float_python_compat_wrappers() {
        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let mut trial = study.ask(None).unwrap();
        let _ = trial.suggest_float_default("x", 0.0, 1.0).unwrap();
        let _ = trial.suggest_loguniform("y", 0.1, 10.0).unwrap();
        let _ = trial.suggest_discrete_uniform("z", 0.0, 1.0, 0.25).unwrap();
        let _ = trial.suggest_float_py("w", -1.0, 1.0, Some(0.5), false).unwrap();

        let dists = trial.distributions().unwrap();

        match dists.get("x").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, None);
            }
            _ => panic!("x should use float distribution"),
        }
        match dists.get("y").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(dist.log);
                assert_eq!(dist.step, None);
            }
            _ => panic!("y should use float distribution"),
        }
        match dists.get("z").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, Some(0.25));
            }
            _ => panic!("z should use float distribution"),
        }
        match dists.get("w").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, Some(0.5));
            }
            _ => panic!("w should use float distribution"),
        }
    }

    #[test]
    fn test_trial_int_python_compat_wrappers() {
        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let mut trial = study.ask(None).unwrap();
        let _ = trial.suggest_int_default("a", 1, 5).unwrap();
        let _ = trial.suggest_int_log("b", 1, 8).unwrap();
        let _ = trial.suggest_int_step("c", 0, 10, 2).unwrap();
        let _ = trial.suggest_int_py("d", 3, 9, 3, false).unwrap();

        let dists = trial.distributions().unwrap();

        match dists.get("a").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, 1);
            }
            _ => panic!("a should use int distribution"),
        }
        match dists.get("b").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(dist.log);
                assert_eq!(dist.step, 1);
            }
            _ => panic!("b should use int distribution"),
        }
        match dists.get("c").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, 2);
            }
            _ => panic!("c should use int distribution"),
        }
        match dists.get("d").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, 3);
            }
            _ => panic!("d should use int distribution"),
        }
    }

    #[test]
    fn test_report_same_step_ignored_and_keeps_first_value() {
        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let trial = study.ask(None).unwrap();
        trial.report(0.1, 0).unwrap();
        trial.report(0.9, 0).unwrap();

        let frozen = study
            .tell(trial.trial_id(), crate::trial::TrialState::Pruned, None)
            .unwrap();
        assert_eq!(frozen.intermediate_values.len(), 1);
        assert_eq!(frozen.intermediate_values.get(&0).copied(), Some(0.1));
    }

    #[test]
    fn test_resuggest_same_param_with_different_range_keeps_first_value() {
        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let mut trial = study.ask(None).unwrap();
        let first = trial.suggest_float_default("x", 0.0, 1.0).unwrap();
        let second = trial.suggest_float_default("x", -10.0, 10.0).unwrap();
        assert_eq!(first, second);

        let dists = trial.distributions().unwrap();
        match dists.get("x").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert_eq!(dist.low, 0.0);
                assert_eq!(dist.high, 1.0);
            }
            _ => panic!("x should use float distribution"),
        }
    }

    #[test]
    fn test_relative_params_accessor() {
        let mut fixed = HashMap::new();
        fixed.insert(
            "x".to_string(),
            Distribution::FloatDistribution(
                crate::distributions::FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
            ),
        );

        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let trial = study.ask(Some(&fixed)).unwrap();
        let rel = trial.relative_params_internal();
        assert!(rel.contains_key("x"));
    }

    #[test]
    fn test_fixed_params_from_enqueue_trial() {
        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let mut params = HashMap::new();
        params.insert("x".to_string(), crate::distributions::ParamValue::Float(0.8));
        study.enqueue_trial(params, None, false).unwrap();

        let mut trial = study.ask(None).unwrap();
        let x = trial.suggest_float_default("x", 0.0, 1.0).unwrap();
        assert!((x - 0.8).abs() < 1e-12);

        let fixed = trial.fixed_params();
        assert!(fixed.contains_key("x"));
    }

    #[test]
    fn test_fixed_param_out_of_range_still_used() {
        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let mut params = HashMap::new();
        params.insert("x".to_string(), crate::distributions::ParamValue::Float(2.0));
        study.enqueue_trial(params, None, false).unwrap();

        let mut trial = study.ask(None).unwrap();
        let x = trial.suggest_float_default("x", 0.0, 1.0).unwrap();
        assert!((x - 2.0).abs() < 1e-12);
    }

    #[derive(Debug)]
    struct RelativeFallbackSampler {
        independent_calls: Arc<AtomicUsize>,
    }

    impl Sampler for RelativeFallbackSampler {
        fn infer_relative_search_space(
            &self,
            _trials: &[FrozenTrial],
        ) -> HashMap<String, Distribution> {
            let mut m = HashMap::new();
            m.insert(
                "x".to_string(),
                Distribution::FloatDistribution(
                    crate::distributions::FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
                ),
            );
            m
        }

        fn sample_relative(
            &self,
            _trials: &[FrozenTrial],
            _search_space: &HashMap<String, Distribution>,
        ) -> Result<HashMap<String, f64>> {
            let mut m = HashMap::new();
            // 故意返回超范围值，触发 Trial::suggest 的 independent 回退路径。
            m.insert("x".to_string(), 2.0);
            Ok(m)
        }

        fn sample_independent(
            &self,
            _trials: &[FrozenTrial],
            _trial: &FrozenTrial,
            _param_name: &str,
            _distribution: &Distribution,
        ) -> Result<f64> {
            self.independent_calls.fetch_add(1, Ordering::Relaxed);
            Ok(0.25)
        }
    }

    #[derive(Debug)]
    struct NeverPrune;
    impl Pruner for NeverPrune {
        fn prune(
            &self,
            _study_trials: &[FrozenTrial],
            _trial: &FrozenTrial,
            _storage: Option<&dyn crate::storage::Storage>,
        ) -> Result<bool> {
            Ok(false)
        }
    }

    #[test]
    fn test_relative_param_out_of_range_falls_back_to_independent() {
        let calls = Arc::new(AtomicUsize::new(0));
        let sampler: Arc<dyn Sampler> = Arc::new(RelativeFallbackSampler {
            independent_calls: Arc::clone(&calls),
        });
        let pruner: Arc<dyn Pruner> = Arc::new(NeverPrune);

        let study = create_study(
            None,
            Some(sampler),
            Some(pruner),
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let mut trial = study.ask(None).unwrap();
        let x = trial.suggest_float_default("x", 0.0, 1.0).unwrap();
        assert!((x - 0.25).abs() < 1e-12);
        assert_eq!(calls.load(Ordering::Relaxed), 1);
    }
}
