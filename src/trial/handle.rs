use std::collections::HashMap;
use std::sync::Arc;

use crate::distributions::{
    CategoricalChoice, CategoricalDistribution, Distribution, FloatDistribution, IntDistribution,
};
use crate::error::{OptunaError, Result};
use crate::pruners::Pruner;
use crate::samplers::Sampler;
use crate::storage::Storage;
use crate::study::StudyDirection;
use crate::trial::TrialState;

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
            // 兼容性检查：允许同类型不同 range（对齐 Python）
            crate::distributions::check_distribution_compatibility(existing_dist, dist)?;
            let val = existing.params.get(name).unwrap();
            return dist.to_internal_repr(val);
        }

        // Check if we have a pre-sampled relative param
        let internal = if let Some(&v) = self.relative_params.get(name) {
            v
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
                "step must be non-negative, got {step}"
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
}
