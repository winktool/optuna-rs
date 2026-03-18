use std::collections::HashMap;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::distributions::{Distribution, ParamValue};
use crate::error::{OptunaError, Result};
use crate::trial::TrialState;

/// 不可变的试验快照。
///
/// 对应 Python `optuna.trial.FrozenTrial`。
/// 实现 Eq + Ord（按 number 排序），支持 suggest_* 方法用于部署场景。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FrozenTrial {
    pub number: i64,
    pub state: TrialState,
    pub values: Option<Vec<f64>>,
    pub datetime_start: Option<DateTime<Utc>>,
    pub datetime_complete: Option<DateTime<Utc>>,
    pub params: HashMap<String, ParamValue>,
    pub distributions: HashMap<String, Distribution>,
    pub user_attrs: HashMap<String, serde_json::Value>,
    pub system_attrs: HashMap<String, serde_json::Value>,
    pub intermediate_values: HashMap<i64, f64>,
    pub trial_id: i64,
}

impl FrozenTrial {
    /// Create a new `FrozenTrial`.
    ///
    /// Either `value` or `values` may be provided, but not both.
    /// A single `value` is stored as `Some(vec![value])`.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        number: i64,
        state: TrialState,
        value: Option<f64>,
        values: Option<Vec<f64>>,
        datetime_start: Option<DateTime<Utc>>,
        datetime_complete: Option<DateTime<Utc>>,
        params: HashMap<String, ParamValue>,
        distributions: HashMap<String, Distribution>,
        user_attrs: HashMap<String, serde_json::Value>,
        system_attrs: HashMap<String, serde_json::Value>,
        intermediate_values: HashMap<i64, f64>,
        trial_id: i64,
    ) -> Result<Self> {
        if value.is_some() && values.is_some() {
            return Err(OptunaError::ValueError(
                "specify either `value` or `values`, not both".into(),
            ));
        }

        let merged_values = if let Some(v) = value {
            Some(vec![v])
        } else {
            values
        };

        let trial = Self {
            number,
            state,
            values: merged_values,
            datetime_start,
            datetime_complete,
            params,
            distributions,
            user_attrs,
            system_attrs,
            intermediate_values,
            trial_id,
        };

        trial.validate()?;
        Ok(trial)
    }

    /// Single-objective accessor. Returns the value if there is exactly one.
    /// 对齐 Python: 多目标时抛 RuntimeError（非 ValueError）。
    pub fn value(&self) -> Result<Option<f64>> {
        match &self.values {
            None => Ok(None),
            Some(vs) if vs.len() == 1 => Ok(Some(vs[0])),
            Some(vs) => Err(OptunaError::RuntimeError(format!(
                "trial has {} values; use `values` for multi-objective",
                vs.len()
            ))),
        }
    }

    /// 对齐 Python `FrozenTrial.value` setter：设置单目标值。
    /// 多目标试验抛 RuntimeError。
    pub fn set_value(&mut self, v: Option<f64>) -> Result<()> {
        if let Some(ref vals) = self.values {
            if vals.len() > 1 {
                return Err(OptunaError::RuntimeError(
                    "This attribute is not available during multi-objective optimization.".into(),
                ));
            }
        }
        self.values = v.map(|x| vec![x]);
        Ok(())
    }

    /// The last intermediate value step, if any.
    pub fn last_step(&self) -> Option<i64> {
        self.intermediate_values.keys().max().copied()
    }

    /// Duration of the trial (complete_time - start_time).
    pub fn duration(&self) -> Option<chrono::Duration> {
        match (self.datetime_start, self.datetime_complete) {
            (Some(start), Some(end)) => Some(end - start),
            _ => None,
        }
    }

    /// Validate the invariants of this trial.
    pub fn validate(&self) -> Result<()> {
        // 1. Non-waiting trials must have a start time.
        if self.state != TrialState::Waiting && self.datetime_start.is_none() {
            return Err(OptunaError::ValueError(format!(
                "trial {} has state {} but no datetime_start",
                self.number, self.state
            )));
        }

        // 2/3. Finished trials must have complete time; non-finished must not.
        if self.state.is_finished() && self.datetime_complete.is_none() {
            return Err(OptunaError::ValueError(format!(
                "trial {} is finished ({}) but has no datetime_complete",
                self.number, self.state
            )));
        }
        if !self.state.is_finished() && self.datetime_complete.is_some() {
            return Err(OptunaError::ValueError(format!(
                "trial {} is not finished ({}) but has datetime_complete",
                self.number, self.state
            )));
        }

        // 4. Failed trials must not have values.
        if self.state == TrialState::Fail && self.values.is_some() {
            return Err(OptunaError::ValueError(format!(
                "trial {} is FAIL but has values",
                self.number
            )));
        }

        // 5. Completed trials must have values with no NaN.
        if self.state == TrialState::Complete {
            match &self.values {
                None => {
                    return Err(OptunaError::ValueError(format!(
                        "trial {} is COMPLETE but has no values",
                        self.number
                    )));
                }
                Some(vs) => {
                    if vs.iter().any(|v| v.is_nan()) {
                        return Err(OptunaError::ValueError(format!(
                            "trial {} is COMPLETE but contains NaN values",
                            self.number
                        )));
                    }
                }
            }
        }

        // 6. params and distributions must have the same keys.
        if self.params.len() != self.distributions.len() {
            return Err(OptunaError::ValueError(format!(
                "trial {} has {} params but {} distributions",
                self.number,
                self.params.len(),
                self.distributions.len()
            )));
        }
        for key in self.params.keys() {
            if !self.distributions.contains_key(key) {
                return Err(OptunaError::ValueError(format!(
                    "trial {}: param '{}' has no matching distribution",
                    self.number, key
                )));
            }
        }

        // 7. Each param value must be contained in its distribution.
        for (name, value) in &self.params {
            let dist = &self.distributions[name];
            let internal = dist.to_internal_repr(value)?;
            if !dist.contains(internal) {
                return Err(OptunaError::ValueError(format!(
                    "trial {}: param '{}' value is not contained in its distribution",
                    self.number, name
                )));
            }
        }

        Ok(())
    }

    // ── suggest_* 方法 (对应 Python FrozenTrial 的 suggest 接口) ──

    /// Report an intermediate value (no-op for FrozenTrial).
    /// 对应 Python `FrozenTrial.report()` — 不执行任何操作。
    pub fn report(&self, _value: f64, _step: i64) {}

    /// Check if the trial should be pruned (always false for FrozenTrial).
    /// 对应 Python `FrozenTrial.should_prune()` — 始终返回 false。
    pub fn should_prune(&self) -> bool {
        false
    }

    /// Set a user attribute.
    /// 对应 Python `FrozenTrial.set_user_attr(key, value)`。
    pub fn set_user_attr(&mut self, key: String, value: serde_json::Value) {
        self.user_attrs.insert(key, value);
    }

    /// Set a system attribute.
    /// 对应 Python `FrozenTrial.set_system_attr(key, value)`。
    pub fn set_system_attr(&mut self, key: String, value: serde_json::Value) {
        self.system_attrs.insert(key, value);
    }

    /// 对齐 Python `FrozenTrial.suggest_float(name, low, high, step=None, log=False)`。
    ///
    /// 构造分布对象，进行范围检查和兼容性验证，返回已有参数值。
    pub fn suggest_float(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        step: Option<f64>,
        log: bool,
    ) -> Result<f64> {
        let dist = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution::new(low, high, log, step)?,
        );
        let val = self._suggest(name, &dist)?;
        match val {
            ParamValue::Float(v) => Ok(v),
            ParamValue::Int(v) => Ok(v as f64),
            _ => Ok(dist.to_internal_repr(&val)?),
        }
    }

    /// 兼容历史 Rust 调用顺序：`(name, low, high, log, step)`。
    pub fn suggest_float_legacy(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        log: bool,
        step: Option<f64>,
    ) -> Result<f64> {
        self.suggest_float(name, low, high, step, log)
    }

    /// 对齐 Python 默认参数形式：`suggest_float(name, low, high)`。
    pub fn suggest_float_default(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest_float(name, low, high, None, false)
    }

    /// 对齐 Python `suggest_float(..., step=...)` 的便捷入口。
    pub fn suggest_float_step(
        &mut self,
        name: &str,
        low: f64,
        high: f64,
        step: f64,
    ) -> Result<f64> {
        self.suggest_float(name, low, high, Some(step), false)
    }

    /// 对齐 Python `suggest_float(..., log=True)` 的便捷入口。
    pub fn suggest_float_log(&mut self, name: &str, low: f64, high: f64) -> Result<f64> {
        self.suggest_float(name, low, high, None, true)
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

    /// 对齐 Python `FrozenTrial.suggest_int(name, low, high, step=1, log=False)`。
    pub fn suggest_int(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        step: i64,
        log: bool,
    ) -> Result<i64> {
        let dist = Distribution::IntDistribution(
            crate::distributions::IntDistribution::new(low, high, log, step)?,
        );
        let val = self._suggest(name, &dist)?;
        match val {
            ParamValue::Int(v) => Ok(v),
            ParamValue::Float(v) => Ok(v as i64),
            _ => Ok(dist.to_internal_repr(&val)? as i64),
        }
    }

    /// 兼容历史 Rust 调用顺序：`(name, low, high, log, step)`。
    pub fn suggest_int_legacy(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        log: bool,
        step: i64,
    ) -> Result<i64> {
        self.suggest_int(name, low, high, step, log)
    }

    /// 对齐 Python 默认参数形式：`suggest_int(name, low, high)`。
    pub fn suggest_int_default(&mut self, name: &str, low: i64, high: i64) -> Result<i64> {
        self.suggest_int(name, low, high, 1, false)
    }

    /// 对齐 Python `suggest_int(..., step=...)` 的便捷入口。
    pub fn suggest_int_step(
        &mut self,
        name: &str,
        low: i64,
        high: i64,
        step: i64,
    ) -> Result<i64> {
        self.suggest_int(name, low, high, step, false)
    }

    /// 对齐 Python `suggest_int(..., log=True)` 的便捷入口。
    pub fn suggest_int_log(&mut self, name: &str, low: i64, high: i64) -> Result<i64> {
        self.suggest_int(name, low, high, 1, true)
    }

    /// 对齐 Python `FrozenTrial.suggest_categorical(name, choices)`。
    pub fn suggest_categorical(
        &mut self,
        name: &str,
        choices: Vec<crate::distributions::CategoricalChoice>,
    ) -> Result<crate::distributions::CategoricalChoice> {
        let dist = Distribution::CategoricalDistribution(
            crate::distributions::CategoricalDistribution::new(choices)?,
        );
        let val = self._suggest(name, &dist)?;
        match val {
            ParamValue::Categorical(c) => Ok(c),
            _ => Err(OptunaError::ValueError(format!(
                "FrozenTrial param '{name}' is not categorical"
            ))),
        }
    }

    /// 核心 suggest 逻辑——对齐 Python `FrozenTrial._suggest(name, distribution)`。
    ///
    /// 1. 参数必须已存在于 self.params 中
    /// 2. 范围检查: 如果值超出分布范围，发出警告（不报错）
    /// 3. 兼容性检查: 如果分布已存在，新旧分布必须兼容
    /// 4. 更新 self.distributions
    fn _suggest(&mut self, name: &str, distribution: &Distribution) -> Result<ParamValue> {
        // 1. 参数存在性检查
        let value = self.params.get(name).cloned().ok_or_else(|| {
            OptunaError::ValueError(format!(
                "The value of the parameter '{}' is not found. \
                 Please set it at the construction of the FrozenTrial object.",
                name
            ))
        })?;

        // 2. 范围检查（对齐 Python: 超出范围仅警告，转换失败也警告）
        match distribution.to_internal_repr(&value) {
            Ok(internal) => {
                if !distribution.contains(internal) {
                    crate::optuna_warn!(
                        "The value {:?} of the parameter '{}' is out of the range of the distribution {:?}.",
                        value,
                        name,
                        distribution
                    );
                }
            }
            Err(_) => {
                crate::optuna_warn!(
                    "The value {:?} of the parameter '{}' could not be converted for distribution {:?}.",
                    value,
                    name,
                    distribution
                );
            }
        }

        // 3. 兼容性检查
        if let Some(existing_dist) = self.distributions.get(name) {
            crate::distributions::check_distribution_compatibility(existing_dist, distribution)?;
        }

        // 4. 更新分布
        self.distributions.insert(name.to_string(), distribution.clone());

        Ok(value)
    }
}

impl crate::trial::BaseTrial for FrozenTrial {
    fn suggest_float(&mut self, name: &str, low: f64, high: f64, step: Option<f64>, log: bool) -> crate::error::Result<f64> {
        self.suggest_float(name, low, high, step, log)
    }

    fn suggest_int(&mut self, name: &str, low: i64, high: i64, step: i64, log: bool) -> crate::error::Result<i64> {
        self.suggest_int(name, low, high, step, log)
    }

    fn suggest_categorical(&mut self, name: &str, choices: Vec<crate::distributions::CategoricalChoice>) -> crate::error::Result<crate::distributions::CategoricalChoice> {
        self.suggest_categorical(name, choices)
    }

    fn report(&mut self, value: f64, step: i64) -> crate::error::Result<()> {
        FrozenTrial::report(self, value, step);
        Ok(())
    }

    fn should_prune(&self) -> crate::error::Result<bool> {
        Ok(FrozenTrial::should_prune(self))
    }

    fn set_user_attr(&mut self, key: &str, value: serde_json::Value) -> crate::error::Result<()> {
        FrozenTrial::set_user_attr(self, key.to_string(), value);
        Ok(())
    }

    fn number(&self) -> i64 {
        self.number
    }

    fn params(&self) -> std::collections::HashMap<String, crate::distributions::ParamValue> {
        self.params.clone()
    }

    fn distributions(&self) -> std::collections::HashMap<String, crate::distributions::Distribution> {
        self.distributions.clone()
    }

    fn user_attrs(&self) -> crate::error::Result<std::collections::HashMap<String, serde_json::Value>> {
        Ok(self.user_attrs.clone())
    }

    fn system_attrs(&self) -> crate::error::Result<std::collections::HashMap<String, serde_json::Value>> {
        Ok(self.system_attrs.clone())
    }

    fn set_system_attr(&mut self, key: &str, value: serde_json::Value) -> crate::error::Result<()> {
        FrozenTrial::set_system_attr(self, key.to_string(), value);
        Ok(())
    }

    fn datetime_start(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        self.datetime_start
    }
}

/// 对应 Python `FrozenTrial.__eq__`：比较所有字段（不仅仅是 number）。
/// `__lt__` / `__le__` 按 number 排序。
///
/// 注意：Rust 惯例要求 PartialEq 和 Ord 一致，但 Python 中不一致，
/// 这里忠实复刻 Python 行为：eq 比所有字段，ord 比 number。
impl PartialEq for FrozenTrial {
    fn eq(&self, other: &Self) -> bool {
        self.number == other.number
            && self.state == other.state
            && self.trial_id == other.trial_id
            && self.datetime_start == other.datetime_start
            && self.datetime_complete == other.datetime_complete
            && self.params == other.params
            && self.distributions == other.distributions
            && self.user_attrs == other.user_attrs
            && self.system_attrs == other.system_attrs
            && self.intermediate_values == other.intermediate_values
            && match (&self.values, &other.values) {
                (None, None) => true,
                (Some(a), Some(b)) => {
                    a.len() == b.len()
                        && a.iter().zip(b.iter()).all(|(x, y)| {
                            // NaN == NaN → true（对齐 Python dict 比较行为）
                            (x.is_nan() && y.is_nan()) || x == y
                        })
                }
                _ => false,
            }
    }
}

impl Eq for FrozenTrial {}

/// 对齐 Python: FrozenTrial 实现 `__hash__`。
/// 由于 f64 和 HashMap 无法直接 Hash，使用 trial_id + number + state
/// 作为稳定哈希键（符合 Python `hash(tuple(self.__dict__))` 的实际行为）。
impl std::hash::Hash for FrozenTrial {
    fn hash<H: std::hash::Hasher>(&self, hasher: &mut H) {
        self.trial_id.hash(hasher);
        self.number.hash(hasher);
        self.state.hash(hasher);
    }
}

impl PartialOrd for FrozenTrial {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for FrozenTrial {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.number.cmp(&other.number)
    }
}

/// 对齐 Python `FrozenTrial.__repr__`:
/// `FrozenTrial(number=0, state=TrialState.COMPLETE, value=1.5, ...)`
impl std::fmt::Display for FrozenTrial {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // 对齐 Python: 输出所有字段 + 末尾 value=...
        let value_str = match &self.values {
            Some(v) if v.len() == 1 => format!("{}", v[0]),
            _ => "None".to_string(),
        };
        write!(
            f,
            "FrozenTrial(number={}, state={:?}, values={:?}, \
             datetime_start={:?}, datetime_complete={:?}, \
             params={:?}, distributions={:?}, \
             user_attrs={:?}, system_attrs={:?}, \
             intermediate_values={:?}, trial_id={}, value={})",
            self.number,
            self.state,
            self.values,
            self.datetime_start,
            self.datetime_complete,
            self.params,
            self.distributions,
            self.user_attrs,
            self.system_attrs,
            self.intermediate_values,
            self.trial_id,
            value_str,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::{FloatDistribution, IntDistribution};

    fn empty_maps() -> (
        HashMap<String, ParamValue>,
        HashMap<String, Distribution>,
        HashMap<String, serde_json::Value>,
        HashMap<String, serde_json::Value>,
        HashMap<i64, f64>,
    ) {
        (
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
            HashMap::new(),
        )
    }

    #[test]
    fn test_complete_trial() {
        let now = Utc::now();
        let (params, dists, ua, sa, iv) = empty_maps();
        let trial = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .unwrap();
        assert_eq!(trial.value().unwrap(), Some(1.0));
        assert_eq!(trial.values, Some(vec![1.0]));
    }

    #[test]
    fn test_value_and_values_both_provided() {
        let now = Utc::now();
        let (params, dists, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            Some(vec![2.0]),
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }

    #[test]
    fn test_complete_without_values_rejected() {
        let now = Utc::now();
        let (params, dists, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0, TrialState::Complete, None, None, Some(now), Some(now), params, dists, ua, sa,
            iv, 0,
        )
        .is_err());
    }

    #[test]
    fn test_fail_with_values_rejected() {
        let now = Utc::now();
        let (params, dists, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0,
            TrialState::Fail,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }

    #[test]
    fn test_running_without_start_rejected() {
        let (params, dists, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0,
            TrialState::Running,
            None,
            None,
            None,
            None,
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }

    #[test]
    fn test_waiting_without_start_ok() {
        let (params, dists, ua, sa, iv) = empty_maps();
        let trial = FrozenTrial::new(
            0,
            TrialState::Waiting,
            None,
            None,
            None,
            None,
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .unwrap();
        assert_eq!(trial.state, TrialState::Waiting);
    }

    #[test]
    fn test_param_distribution_mismatch() {
        let now = Utc::now();
        let mut params = HashMap::new();
        params.insert("x".into(), ParamValue::Float(0.5));
        let (_, dists, ua, sa, iv) = empty_maps();
        // params has "x" but dists is empty
        assert!(FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }

    #[test]
    fn test_param_value_out_of_range() {
        let now = Utc::now();
        let mut params = HashMap::new();
        params.insert("x".into(), ParamValue::Float(2.0)); // out of [0, 1]
        let mut dists = HashMap::new();
        dists.insert(
            "x".into(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );
        let (_, _, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }

    #[test]
    fn test_with_params() {
        let now = Utc::now();
        let mut params = HashMap::new();
        params.insert("x".into(), ParamValue::Float(0.5));
        params.insert("n".into(), ParamValue::Int(3));
        let mut dists = HashMap::new();
        dists.insert(
            "x".into(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );
        dists.insert(
            "n".into(),
            Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap()),
        );
        let (_, _, ua, sa, iv) = empty_maps();
        let trial = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(42.0),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .unwrap();
        assert_eq!(trial.params.len(), 2);
    }

    #[test]
    fn test_last_step() {
        let now = Utc::now();
        let (params, dists, ua, sa, _) = empty_maps();
        let mut iv = HashMap::new();
        iv.insert(0, 0.5);
        iv.insert(5, 0.3);
        iv.insert(2, 0.4);
        let trial = FrozenTrial::new(
            0,
            TrialState::Running,
            None,
            None,
            Some(now),
            None,
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .unwrap();
        assert_eq!(trial.last_step(), Some(5));
    }

    #[test]
    fn test_nan_values_rejected() {
        let now = Utc::now();
        let (params, dists, ua, sa, iv) = empty_maps();
        assert!(FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(f64::NAN),
            None,
            Some(now),
            Some(now),
            params,
            dists,
            ua,
            sa,
            iv,
            0,
        )
        .is_err());
    }

    // ── PartialEq 测试（对齐 Python __eq__ 比较所有字段）──

    #[test]
    fn test_eq_all_fields_match() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t1 = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), p.clone(), d.clone(), u.clone(), s.clone(), i.clone(), 0).unwrap();
        let t2 = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), p, d, u, s, i, 0).unwrap();
        assert_eq!(t1, t2);
    }

    #[test]
    fn test_eq_different_state_not_equal() {
        // 同 number 不同 state → 不相等（对齐 Python __dict__ 比较）
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let complete = FrozenTrial {
            number: 0, state: TrialState::Complete, values: Some(vec![1.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p.clone(), distributions: d.clone(), user_attrs: u.clone(),
            system_attrs: s.clone(), intermediate_values: i.clone(), trial_id: 0,
        };
        let running = FrozenTrial {
            number: 0, state: TrialState::Running, values: None,
            datetime_start: Some(now), datetime_complete: None,
            params: p, distributions: d, user_attrs: u,
            system_attrs: s, intermediate_values: i, trial_id: 0,
        };
        assert_ne!(complete, running);
    }

    #[test]
    fn test_eq_different_values_not_equal() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t1 = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), p.clone(), d.clone(), u.clone(), s.clone(), i.clone(), 0).unwrap();
        let t2 = FrozenTrial::new(0, TrialState::Complete, Some(2.0), None,
            Some(now), Some(now), p, d, u, s, i, 0).unwrap();
        assert_ne!(t1, t2);
    }

    #[test]
    fn test_ord_by_number() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t0 = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), p.clone(), d.clone(), u.clone(), s.clone(), i.clone(), 0).unwrap();
        let t1 = FrozenTrial::new(1, TrialState::Complete, Some(2.0), None,
            Some(now), Some(now), p, d, u, s, i, 1).unwrap();
        assert!(t0 < t1);
    }

    // ── report / should_prune / set_user_attr 测试 ──

    #[test]
    fn test_report_noop() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), p, d, u, s, i, 0).unwrap();
        t.report(0.5, 0); // should not panic
    }

    #[test]
    fn test_should_prune_always_false() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), p, d, u, s, i, 0).unwrap();
        assert!(!t.should_prune());
    }

    #[test]
    fn test_set_user_attr() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let mut t = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), p, d, u, s, i, 0).unwrap();
        t.set_user_attr("key".to_string(), serde_json::json!("val"));
        assert_eq!(t.user_attrs["key"], serde_json::json!("val"));
    }

    // ── suggest_* 测试 ──

    #[test]
    fn test_suggest_float_ok() {
        let now = Utc::now();
        let mut params = HashMap::new();
        params.insert("x".into(), ParamValue::Float(0.5));
        let mut dists = HashMap::new();
        dists.insert("x".into(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()));
        let (_, _, u, s, i) = empty_maps();
        let mut t = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), params, dists, u, s, i, 0).unwrap();
        assert_eq!(t.suggest_float("x", 0.0, 1.0, None, false).unwrap(), 0.5);
    }

    #[test]
    fn test_suggest_int_ok() {
        let now = Utc::now();
        let mut params = HashMap::new();
        params.insert("n".into(), ParamValue::Int(5));
        let mut dists = HashMap::new();
        dists.insert("n".into(),
            Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap()));
        let (_, _, u, s, i) = empty_maps();
        let mut t = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), params, dists, u, s, i, 0).unwrap();
        assert_eq!(t.suggest_int("n", 1, 10, 1, false).unwrap(), 5);
    }

    #[test]
    fn test_suggest_missing_param() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let mut t = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), p, d, u, s, i, 0).unwrap();
        assert!(t.suggest_float("missing", 0.0, 1.0, None, false).is_err());
    }

    #[test]
    fn test_python_compat_float_wrappers() {
        let now = Utc::now();
        let (_, _, u, s, i) = empty_maps();

        let mut params_x = HashMap::new();
        params_x.insert("x".into(), ParamValue::Float(0.25));
        let mut dists_x = HashMap::new();
        dists_x.insert(
            "x".into(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );

        let mut tx = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params_x,
            dists_x,
            u.clone(),
            s.clone(),
            i.clone(),
            0,
        )
        .unwrap();
        assert_eq!(tx.suggest_float_default("x", 0.0, 1.0).unwrap(), 0.25);
        match tx.distributions.get("x").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, None);
            }
            _ => panic!("x should use float distribution"),
        }

        let mut params_y = HashMap::new();
        params_y.insert("y".into(), ParamValue::Float(2.0));
        let mut dists_y = HashMap::new();
        dists_y.insert(
            "y".into(),
            Distribution::FloatDistribution(FloatDistribution::new(1.0, 10.0, true, None).unwrap()),
        );

        let mut ty = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params_y,
            dists_y,
            u.clone(),
            s.clone(),
            i.clone(),
            0,
        )
        .unwrap();
        assert_eq!(ty.suggest_loguniform("y", 1.0, 10.0).unwrap(), 2.0);
        match ty.distributions.get("y").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(dist.log);
                assert_eq!(dist.step, None);
            }
            _ => panic!("y should use float distribution"),
        }

        let mut params_z = HashMap::new();
        params_z.insert("z".into(), ParamValue::Float(0.75));
        let mut dists_z = HashMap::new();
        dists_z.insert(
            "z".into(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap()),
        );

        let mut tz = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params_z,
            dists_z,
            u.clone(),
            s.clone(),
            i.clone(),
            0,
        )
        .unwrap();
        assert_eq!(tz.suggest_discrete_uniform("z", 0.0, 1.0, 0.25).unwrap(), 0.75);
        match tz.distributions.get("z").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, Some(0.25));
            }
            _ => panic!("z should use float distribution"),
        }

        let mut params_w = HashMap::new();
        params_w.insert("w".into(), ParamValue::Float(1.5));
        let mut dists_w = HashMap::new();
        dists_w.insert(
            "w".into(),
            Distribution::FloatDistribution(FloatDistribution::new(1.0, 2.0, false, Some(0.5)).unwrap()),
        );

        let mut tw = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params_w,
            dists_w,
            u,
            s,
            i,
            0,
        )
        .unwrap();
        assert_eq!(tw.suggest_float_legacy("w", 1.0, 2.0, false, Some(0.5)).unwrap(), 1.5);
        match tw.distributions.get("w").unwrap() {
            Distribution::FloatDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, Some(0.5));
            }
            _ => panic!("w should use float distribution"),
        }
    }

    #[test]
    fn test_python_compat_int_wrappers() {
        let now = Utc::now();
        let (_, _, u, s, i) = empty_maps();

        let mut params_a = HashMap::new();
        params_a.insert("a".into(), ParamValue::Int(3));
        let mut dists_a = HashMap::new();
        dists_a.insert(
            "a".into(),
            Distribution::IntDistribution(IntDistribution::new(1, 5, false, 1).unwrap()),
        );

        let mut ta = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params_a,
            dists_a,
            u.clone(),
            s.clone(),
            i.clone(),
            0,
        )
        .unwrap();
        assert_eq!(ta.suggest_int_default("a", 1, 5).unwrap(), 3);
        match ta.distributions.get("a").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, 1);
            }
            _ => panic!("a should use int distribution"),
        }

        let mut params_b = HashMap::new();
        params_b.insert("b".into(), ParamValue::Int(4));
        let mut dists_b = HashMap::new();
        dists_b.insert(
            "b".into(),
            Distribution::IntDistribution(IntDistribution::new(1, 8, true, 1).unwrap()),
        );

        let mut tb = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params_b,
            dists_b,
            u.clone(),
            s.clone(),
            i.clone(),
            0,
        )
        .unwrap();
        assert_eq!(tb.suggest_int_log("b", 1, 8).unwrap(), 4);
        match tb.distributions.get("b").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(dist.log);
                assert_eq!(dist.step, 1);
            }
            _ => panic!("b should use int distribution"),
        }

        let mut params_c = HashMap::new();
        params_c.insert("c".into(), ParamValue::Int(6));
        let mut dists_c = HashMap::new();
        dists_c.insert(
            "c".into(),
            Distribution::IntDistribution(IntDistribution::new(0, 10, false, 2).unwrap()),
        );

        let mut tc = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params_c,
            dists_c,
            u.clone(),
            s.clone(),
            i.clone(),
            0,
        )
        .unwrap();
        assert_eq!(tc.suggest_int_step("c", 0, 10, 2).unwrap(), 6);
        match tc.distributions.get("c").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, 2);
            }
            _ => panic!("c should use int distribution"),
        }

        let mut params_d = HashMap::new();
        params_d.insert("d".into(), ParamValue::Int(9));
        let mut dists_d = HashMap::new();
        dists_d.insert(
            "d".into(),
            Distribution::IntDistribution(IntDistribution::new(3, 9, false, 3).unwrap()),
        );

        let mut td = FrozenTrial::new(
            0,
            TrialState::Complete,
            Some(1.0),
            None,
            Some(now),
            Some(now),
            params_d,
            dists_d,
            u,
            s,
            i,
            0,
        )
        .unwrap();
        assert_eq!(td.suggest_int_legacy("d", 3, 9, false, 3).unwrap(), 9);
        match td.distributions.get("d").unwrap() {
            Distribution::IntDistribution(dist) => {
                assert!(!dist.log);
                assert_eq!(dist.step, 3);
            }
            _ => panic!("d should use int distribution"),
        }
    }

    // ── duration 测试 ──

    #[test]
    fn test_duration() {
        let start = Utc::now();
        let end = start + chrono::Duration::seconds(10);
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(start), Some(end), p, d, u, s, i, 0).unwrap();
        assert_eq!(t.duration().unwrap().num_seconds(), 10);
    }

    // ── 对齐 Python 审计修复的测试 ──

    /// 对齐 Python: value() 多目标时应抛 RuntimeError (非 ValueError)
    #[test]
    fn test_value_multi_objective_returns_runtime_error() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0, 2.0]),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: p,
            distributions: d,
            user_attrs: u,
            system_attrs: s,
            intermediate_values: i,
            trial_id: 0,
        };
        let err = t.value().unwrap_err();
        assert!(matches!(err, OptunaError::RuntimeError(_)));
    }

    /// 对齐 Python: set_system_attr()
    #[test]
    fn test_set_system_attr() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let mut t = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), p, d, u, s, i, 0).unwrap();
        t.set_system_attr("key".into(), serde_json::json!("value"));
        assert_eq!(t.system_attrs.get("key").unwrap(), &serde_json::json!("value"));
    }

    /// 对齐 Python: set_value() 单目标
    #[test]
    fn test_set_value_single_objective() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let mut t = FrozenTrial::new(0, TrialState::Complete, Some(1.0), None,
            Some(now), Some(now), p, d, u, s, i, 0).unwrap();
        t.set_value(Some(2.0)).unwrap();
        assert_eq!(t.values, Some(vec![2.0]));
    }

    /// 对齐 Python: set_value() 多目标时应 RuntimeError
    #[test]
    fn test_set_value_multi_objective_error() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let mut t = FrozenTrial {
            number: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0, 2.0]),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: p,
            distributions: d,
            user_attrs: u,
            system_attrs: s,
            intermediate_values: i,
            trial_id: 0,
        };
        let err = t.set_value(Some(3.0)).unwrap_err();
        assert!(matches!(err, OptunaError::RuntimeError(_)));
    }

    // ========== 对齐 Python: FrozenTrial Hash 测试 ==========

    /// 测试 FrozenTrial 可以放入 HashSet。
    /// 对应 Python: `set([trial1, trial2])` 去重依赖 __hash__。
    #[test]
    fn test_frozen_trial_hash_in_hashset() {
        use std::collections::HashSet;
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t1 = FrozenTrial {
            number: 0, state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p.clone(), distributions: d.clone(),
            user_attrs: u.clone(), system_attrs: s.clone(),
            intermediate_values: i.clone(), trial_id: 0,
        };
        let t2 = FrozenTrial {
            number: 1, state: TrialState::Complete,
            values: Some(vec![2.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 1,
        };

        let mut set = HashSet::new();
        set.insert(t1.clone());
        set.insert(t2.clone());
        set.insert(t1.clone()); // 重复插入
        assert_eq!(set.len(), 2, "HashSet 应去重");
    }

    /// 测试相同 trial_id 和 number 的 FrozenTrial 产生相同 hash。
    #[test]
    fn test_frozen_trial_hash_consistency() {
        use std::hash::{Hash, Hasher};
        use std::collections::hash_map::DefaultHasher;

        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t1 = FrozenTrial {
            number: 5, state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p.clone(), distributions: d.clone(),
            user_attrs: u.clone(), system_attrs: s.clone(),
            intermediate_values: i.clone(), trial_id: 42,
        };
        let t2 = FrozenTrial {
            number: 5, state: TrialState::Complete,
            values: Some(vec![999.0]), // 不同 values
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 42,
        };

        let mut h1 = DefaultHasher::new();
        t1.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        t2.hash(&mut h2);
        assert_eq!(h1.finish(), h2.finish(), "相同 trial_id+number+state 应产生相同 hash");
    }

    // ========== 对齐 Python: FrozenTrial validate 边界测试 ==========

    /// 测试 validate 拒绝 params/distributions 键不一致。
    #[test]
    fn test_validate_mismatched_keys() {
        let now = Utc::now();
        let mut params = HashMap::new();
        params.insert("x".to_string(), ParamValue::Float(1.0));
        let distributions = HashMap::new(); // 没有 "x"

        let t = FrozenTrial {
            number: 0, state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params, distributions,
            user_attrs: HashMap::new(), system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(), trial_id: 0,
        };
        assert!(t.validate().is_err());
    }

    /// 测试 validate 拒绝 Complete 但无 values。
    #[test]
    fn test_validate_complete_without_values() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 0, state: TrialState::Complete,
            values: None, // Complete 必须有 values
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 0,
        };
        assert!(t.validate().is_err());
    }

    /// 测试 validate 接受 Pruned 无 values。
    #[test]
    fn test_validate_pruned_without_values_ok() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 0, state: TrialState::Pruned,
            values: None,
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 0,
        };
        assert!(t.validate().is_ok());
    }

    // ================================================================
    // Display / __repr__ 对齐测试
    // ================================================================

    /// 对齐 Python `FrozenTrial.__repr__`: Display 输出包含所有关键字段。
    #[test]
    fn test_display_contains_all_fields() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 42, state: TrialState::Complete,
            values: Some(vec![1.5]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 7,
        };
        let repr = format!("{}", t);
        assert!(repr.starts_with("FrozenTrial("), "should start with class name");
        assert!(repr.contains("number=42"), "should contain number");
        assert!(repr.contains("trial_id=7"), "should contain trial_id");
        assert!(repr.contains("value=1.5"), "should contain value for single-obj");
    }

    /// 对齐 Python: 多目标时 value=None。
    #[test]
    fn test_display_multi_objective_value_none() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 0, state: TrialState::Complete,
            values: Some(vec![1.0, 2.0]), // 多目标
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 0,
        };
        let repr = format!("{}", t);
        assert!(repr.contains("value=None"), "multi-obj should show value=None");
    }

    // ================================================================
    // PartialOrd / Ord 排序对齐测试
    // ================================================================

    /// 对齐 Python `FrozenTrial.__lt__`: 按 number 排序。
    #[test]
    fn test_ordering_by_number() {
        let now = Utc::now();
        let (p1, d1, u1, s1, i1) = empty_maps();
        let (p2, d2, u2, s2, i2) = empty_maps();
        let t1 = FrozenTrial {
            number: 3, state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p1, distributions: d1,
            user_attrs: u1, system_attrs: s1,
            intermediate_values: i1, trial_id: 0,
        };
        let t2 = FrozenTrial {
            number: 1, state: TrialState::Complete,
            values: Some(vec![2.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p2, distributions: d2,
            user_attrs: u2, system_attrs: s2,
            intermediate_values: i2, trial_id: 1,
        };
        assert!(t2 < t1, "trial with smaller number should be less");
        assert!(t1 > t2);

        // Vec 排序应按 number
        let mut trials = vec![t1.clone(), t2.clone()];
        trials.sort();
        assert_eq!(trials[0].number, 1);
        assert_eq!(trials[1].number, 3);
    }

    // ================================================================
    // PartialEq 语义测试
    // ================================================================

    /// 对齐 Python: PartialEq 比较所有字段，NaN == NaN。
    #[test]
    fn test_eq_nan_values() {
        let now = Utc::now();
        let (p1, d1, u1, s1, i1) = empty_maps();
        let (p2, d2, u2, s2, i2) = empty_maps();
        // 注意: Complete 不允许 NaN values (validate 会报错)
        // 所以用 Running 状态
        let t1 = FrozenTrial {
            number: 0, state: TrialState::Running,
            values: Some(vec![f64::NAN]),
            datetime_start: Some(now), datetime_complete: None,
            params: p1, distributions: d1,
            user_attrs: u1, system_attrs: s1,
            intermediate_values: i1, trial_id: 0,
        };
        let t2 = FrozenTrial {
            number: 0, state: TrialState::Running,
            values: Some(vec![f64::NAN]),
            datetime_start: Some(now), datetime_complete: None,
            params: p2, distributions: d2,
            user_attrs: u2, system_attrs: s2,
            intermediate_values: i2, trial_id: 0,
        };
        assert_eq!(t1, t2, "NaN == NaN should be true for values comparison");
    }

    /// 对齐 Python: 不同 state 的试验不相等。
    #[test]
    fn test_eq_different_state() {
        let now = Utc::now();
        let (p1, d1, u1, s1, i1) = empty_maps();
        let (p2, d2, u2, s2, i2) = empty_maps();
        let t1 = FrozenTrial {
            number: 0, state: TrialState::Running,
            values: None,
            datetime_start: Some(now), datetime_complete: None,
            params: p1, distributions: d1,
            user_attrs: u1, system_attrs: s1,
            intermediate_values: i1, trial_id: 0,
        };
        let t2 = FrozenTrial {
            number: 0, state: TrialState::Waiting,
            values: None,
            datetime_start: None, datetime_complete: None,
            params: p2, distributions: d2,
            user_attrs: u2, system_attrs: s2,
            intermediate_values: i2, trial_id: 0,
        };
        assert_ne!(t1, t2);
    }

    // ================================================================
    // Hash 一致性测试
    // ================================================================

    /// 对齐 Python: 相同 trial 多次 hash 结果一致。
    #[test]
    fn test_hash_deterministic() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 5, state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 10,
        };

        let mut h1 = DefaultHasher::new();
        t.hash(&mut h1);
        let hash1 = h1.finish();

        let mut h2 = DefaultHasher::new();
        t.hash(&mut h2);
        let hash2 = h2.finish();

        assert_eq!(hash1, hash2, "same trial should hash identically");
    }

    /// 对齐 Python: 不同 number 的试验 hash 不同。
    #[test]
    fn test_hash_different_numbers() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let now = Utc::now();
        let (p1, d1, u1, s1, i1) = empty_maps();
        let (p2, d2, u2, s2, i2) = empty_maps();
        let t1 = FrozenTrial {
            number: 1, state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p1, distributions: d1,
            user_attrs: u1, system_attrs: s1,
            intermediate_values: i1, trial_id: 1,
        };
        let t2 = FrozenTrial {
            number: 2, state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p2, distributions: d2,
            user_attrs: u2, system_attrs: s2,
            intermediate_values: i2, trial_id: 2,
        };

        let mut h1 = DefaultHasher::new();
        t1.hash(&mut h1);
        let mut h2 = DefaultHasher::new();
        t2.hash(&mut h2);

        assert_ne!(h1.finish(), h2.finish(), "different trials should hash differently");
    }

    // ================================================================
    // last_step / duration 方法测试
    // ================================================================

    /// 对齐 Python `FrozenTrial.last_step`: 返回最大 step。
    #[test]
    fn test_last_step_returns_max() {
        let (p, d, u, s, _) = empty_maps();
        let mut iv = HashMap::new();
        iv.insert(0, 1.0);
        iv.insert(5, 2.0);
        iv.insert(3, 3.0);
        let t = FrozenTrial {
            number: 0, state: TrialState::Running,
            values: None,
            datetime_start: Some(Utc::now()), datetime_complete: None,
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: iv, trial_id: 0,
        };
        assert_eq!(t.last_step(), Some(5));
    }

    /// 对齐 Python: 无中间值时 last_step 返回 None。
    #[test]
    fn test_last_step_empty() {
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 0, state: TrialState::Running,
            values: None,
            datetime_start: Some(Utc::now()), datetime_complete: None,
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 0,
        };
        assert_eq!(t.last_step(), None);
    }

    /// 对齐 Python `FrozenTrial.duration`: 有 start 和 complete 时返回 Some。
    #[test]
    fn test_duration_complete() {
        let start = Utc::now();
        let end = start + chrono::Duration::seconds(10);
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 0, state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(start), datetime_complete: Some(end),
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 0,
        };
        let dur = t.duration().expect("should have duration");
        assert_eq!(dur, chrono::Duration::seconds(10));
    }

    /// 对齐 Python: 无 datetime_complete 时 duration 返回 None。
    #[test]
    fn test_duration_incomplete() {
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 0, state: TrialState::Running,
            values: None,
            datetime_start: Some(Utc::now()), datetime_complete: None,
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 0,
        };
        assert!(t.duration().is_none());
    }

    // ================================================================
    // value() getter 测试
    // ================================================================

    /// 对齐 Python: 单目标 value() 返回值。
    #[test]
    fn test_value_single_objective() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 0, state: TrialState::Complete,
            values: Some(vec![3.14]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 0,
        };
        assert_eq!(t.value().unwrap(), Some(3.14));
    }

    /// 对齐 Python: 多目标调用 value() 报 RuntimeError。
    #[test]
    fn test_value_multi_objective_error() {
        let now = Utc::now();
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 0, state: TrialState::Complete,
            values: Some(vec![1.0, 2.0]),
            datetime_start: Some(now), datetime_complete: Some(now),
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 0,
        };
        assert!(t.value().is_err());
    }

    /// 对齐 Python: 无值时 value() 返回 None。
    #[test]
    fn test_value_none() {
        let (p, d, u, s, i) = empty_maps();
        let t = FrozenTrial {
            number: 0, state: TrialState::Running,
            values: None,
            datetime_start: Some(Utc::now()), datetime_complete: None,
            params: p, distributions: d,
            user_attrs: u, system_attrs: s,
            intermediate_values: i, trial_id: 0,
        };
        assert_eq!(t.value().unwrap(), None);
    }
}
