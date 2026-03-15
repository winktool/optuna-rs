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
    pub fn value(&self) -> Result<Option<f64>> {
        match &self.values {
            None => Ok(None),
            Some(vs) if vs.len() == 1 => Ok(Some(vs[0])),
            Some(vs) => Err(OptunaError::ValueError(format!(
                "trial has {} values; use `values` for multi-objective",
                vs.len()
            ))),
        }
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

        // 2. 范围检查（超出范围仅警告）
        if let Ok(internal) = distribution.to_internal_repr(&value) {
            if !distribution.contains(internal) {
                crate::optuna_warn!(
                    "The value {} of the parameter '{}' is out of the range of the distribution {:?}.",
                    internal, name, distribution
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
}
