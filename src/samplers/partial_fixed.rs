use std::collections::HashMap;
use std::sync::Arc;

use crate::distributions::{Distribution, ParamValue};
use crate::error::Result;
use crate::optuna_warn;
use crate::samplers::Sampler;
use crate::trial::{FrozenTrial, TrialState};

/// A sampler that fixes some parameters to constant values and delegates
/// the rest to a base sampler.
pub struct PartialFixedSampler {
    fixed_params: HashMap<String, f64>,
    base_sampler: Arc<dyn Sampler>,
}

impl std::fmt::Debug for PartialFixedSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PartialFixedSampler")
            .field("n_fixed", &self.fixed_params.len())
            .finish()
    }
}

impl PartialFixedSampler {
    /// Create a new `PartialFixedSampler`.
    ///
    /// # Arguments
    /// * `fixed_params` - Map from param name to fixed internal value.
    /// * `base_sampler` - The sampler to use for non-fixed parameters.
    pub fn new(fixed_params: HashMap<String, f64>, base_sampler: Arc<dyn Sampler>) -> Self {
        Self {
            fixed_params,
            base_sampler,
        }
    }

    /// Create from external ParamValue representations.
    ///
    /// Converts each `ParamValue` to its internal `f64` representation using
    /// the provided distributions.
    pub fn from_param_values(
        fixed_params: HashMap<String, ParamValue>,
        distributions: &HashMap<String, Distribution>,
        base_sampler: Arc<dyn Sampler>,
    ) -> Result<Self> {
        let mut internal_params = HashMap::new();
        for (name, value) in &fixed_params {
            if let Some(dist) = distributions.get(name) {
                let internal = dist.to_internal_repr(value)?;
                internal_params.insert(name.clone(), internal);
            } else {
                // No distribution found; store Float/Int directly
                let internal = match value {
                    ParamValue::Float(v) => *v,
                    ParamValue::Int(v) => *v as f64,
                    ParamValue::Categorical(_) => {
                        return Err(crate::error::OptunaError::ValueError(format!(
                            "cannot fix categorical param '{name}' without a distribution"
                        )));
                    }
                };
                internal_params.insert(name.clone(), internal);
            }
        }
        Ok(Self::new(internal_params, base_sampler))
    }
}

impl Sampler for PartialFixedSampler {
    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> HashMap<String, Distribution> {
        let mut space = self.base_sampler.infer_relative_search_space(trials);
        // Remove fixed params from the relative search space
        for name in self.fixed_params.keys() {
            space.remove(name);
        }
        space
    }

    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        // 对齐 Python: sample_relative 中不注入固定参数
        // 固定参数通过 sample_independent 路径返回
        self.base_sampler.sample_relative(trials, search_space)
    }

    fn sample_independent(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        if let Some(&value) = self.fixed_params.get(param_name) {
            // 对齐 Python: 检查固定值是否在分布范围内，不在则发出警告
            if !distribution.contains(value) {
                optuna_warn!(
                    "Fixed param '{}' value {} is not contained in distribution {:?}.",
                    param_name, value, distribution
                );
            }
            return Ok(value);
        }
        self.base_sampler
            .sample_independent(trials, trial, param_name, distribution)
    }

    fn before_trial(&self, trials: &[FrozenTrial]) {
        self.base_sampler.before_trial(trials);
    }

    fn after_trial(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        state: TrialState,
        values: Option<&[f64]>,
    ) {
        self.base_sampler.after_trial(trials, trial, state, values);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samplers::RandomSampler;
    use crate::study::{create_study, StudyDirection};

    #[test]
    fn test_partial_fixed_basic() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
        let mut fixed = HashMap::new();
        fixed.insert("x".to_string(), 0.5);

        let sampler: Arc<dyn Sampler> =
            Arc::new(PartialFixedSampler::new(fixed, base));

        let study = create_study(
            None,
            Some(sampler),
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
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                    Ok(x * x + y * y)
                },
                Some(10),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 10);
        // x should always be 0.5
        for trial in &trials {
            let x = match trial.params.get("x") {
                Some(ParamValue::Float(v)) => *v,
                _ => panic!("expected float param x"),
            };
            assert!(
                (x - 0.5).abs() < 1e-10,
                "x should be fixed at 0.5, got {x}"
            );
        }
    }

    #[test]
    fn test_partial_fixed_non_fixed_varies() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
        let mut fixed = HashMap::new();
        fixed.insert("x".to_string(), 0.5);

        let sampler: Arc<dyn Sampler> =
            Arc::new(PartialFixedSampler::new(fixed, base));

        let study = create_study(
            None,
            Some(sampler),
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
                    let _x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                    Ok(y * y)
                },
                Some(20),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        let y_values: Vec<f64> = trials
            .iter()
            .map(|t| match t.params.get("y") {
                Some(ParamValue::Float(v)) => *v,
                _ => panic!("expected float param y"),
            })
            .collect();

        // y should vary (not all the same)
        let first = y_values[0];
        assert!(
            y_values.iter().any(|&v| (v - first).abs() > 1e-10),
            "y values should vary"
        );
    }

    #[test]
    fn test_from_param_values_with_distribution_for_categorical() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(1)));
        let mut fixed_params = HashMap::new();
        fixed_params.insert(
            "cat".to_string(),
            ParamValue::Categorical(crate::distributions::CategoricalChoice::Str("b".into())),
        );

        let mut distributions = HashMap::new();
        distributions.insert(
            "cat".to_string(),
            Distribution::CategoricalDistribution(
                crate::distributions::CategoricalDistribution::new(vec![
                    crate::distributions::CategoricalChoice::Str("a".into()),
                    crate::distributions::CategoricalChoice::Str("b".into()),
                ])
                .unwrap(),
            ),
        );

        let sampler = PartialFixedSampler::from_param_values(fixed_params, &distributions, base).unwrap();
        let trial = FrozenTrial {
            number: 0,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };

        let v = sampler
            .sample_independent(
                &[],
                &trial,
                "cat",
                distributions.get("cat").unwrap(),
            )
            .unwrap();
        assert_eq!(v, 1.0);
    }

    #[test]
    fn test_from_param_values_without_distribution_for_categorical_errors() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(1)));
        let mut fixed_params = HashMap::new();
        fixed_params.insert(
            "cat".to_string(),
            ParamValue::Categorical(crate::distributions::CategoricalChoice::Str("b".into())),
        );
        let distributions = HashMap::new();

        let err = PartialFixedSampler::from_param_values(fixed_params, &distributions, base)
            .unwrap_err();
        match err {
            crate::error::OptunaError::ValueError(msg) => {
                assert!(msg.contains("cannot fix categorical param 'cat' without a distribution"));
            }
            _ => panic!("unexpected error type"),
        }
    }

    /// 验证 from_param_values 对 Float/Int 类型无分布时的直接转换
    #[test]
    fn test_from_param_values_float_int_without_distribution() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(1)));
        let mut fixed_params = HashMap::new();
        fixed_params.insert("f".to_string(), ParamValue::Float(3.14));
        fixed_params.insert("i".to_string(), ParamValue::Int(42));
        let distributions = HashMap::new();

        let sampler = PartialFixedSampler::from_param_values(fixed_params, &distributions, base).unwrap();
        assert!((sampler.fixed_params["f"] - 3.14).abs() < 1e-15);
        assert!((sampler.fixed_params["i"] - 42.0).abs() < 1e-15);
    }

    /// 验证 infer_relative_search_space 排除固定参数
    #[test]
    fn test_infer_relative_search_space_excludes_fixed() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
        let mut fixed = HashMap::new();
        fixed.insert("x".to_string(), 0.5);

        let sampler = PartialFixedSampler::new(fixed, base);
        // RandomSampler 默认返回空搜索空间
        let space = sampler.infer_relative_search_space(&[]);
        assert!(!space.contains_key("x"), "固定参数应被排除");
    }

    /// 对齐 Python: sample_relative 不注入固定值，固定参数通过 sample_independent 返回
    #[test]
    fn test_sample_relative_does_not_inject_fixed() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
        let mut fixed = HashMap::new();
        fixed.insert("x".to_string(), 0.5);

        let sampler = PartialFixedSampler::new(fixed, base);
        let result = sampler.sample_relative(&[], &HashMap::new()).unwrap();
        // 固定参数不应在 sample_relative 结果中
        assert!(!result.contains_key("x"), "固定参数不应被注入到 sample_relative");
    }

    /// 验证 Debug trait 实现
    #[test]
    fn test_debug_display() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
        let mut fixed = HashMap::new();
        fixed.insert("x".to_string(), 0.5);
        let sampler = PartialFixedSampler::new(fixed, base);
        let debug_str = format!("{:?}", sampler);
        assert!(debug_str.contains("PartialFixedSampler"));
        assert!(debug_str.contains("n_fixed"));
    }

    /// 对齐 Python: 固定值不在分布范围内应发出警告（不报错）
    #[test]
    fn test_fixed_value_out_of_range_warns_but_succeeds() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
        let mut fixed = HashMap::new();
        // 值 99.0 超出 [0, 1] 范围
        fixed.insert("x".to_string(), 99.0);

        let sampler = PartialFixedSampler::new(fixed, base);
        let dist = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        let trial = FrozenTrial {
            number: 0,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        // 应成功返回（Python 行为: 发出警告但返回固定值）
        let result = sampler.sample_independent(&[], &trial, "x", &dist);
        assert!(result.is_ok());
        assert!((result.unwrap() - 99.0).abs() < 1e-15);
    }

    /// 对齐 Python: 固定值在范围内不产生任何问题
    #[test]
    fn test_fixed_value_in_range_no_warn() {
        let base: Arc<dyn Sampler> = Arc::new(RandomSampler::new(Some(42)));
        let mut fixed = HashMap::new();
        fixed.insert("x".to_string(), 0.5);

        let sampler = PartialFixedSampler::new(fixed, base);
        let dist = Distribution::FloatDistribution(
            crate::distributions::FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        let trial = FrozenTrial {
            number: 0,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        let result = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
        assert!((result - 0.5).abs() < 1e-15);
    }
}
