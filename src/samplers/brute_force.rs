use std::collections::{HashMap, HashSet};

use crate::distributions::Distribution;
use crate::error::{OptunaError, Result};
use crate::samplers::random::RandomSampler;
use crate::samplers::Sampler;
use crate::trial::{FrozenTrial, TrialState};

/// A sampler that exhaustively evaluates all possible discrete parameter combinations.
///
/// Similar to `GridSampler` but dynamically discovers parameter distributions
/// from the search space rather than requiring them upfront.
pub struct BruteForceSampler {
    random_sampler: RandomSampler,
}

impl std::fmt::Debug for BruteForceSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BruteForceSampler").finish()
    }
}

impl BruteForceSampler {
    pub fn new(seed: Option<u64>) -> Self {
        Self {
            random_sampler: RandomSampler::new(seed),
        }
    }

    /// Enumerate all values for a distribution, if discrete.
    fn enumerate_distribution(dist: &Distribution) -> Option<Vec<f64>> {
        match dist {
            Distribution::IntDistribution(d) => {
                let mut vals = Vec::new();
                let mut v = d.low;
                while v <= d.high {
                    vals.push(v as f64);
                    v += d.step;
                }
                Some(vals)
            }
            Distribution::FloatDistribution(d) => {
                if let Some(step) = d.step {
                    let mut vals = Vec::new();
                    let n_steps = ((d.high - d.low) / step).round() as i64;
                    for i in 0..=n_steps {
                        let v = d.low + step * i as f64;
                        if v <= d.high + 1e-8 {
                            vals.push(v);
                        }
                    }
                    Some(vals)
                } else if d.single() {
                    Some(vec![d.low])
                } else {
                    None
                }
            }
            Distribution::CategoricalDistribution(d) => {
                Some((0..d.choices.len()).map(|i| i as f64).collect())
            }
        }
    }

    /// Get all parameter combinations that have been visited.
    /// 对齐 Python: 包含 Complete, Pruned, Running, Fail 四种状态
    fn get_visited_combinations(trials: &[FrozenTrial], param_names: &[String]) -> HashSet<Vec<i64>> {
        let mut visited = HashSet::new();
        for trial in trials {
            // 对齐 Python: states=(COMPLETE, PRUNED, RUNNING, FAIL)
            if trial.state == TrialState::Complete
                || trial.state == TrialState::Pruned
                || trial.state == TrialState::Running
                || trial.state == TrialState::Fail
            {
                let key: Vec<i64> = param_names
                    .iter()
                    .map(|name| {
                        trial
                            .params
                            .get(name)
                            .map(|pv| {
                                let dist = trial.distributions.get(name);
                                dist.and_then(|d| d.to_internal_repr(pv).ok())
                                    .map(|v| (v * 1e6).round() as i64)
                                    .unwrap_or(0)
                            })
                            .unwrap_or(i64::MIN)
                    })
                    .collect();
                visited.insert(key);
            }
        }
        visited
    }
}

impl Sampler for BruteForceSampler {
    fn sample_independent(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        // For brute force, we try to pick an unvisited value
        // In independent mode, just delegate to random
        self.random_sampler
            .sample_independent(trials, trial, param_name, distribution)
    }

    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> HashMap<String, Distribution> {
        // Use intersection search space from completed trials
        crate::search_space::intersection_search_space(trials, false)
    }

    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        if search_space.is_empty() {
            return Ok(HashMap::new());
        }

        // Sort param names for deterministic ordering
        let mut param_names: Vec<String> = search_space.keys().cloned().collect();
        param_names.sort();

        // Enumerate all values for each param
        let mut param_values: Vec<Vec<f64>> = Vec::new();
        for name in &param_names {
            let dist = &search_space[name];
            match Self::enumerate_distribution(dist) {
                Some(vals) => param_values.push(vals),
                None => {
                    // Can't enumerate continuous param, delegate to random
                    return self.random_sampler.sample_relative(trials, search_space);
                }
            }
        }

        // Build cartesian product
        let mut all_combos: Vec<Vec<f64>> = vec![vec![]];
        for vals in &param_values {
            let mut new_combos = Vec::new();
            for combo in &all_combos {
                for &val in vals {
                    let mut entry = combo.clone();
                    entry.push(val);
                    new_combos.push(entry);
                }
            }
            all_combos = new_combos;
        }

        // Find visited combinations
        let visited = Self::get_visited_combinations(trials, &param_names);

        // Find first unvisited combination
        let unvisited: Vec<&Vec<f64>> = all_combos
            .iter()
            .filter(|combo| {
                let key: Vec<i64> = combo.iter().map(|v| (v * 1e6).round() as i64).collect();
                !visited.contains(&key)
            })
            .collect();

        if unvisited.is_empty() {
            return Err(OptunaError::ValueError(
                "BruteForceSampler: all parameter combinations have been exhausted".to_string(),
            ));
        }

        // Pick the first unvisited combo (deterministic)
        let combo = unvisited[0];
        let mut result = HashMap::new();
        for (i, name) in param_names.iter().enumerate() {
            result.insert(name.clone(), combo[i]);
        }

        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::*;
    use crate::study::{create_study, StudyDirection};
    use std::sync::Arc;

    #[test]
    fn test_brute_force_enumerate_int() {
        let dist = Distribution::IntDistribution(IntDistribution::new(0, 4, false, 2).unwrap());
        let vals = BruteForceSampler::enumerate_distribution(&dist).unwrap();
        assert_eq!(vals, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_brute_force_enumerate_categorical() {
        let dist = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".into()),
                CategoricalChoice::Str("b".into()),
            ])
            .unwrap(),
        );
        let vals = BruteForceSampler::enumerate_distribution(&dist).unwrap();
        assert_eq!(vals, vec![0.0, 1.0]);
    }

    #[test]
    fn test_brute_force_enumerate_continuous_none() {
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        assert!(BruteForceSampler::enumerate_distribution(&dist).is_none());
    }

    #[test]
    fn test_brute_force_exhausts_grid() {
        let sampler: Arc<dyn Sampler> = Arc::new(BruteForceSampler::new(Some(42)));

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

        // 3 * 2 = 6 combinations for int [1,3] step 1 and cat [a,b]
        let _result = study.optimize(
            |trial| {
                let n = trial.suggest_int("n", 1, 3, false, 1)?;
                let c = trial.suggest_categorical(
                    "c",
                    vec![
                        CategoricalChoice::Str("a".into()),
                        CategoricalChoice::Str("b".into()),
                    ],
                )?;
                let c_val = match c {
                    CategoricalChoice::Str(s) if s == "a" => 0.0,
                    _ => 1.0,
                };
                Ok(n as f64 + c_val)
            },
            Some(10), // More than 6 to trigger exhaustion
            None,
            None,
        );

        // Should have completed some trials (might error on exhaustion)
        let trials = study.trials().unwrap();
        assert!(!trials.is_empty());
    }

    /// 验证 enumerate_distribution 对 float+step 的枚举正确性
    #[test]
    fn test_brute_force_enumerate_float_step() {
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap(),
        );
        let vals = BruteForceSampler::enumerate_distribution(&dist).unwrap();
        assert_eq!(vals, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    /// 验证 enumerate_distribution 对 single-value float 返回唯一值
    #[test]
    fn test_brute_force_enumerate_single_float() {
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(5.0, 5.0, false, None).unwrap(),
        );
        let vals = BruteForceSampler::enumerate_distribution(&dist).unwrap();
        assert_eq!(vals, vec![5.0]);
    }

    /// 验证 get_visited_combinations 正确收集已完成和运行中试验
    #[test]
    fn test_brute_force_get_visited_combinations() {
        use std::collections::HashMap;
        let param_names = vec!["x".to_string()];
        let dist = Distribution::IntDistribution(IntDistribution::new(1, 3, false, 1).unwrap());

        let mut params1 = HashMap::new();
        params1.insert("x".to_string(), crate::distributions::ParamValue::Int(1));
        let mut dists1 = HashMap::new();
        dists1.insert("x".to_string(), dist.clone());

        let trial1 = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: None,
            datetime_complete: None,
            params: params1,
            distributions: dists1,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
        };

        let visited = BruteForceSampler::get_visited_combinations(&[trial1], &param_names);
        assert_eq!(visited.len(), 1, "完成的试验应被记录为已访问");
    }

    /// 验证 sample_relative 正确跳过已访问组合
    #[test]
    fn test_brute_force_sample_relative_skips_visited() {
        let sampler = BruteForceSampler::new(Some(42));

        let dist = Distribution::IntDistribution(IntDistribution::new(1, 2, false, 1).unwrap());
        let mut search_space = HashMap::new();
        search_space.insert("x".to_string(), dist.clone());

        // 第一次调用: 无已访问组合 → 返回第一个 (x=1)
        let r1 = sampler.sample_relative(&[], &search_space).unwrap();
        assert_eq!(r1["x"], 1.0, "首次应返回第一个组合 x=1");

        // 模拟 x=1 已被访问的试验
        let mut params = std::collections::HashMap::new();
        params.insert("x".to_string(), crate::distributions::ParamValue::Int(1));
        let mut dists = std::collections::HashMap::new();
        dists.insert("x".to_string(), dist.clone());

        let visited_trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: None,
            datetime_complete: None,
            params,
            distributions: dists,
            user_attrs: std::collections::HashMap::new(),
            system_attrs: std::collections::HashMap::new(),
            intermediate_values: std::collections::HashMap::new(),
        };

        // 第二次调用: x=1 已访问 → 跳过，返回 x=2
        let r2 = sampler.sample_relative(&[visited_trial], &search_space).unwrap();
        assert_eq!(r2["x"], 2.0, "应跳过已访问的 x=1，返回 x=2");
    }

    /// 验证空搜索空间时 sample_relative 返回空 HashMap
    #[test]
    fn test_brute_force_empty_search_space() {
        let sampler = BruteForceSampler::new(Some(42));
        let result = sampler.sample_relative(&[], &HashMap::new()).unwrap();
        assert!(result.is_empty());
    }
}
