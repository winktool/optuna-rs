use std::collections::HashMap;
use std::sync::Arc;
use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::multi_objective::{
    constrained_fast_non_dominated_sort, crowding_distance, fast_non_dominated_sort,
};
use crate::samplers::random::RandomSampler;
use crate::samplers::Sampler;
use crate::search_space::{IntersectionSearchSpace, SearchSpaceTransform};
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

use super::crossover::{Crossover, UniformCrossover};

/// Type alias for constraint evaluation function.
pub type ConstraintsFn = Arc<dyn Fn(&FrozenTrial) -> Vec<f64> + Send + Sync>;

/// Type alias for elite population selection strategy.
pub type EliteSelectionStrategy =
    Arc<dyn Fn(&[FrozenTrial], &[StudyDirection], usize) -> Vec<FrozenTrial> + Send + Sync>;

/// Type alias for child generation strategy.
pub type ChildGenerationStrategy =
    Arc<dyn Fn(&IndexMap<String, Distribution>, &[FrozenTrial]) -> HashMap<String, f64> + Send + Sync>;

/// after_trial 回调策略。
/// 对应 Python `NSGAIIAfterTrialStrategy`。
/// 参数: (trials, trial, state, values)
pub type AfterTrialStrategy =
    Arc<dyn Fn(&[FrozenTrial], &FrozenTrial, TrialState, Option<&[f64]>) + Send + Sync>;

/// NSGA-II sampler for multi-objective optimization.
///
/// Implements the Non-dominated Sorting Genetic Algorithm II.
pub struct NSGAIISampler {
    directions: Vec<StudyDirection>,
    population_size: usize,
    crossover: Box<dyn Crossover>,
    crossover_prob: f64,
    swapping_prob: f64,
    mutation_prob: Option<f64>,
    rng: Mutex<ChaCha8Rng>,
    random_sampler: RandomSampler,
    search_space: Mutex<IntersectionSearchSpace>,
    /// Constraints function for constrained optimization.
    constraints_func: Option<ConstraintsFn>,
    /// Custom elite population selection strategy.
    elite_population_selection_strategy: Option<EliteSelectionStrategy>,
    /// Custom child generation strategy.
    child_generation_strategy: Option<ChildGenerationStrategy>,
    /// Custom after-trial strategy.
    /// 对应 Python `after_trial_strategy` 参数。
    after_trial_strategy: Option<AfterTrialStrategy>,
}

impl std::fmt::Debug for NSGAIISampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NSGAIISampler")
            .field("population_size", &self.population_size)
            .field("crossover_prob", &self.crossover_prob)
            .finish()
    }
}

impl NSGAIISampler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        directions: Vec<StudyDirection>,
        population_size: Option<usize>,
        crossover: Option<Box<dyn Crossover>>,
        crossover_prob: Option<f64>,
        swapping_prob: Option<f64>,
        mutation_prob: Option<f64>,
        seed: Option<u64>,
        constraints_func: Option<ConstraintsFn>,
        elite_population_selection_strategy: Option<EliteSelectionStrategy>,
        child_generation_strategy: Option<ChildGenerationStrategy>,
        after_trial_strategy: Option<AfterTrialStrategy>,
    ) -> Self {
        let pop_size = population_size.unwrap_or(50);
        // 对齐 Python: population_size >= 2
        assert!(
            pop_size >= 2,
            "`population_size` must be greater than or equal to 2."
        );
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        let crossover_box = crossover.unwrap_or_else(|| {
            // 对应 Python: if crossover is None: crossover = UniformCrossover(swapping_prob)
            Box::new(UniformCrossover::new(swapping_prob))
        });
        // 对齐 Python: population_size >= crossover.n_parents
        assert!(
            pop_size >= crossover_box.n_parents(),
            "population_size ({}) must be >= crossover.n_parents ({})",
            pop_size,
            crossover_box.n_parents()
        );
        Self {
            directions,
            population_size: pop_size,
            crossover: crossover_box,
            crossover_prob: crossover_prob.unwrap_or(0.9),
            swapping_prob: swapping_prob.unwrap_or(0.5),
            mutation_prob,
            rng: Mutex::new(rng),
            random_sampler: RandomSampler::new(seed),
            search_space: Mutex::new(IntersectionSearchSpace::new(false)),
            constraints_func,
            elite_population_selection_strategy,
            child_generation_strategy,
            after_trial_strategy,
        }
    }

    /// Select parent via binary tournament.
    ///
    /// Compare by rank first, then crowding distance.
    fn binary_tournament(
        &self,
        ranks: &[usize],
        crowd_dist: &[f64],
        rng: &mut ChaCha8Rng,
    ) -> usize {
        let n = ranks.len();
        let a = rng.gen_range(0..n);
        let b = rng.gen_range(0..n);

        if ranks[a] < ranks[b] {
            a
        } else if ranks[b] < ranks[a] {
            b
        } else if crowd_dist[a] > crowd_dist[b] {
            a
        } else {
            b
        }
    }

    /// Compute per-individual rank from non-dominated sort fronts.
    fn compute_ranks(fronts: &[Vec<usize>], n: usize) -> Vec<usize> {
        let mut ranks = vec![0usize; n];
        for (rank, front) in fronts.iter().enumerate() {
            for &idx in front {
                ranks[idx] = rank;
            }
        }
        ranks
    }
}

impl Sampler for NSGAIISampler {
    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> IndexMap<String, Distribution> {
        // 对齐 Python: 过滤 single() 分布（单值分布不参与搜索空间）
        let space = self.search_space.lock().calculate(trials);
        space
            .into_iter()
            .filter(|(_, d)| !d.single())
            .collect()
    }

    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &IndexMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        if search_space.is_empty() {
            return Ok(HashMap::new());
        }

        let complete: Vec<&FrozenTrial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.values.is_some())
            .collect();

        // Startup phase: delegate to random sampler
        if complete.len() < self.population_size {
            return self.random_sampler.sample_relative(trials, search_space);
        }

        // ── 步骤1: 精英种群选择 ──
        // 对应 Python `_elite_population_selection_strategy(...)`
        let parent_trials: Vec<FrozenTrial> = if let Some(ref strategy) = self.elite_population_selection_strategy {
            // 使用自定义精英选择策略
            let all_trials: Vec<FrozenTrial> = complete.iter().map(|t| (*t).clone()).collect();
            strategy(&all_trials, &self.directions, self.population_size)
        } else {
            // 对齐 Python 默认精英选择: NSGA-II 非支配排序 + 拥挤距离
            // Python: population_per_rank = _rank_population(population, study.directions, ...)
            //         按 rank 填充精英池，最后一级用拥挤距离截断
            let population: Vec<FrozenTrial> = {
                // 取最后 population_size 个完成的试验作为候选种群
                let start = if complete.len() > self.population_size {
                    complete.len() - self.population_size
                } else {
                    0
                };
                complete[start..].iter().map(|t| (*t).clone()).collect()
            };

            // 非支配排序
            let pop_refs: Vec<&FrozenTrial> = population.iter().collect();
            let ranks_by_rank = if self.constraints_func.is_some() {
                constrained_fast_non_dominated_sort(&pop_refs, &self.directions)
            } else {
                fast_non_dominated_sort(&pop_refs, &self.directions)
            };

            let mut elite = Vec::new();
            for rank_indices in &ranks_by_rank {
                if elite.len() + rank_indices.len() <= self.population_size {
                    // 整级都放入精英池
                    for &idx in rank_indices {
                        elite.push(population[idx].clone());
                    }
                } else {
                    // 最后一级: 按拥挤距离排序，取 remaining 个
                    let remaining = self.population_size - elite.len();
                    let rank_trials: Vec<&FrozenTrial> = rank_indices.iter()
                        .map(|&i| &population[i])
                        .collect();
                    let cd = crowding_distance(&rank_trials, &self.directions);
                    // 按拥挤距离降序排序
                    let mut indexed: Vec<(usize, f64)> = cd.into_iter().enumerate().collect();
                    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
                    for (local_idx, _) in indexed.into_iter().take(remaining) {
                        elite.push(population[rank_indices[local_idx]].clone());
                    }
                    break;
                }
            }
            elite
        };

        if parent_trials.is_empty() {
            return self.random_sampler.sample_relative(trials, search_space);
        }

        // ── 步骤2: 子代生成 ──
        // 对应 Python `sample_relative()` → `self._child_generation_strategy(...)`
        if let Some(ref strategy) = self.child_generation_strategy {
            // 使用自定义子代生成策略
            return Ok(strategy(search_space, &parent_trials));
        }

        // 默认子代生成: tournament + crossover + mutation
        let mut ordered_space = IndexMap::new();
        let mut param_names: Vec<String> = search_space.keys().cloned().collect();
        param_names.sort();
        for name in &param_names {
            ordered_space.insert(name.clone(), search_space[name].clone());
        }

        let transform = SearchSpaceTransform::new(ordered_space.clone(), true, true, true);
        let n_dims = transform.n_encoded();

        let parent_refs: Vec<&FrozenTrial> = parent_trials.iter().collect();

        // Non-dominated sort + crowding distance
        let fronts = if self.constraints_func.is_some() {
            constrained_fast_non_dominated_sort(&parent_refs, &self.directions)
        } else {
            fast_non_dominated_sort(&parent_refs, &self.directions)
        };
        let ranks = Self::compute_ranks(&fronts, parent_refs.len());

        let mut crowd_dist = vec![0.0_f64; parent_refs.len()];
        for front in &fronts {
            let front_trials: Vec<&FrozenTrial> = front.iter().map(|&i| parent_refs[i]).collect();
            let front_cd = crowding_distance(&front_trials, &self.directions);
            for (fi, &idx) in front.iter().enumerate() {
                crowd_dist[idx] = front_cd[fi];
            }
        }

        // Transform parent params to [0,1] space
        let parent_vecs: Vec<Vec<f64>> = parent_refs
            .iter()
            .map(|t| {
                let mut params = IndexMap::new();
                for name in &param_names {
                    if let Some(pv) = t.params.get(name) {
                        params.insert(name.clone(), pv.clone());
                    }
                }
                if params.len() == ordered_space.len() {
                    transform.transform(&params)
                } else {
                    vec![0.5; n_dims]
                }
            })
            .collect();

        let mut rng = self.rng.lock();

        // Select parents via binary tournament
        let n_parents_needed = self.crossover.n_parents();
        let mut parent_indices = Vec::with_capacity(n_parents_needed);
        for _ in 0..n_parents_needed {
            parent_indices.push(self.binary_tournament(&ranks, &crowd_dist, &mut rng));
        }

        let parents: Vec<Vec<f64>> = parent_indices
            .iter()
            .map(|&i| parent_vecs[i].clone())
            .collect();

        // Crossover
        let mut child = if rng.r#gen::<f64>() < self.crossover_prob {
            self.crossover.crossover(&parents, &mut *rng)
        } else {
            parents[0].clone()
        };

        // Mutation: 对齐 Python — 被 mutate 的参数从返回结果中排除，
        // 由 sample_independent 重新采样（保证分布约束）
        let mutation_prob = self
            .mutation_prob
            .unwrap_or_else(|| 1.0 / n_dims.max(1) as f64);
        let mut mutated = vec![false; n_dims];
        for (i, m) in mutated.iter_mut().enumerate() {
            if rng.r#gen::<f64>() < mutation_prob {
                *m = true;
            }
            let _ = i; // suppress unused warning
        }

        // Clamp to [0,1]
        for v in &mut child {
            *v = v.clamp(0.0, 1.0);
        }

        drop(rng);

        // Untransform back to parameter space
        let decoded = transform.untransform(&child)?;
        let mut result = HashMap::new();
        for (i, (name, dist)) in ordered_space.iter().enumerate() {
            // 对齐 Python: 被 mutate 的参数不加入结果, 由 sample_independent 处理
            if mutated[i] {
                continue;
            }
            if let Some(pv) = decoded.get(name) {
                let internal = dist.to_internal_repr(pv)?;
                result.insert(name.clone(), internal);
            }
        }

        Ok(result)
    }

    fn sample_independent(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        self.random_sampler
            .sample_independent(trials, trial, param_name, distribution)
    }

    fn after_trial(
        &self,
        _trials: &[FrozenTrial],
        trial: &FrozenTrial,
        state: TrialState,
        _values: Option<&[f64]>,
    ) {
        // 如果设置了自定义 after_trial_strategy，优先使用
        if let Some(ref strategy) = self.after_trial_strategy {
            strategy(_trials, trial, state, _values);
            return;
        }
        // 默认行为: 评估约束函数
        if let Some(ref cf) = self.constraints_func {
            if state == TrialState::Complete || state == TrialState::Pruned {
                let _constraints = cf(trial);
            }
        }
    }

    /// 对齐 Python `NSGAIISampler.reseed_rng(seed)`: 重新设置随机种子。
    fn reseed_rng(&self, seed: u64) {
        *self.rng.lock() = ChaCha8Rng::seed_from_u64(seed);
    }
}

/// Builder for constructing an [`NSGAIISampler`] with custom parameters.
///
/// # Example
///
/// ```
/// use optuna_rs::{NSGAIISamplerBuilder, StudyDirection};
///
/// let sampler = NSGAIISamplerBuilder::new(vec![
///     StudyDirection::Minimize,
///     StudyDirection::Minimize,
/// ])
/// .population_size(100)
/// .seed(42)
/// .build();
/// ```
pub struct NSGAIISamplerBuilder {
    directions: Vec<StudyDirection>,
    population_size: Option<usize>,
    crossover: Option<Box<dyn Crossover>>,
    crossover_prob: Option<f64>,
    swapping_prob: Option<f64>,
    mutation_prob: Option<f64>,
    seed: Option<u64>,
    constraints_func: Option<ConstraintsFn>,
    elite_population_selection_strategy: Option<EliteSelectionStrategy>,
    child_generation_strategy: Option<ChildGenerationStrategy>,
    after_trial_strategy: Option<AfterTrialStrategy>,
}

impl NSGAIISamplerBuilder {
    /// Create a new builder with the given optimization directions.
    pub fn new(directions: Vec<StudyDirection>) -> Self {
        Self {
            directions,
            population_size: None,
            crossover: None,
            crossover_prob: None,
            swapping_prob: None,
            mutation_prob: None,
            seed: None,
            constraints_func: None,
            elite_population_selection_strategy: None,
            child_generation_strategy: None,
            after_trial_strategy: None,
        }
    }

    /// Set the population size.
    pub fn population_size(mut self, size: usize) -> Self {
        self.population_size = Some(size);
        self
    }

    /// Set the crossover operator.
    pub fn crossover(mut self, crossover: Box<dyn Crossover>) -> Self {
        self.crossover = Some(crossover);
        self
    }

    /// Set the crossover probability.
    pub fn crossover_prob(mut self, prob: f64) -> Self {
        self.crossover_prob = Some(prob);
        self
    }

    /// Set the swapping probability for uniform crossover.
    pub fn swapping_prob(mut self, prob: f64) -> Self {
        self.swapping_prob = Some(prob);
        self
    }

    /// Set the mutation probability.
    pub fn mutation_prob(mut self, prob: f64) -> Self {
        self.mutation_prob = Some(prob);
        self
    }

    /// Set the random seed.
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Set the constraints function for constrained optimization.
    pub fn constraints_func(mut self, func: ConstraintsFn) -> Self {
        self.constraints_func = Some(func);
        self
    }

    /// Set a custom elite population selection strategy.
    pub fn elite_population_selection_strategy(mut self, strategy: EliteSelectionStrategy) -> Self {
        self.elite_population_selection_strategy = Some(strategy);
        self
    }

    /// Set a custom child generation strategy.
    pub fn child_generation_strategy(mut self, strategy: ChildGenerationStrategy) -> Self {
        self.child_generation_strategy = Some(strategy);
        self
    }

    /// 设置自定义 after_trial 策略。
    /// 对应 Python `after_trial_strategy` 参数。
    pub fn after_trial_strategy(mut self, strategy: AfterTrialStrategy) -> Self {
        self.after_trial_strategy = Some(strategy);
        self
    }

    /// Build the [`NSGAIISampler`].
    pub fn build(self) -> NSGAIISampler {
        NSGAIISampler::new(
            self.directions,
            self.population_size,
            self.crossover,
            self.crossover_prob,
            self.swapping_prob,
            self.mutation_prob,
            self.seed,
            self.constraints_func,
            self.elite_population_selection_strategy,
            self.child_generation_strategy,
            self.after_trial_strategy,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use crate::study::{create_study, StudyDirection};

    #[test]
    fn test_nsgaii_sampler_creation() {
        let sampler = NSGAIISampler::new(
            vec![StudyDirection::Minimize, StudyDirection::Minimize],
            Some(10),
            None,
            None,
            None,
            None,
            Some(42),
            None,
            None,
            None,
            None,
        );
        assert_eq!(sampler.population_size, 10);
        assert!((sampler.crossover_prob - 0.9).abs() < 1e-10);
    }

    #[test]
    fn test_nsgaii_startup_random() {
        let sampler: Arc<dyn Sampler> = Arc::new(NSGAIISampler::new(
            vec![StudyDirection::Minimize, StudyDirection::Minimize],
            Some(10),
            None,
            None,
            None,
            None,
            Some(42),
            None,
            None,
            None,
            None,
        ));

        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            None,
            Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]),
            false,
        )
        .unwrap();

        // Should work during startup phase (random sampling)
        study
            .optimize_multi(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                    Ok(vec![x, y])
                },
                Some(5),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 5);
    }

    #[test]
    fn test_nsgaii_full_run() {
        let sampler: Arc<dyn Sampler> = Arc::new(NSGAIISampler::new(
            vec![StudyDirection::Minimize, StudyDirection::Minimize],
            Some(10),
            None,
            None,
            None,
            None,
            Some(42),
            None,
            None,
            None,
            None,
        ));

        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            None,
            Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]),
            false,
        )
        .unwrap();

        // Run enough trials to get past startup phase
        study
            .optimize_multi(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                    Ok(vec![x * x, (1.0 - x) * (1.0 - x) + y * y])
                },
                Some(30),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 30);

        // Check that best_trials returns a Pareto front
        let best = study.best_trials().unwrap();
        assert!(!best.is_empty());
    }

    /// 对齐 Python: population_size < 2 应 panic
    #[test]
    #[should_panic(expected = "population_size")]
    fn test_population_size_too_small() {
        NSGAIISampler::new(
            vec![StudyDirection::Minimize, StudyDirection::Minimize],
            Some(1), None, None, None, None, None, None, None, None, None,
        );
    }
}
