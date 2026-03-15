use std::collections::HashMap;
use std::sync::Arc;
use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::multi_objective::{constrained_fast_non_dominated_sort, fast_non_dominated_sort};
use crate::samplers::nsgaii::crossover::{Crossover, UniformCrossover};
use crate::samplers::random::RandomSampler;
use crate::samplers::Sampler;
use crate::search_space::{IntersectionSearchSpace, SearchSpaceTransform};
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

/// Type alias for constraint evaluation function.
pub type ConstraintsFn = Arc<dyn Fn(&FrozenTrial) -> Vec<f64> + Send + Sync>;

/// 精英种群选择策略。
/// 对应 Python `NSGAIIIElitePopulationSelectionStrategy`。
pub type EliteSelectionStrategy =
    Arc<dyn Fn(&[FrozenTrial], &[StudyDirection], usize) -> Vec<FrozenTrial> + Send + Sync>;

/// 子代生成策略。
/// 对应 Python `NSGAIIChildGenerationStrategy`。
pub type ChildGenerationStrategy =
    Arc<dyn Fn(&HashMap<String, Distribution>, &[FrozenTrial]) -> HashMap<String, f64> + Send + Sync>;

/// after_trial 回调策略。
/// 对应 Python `NSGAIIAfterTrialStrategy`。
pub type AfterTrialStrategy =
    Arc<dyn Fn(&[FrozenTrial], &FrozenTrial, TrialState, Option<&[f64]>) + Send + Sync>;

/// NSGA-III sampler for many-objective optimization.
///
/// Uses reference-point-based selection instead of crowding distance.
pub struct NSGAIIISampler {
    directions: Vec<StudyDirection>,
    population_size: usize,
    crossover: Box<dyn Crossover>,
    crossover_prob: f64,
    swapping_prob: f64,
    mutation_prob: Option<f64>,
    reference_points: Vec<Vec<f64>>,
    rng: Mutex<ChaCha8Rng>,
    random_sampler: RandomSampler,
    search_space: Mutex<IntersectionSearchSpace>,
    /// Constraints function for constrained optimization.
    constraints_func: Option<ConstraintsFn>,
    /// 精英种群选择策略。
    elite_population_selection_strategy: Option<EliteSelectionStrategy>,
    /// 子代生成策略。
    child_generation_strategy: Option<ChildGenerationStrategy>,
    /// after_trial 回调策略。
    after_trial_strategy: Option<AfterTrialStrategy>,
}

impl std::fmt::Debug for NSGAIIISampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("NSGAIIISampler")
            .field("population_size", &self.population_size)
            .field("n_reference_points", &self.reference_points.len())
            .finish()
    }
}

/// Generate reference points using the Das-Dennis method.
///
/// Generates uniformly distributed points on a simplex in `n_objectives` dimensions
/// with `dividing_parameter` divisions.
pub fn generate_reference_points(n_objectives: usize, dividing_parameter: usize) -> Vec<Vec<f64>> {
    let mut points = Vec::new();
    let mut point = vec![0.0; n_objectives];
    das_dennis_recursive(
        &mut points,
        &mut point,
        n_objectives,
        dividing_parameter,
        dividing_parameter,
        0,
    );
    points
}

fn das_dennis_recursive(
    points: &mut Vec<Vec<f64>>,
    point: &mut Vec<f64>,
    n_objectives: usize,
    dividing_parameter: usize,
    remaining: usize,
    depth: usize,
) {
    if depth == n_objectives - 1 {
        point[depth] = remaining as f64 / dividing_parameter as f64;
        points.push(point.clone());
        return;
    }

    for i in 0..=remaining {
        point[depth] = i as f64 / dividing_parameter as f64;
        das_dennis_recursive(points, point, n_objectives, dividing_parameter, remaining - i, depth + 1);
    }
}

impl NSGAIIISampler {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        directions: Vec<StudyDirection>,
        population_size: Option<usize>,
        crossover: Option<Box<dyn Crossover>>,
        crossover_prob: Option<f64>,
        swapping_prob: Option<f64>,
        mutation_prob: Option<f64>,
        dividing_parameter: Option<usize>,
        reference_points: Option<Vec<Vec<f64>>>,
        seed: Option<u64>,
        constraints_func: Option<ConstraintsFn>,
        elite_population_selection_strategy: Option<EliteSelectionStrategy>,
        child_generation_strategy: Option<ChildGenerationStrategy>,
        after_trial_strategy: Option<AfterTrialStrategy>,
    ) -> Self {
        let n_obj = directions.len();
        let divs = dividing_parameter.unwrap_or(3);
        let ref_pts = reference_points.unwrap_or_else(|| generate_reference_points(n_obj, divs));
        let pop_size = population_size.unwrap_or_else(|| ref_pts.len().max(10));

        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };

        Self {
            directions,
            population_size: pop_size,
            crossover: crossover.unwrap_or_else(|| {
                // 对应 Python: if crossover is None: crossover = UniformCrossover(swapping_prob)
                Box::new(UniformCrossover::new(swapping_prob))
            }),
            crossover_prob: crossover_prob.unwrap_or(0.9),
            swapping_prob: swapping_prob.unwrap_or(0.5),
            mutation_prob,
            reference_points: ref_pts,
            rng: Mutex::new(rng),
            random_sampler: RandomSampler::new(seed),
            search_space: Mutex::new(IntersectionSearchSpace::new(false)),
            constraints_func,
            elite_population_selection_strategy,
            child_generation_strategy,
            after_trial_strategy,
        }
    }

    /// Associate a trial's objective values with the nearest reference point.
    /// Returns the index of the nearest reference point.
    fn associate_to_reference_point(&self, normalized_values: &[f64]) -> usize {
        let mut best_dist = f64::INFINITY;
        let mut best_idx = 0;

        for (i, ref_point) in self.reference_points.iter().enumerate() {
            let dist = perpendicular_distance(normalized_values, ref_point);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Normalize objective values to [0, 1] range.
    fn normalize_values(
        trials: &[&FrozenTrial],
        directions: &[StudyDirection],
    ) -> Vec<Vec<f64>> {
        let n_obj = directions.len();
        if trials.is_empty() {
            return vec![];
        }

        // Find ideal (min) and nadir (max) for each objective
        let mut ideal = vec![f64::INFINITY; n_obj];
        let mut nadir = vec![f64::NEG_INFINITY; n_obj];

        for t in trials {
            if let Some(values) = &t.values {
                for (m, &v) in values.iter().enumerate() {
                    let v_normalized = match directions[m] {
                        StudyDirection::Maximize => -v,
                        _ => v,
                    };
                    ideal[m] = ideal[m].min(v_normalized);
                    nadir[m] = nadir[m].max(v_normalized);
                }
            }
        }

        trials
            .iter()
            .map(|t| {
                let values = t.values.as_deref().unwrap_or(&[]);
                (0..n_obj)
                    .map(|m| {
                        let v = if m < values.len() {
                            match directions[m] {
                                StudyDirection::Maximize => -values[m],
                                _ => values[m],
                            }
                        } else {
                            0.0
                        };
                        let range = nadir[m] - ideal[m];
                        if range < 1e-10 {
                            0.0
                        } else {
                            (v - ideal[m]) / range
                        }
                    })
                    .collect()
            })
            .collect()
    }

    /// Niche-count-based tournament selection.
    fn select_parent(
        &self,
        ranks: &[usize],
        niche_counts: &[usize],
        rng: &mut ChaCha8Rng,
    ) -> usize {
        let n = ranks.len();
        let a = rng.gen_range(0..n);
        let b = rng.gen_range(0..n);

        if ranks[a] < ranks[b] {
            a
        } else if ranks[b] < ranks[a] {
            b
        } else if niche_counts[a] < niche_counts[b] {
            a
        } else {
            b
        }
    }
}

/// Perpendicular distance from a point to a reference direction line.
fn perpendicular_distance(point: &[f64], direction: &[f64]) -> f64 {
    let norm_sq: f64 = direction.iter().map(|d| d * d).sum();
    if norm_sq < 1e-14 {
        return f64::INFINITY;
    }

    let dot: f64 = point.iter().zip(direction.iter()).map(|(p, d)| p * d).sum();
    let proj_scale = dot / norm_sq;

    let dist_sq: f64 = point
        .iter()
        .zip(direction.iter())
        .map(|(p, d)| {
            let diff = p - proj_scale * d;
            diff * diff
        })
        .sum();

    dist_sq.sqrt()
}

impl Sampler for NSGAIIISampler {
    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> HashMap<String, Distribution> {
        self.search_space.lock().calculate(trials)
    }

    fn sample_relative(
        &self,
        trials: &[FrozenTrial],
        search_space: &HashMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        if search_space.is_empty() {
            return Ok(HashMap::new());
        }

        let complete: Vec<&FrozenTrial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete && t.values.is_some())
            .collect();

        // Startup phase
        if complete.len() < self.population_size {
            return self.random_sampler.sample_relative(trials, search_space);
        }

        // ── 步骤1: 精英种群选择 ──
        // 对应 Python `select_parent()` → `self._elite_population_selection_strategy(...)`
        let parent_trials: Vec<FrozenTrial> = if let Some(ref strategy) = self.elite_population_selection_strategy {
            let all_trials: Vec<FrozenTrial> = complete.iter().map(|t| (*t).clone()).collect();
            strategy(&all_trials, &self.directions, self.population_size)
        } else {
            // 默认: 取最后 population_size 个完成试验
            let start = if complete.len() > self.population_size {
                complete.len() - self.population_size
            } else {
                0
            };
            complete[start..].iter().map(|t| (*t).clone()).collect()
        };

        if parent_trials.is_empty() {
            return self.random_sampler.sample_relative(trials, search_space);
        }

        // ── 步骤2: 子代生成 ──
        // 对应 Python `sample_relative()` → `self._child_generation_strategy(...)`
        if let Some(ref strategy) = self.child_generation_strategy {
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

        // Non-dominated sort
        let fronts = if self.constraints_func.is_some() {
            constrained_fast_non_dominated_sort(&parent_refs, &self.directions)
        } else {
            fast_non_dominated_sort(&parent_refs, &self.directions)
        };
        let mut ranks = vec![0usize; parent_refs.len()];
        for (rank, front) in fronts.iter().enumerate() {
            for &idx in front {
                ranks[idx] = rank;
            }
        }

        // Normalize values and compute niche counts
        let normalized = Self::normalize_values(&parent_refs, &self.directions);
        let mut niche_counts = vec![0usize; self.reference_points.len()];
        let mut trial_niches = vec![0usize; parent_refs.len()];
        for (i, nv) in normalized.iter().enumerate() {
            let niche = self.associate_to_reference_point(nv);
            trial_niches[i] = niche;
            niche_counts[niche] += 1;
        }

        let individual_niche_counts: Vec<usize> = trial_niches
            .iter()
            .map(|&niche| niche_counts[niche])
            .collect();

        // Transform parents
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

        // Select parents
        let n_parents_needed = self.crossover.n_parents();
        let mut parent_indices = Vec::with_capacity(n_parents_needed);
        for _ in 0..n_parents_needed {
            parent_indices.push(self.select_parent(&ranks, &individual_niche_counts, &mut rng));
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

        // Mutation
        let mutation_prob = self
            .mutation_prob
            .unwrap_or_else(|| 1.0 / n_dims.max(1) as f64);
        for v in &mut child {
            if rng.r#gen::<f64>() < mutation_prob {
                *v = rng.r#gen::<f64>();
            }
        }

        for v in &mut child {
            *v = v.clamp(0.0, 1.0);
        }

        drop(rng);

        // Untransform
        let decoded = transform.untransform(&child)?;
        let mut result = HashMap::new();
        for (name, dist) in &ordered_space {
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
}

/// Builder for constructing an [`NSGAIIISampler`] with custom parameters.
///
/// # Example
///
/// ```
/// use optuna_rs::{NSGAIIISamplerBuilder, StudyDirection};
///
/// let sampler = NSGAIIISamplerBuilder::new(vec![
///     StudyDirection::Minimize,
///     StudyDirection::Minimize,
///     StudyDirection::Minimize,
/// ])
/// .population_size(100)
/// .dividing_parameter(4)
/// .seed(42)
/// .build();
/// ```
pub struct NSGAIIISamplerBuilder {
    directions: Vec<StudyDirection>,
    population_size: Option<usize>,
    crossover: Option<Box<dyn Crossover>>,
    crossover_prob: Option<f64>,
    swapping_prob: Option<f64>,
    mutation_prob: Option<f64>,
    dividing_parameter: Option<usize>,
    reference_points: Option<Vec<Vec<f64>>>,
    seed: Option<u64>,
    constraints_func: Option<ConstraintsFn>,
    elite_population_selection_strategy: Option<EliteSelectionStrategy>,
    child_generation_strategy: Option<ChildGenerationStrategy>,
    after_trial_strategy: Option<AfterTrialStrategy>,
}

impl NSGAIIISamplerBuilder {
    /// Create a new builder with the given optimization directions.
    pub fn new(directions: Vec<StudyDirection>) -> Self {
        Self {
            directions,
            population_size: None,
            crossover: None,
            crossover_prob: None,
            swapping_prob: None,
            mutation_prob: None,
            dividing_parameter: None,
            reference_points: None,
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

    /// Set the mutation probability.
    pub fn mutation_prob(mut self, prob: f64) -> Self {
        self.mutation_prob = Some(prob);
        self
    }

    /// Set the dividing parameter for Das-Dennis reference point generation.
    pub fn dividing_parameter(mut self, divs: usize) -> Self {
        self.dividing_parameter = Some(divs);
        self
    }

    /// Set custom reference points (overrides dividing_parameter).
    pub fn reference_points(mut self, points: Vec<Vec<f64>>) -> Self {
        self.reference_points = Some(points);
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

    /// 设置 swapping 概率（uniform 交叉时使用）。
    pub fn swapping_prob(mut self, prob: f64) -> Self {
        self.swapping_prob = Some(prob);
        self
    }

    /// 设置精英种群选择策略。
    pub fn elite_population_selection_strategy(mut self, strategy: EliteSelectionStrategy) -> Self {
        self.elite_population_selection_strategy = Some(strategy);
        self
    }

    /// 设置子代生成策略。
    pub fn child_generation_strategy(mut self, strategy: ChildGenerationStrategy) -> Self {
        self.child_generation_strategy = Some(strategy);
        self
    }

    /// 设置 after_trial 策略。
    pub fn after_trial_strategy(mut self, strategy: AfterTrialStrategy) -> Self {
        self.after_trial_strategy = Some(strategy);
        self
    }

    /// Build the [`NSGAIIISampler`].
    pub fn build(self) -> NSGAIIISampler {
        NSGAIIISampler::new(
            self.directions,
            self.population_size,
            self.crossover,
            self.crossover_prob,
            self.swapping_prob,
            self.mutation_prob,
            self.dividing_parameter,
            self.reference_points,
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
    fn test_das_dennis_2d() {
        let pts = generate_reference_points(2, 4);
        // For 2 objectives with 4 divisions: 5 points
        assert_eq!(pts.len(), 5);
        // Each point should sum to 1.0
        for pt in &pts {
            let sum: f64 = pt.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_das_dennis_3d() {
        let pts = generate_reference_points(3, 3);
        // C(3+3-1, 3-1) = C(5,2) = 10
        assert_eq!(pts.len(), 10);
        for pt in &pts {
            let sum: f64 = pt.iter().sum();
            assert!((sum - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_perpendicular_distance() {
        let dist = perpendicular_distance(&[1.0, 0.0], &[1.0, 1.0]);
        // Distance from (1,0) to line through origin in direction (1,1)
        assert!(dist > 0.0);
    }

    #[test]
    fn test_nsgaiii_basic() {
        let sampler: Arc<dyn Sampler> = Arc::new(NSGAIIISampler::new(
            vec![StudyDirection::Minimize, StudyDirection::Minimize],
            Some(10),
            None,
            None,
            None,
            None,
            Some(3),
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

        study
            .optimize_multi(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
                    Ok(vec![x, y])
                },
                Some(25),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 25);
    }
}
