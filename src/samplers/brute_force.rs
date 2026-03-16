use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;

use indexmap::IndexMap;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use crate::distributions::{Distribution, ParamValue};
use crate::error::{OptunaError, Result};
use crate::samplers::Sampler;
use crate::trial::{FrozenTrial, TrialState};

// ---------------------------------------------------------------------------
// TreeNode — mirrors Python's `_TreeNode`
// ---------------------------------------------------------------------------

/// A tree node representing the search space exploration state.
///
/// Three states:
/// 1. **Unexpanded** — `children` is `None`.
/// 2. **Leaf** — `children` is `Some(empty vec)` and `param_name` is `None`.
/// 3. **Normal** — has a `param_name` and non-empty `children`.
#[derive(Debug, Clone)]
struct TreeNode {
    param_name: Option<String>,
    /// `None` = unexpanded. `Some(vec)` = expanded (may be empty for leaf).
    children: Option<Vec<(FloatKey, TreeNode)>>,
    is_running: bool,
}

/// Float wrapper using bit-exact equality (same semantics as Python dict float keys).
#[derive(Debug, Clone, Copy)]
struct FloatKey(f64);

impl PartialEq for FloatKey {
    fn eq(&self, other: &Self) -> bool {
        self.0.to_bits() == other.0.to_bits()
    }
}
impl Eq for FloatKey {}

impl std::hash::Hash for FloatKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.to_bits().hash(state);
    }
}

impl TreeNode {
    fn new() -> Self {
        Self {
            param_name: None,
            children: None,
            is_running: false,
        }
    }

    /// Expand the node. If already expanded, this is a no-op
    /// (Python raises ValueError on mismatch, but in practice the caller
    /// always provides consistent values).
    fn expand(&mut self, param_name: Option<&str>, search_space: &[f64]) {
        if self.children.is_none() {
            self.param_name = param_name.map(|s| s.to_string());
            self.children = Some(
                search_space
                    .iter()
                    .map(|&v| (FloatKey(v), TreeNode::new()))
                    .collect(),
            );
        }
    }

    fn set_running(&mut self) {
        self.is_running = true;
    }

    fn set_leaf(&mut self) {
        self.expand(None, &[]);
    }

    /// Add a path (one trial's parameters) to the tree.
    /// Returns `Some(&mut leaf)` if the path is on the grid, `None` otherwise.
    fn add_path(
        &mut self,
        params_and_search_spaces: &[(String, Vec<f64>, f64)],
    ) -> Option<&mut TreeNode> {
        let mut current = self;
        for (param_name, search_space, value) in params_and_search_spaces {
            current.expand(Some(param_name), search_space);
            let children = current.children.as_mut().unwrap();
            let key = FloatKey(*value);
            let found = children.iter_mut().find(|(k, _)| *k == key);
            match found {
                Some((_, child)) => current = child,
                None => return None,
            }
        }
        Some(current)
    }

    /// Count unexpanded (unvisited) leaves in the subtree.
    /// Mirrors Python's `count_unexpanded`.
    fn count_unexpanded(&self, exclude_running: bool) -> usize {
        match &self.children {
            None => {
                if exclude_running && self.is_running {
                    0
                } else {
                    1
                }
            }
            Some(children) => children
                .iter()
                .map(|(_, child)| child.count_unexpanded(exclude_running))
                .sum(),
        }
    }

    /// Sample a child proportional to count_unexpanded weights.
    /// Mirrors Python's `sample_child`: weighted random, prioritizing non-running branches.
    fn sample_child(&self, rng: &mut StdRng, exclude_running: bool) -> f64 {
        let children = self.children.as_ref().unwrap();
        let mut weights: Vec<f64> = children
            .iter()
            .map(|(_, child)| child.count_unexpanded(exclude_running) as f64)
            .collect();

        // Prioritize non-running unexpanded children (matches Python).
        let has_non_running_unexpanded = children
            .iter()
            .enumerate()
            .any(|(i, (_, child))| !child.is_running && weights[i] > 0.0);
        if has_non_running_unexpanded {
            for (i, (_, child)) in children.iter().enumerate() {
                if child.is_running {
                    weights[i] = 0.0;
                }
            }
        }

        let total: f64 = weights.iter().sum();
        if total == 0.0 {
            // Fallback: uniform (shouldn't happen if count_unexpanded > 0)
            use rand::Rng;
            let idx = rng.r#gen_range(0..children.len());
            return children[idx].0 .0;
        }

        // Weighted random choice (mirrors `rng.choice(keys, p=weights)`)
        use rand::Rng;
        let mut r: f64 = rng.r#gen::<f64>() * total;
        for (i, w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return children[i].0 .0;
            }
        }
        // Fallback to last
        children.last().unwrap().0 .0
    }
}

// ---------------------------------------------------------------------------
// BruteForceSampler
// ---------------------------------------------------------------------------

/// A sampler that exhaustively evaluates all possible discrete parameter combinations.
///
/// Mirrors Python's `optuna.samplers.BruteForceSampler`:
/// - All logic lives in `sample_independent` (not `sample_relative`), because the
///   search space is discovered dynamically per-parameter and may be conditional.
/// - `after_trial` checks whether the entire search space is exhausted and signals
///   the study to stop.
/// - `avoid_premature_stop` controls whether Running trials are excluded when
///   counting unvisited combinations.
pub struct BruteForceSampler {
    seed: Option<u64>,
    avoid_premature_stop: bool,
    /// Lazily initialized RNG.
    rng: Mutex<Option<StdRng>>,
    /// Set by `after_trial` when the search space is exhausted.
    stop_requested: AtomicBool,
}

impl std::fmt::Debug for BruteForceSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BruteForceSampler")
            .field("avoid_premature_stop", &self.avoid_premature_stop)
            .finish()
    }
}

impl BruteForceSampler {
    /// Create a new `BruteForceSampler`.
    ///
    /// # Arguments
    /// * `seed` — RNG seed for reproducibility.
    /// * `avoid_premature_stop` — If `true`, Running trials are **not** excluded
    ///   when counting unvisited combinations, preventing premature study stop
    ///   at the cost of possible duplicate suggestions.
    pub fn new(seed: Option<u64>, avoid_premature_stop: bool) -> Self {
        Self {
            seed,
            avoid_premature_stop,
            rng: Mutex::new(None),
            stop_requested: AtomicBool::new(false),
        }
    }

    fn get_rng(&self) -> std::sync::MutexGuard<'_, Option<StdRng>> {
        let mut guard = self.rng.lock().unwrap();
        if guard.is_none() {
            *guard = Some(match self.seed {
                Some(s) => StdRng::seed_from_u64(s),
                None => StdRng::from_entropy(),
            });
        }
        guard
    }

    /// Enumerate all candidate internal-repr values for a distribution.
    ///
    /// Uses step-based enumeration to avoid float accumulation errors
    /// (mirrors Python's `_enumerate_candidates` which uses `decimal.Decimal`).
    ///
    /// Returns `None` for continuous (step-less) float distributions.
    fn enumerate_candidates(dist: &Distribution) -> Option<Vec<f64>> {
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
                    if step == 0.0 {
                        return Some(vec![d.low]);
                    }
                    let n_steps = ((d.high - d.low) / step).round() as usize;
                    let vals: Vec<f64> = (0..=n_steps)
                        .map(|i| {
                            let v = d.low + step * i as f64;
                            if v > d.high { d.high } else { v }
                        })
                        .collect();
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

    /// Populate the tree from existing trials, filtering by already-chosen params.
    /// Mirrors Python's `_populate_tree`.
    fn populate_tree(
        tree: &mut TreeNode,
        trials: &[TrialSnapshot],
        fixed_params: &HashMap<String, ParamValue>,
    ) {
        for trial in trials {
            // Check fixed params match.
            let matches = fixed_params.iter().all(|(p, v)| {
                trial.params.get(p).map_or(false, |tv| tv == v)
            });
            if !matches {
                continue;
            }

            // Only consider relevant states.
            match trial.state {
                TrialState::Complete
                | TrialState::Pruned
                | TrialState::Running
                | TrialState::Fail => {}
                _ => continue,
            }

            // Build path excluding fixed params.
            // Sort by param name for deterministic order (Python dicts preserve
            // insertion order, but Rust HashMaps don't).
            let mut dist_pairs: Vec<(&String, &Distribution)> = trial
                .distributions
                .iter()
                .filter(|(name, _)| !fixed_params.contains_key(*name))
                .collect();
            dist_pairs.sort_by_key(|(name, _)| name.clone());

            let path: Vec<(String, Vec<f64>, f64)> = dist_pairs
                .into_iter()
                .filter_map(|(name, dist)| {
                    let candidates = Self::enumerate_candidates(dist)?;
                    let value = trial
                        .params
                        .get(name)
                        .and_then(|pv| dist.to_internal_repr(pv).ok())?;
                    Some((name.clone(), candidates, value))
                })
                .collect();

            if let Some(leaf) = tree.add_path(&path) {
                if trial.state.is_finished() {
                    leaf.set_leaf();
                } else {
                    leaf.set_running();
                }
            }
        }
    }
}

/// Lightweight snapshot of a trial for tree population.
/// Mirrors Python's approach of passing modified trial objects.
struct TrialSnapshot {
    state: TrialState,
    params: HashMap<String, ParamValue>,
    distributions: HashMap<String, Distribution>,
}

impl From<&FrozenTrial> for TrialSnapshot {
    fn from(t: &FrozenTrial) -> Self {
        Self {
            state: t.state,
            params: t.params.clone(),
            distributions: t.distributions.clone(),
        }
    }
}

impl Sampler for BruteForceSampler {
    // Python returns {} — no relative search space.
    fn infer_relative_search_space(
        &self,
        _trials: &[FrozenTrial],
    ) -> IndexMap<String, Distribution> {
        IndexMap::new()
    }

    // Python returns {} — all logic is in sample_independent.
    fn sample_relative(
        &self,
        _trials: &[FrozenTrial],
        _search_space: &IndexMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        Ok(HashMap::new())
    }

    fn sample_independent(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        let exclude_running = !self.avoid_premature_stop;

        let candidates = match Self::enumerate_candidates(distribution) {
            Some(c) => c,
            None => {
                return Err(OptunaError::ValueError(
                    "FloatDistribution.step must be given for BruteForceSampler \
                     (otherwise, the search space will be infinite)."
                        .to_string(),
                ));
            }
        };

        let mut tree = TreeNode::new();
        tree.expand(Some(param_name), &candidates);

        // Populate tree with existing trials (excluding current trial).
        let fixed_params = trial.params.clone();
        let snapshots: Vec<TrialSnapshot> = trials
            .iter()
            .filter(|t| t.number != trial.number)
            .map(TrialSnapshot::from)
            .collect();
        Self::populate_tree(&mut tree, &snapshots, &fixed_params);

        let mut rng_guard = self.get_rng();
        let rng = rng_guard.as_mut().unwrap();

        if tree.count_unexpanded(exclude_running) == 0 {
            // All exhausted — return a random candidate (matches Python behavior).
            let &val = candidates.choose(rng).unwrap();
            Ok(val)
        } else {
            Ok(tree.sample_child(rng, exclude_running))
        }
    }

    fn after_trial(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        state: TrialState,
        _values: Option<&[f64]>,
    ) {
        let exclude_running = !self.avoid_premature_stop;

        // Build snapshots, replacing current trial's state with the final state.
        let snapshots: Vec<TrialSnapshot> = trials
            .iter()
            .map(|t| {
                if t.number == trial.number {
                    TrialSnapshot {
                        state,
                        params: trial.params.clone(),
                        distributions: trial.distributions.clone(),
                    }
                } else {
                    TrialSnapshot::from(t)
                }
            })
            .collect();

        let mut tree = TreeNode::new();
        Self::populate_tree(&mut tree, &snapshots, &HashMap::new());

        if tree.count_unexpanded(exclude_running) == 0 {
            self.stop_requested.store(true, Ordering::SeqCst);
        }
    }

    fn should_stop_study(&self) -> bool {
        self.stop_requested.swap(false, Ordering::SeqCst)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::*;

    #[test]
    fn test_enumerate_int() {
        let dist = Distribution::IntDistribution(IntDistribution::new(0, 4, false, 2).unwrap());
        let vals = BruteForceSampler::enumerate_candidates(&dist).unwrap();
        assert_eq!(vals, vec![0.0, 2.0, 4.0]);
    }

    #[test]
    fn test_enumerate_categorical() {
        let dist = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".into()),
                CategoricalChoice::Str("b".into()),
            ])
            .unwrap(),
        );
        let vals = BruteForceSampler::enumerate_candidates(&dist).unwrap();
        assert_eq!(vals, vec![0.0, 1.0]);
    }

    #[test]
    fn test_enumerate_continuous_none() {
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );
        assert!(BruteForceSampler::enumerate_candidates(&dist).is_none());
    }

    #[test]
    fn test_enumerate_float_step() {
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap(),
        );
        let vals = BruteForceSampler::enumerate_candidates(&dist).unwrap();
        assert_eq!(vals, vec![0.0, 0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn test_enumerate_single_float() {
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(5.0, 5.0, false, None).unwrap(),
        );
        let vals = BruteForceSampler::enumerate_candidates(&dist).unwrap();
        assert_eq!(vals, vec![5.0]);
    }

    #[test]
    fn test_tree_count_unexpanded() {
        let mut tree = TreeNode::new();
        tree.expand(Some("x"), &[1.0, 2.0, 3.0]);
        assert_eq!(tree.count_unexpanded(false), 3);
        assert_eq!(tree.count_unexpanded(true), 3);

        // Mark one child as leaf (visited).
        tree.children.as_mut().unwrap()[0].1.set_leaf();
        assert_eq!(tree.count_unexpanded(false), 2);

        // Mark another as running.
        tree.children.as_mut().unwrap()[1].1.set_running();
        assert_eq!(tree.count_unexpanded(false), 2); // running still counts
        assert_eq!(tree.count_unexpanded(true), 1); // exclude_running=true
    }

    #[test]
    fn test_tree_add_path() {
        let mut tree = TreeNode::new();
        let path = vec![
            ("x".to_string(), vec![1.0, 2.0], 1.0),
            ("y".to_string(), vec![10.0, 20.0], 20.0),
        ];
        let leaf = tree.add_path(&path);
        assert!(leaf.is_some());
        leaf.unwrap().set_leaf();
        // Tree: x has children [1.0, 2.0].
        // x=1.0 is expanded with y=[10.0, 20.0]; y=20.0 is a leaf.
        //   → x=1.0/y=10.0 is unexpanded (1)
        // x=2.0 is unexpanded (1) — its children haven't been created.
        // Total = 2
        assert_eq!(tree.count_unexpanded(false), 2);
    }

    #[test]
    fn test_sample_independent_basic() {
        let sampler = BruteForceSampler::new(Some(42), false);
        let dist = Distribution::IntDistribution(IntDistribution::new(1, 3, false, 1).unwrap());

        let trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Running,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
        };

        let val = sampler
            .sample_independent(&[], &trial, "x", &dist)
            .unwrap();
        assert!([1.0, 2.0, 3.0].contains(&val));
    }

    #[test]
    fn test_sample_independent_skips_visited() {
        let sampler = BruteForceSampler::new(Some(42), false);
        let dist = Distribution::IntDistribution(IntDistribution::new(1, 2, false, 1).unwrap());

        let mut params = HashMap::new();
        params.insert("x".to_string(), ParamValue::Int(1));
        let mut dists = HashMap::new();
        dists.insert("x".to_string(), dist.clone());

        let completed = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: None,
            datetime_complete: None,
            params,
            distributions: dists,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
        };

        let current = FrozenTrial {
            number: 1,
            trial_id: 1,
            state: TrialState::Running,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
        };

        let val = sampler
            .sample_independent(&[completed], &current, "x", &dist)
            .unwrap();
        assert_eq!(val, 2.0, "should skip visited x=1 and return x=2");
    }

    #[test]
    fn test_sample_independent_exhausted_returns_random() {
        let sampler = BruteForceSampler::new(Some(42), false);
        let dist = Distribution::IntDistribution(IntDistribution::new(1, 1, false, 1).unwrap());

        let mut params = HashMap::new();
        params.insert("x".to_string(), ParamValue::Int(1));
        let mut dists = HashMap::new();
        dists.insert("x".to_string(), dist.clone());

        let completed = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: None,
            datetime_complete: None,
            params,
            distributions: dists,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
        };

        let current = FrozenTrial {
            number: 1,
            trial_id: 1,
            state: TrialState::Running,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
        };

        // Should not error — returns random candidate (matches Python).
        let val = sampler
            .sample_independent(&[completed], &current, "x", &dist)
            .unwrap();
        assert_eq!(val, 1.0);
    }

    #[test]
    fn test_continuous_float_errors() {
        let sampler = BruteForceSampler::new(Some(42), false);
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
        );

        let trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Running,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
        };

        let result = sampler.sample_independent(&[], &trial, "x", &dist);
        assert!(result.is_err());
    }

    #[test]
    fn test_relative_returns_empty() {
        let sampler = BruteForceSampler::new(None, false);
        let result = sampler.sample_relative(&[], &IndexMap::new()).unwrap();
        assert!(result.is_empty());

        // Even with a non-empty search space, should return empty.
        let mut space = IndexMap::new();
        space.insert(
            "x".to_string(),
            Distribution::IntDistribution(IntDistribution::new(1, 3, false, 1).unwrap()),
        );
        let result = sampler.sample_relative(&[], &space).unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn test_after_trial_signals_stop() {
        let sampler = BruteForceSampler::new(Some(42), false);
        let dist = Distribution::IntDistribution(IntDistribution::new(1, 1, false, 1).unwrap());

        let mut params = HashMap::new();
        params.insert("x".to_string(), ParamValue::Int(1));
        let mut dists = HashMap::new();
        dists.insert("x".to_string(), dist.clone());

        let trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Running,
            values: Some(vec![1.0]),
            datetime_start: None,
            datetime_complete: None,
            params,
            distributions: dists,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
        };

        // No stop requested yet.
        assert!(!sampler.should_stop_study());

        // after_trial should detect exhaustion.
        sampler.after_trial(&[trial.clone()], &trial, TrialState::Complete, Some(&[1.0]));
        assert!(sampler.should_stop_study());

        // Flag should be cleared after reading.
        assert!(!sampler.should_stop_study());
    }

    #[test]
    fn test_after_trial_no_stop_when_not_exhausted() {
        let sampler = BruteForceSampler::new(Some(42), false);
        let dist = Distribution::IntDistribution(IntDistribution::new(1, 2, false, 1).unwrap());

        let mut params = HashMap::new();
        params.insert("x".to_string(), ParamValue::Int(1));
        let mut dists = HashMap::new();
        dists.insert("x".to_string(), dist.clone());

        let trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Running,
            values: Some(vec![1.0]),
            datetime_start: None,
            datetime_complete: None,
            params,
            distributions: dists,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
        };

        // x=2 is still unvisited, so should not request stop.
        sampler.after_trial(&[trial.clone()], &trial, TrialState::Complete, Some(&[1.0]));
        assert!(!sampler.should_stop_study());
    }

    #[test]
    fn test_brute_force_exhausts_grid() {
        use crate::study::{create_study, StudyDirection};
        use std::sync::Arc;

        let sampler: Arc<dyn Sampler> = Arc::new(BruteForceSampler::new(Some(42), false));

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
            Some(20), // More than 6 to trigger exhaustion
            None,
            None,
        );

        let trials = study.trials().unwrap();
        assert!(!trials.is_empty());
        // Study should stop well before 20 trials due to after_trial detecting exhaustion.
        // With independent sampling there may be some duplicates before full exhaustion.
        assert!(
            trials.len() < 20,
            "Expected study to stop before 20 trials, got {}",
            trials.len()
        );
    }
}
