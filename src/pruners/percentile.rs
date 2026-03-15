use crate::error::Result;
use crate::pruners::Pruner;
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

/// A pruner that prunes trials whose best intermediate value is worse than
/// a given percentile of completed trials at the same step.
///
/// Corresponds to Python `optuna.pruners.PercentilePruner`.
#[derive(Debug, Clone)]
pub struct PercentilePruner {
    percentile: f64,
    n_startup_trials: usize,
    n_warmup_steps: i64,
    interval_steps: i64,
    n_min_trials: usize,
    direction: StudyDirection,
}

impl PercentilePruner {
    /// Create a new `PercentilePruner`.
    ///
    /// # Arguments
    /// * `percentile` - Percentile threshold (0.0–100.0). E.g., 25.0 keeps top 25%.
    /// * `n_startup_trials` - Disable pruning until this many trials complete.
    /// * `n_warmup_steps` - Disable pruning until step >= this.
    /// * `interval_steps` - Only check every N steps (>= 1).
    /// * `n_min_trials` - Need at least this many values at a step before pruning.
    /// * `direction` - Study direction (Minimize or Maximize).
    pub fn new(
        percentile: f64,
        n_startup_trials: usize,
        n_warmup_steps: i64,
        interval_steps: i64,
        n_min_trials: usize,
        direction: StudyDirection,
    ) -> Self {
        assert!(
            (0.0..=100.0).contains(&percentile),
            "percentile must be in [0, 100]"
        );
        assert!(interval_steps >= 1, "interval_steps must be >= 1");
        assert!(n_min_trials >= 1, "n_min_trials must be >= 1");
        Self {
            percentile,
            n_startup_trials,
            n_warmup_steps,
            interval_steps,
            n_min_trials,
            direction,
        }
    }
}

impl Pruner for PercentilePruner {
    fn prune(&self, study_trials: &[FrozenTrial], trial: &FrozenTrial, _storage: Option<&dyn crate::storage::Storage>) -> Result<bool> {
        let completed: Vec<&FrozenTrial> = study_trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();
        let n_trials = completed.len();

        if n_trials == 0 {
            return Ok(false);
        }
        if n_trials < self.n_startup_trials {
            return Ok(false);
        }

        let step = match trial.last_step() {
            Some(s) => s,
            None => return Ok(false),
        };

        if step < self.n_warmup_steps {
            return Ok(false);
        }

        if !super::is_first_in_interval_step(
            step,
            &trial.intermediate_values,
            self.n_warmup_steps,
            self.interval_steps,
        ) {
            return Ok(false);
        }

        let best = get_best_intermediate_result_over_steps(trial, self.direction);
        if best.is_nan() {
            return Ok(true); // all NaN → prune
        }

        let p = get_percentile_intermediate_result_over_trials(
            &completed,
            self.direction,
            step,
            self.percentile,
            self.n_min_trials,
        );
        if p.is_nan() {
            return Ok(false); // not enough data at step → keep
        }

        Ok(match self.direction {
            StudyDirection::Maximize => best < p,
            _ => best > p,
        })
    }
}

/// Get the best (min or max) intermediate value so far in a trial.
fn get_best_intermediate_result_over_steps(trial: &FrozenTrial, direction: StudyDirection) -> f64 {
    let values: Vec<f64> = trial
        .intermediate_values
        .values()
        .copied()
        .filter(|v| !v.is_nan())
        .collect();

    if values.is_empty() {
        return f64::NAN;
    }

    match direction {
        StudyDirection::Maximize => values.iter().copied().fold(f64::NEG_INFINITY, f64::max),
        _ => values.iter().copied().fold(f64::INFINITY, f64::min),
    }
}

/// Get the percentile of intermediate values at a given step across completed trials.
///
/// 对齐 Python: n_min_trials 计数包括 NaN 值，但 percentile 计算忽略 NaN。
fn get_percentile_intermediate_result_over_trials(
    completed_trials: &[&FrozenTrial],
    direction: StudyDirection,
    step: i64,
    percentile: f64,
    n_min_trials: usize,
) -> f64 {
    // 收集所有试验在该 step 的中间值（包括 NaN）
    let all_values: Vec<f64> = completed_trials
        .iter()
        .filter_map(|t| t.intermediate_values.get(&step))
        .copied()
        .collect();

    // 对齐 Python: n_min_trials 检查在 NaN 过滤之前
    if all_values.len() < n_min_trials {
        return f64::NAN;
    }

    // 过滤 NaN 后排序（对应 np.nanpercentile）
    let mut values: Vec<f64> = all_values.into_iter().filter(|v| !v.is_nan()).collect();
    if values.is_empty() {
        return f64::NAN;
    }

    values.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let effective_percentile = match direction {
        StudyDirection::Maximize => 100.0 - percentile,
        _ => percentile,
    };

    nan_percentile(&values, effective_percentile)
}

/// Compute the p-th percentile of a sorted slice (linear interpolation).
fn nan_percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    if sorted.len() == 1 {
        return sorted[0];
    }

    let n = sorted.len() as f64;
    let idx = p / 100.0 * (n - 1.0);
    let lo = idx.floor() as usize;
    let hi = idx.ceil() as usize;
    let frac = idx - lo as f64;

    if hi >= sorted.len() {
        sorted[sorted.len() - 1]
    } else {
        sorted[lo] * (1.0 - frac) + sorted[hi] * frac
    }
}

// is_first_in_interval_step 已移至 pruners/mod.rs 作为模块共享函数

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_complete_trial(
        number: i64,
        intermediate_values: Vec<(i64, f64)>,
    ) -> FrozenTrial {
        let now = chrono::Utc::now();
        FrozenTrial {
            number,
            state: TrialState::Complete,
            values: Some(vec![0.0]),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: intermediate_values.into_iter().collect(),
            trial_id: number,
        }
    }

    fn make_running_trial(
        number: i64,
        intermediate_values: Vec<(i64, f64)>,
    ) -> FrozenTrial {
        let now = chrono::Utc::now();
        FrozenTrial {
            number,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(now),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: intermediate_values.into_iter().collect(),
            trial_id: number,
        }
    }

    #[test]
    fn test_no_pruning_before_startup() {
        let pruner = PercentilePruner::new(50.0, 5, 0, 1, 1, StudyDirection::Minimize);
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, 2.0)]),
        ];
        let trial = make_running_trial(2, vec![(0, 100.0)]);
        assert!(!pruner.prune(&completed, &trial, None).unwrap());
    }

    #[test]
    fn test_no_pruning_before_warmup() {
        let pruner = PercentilePruner::new(50.0, 0, 5, 1, 1, StudyDirection::Minimize);
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, 2.0)]),
        ];
        let trial = make_running_trial(2, vec![(2, 100.0)]);
        assert!(!pruner.prune(&completed, &trial, None).unwrap());
    }

    #[test]
    fn test_prune_minimize_bad_trial() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, 2.0)]),
            make_complete_trial(2, vec![(0, 3.0)]),
        ];
        // Trial with very high value should be pruned (minimize)
        let trial = make_running_trial(3, vec![(0, 100.0)]);
        assert!(pruner.prune(&completed, &trial, None).unwrap());
    }

    #[test]
    fn test_keep_minimize_good_trial() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, 2.0)]),
            make_complete_trial(2, vec![(0, 3.0)]),
        ];
        // Trial with low value should be kept (minimize)
        let trial = make_running_trial(3, vec![(0, 0.5)]);
        assert!(!pruner.prune(&completed, &trial, None).unwrap());
    }

    #[test]
    fn test_prune_maximize_bad_trial() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Maximize);
        let completed = vec![
            make_complete_trial(0, vec![(0, 10.0)]),
            make_complete_trial(1, vec![(0, 20.0)]),
            make_complete_trial(2, vec![(0, 30.0)]),
        ];
        // Trial with low value should be pruned (maximize)
        let trial = make_running_trial(3, vec![(0, 1.0)]);
        assert!(pruner.prune(&completed, &trial, None).unwrap());
    }

    #[test]
    fn test_no_intermediate_values() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);
        let completed = vec![make_complete_trial(0, vec![(0, 1.0)])];
        let trial = make_running_trial(1, vec![]);
        assert!(!pruner.prune(&completed, &trial, None).unwrap());
    }

    #[test]
    fn test_n_min_trials() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 5, StudyDirection::Minimize);
        // Only 2 completed trials at step 0, but n_min_trials=5
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, 2.0)]),
        ];
        let trial = make_running_trial(2, vec![(0, 100.0)]);
        assert!(!pruner.prune(&completed, &trial, None).unwrap());
    }

    #[test]
    fn test_interval_steps() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 3, 1, StudyDirection::Minimize);
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0), (1, 1.0), (2, 1.0), (3, 1.0)]),
        ];
        // Step 1 is not at an interval boundary (interval=3, warmup=0)
        // The first pruning steps are at 0, 3, 6, ...
        let trial_at_step_1 = make_running_trial(1, vec![(0, 100.0), (1, 100.0)]);
        assert!(!pruner.prune(&completed, &trial_at_step_1, None).unwrap());

        // Step 3 IS at an interval boundary
        let trial_at_step_3 =
            make_running_trial(1, vec![(0, 100.0), (1, 100.0), (2, 100.0), (3, 100.0)]);
        assert!(pruner.prune(&completed, &trial_at_step_3, None).unwrap());
    }

    #[test]
    fn test_nan_percentile_fn() {
        assert_eq!(nan_percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 50.0), 3.0);
        assert_eq!(nan_percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 0.0), 1.0);
        assert_eq!(nan_percentile(&[1.0, 2.0, 3.0, 4.0, 5.0], 100.0), 5.0);
        assert_eq!(nan_percentile(&[10.0], 50.0), 10.0);
        assert!(nan_percentile(&[], 50.0).is_nan());
    }

    // ========================================================================
    // Python 交叉验证测试: numpy.percentile 精确值
    // ========================================================================

    /// Python: np.percentile([1..10], 25) = 3.25
    /// Python: np.percentile([1..10], 50) = 5.5
    /// Python: np.percentile([1..10], 75) = 7.75
    #[test]
    fn test_python_cross_percentile_values() {
        let vals: Vec<f64> = (1..=10).map(|x| x as f64).collect();
        let p25 = nan_percentile(&vals, 25.0);
        let p50 = nan_percentile(&vals, 50.0);
        let p75 = nan_percentile(&vals, 75.0);
        assert!((p25 - 3.25).abs() < 1e-10, "Python: percentile_25=3.25, got {p25}");
        assert!((p50 - 5.5).abs()  < 1e-10, "Python: percentile_50=5.5, got {p50}");
        assert!((p75 - 7.75).abs() < 1e-10, "Python: percentile_75=7.75, got {p75}");
    }

    /// Python 交叉验证: 边界情况
    /// Python: np.percentile([5.0], 50) = 5.0
    /// Python: np.percentile([1.0, 3.0], 50) = 2.0
    /// Python: np.percentile([1.0, 3.0], 25) = 1.5
    #[test]
    fn test_python_cross_percentile_edge() {
        assert!((nan_percentile(&[5.0], 50.0) - 5.0).abs() < 1e-12);
        assert!((nan_percentile(&[5.0], 0.0) - 5.0).abs() < 1e-12);
        assert!((nan_percentile(&[5.0], 100.0) - 5.0).abs() < 1e-12);
        assert!((nan_percentile(&[1.0, 3.0], 50.0) - 2.0).abs() < 1e-12);
        assert!((nan_percentile(&[1.0, 3.0], 25.0) - 1.5).abs() < 1e-12);
        let p50 = nan_percentile(&[1.0, 10.0, 100.0], 50.0);
        assert!((p50 - 10.0).abs() < 1e-12, "Python: p50=10.0, got {p50}");
    }

    /// 对齐 Python: 全 NaN 中间值 → 剪枝
    #[test]
    fn test_all_nan_intermediate_prunes() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
        ];
        let trial = make_running_trial(1, vec![(0, f64::NAN)]);
        // 全 NaN → best = NaN → prune = true
        assert!(pruner.prune(&completed, &trial, None).unwrap());
    }

    /// 对齐 Python: completed trial 在某步有 NaN 时的 percentile 忽略 NaN
    #[test]
    fn test_nan_in_completed_trials() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, f64::NAN)]),
            make_complete_trial(2, vec![(0, 3.0)]),
        ];
        // NaN 被过滤后，有效值 = [1.0, 3.0]，p50 = 2.0
        let good = make_running_trial(3, vec![(0, 1.5)]);
        assert!(!pruner.prune(&completed, &good, None).unwrap());
        let bad = make_running_trial(3, vec![(0, 5.0)]);
        assert!(pruner.prune(&completed, &bad, None).unwrap());
    }

    /// 对齐 Python: 不同 percentile 值（25, 75）
    #[test]
    fn test_different_percentiles() {
        let completed = vec![
            make_complete_trial(0, vec![(0, 1.0)]),
            make_complete_trial(1, vec![(0, 2.0)]),
            make_complete_trial(2, vec![(0, 3.0)]),
            make_complete_trial(3, vec![(0, 4.0)]),
        ];
        // p25 = 1.75, 值 2.5 > 1.75 → 剪枝（低容忍）
        let p25 = PercentilePruner::new(25.0, 0, 0, 1, 1, StudyDirection::Minimize);
        let trial = make_running_trial(4, vec![(0, 2.5)]);
        assert!(p25.prune(&completed, &trial, None).unwrap());
        // p75 = 3.25, 值 2.5 < 3.25 → 不剪枝（高容忍）
        let p75 = PercentilePruner::new(75.0, 0, 0, 1, 1, StudyDirection::Minimize);
        assert!(!p75.prune(&completed, &trial, None).unwrap());
    }

    /// 对齐 Python: 多步场景的 best intermediate
    #[test]
    fn test_multi_step_best_intermediate() {
        let pruner = PercentilePruner::new(50.0, 0, 0, 1, 1, StudyDirection::Minimize);
        let completed = vec![
            make_complete_trial(0, vec![(0, 10.0), (1, 5.0), (2, 8.0)]),
            make_complete_trial(1, vec![(0, 10.0), (1, 6.0), (2, 7.0)]),
        ];
        // Running trial: best over steps = min(100, 4, 50) = 4.0
        // At step 2, p50 of [8.0, 7.0] = 7.5
        // best(4.0) < p50(7.5) → 不剪枝
        let trial = make_running_trial(2, vec![(0, 100.0), (1, 4.0), (2, 50.0)]);
        assert!(!pruner.prune(&completed, &trial, None).unwrap());
    }
}
