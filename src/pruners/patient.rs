// 耐心剪枝器 - 包装另一个剪枝器，在连续 patience 步没有改善时才触发剪枝
// 对应 Python `optuna.pruners.PatientPruner`
//
// 算法逻辑：
// 1. 获取试验的所有中间值，按步骤排序
// 2. 如果步骤数 <= patience + 1，不剪枝（数据不足）
// 3. 将步骤分为两部分：patience 窗口之前和之内
// 4. 比较两个窗口的最佳值：
//    - 最小化：如果 min(before) + min_delta < min(after)，说明没有改善
//    - 最大化：如果 max(before) - min_delta > max(after)，说明没有改善
// 5. 如果没有改善，委托给内部剪枝器（或直接剪枝）

use crate::error::Result;
use crate::pruners::Pruner;
use crate::study::StudyDirection;
use crate::trial::FrozenTrial;

/// 耐心剪枝器：包装另一个剪枝器，容忍一定步数的停滞。
///
/// 对应 Python `optuna.pruners.PatientPruner`。
///
/// # 参数
/// * `wrapped_pruner` - 被包装的内部剪枝器。`None` 表示直接剪枝。
/// * `patience` - 容忍的停滞步数。
/// * `min_delta` - 最小改善阈值。
/// * `direction` - 优化方向。
pub struct PatientPruner {
    /// 被包装的内部剪枝器（可选）
    wrapped_pruner: Option<Box<dyn Pruner>>,
    /// 容忍的停滞步数
    patience: usize,
    /// 最小改善阈值
    min_delta: f64,
    /// 优化方向
    direction: StudyDirection,
}

impl PatientPruner {
    /// 创建新的耐心剪枝器。
    ///
    /// # 参数
    /// * `wrapped_pruner` - 内部剪枝器。为 `None` 时，一旦检测到停滞则直接剪枝。
    /// * `patience` - 容忍步数 (>= 0)。
    /// * `min_delta` - 最小改善量 (>= 0.0)。
    /// * `direction` - 优化方向（Minimize 或 Maximize）。
    pub fn new(
        wrapped_pruner: Option<Box<dyn Pruner>>,
        patience: usize,
        min_delta: f64,
        direction: StudyDirection,
    ) -> Self {
        assert!(min_delta >= 0.0, "`min_delta` must be >= 0.0.");
        Self {
            wrapped_pruner,
            patience,
            min_delta,
            direction,
        }
    }
}

impl Pruner for PatientPruner {
    fn prune(&self, study_trials: &[FrozenTrial], trial: &FrozenTrial, storage: Option<&dyn crate::storage::Storage>) -> Result<bool> {
        // 获取最新步骤
        let _step = match trial.last_step() {
            Some(s) => s,
            None => return Ok(false), // 没有中间值 → 不剪枝
        };

        // 获取所有中间值的步骤，排序
        let mut steps: Vec<i64> = trial.intermediate_values.keys().copied().collect();
        steps.sort();

        // 步骤数不足 → 不剪枝
        // 与 Python 一致：需要 len(steps) > patience + 1
        if steps.len() <= self.patience + 1 {
            return Ok(false);
        }

        // 分割步骤为 patience 窗口之前和之内
        // steps_before = steps[: -(patience + 1)]
        // steps_after  = steps[-(patience + 1) :]
        let split_idx = steps.len() - (self.patience + 1);
        let steps_before = &steps[..split_idx];
        let steps_after = &steps[split_idx..];

        // 收集两个窗口的中间值（忽略 NaN）
        let scores_before: Vec<f64> = steps_before
            .iter()
            .filter_map(|s| trial.intermediate_values.get(s).copied())
            .filter(|v| !v.is_nan())
            .collect();

        let scores_after: Vec<f64> = steps_after
            .iter()
            .filter_map(|s| trial.intermediate_values.get(s).copied())
            .filter(|v| !v.is_nan())
            .collect();

        // 如果某个窗口全是 NaN → 不剪枝（与 Python nanmin/nanmax 行为一致）
        if scores_before.is_empty() || scores_after.is_empty() {
            return Ok(false);
        }

        // 判断是否没有改善
        let maybe_prune = match self.direction {
            StudyDirection::Minimize => {
                // 最小化：检查 min(before) + min_delta < min(after)
                let best_before = scores_before.iter().copied().fold(f64::INFINITY, f64::min);
                let best_after = scores_after.iter().copied().fold(f64::INFINITY, f64::min);
                best_before + self.min_delta < best_after
            }
            StudyDirection::Maximize => {
                // 最大化：检查 max(before) - min_delta > max(after)
                let best_before = scores_before
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                let best_after = scores_after
                    .iter()
                    .copied()
                    .fold(f64::NEG_INFINITY, f64::max);
                best_before - self.min_delta > best_after
            }
            _ => return Ok(false), // NotSet → 不剪枝
        };

        if maybe_prune {
            // 检测到停滞 → 委托给内部剪枝器或直接剪枝
            match &self.wrapped_pruner {
                Some(pruner) => pruner.prune(study_trials, trial, storage),
                None => Ok(true),
            }
        } else {
            // 仍有改善 → 不剪枝
            Ok(false)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pruners::NopPruner;
    use crate::trial::TrialState;
    use std::collections::HashMap;

    /// 辅助函数：创建带有中间值的 FrozenTrial
    fn make_trial(intermediate: Vec<(i64, f64)>) -> FrozenTrial {
        let mut iv = HashMap::new();
        for (step, val) in intermediate {
            iv.insert(step, val);
        }
        FrozenTrial {
            number: 0,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: iv,
            trial_id: 0,
        }
    }

    #[test]
    fn test_no_intermediate_values() {
        // 没有中间值 → 不剪枝
        let pruner = PatientPruner::new(None, 3, 0.0, StudyDirection::Minimize);
        let trial = make_trial(vec![]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_insufficient_steps() {
        // 步骤不足 → 不剪枝（patience=3，需要至少 5 步）
        let pruner = PatientPruner::new(None, 3, 0.0, StudyDirection::Minimize);
        let trial = make_trial(vec![(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.7)]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_minimize_no_improvement() {
        // 最小化：后期没有改善 → 剪枝
        let pruner = PatientPruner::new(None, 2, 0.0, StudyDirection::Minimize);
        // steps = [0, 1, 2, 3, 4]
        // before = [0, 1] → 值 [1.0, 0.5]，min = 0.5
        // after  = [2, 3, 4] → 值 [0.8, 0.9, 1.0]，min = 0.8
        // 0.5 + 0.0 < 0.8 → true → 剪枝
        let trial = make_trial(vec![(0, 1.0), (1, 0.5), (2, 0.8), (3, 0.9), (4, 1.0)]);
        assert!(pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_minimize_has_improvement() {
        // 最小化：后期有改善 → 不剪枝
        let pruner = PatientPruner::new(None, 2, 0.0, StudyDirection::Minimize);
        // before = [0] → 值 [1.0]，min = 1.0
        // after  = [1, 2, 3] → 值 [0.9, 0.8, 0.7]，min = 0.7
        // 1.0 + 0.0 < 0.7 → false → 不剪枝
        let trial = make_trial(vec![(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.7)]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_maximize_no_improvement() {
        // 最大化：后期没有改善 → 剪枝
        let pruner = PatientPruner::new(None, 2, 0.0, StudyDirection::Maximize);
        // before = [0, 1] → 值 [1.0, 2.0]，max = 2.0
        // after  = [2, 3, 4] → 值 [1.5, 1.0, 0.5]，max = 1.5
        // 2.0 - 0.0 > 1.5 → true → 剪枝
        let trial = make_trial(vec![(0, 1.0), (1, 2.0), (2, 1.5), (3, 1.0), (4, 0.5)]);
        assert!(pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_with_wrapped_pruner_nop() {
        // 包装 NopPruner → 即使停滞也不剪枝
        let wrapped = Box::new(NopPruner::new());
        let pruner = PatientPruner::new(Some(wrapped), 2, 0.0, StudyDirection::Minimize);
        let trial = make_trial(vec![(0, 1.0), (1, 0.5), (2, 0.8), (3, 0.9), (4, 1.0)]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_min_delta() {
        // 使用 min_delta 阈值
        let pruner = PatientPruner::new(None, 2, 0.1, StudyDirection::Minimize);
        // before = [0] → 值 [1.0]，min = 1.0
        // after  = [1, 2, 3] → 值 [0.95, 0.92, 0.91]，min = 0.91
        // 1.0 + 0.1 < 0.91 → 1.1 < 0.91 → false → 不剪枝
        let trial = make_trial(vec![(0, 1.0), (1, 0.95), (2, 0.92), (3, 0.91)]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    /// 对齐 Python: patience=0 时只要最新值不如之前就剪枝
    #[test]
    fn test_patience_zero() {
        let pruner = PatientPruner::new(None, 0, 0.0, StudyDirection::Minimize);
        // before = [0] → min = 0.5
        // after  = [1] → min = 0.8
        // 0.5 + 0.0 < 0.8 → true → 剪枝
        let trial = make_trial(vec![(0, 0.5), (1, 0.8)]);
        assert!(pruner.prune(&[], &trial, None).unwrap());
    }

    /// 对齐 Python: 全 NaN before/after 窗口 → 不剪枝
    #[test]
    fn test_nan_in_windows() {
        let pruner = PatientPruner::new(None, 1, 0.0, StudyDirection::Minimize);
        // before 窗口全 NaN
        let trial = make_trial(vec![
            (0, f64::NAN),
            (1, 1.0),
            (2, 2.0),
        ]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    /// 对齐 Python: min_delta + Maximize 方向
    #[test]
    fn test_min_delta_maximize() {
        let pruner = PatientPruner::new(None, 2, 0.5, StudyDirection::Maximize);
        // before = [0, 1] → max = 10.0
        // after  = [2, 3, 4] → max = 9.8
        // 10.0 - 0.5 > 9.8 → 9.5 > 9.8 → false → 不剪枝
        let trial = make_trial(vec![(0, 8.0), (1, 10.0), (2, 9.5), (3, 9.8), (4, 9.2)]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    /// 对齐 Python: 不连续步骤
    #[test]
    fn test_non_contiguous_steps() {
        let pruner = PatientPruner::new(None, 2, 0.0, StudyDirection::Minimize);
        // steps = [0, 5, 10, 15, 20]，patience=2
        // before = [0, 5] → min = 1.0
        // after  = [10, 15, 20] → min = 2.0
        // 1.0 < 2.0 → 剪枝
        let trial = make_trial(vec![(0, 1.0), (5, 3.0), (10, 2.0), (15, 4.0), (20, 5.0)]);
        assert!(pruner.prune(&[], &trial, None).unwrap());
    }
}
