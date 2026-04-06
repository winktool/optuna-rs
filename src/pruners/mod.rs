mod hyperband;
mod median;
mod nop;
mod patient;
mod percentile;
pub(crate) mod successive_halving;
mod threshold;
mod wilcoxon;

pub use hyperband::HyperbandPruner;
pub use median::MedianPruner;
pub use nop::NopPruner;
pub use patient::PatientPruner;
pub use percentile::PercentilePruner;
pub use successive_halving::SuccessiveHalvingPruner;
pub use threshold::ThresholdPruner;
pub use wilcoxon::WilcoxonPruner;
pub use wilcoxon::wilcoxon_signed_rank_test;

use crate::error::Result;
use crate::storage::Storage;
use crate::trial::FrozenTrial;

/// 剪枝器 trait：决定运行中的试验是否应被提前终止。
///
/// 对应 Python `optuna.pruners.BasePruner`。
/// `prune` 方法接受可选的 storage 引用，以便 SuccessiveHalvingPruner 等
/// 需要写入 system_attrs 的剪枝器使用。
pub trait Pruner: Send + Sync {
    /// 如果试验应被剪枝则返回 `true`。
    ///
    /// `study_trials` 是来自 study 的已完成/已剪枝试验。
    /// `trial` 是当前运行中的试验，包含已报告的中间值。
    /// `storage` 是可选的存储引用，用于写入 system_attrs（SuccessiveHalving 需要）。
    fn prune(
        &self,
        study_trials: &[FrozenTrial],
        trial: &FrozenTrial,
        storage: Option<&dyn Storage>,
    ) -> Result<bool>;

    /// 对齐 Python `_filter_study`: 为采样器过滤试验列表。
    ///
    /// 默认返回所有试验（无过滤）。HyperbandPruner 覆盖此方法，
    /// 仅返回与指定试验同一括号的试验，使采样器只看到同括号数据。
    fn filter_trials(&self, trials: &[FrozenTrial], trial: &FrozenTrial) -> Vec<FrozenTrial> {
        let _ = trial;
        trials.to_vec()
    }
}

/// 检查当前步骤是否是剪枝检查间隔中的第一个步骤。
/// 被 PercentilePruner 和 ThresholdPruner 共用。
///
/// 对应 Python `_is_first_in_interval_step()`。
pub(crate) fn is_first_in_interval_step(
    step: i64,
    intermediate_values: &std::collections::HashMap<i64, f64>,
    n_warmup_steps: i64,
    interval_steps: i64,
) -> bool {
    // 计算当前间隔的下界
    let nearest_lower_pruning_step =
        (step - n_warmup_steps) / interval_steps * interval_steps + n_warmup_steps;

    // 找到当前步骤之前的最大已报告步骤
    let second_last_step = intermediate_values
        .keys()
        .filter(|&&s| s < step)
        .max()
        .copied()
        .unwrap_or(-1);

    // 只有在前一个步骤在间隔边界之前时才检查
    second_last_step < nearest_lower_pruning_step
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// 对齐 Python: is_first_in_interval_step 基本场景
    #[test]
    fn test_is_first_interval_step_basic() {
        let mut iv = HashMap::new();
        iv.insert(0, 1.0);
        iv.insert(1, 1.0);
        iv.insert(2, 1.0);
        // step=2, warmup=0, interval=1 → 前一步是 1, 边界=2 → 1 < 2 → true
        assert!(is_first_in_interval_step(2, &iv, 0, 1));
    }

    /// 对齐 Python: step == warmup 时应检查
    #[test]
    fn test_is_first_step_at_warmup() {
        let mut iv = HashMap::new();
        iv.insert(5, 1.0);
        // step=5, warmup=5, interval=1 → 边界=5, 前一步=-1 → -1 < 5 → true
        assert!(is_first_in_interval_step(5, &iv, 5, 1));
    }

    /// 对齐 Python: 无前序步骤
    #[test]
    fn test_is_first_no_previous_step() {
        let mut iv = HashMap::new();
        iv.insert(0, 1.0);
        // step=0, 没有 <0 的步骤 → second_last = -1 → -1 < 0 → true
        assert!(is_first_in_interval_step(0, &iv, 0, 1));
    }

    /// 对齐 Python: interval=3 时跳过中间步骤
    #[test]
    fn test_interval_skip() {
        let mut iv = HashMap::new();
        iv.insert(0, 1.0);
        iv.insert(1, 1.0);
        iv.insert(2, 1.0);
        iv.insert(3, 1.0);
        // step=1, warmup=0, interval=3 → 边界=0, 前一步=0 → 0 < 0 → false
        assert!(!is_first_in_interval_step(1, &iv, 0, 3));
        // step=3, warmup=0, interval=3 → 边界=3, 前一步=2 → 2 < 3 → true
        assert!(is_first_in_interval_step(3, &iv, 0, 3));
    }

    /// 对齐 Python: warmup + interval 组合
    #[test]
    fn test_warmup_and_interval() {
        let mut iv = HashMap::new();
        for i in 0..10 {
            iv.insert(i, 1.0);
        }
        // warmup=3, interval=2 → 检查步骤: 3, 5, 7, 9, ...
        // step=4, 边界=3, 前一步=3 → 3 < 3 → false
        assert!(!is_first_in_interval_step(4, &iv, 3, 2));
        // step=5, 边界=5, 前一步=4 → 4 < 5 → true
        assert!(is_first_in_interval_step(5, &iv, 3, 2));
    }
}
