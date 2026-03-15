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
