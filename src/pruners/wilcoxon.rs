// Wilcoxon 符号秩检验剪枝器
// 对应 Python `optuna.pruners.WilcoxonPruner`
//
// 算法原理：
// - 比较当前试验和最优试验在共同步骤上的中间值
// - 使用 Wilcoxon 符号秩检验判断差异显著性
// - 如果 p 值 < 阈值且当前试验平均表现更差 → 剪枝
//
// Python 版本使用 scipy.stats.wilcoxon，Rust 需要自行实现

use crate::error::Result;
use crate::pruners::Pruner;
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};

/// Wilcoxon 符号秩检验剪枝器：基于统计检验决定是否剪枝。
///
/// 对应 Python `optuna.pruners.WilcoxonPruner`。
///
/// 适用于优化均值/中位数的场景（如 k-fold 交叉验证）。
#[derive(Debug, Clone)]
pub struct WilcoxonPruner {
    /// p 值阈值 (0.0 - 1.0)
    p_threshold: f64,
    /// 最小启动步数
    n_startup_steps: usize,
    /// 优化方向
    direction: StudyDirection,
}

impl WilcoxonPruner {
    /// 创建新的 Wilcoxon 剪枝器。
    ///
    /// # 参数
    /// * `p_threshold` - p 值阈值 (0.0 - 1.0)。低于此值认为差异显著。
    /// * `n_startup_steps` - 在进行检验前需要的最少公共步骤数。
    /// * `direction` - 优化方向。
    pub fn new(p_threshold: f64, n_startup_steps: usize, direction: StudyDirection) -> Self {
        assert!(
            (0.0..=1.0).contains(&p_threshold),
            "p_threshold 必须在 [0.0, 1.0] 范围内"
        );
        Self {
            p_threshold,
            n_startup_steps,
            direction,
        }
    }

    /// 找到当前 study 中的最优试验。
    ///
    /// 对齐 Python: 按 objective value 找最优试验（不过滤 intermediate_values）。
    fn find_best_trial<'a>(
        &self,
        study_trials: &'a [FrozenTrial],
    ) -> Option<&'a FrozenTrial> {
        study_trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .filter(|t| t.values.is_some())
            .min_by(|a, b| {
                let va = a.values.as_ref().unwrap()[0];
                let vb = b.values.as_ref().unwrap()[0];
                match self.direction {
                    StudyDirection::Maximize => {
                        vb.partial_cmp(&va).unwrap_or(std::cmp::Ordering::Equal)
                    }
                    _ => va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal),
                }
            })
    }
}

impl Pruner for WilcoxonPruner {
    fn prune(&self, study_trials: &[FrozenTrial], trial: &FrozenTrial, _storage: Option<&dyn crate::storage::Storage>) -> Result<bool> {
        // 当前试验没有中间值 → 不剪枝
        if trial.intermediate_values.is_empty() {
            return Ok(false);
        }

        // 检查当前试验的所有中间值是否有限
        let current_steps: Vec<i64> = {
            let mut steps: Vec<i64> = trial.intermediate_values.keys().copied().collect();
            steps.sort();
            steps
        };
        let current_values: Vec<f64> = current_steps
            .iter()
            .map(|s| trial.intermediate_values[s])
            .collect();

        // 检查有限性
        if current_values.iter().any(|v| !v.is_finite()) {
            return Ok(false); // 含有 inf/NaN → 不剪枝
        }

        // 找到最优试验
        let best_trial = match self.find_best_trial(study_trials) {
            Some(t) => t,
            None => return Ok(false), // 没有最优试验 → 不剪枝
        };

        // 最优试验没有中间值 → 不剪枝
        if best_trial.intermediate_values.is_empty() {
            return Ok(false);
        }

        // 获取最优试验的步骤和值
        let best_steps: Vec<i64> = {
            let mut steps: Vec<i64> = best_trial.intermediate_values.keys().copied().collect();
            steps.sort();
            steps
        };
        let best_values: Vec<f64> = best_steps
            .iter()
            .map(|s| best_trial.intermediate_values[s])
            .collect();

        // 检查有限性
        if best_values.iter().any(|v| !v.is_finite()) {
            return Ok(false);
        }

        // 找到公共步骤并计算差值
        let mut diff_values = Vec::new();
        let mut common_current = Vec::new();
        let mut common_best = Vec::new();

        for &step in &current_steps {
            if let Some(&best_val) = best_trial.intermediate_values.get(&step) {
                let curr_val = trial.intermediate_values[&step];
                diff_values.push(curr_val - best_val);
                common_current.push(curr_val);
                common_best.push(best_val);
            }
        }

        // 公共步骤不足 → 不剪枝
        let min_samples = self.n_startup_steps.max(2);
        if diff_values.len() < min_samples {
            return Ok(false);
        }

        // 计算平均值比较（安全检查）
        // 对齐 Python: 使用各试验的 ALL 中间值计算平均，不仅是公共步骤
        let avg_current: f64 =
            current_values.iter().sum::<f64>() / current_values.len() as f64;
        let avg_best: f64 = best_values.iter().sum::<f64>() / best_values.len() as f64;

        let average_is_best = match self.direction {
            StudyDirection::Maximize => avg_best <= avg_current,
            _ => avg_best >= avg_current,
        };

        // 执行 Wilcoxon 符号秩检验
        let p_value = wilcoxon_signed_rank_test(&diff_values, self.direction);

        // 决策逻辑
        if p_value < self.p_threshold && average_is_best {
            // 安全检查：即使 p 值显著，但当前平均更优 → 不剪枝
            return Ok(false);
        }

        Ok(p_value < self.p_threshold)
    }
}

/// Wilcoxon 符号秩检验实现。
///
/// 使用 zsplit 零值处理方法（与 Python scipy.stats.wilcoxon 的 zero_method="zsplit" 一致）。
///
/// # 参数
/// * `diff_values` - 差值序列 (current - best)
/// * `direction` - 优化方向：
///   - Minimize: alternative="greater" (检验 current > best)
///   - Maximize: alternative="less" (检验 current < best)
///
/// # 返回值
/// p 值。p < threshold 表示差异显著。
fn wilcoxon_signed_rank_test(diff_values: &[f64], direction: StudyDirection) -> f64 {
    let n = diff_values.len();
    if n == 0 {
        return 1.0;
    }

    // zsplit 方法：将零值一半加到正秩，一半加到负秩
    let mut abs_diffs: Vec<(f64, usize)> = diff_values
        .iter()
        .enumerate()
        .map(|(i, &d)| (d.abs(), i))
        .collect();

    // 对齐 Python scipy: 如果所有差值都是零，直接返回 1.0（NullHypothesisWarning）
    if abs_diffs.iter().all(|(a, _)| *a < 1e-15) {
        return 1.0;
    }

    // 按绝对值排序
    abs_diffs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    // 分配秩次（处理并列）
    let ranks = assign_ranks(&abs_diffs);

    // 计算正秩和与负秩和
    let mut r_plus = 0.0; // 正差值的秩和
    let mut r_minus = 0.0; // 负差值的秩和

    for (rank, &(_abs_d, orig_idx)) in ranks.iter().zip(abs_diffs.iter()) {
        let d = diff_values[orig_idx];
        if d > 0.0 {
            r_plus += rank;
        } else if d < 0.0 {
            r_minus += rank;
        } else {
            // zsplit：零值平分到两边
            r_plus += rank / 2.0;
            r_minus += rank / 2.0;
        }
    }

    // 使用正态近似（适用于 n >= 10 的情况）
    // 并列修正：减去每组并列秩次的 (t^3 - t) / 2
    let nn = n as f64;
    let mean = nn * (nn + 1.0) / 4.0;
    let tie_correction = compute_tie_correction(&abs_diffs);
    let se_raw = nn * (nn + 1.0) * (2.0 * nn + 1.0) - tie_correction;
    let var = se_raw.max(0.0) / 24.0;

    if var == 0.0 {
        return 1.0; // 所有差值相同 → 无法检验
    }

    let std_dev = var.sqrt();

    // 根据方向选择检验统计量
    let t_stat = match direction {
        StudyDirection::Maximize => {
            // alternative="less": 检验 current < best → 使用 R-
            (r_minus - mean) / std_dev
        }
        _ => {
            // alternative="greater": 检验 current > best → 使用 R+
            (r_plus - mean) / std_dev
        }
    };

    // 单尾 p 值（正态近似）
    1.0 - normal_cdf(t_stat)
}

/// 为排序后的绝对差值分配秩次，处理并列情况。
fn assign_ranks(sorted: &[(f64, usize)]) -> Vec<f64> {
    let n = sorted.len();
    let mut ranks = vec![0.0; n];
    let mut i = 0;

    while i < n {
        let mut j = i;
        // 找到并列组的结束位置
        while j < n && (sorted[j].0 - sorted[i].0).abs() < 1e-14 {
            j += 1;
        }
        // 平均秩次
        let avg_rank = (i + 1..=j).map(|r| r as f64).sum::<f64>() / (j - i) as f64;
        for k in i..j {
            ranks[k] = avg_rank;
        }
        i = j;
    }

    ranks
}

/// 计算并列修正项。
/// 对每组大小为 t 的并列秩次：修正量 += (t^3 - t) / 2
/// 与 scipy.stats.wilcoxon 的并列修正一致。
fn compute_tie_correction(sorted: &[(f64, usize)]) -> f64 {
    let mut correction = 0.0;
    let mut i = 0;
    while i < sorted.len() {
        let mut j = i;
        while j < sorted.len() && (sorted[j].0 - sorted[i].0).abs() < 1e-14 {
            j += 1;
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            correction += t * t * t - t;
        }
        i = j;
    }
    correction / 2.0
}

/// 标准正态分布的累积分布函数（高精度近似）。
/// 使用 Abramowitz and Stegun 近似公式。
fn normal_cdf(x: f64) -> f64 {
    // 使用互补误差函数的近似
    0.5 * erfc(-x / std::f64::consts::SQRT_2)
}

/// 互补误差函数的近似（Horner 形式）。
/// 基于 Abramowitz and Stegun, equation 7.1.26。
fn erfc(x: f64) -> f64 {
    if x >= 0.0 {
        erfc_inner(x)
    } else {
        2.0 - erfc_inner(-x)
    }
}

/// erfc(x) 对 x >= 0 的近似。
fn erfc_inner(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.3275911 * x);
    let poly = t
        * (0.254829592
            + t * (-0.284496736
                + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    poly * (-x * x).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_completed_trial(
        number: i64,
        value: f64,
        intermediate: Vec<(i64, f64)>,
    ) -> FrozenTrial {
        let mut iv = HashMap::new();
        for (step, val) in intermediate {
            iv.insert(step, val);
        }
        FrozenTrial {
            number,
            state: TrialState::Complete,
            values: Some(vec![value]),
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: Some(chrono::Utc::now()),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: iv,
            trial_id: number,
        }
    }

    fn make_running_trial(number: i64, intermediate: Vec<(i64, f64)>) -> FrozenTrial {
        let mut iv = HashMap::new();
        for (step, val) in intermediate {
            iv.insert(step, val);
        }
        FrozenTrial {
            number,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: iv,
            trial_id: number,
        }
    }

    #[test]
    fn test_no_intermediate_values() {
        // 没有中间值 → 不剪枝
        let pruner = WilcoxonPruner::new(0.1, 2, StudyDirection::Minimize);
        let trial = make_running_trial(0, vec![]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_no_best_trial() {
        // 没有最优试验 → 不剪枝
        let pruner = WilcoxonPruner::new(0.1, 2, StudyDirection::Minimize);
        let trial = make_running_trial(0, vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_insufficient_common_steps() {
        // 公共步骤不足 → 不剪枝
        let pruner = WilcoxonPruner::new(0.1, 5, StudyDirection::Minimize);
        let best = make_completed_trial(0, 1.0, vec![(0, 0.5), (1, 0.5)]);
        let trial = make_running_trial(1, vec![(0, 1.0), (1, 2.0)]);
        assert!(!pruner.prune(&[best], &trial, None).unwrap());
    }

    #[test]
    fn test_clearly_worse_trial_minimize() {
        // 最小化：明显更差的试验 → 应该被剪枝
        let pruner = WilcoxonPruner::new(0.1, 2, StudyDirection::Minimize);
        let best = make_completed_trial(
            0,
            0.1,
            (0..20).map(|s| (s, 0.1)).collect(),
        );
        let trial = make_running_trial(
            1,
            (0..20).map(|s| (s, 10.0)).collect(),
        );
        assert!(pruner.prune(&[best], &trial, None).unwrap());
    }

    #[test]
    fn test_similar_trial_no_prune() {
        // 相似的试验 → 不剪枝
        let pruner = WilcoxonPruner::new(0.01, 2, StudyDirection::Minimize);
        let best = make_completed_trial(
            0,
            1.0,
            (0..10).map(|s| (s, 1.0 + 0.001 * s as f64)).collect(),
        );
        let trial = make_running_trial(
            1,
            (0..10).map(|s| (s, 1.0 + 0.001 * s as f64)).collect(),
        );
        assert!(!pruner.prune(&[best], &trial, None).unwrap());
    }

    #[test]
    fn test_inf_values_no_prune() {
        // 含有 inf 值 → 不剪枝
        let pruner = WilcoxonPruner::new(0.1, 2, StudyDirection::Minimize);
        let trial =
            make_running_trial(0, vec![(0, 1.0), (1, f64::INFINITY), (2, 3.0)]);
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    #[test]
    fn test_normal_cdf() {
        // 验证正态 CDF 近似精度
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-6);
        assert!((normal_cdf(1.96) - 0.975).abs() < 1e-3);
        assert!((normal_cdf(-1.96) - 0.025).abs() < 1e-3);
    }

    #[test]
    fn test_assign_ranks() {
        // 验证秩次分配
        let data = vec![(1.0, 0), (2.0, 1), (2.0, 2), (3.0, 3)];
        let ranks = assign_ranks(&data);
        assert_eq!(ranks[0], 1.0); // 值 1.0 → 秩 1
        assert_eq!(ranks[1], 2.5); // 值 2.0 并列 → 秩 (2+3)/2 = 2.5
        assert_eq!(ranks[2], 2.5);
        assert_eq!(ranks[3], 4.0); // 值 3.0 → 秩 4
    }

    // ========================================================================
    // Python 交叉验证测试: 使用 scipy.stats.wilcoxon 精确参考值
    // ========================================================================

    /// Python: wilcoxon([10]*20, alt='greater', zero_method='zsplit').pvalue ≈ 3.87e-06
    /// 全部正差值 → p 值极小 → 方向一致
    #[test]
    fn test_python_cross_wilcoxon_clear_diff() {
        let diff: Vec<f64> = vec![10.0; 20];
        let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
        // Python 精确值: 3.872108215522035e-06
        assert!(p < 0.001, "Python p≈3.87e-06, got {p}");
    }

    /// Python: wilcoxon([0]*10, alt='greater', zero_method='zsplit').pvalue = 1.0
    /// 全部零差值 → p = 1.0 → 不能拒绝原假设
    #[test]
    fn test_python_cross_wilcoxon_all_zero() {
        let diff: Vec<f64> = vec![0.0; 10];
        let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
        // Python 精确值: 1.0
        assert!((p - 1.0).abs() < 0.05, "Python p=1.0, got {p}");
    }

    /// Python: wilcoxon(mixed, alt='greater', zero_method='zsplit').pvalue ≈ 0.0439
    /// 混合正负差值 → 中等 p 值
    #[test]
    fn test_python_cross_wilcoxon_mixed() {
        let diff = vec![1.0, -0.5, 2.0, -0.3, 1.5, 0.8, -0.1, 1.2, 0.5, -0.2];
        let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
        // Python 精确值: 0.0439453125
        // 正态近似可能有 ~0.01 偏移，允许 0.03 容差
        assert!((p - 0.044).abs() < 0.03, "Python p≈0.044, got {p}");
    }

    /// Python: wilcoxon(mixed, alt='less', zero_method='zsplit').pvalue ≈ 0.9600
    /// 同数据用 Maximize 方向 → p 值大 → 不剪枝
    #[test]
    fn test_python_cross_wilcoxon_mixed_maximize() {
        let diff = vec![1.0, -0.5, 2.0, -0.3, 1.5, 0.8, -0.1, 1.2, 0.5, -0.2];
        let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Maximize);
        // Python 精确值: 0.9599609375
        assert!((p - 0.96).abs() < 0.03, "Python p≈0.96, got {p}");
    }

    #[test]
    fn test_tie_correction() {
        // 无并列 → 修正量为 0
        let data = vec![(1.0, 0), (2.0, 1), (3.0, 2)];
        assert_eq!(compute_tie_correction(&data), 0.0);

        // 3 个并列 → (3^3 - 3) / 2 = 12
        let data = vec![(1.0, 0), (1.0, 1), (1.0, 2), (2.0, 3)];
        assert_eq!(compute_tie_correction(&data), 12.0);

        // 2 组并列 (2个 + 2个) → (8-2)/2 + (8-2)/2 = 6
        let data = vec![(1.0, 0), (1.0, 1), (2.0, 2), (2.0, 3)];
        assert_eq!(compute_tie_correction(&data), 6.0);
    }

    #[test]
    fn test_wilcoxon_with_ties() {
        // 含并列值的 Wilcoxon 检验应返回有效 p 值
        // 并列修正使方差更小 → p 值与无修正时不同
        let diffs = vec![1.0, 1.0, 1.0, 2.0, 3.0]; // 3 个并列值
        let p = wilcoxon_signed_rank_test(&diffs, StudyDirection::Minimize);
        assert!(p >= 0.0 && p <= 1.0, "p 值超出 [0,1] 范围: {}", p);
        assert!(p < 0.1, "全正差值应显著 (p < 0.1), 实际 p = {}", p);
    }

    /// Python 交叉验证: 大量并列值
    /// Python: wilcoxon([1,1,1,-1,-1,2,2,-2,3,0], alt='greater', zsplit).pvalue ≈ 0.171875
    #[test]
    fn test_python_cross_wilcoxon_tied() {
        let diff = vec![1.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0, -2.0, 3.0, 0.0];
        let p = wilcoxon_signed_rank_test(&diff, StudyDirection::Minimize);
        // Python 精确值: 0.171875; 正态近似允许 0.05 容差
        assert!(
            (p - 0.172).abs() < 0.05,
            "Python p≈0.172, got {p}"
        );
    }

    /// Python 交叉验证: normal_cdf 精度
    /// 对比 scipy.stats.norm.cdf 的精确值
    /// Abramowitz & Stegun 公式精度约 1.5e-7
    #[test]
    fn test_python_cross_normal_cdf() {
        // Python: norm.cdf(0) = 0.5
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-7, "cdf(0)={}", normal_cdf(0.0));
        // Python: norm.cdf(-3) = 0.001349898031630093
        assert!((normal_cdf(-3.0) - 0.001349898031630093).abs() < 1e-7);
        // Python: norm.cdf(3) = 0.9986501019683699
        assert!((normal_cdf(3.0) - 0.9986501019683699).abs() < 1e-7);
        // Python: norm.cdf(-1) = 0.15865525393145707
        assert!((normal_cdf(-1.0) - 0.15865525393145707).abs() < 1e-7);
        // Python: norm.cdf(1) = 0.8413447460685429
        assert!((normal_cdf(1.0) - 0.8413447460685429).abs() < 1e-7);
        // Python: norm.cdf(-2) = 0.022750131948179198
        assert!((normal_cdf(-2.0) - 0.022750131948179198).abs() < 1e-7);
        // Python: norm.cdf(2) = 0.9772498680518208
        assert!((normal_cdf(2.0) - 0.9772498680518208).abs() < 1e-7);
    }
}
