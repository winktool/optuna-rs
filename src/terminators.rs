//! 早停终止器与评估器框架。
//!
//! 对应 Python `optuna.terminator` 模块。
//!
//! 提供两套互补的终止机制：
//! 1. **基础终止器** (`Terminator` trait)：直接判断是否停止。
//! 2. **评估器框架** (`ErrorEvaluator` / `ImprovementEvaluator`)：
//!    Python 风格的误差-改善比较终止策略。
//!
//! ## 架构对齐
//!
//! Python 的终止器框架使用 `Terminator(improvement_evaluator, error_evaluator, min_n_trials)`:
//! - `improvement_evaluator.evaluate(trials, direction) -> f64`
//! - `error_evaluator.evaluate(trials, direction) -> f64`
//! - 当 `improvement < error` 时终止
//!
//! Rust 版本完整对齐此架构。

use crate::study::Study;
use crate::study::StudyDirection;
use crate::trial::{FrozenTrial, TrialState};
use crate::distributions::Distribution;
use crate::samplers::gp::{
    GPRegressor, fit_kernel_params, normalize_param,
    normal_cdf, normal_pdf, DEFAULT_MINIMUM_NOISE_VAR,
};
use crate::search_space::IntersectionSearchSpace;

use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

// ============================================================================
// 核心 Trait 定义
// ============================================================================

/// 终止器 trait：决定优化是否应提前停止。
///
/// 对应 Python `optuna.terminator.BaseTerminator`。
pub trait Terminator: Send + Sync {
    /// 返回 `true` 表示优化应停止。
    fn should_terminate(&self, study: &Study) -> bool;
}

/// 误差评估器 trait。
///
/// 对应 Python `optuna.terminator.BaseErrorEvaluator`。
/// 评估目标函数的统计误差（如交叉验证方差）。
pub trait ErrorEvaluator: Send + Sync {
    /// 评估当前试验集的误差。
    ///
    /// # 参数
    /// * `trials` - 所有试验列表
    /// * `study_direction` - 优化方向
    fn evaluate(&self, trials: &[FrozenTrial], study_direction: StudyDirection) -> f64;
}

/// 改善评估器 trait。
///
/// 对应 Python `optuna.terminator.BaseImprovementEvaluator`。
/// 评估优化的剩余改善空间。
pub trait ImprovementEvaluator: Send + Sync {
    /// 评估剩余改善空间。
    ///
    /// 返回值越大表示还有更多优化空间；返回值 <= 0 表示应终止。
    fn evaluate(&self, trials: &[FrozenTrial], study_direction: StudyDirection) -> f64;
}

// ============================================================================
// Python 风格组合终止器
// ============================================================================

/// 默认最小试验数（对应 Python `DEFAULT_MIN_N_TRIALS = 20`）。
pub const DEFAULT_MIN_N_TRIALS: usize = 20;

/// 组合终止器：比较改善空间与统计误差。
///
/// 对应 Python `optuna.terminator.Terminator`。
///
/// 当 `improvement_evaluator.evaluate() < error_evaluator.evaluate()` 时终止优化。
/// 这意味着剩余改善空间已小于统计噪声，继续优化不太可能有意义。
pub struct EvaluatorTerminator {
    /// 改善评估器
    improvement_evaluator: Box<dyn ImprovementEvaluator>,
    /// 误差评估器
    error_evaluator: Box<dyn ErrorEvaluator>,
    /// 最小试验数，至少需要这么多完成的试验才开始评估
    min_n_trials: usize,
}

impl EvaluatorTerminator {
    /// 创建新的组合终止器。
    ///
    /// # 参数
    /// * `improvement_evaluator` - 改善评估器
    /// * `error_evaluator` - 误差评估器
    /// * `min_n_trials` - 最小试验数（默认 20）
    ///
    /// # 错误
    /// 如果 `min_n_trials` <= 0 则 panic。
    pub fn new(
        improvement_evaluator: Box<dyn ImprovementEvaluator>,
        error_evaluator: Box<dyn ErrorEvaluator>,
        min_n_trials: usize,
    ) -> Self {
        assert!(min_n_trials > 0, "`min_n_trials` 必须为正整数");
        Self {
            improvement_evaluator,
            error_evaluator,
            min_n_trials,
        }
    }

    /// 使用 BestValueStagnationEvaluator + StaticErrorEvaluator(0) 的便捷构造。
    ///
    /// 这是 Python 中 `Terminator(BestValueStagnationEvaluator(...))` 的等效写法。
    pub fn with_stagnation(max_stagnation_trials: usize) -> Self {
        Self {
            improvement_evaluator: Box::new(BestValueStagnationEvaluator::new(
                max_stagnation_trials,
            )),
            error_evaluator: Box::new(StaticErrorEvaluator::new(0.0)),
            min_n_trials: 1,
        }
    }
}

impl Terminator for EvaluatorTerminator {
    fn should_terminate(&self, study: &Study) -> bool {
        // 获取所有试验
        let Ok(trials) = study.trials() else {
            return false;
        };
        // 过滤出完成的试验进行计数
        let complete_count = trials.iter().filter(|t| t.state == TrialState::Complete).count();
        if complete_count < self.min_n_trials {
            return false;
        }
        // 获取优化方向
        let Ok(direction) = study.direction() else {
            return false;
        };
        // 计算改善量和误差
        let improvement = self.improvement_evaluator.evaluate(&trials, direction);
        let error = self.error_evaluator.evaluate(&trials, direction);
        // 当改善量 < 误差时终止
        improvement < error
    }
}

// ============================================================================
// 误差评估器实现
// ============================================================================

/// 静态误差评估器：始终返回固定常量。
///
/// 对应 Python `optuna.terminator.StaticErrorEvaluator`。
///
/// 当与 `BestValueStagnationEvaluator` 配合使用时，`constant=0.0` 意味着
/// 只要改善评估器返回 <= 0 即终止。
pub struct StaticErrorEvaluator {
    /// 固定返回的常量值
    constant: f64,
}

impl StaticErrorEvaluator {
    /// 创建静态误差评估器。
    ///
    /// # 参数
    /// * `constant` - 始终返回的误差值
    pub fn new(constant: f64) -> Self {
        Self { constant }
    }
}

impl ErrorEvaluator for StaticErrorEvaluator {
    fn evaluate(&self, _trials: &[FrozenTrial], _study_direction: StudyDirection) -> f64 {
        self.constant
    }
}

/// 交叉验证误差评估器。
///
/// 对应 Python `optuna.terminator.CrossValidationErrorEvaluator`。
///
/// 使用最佳试验的交叉验证分数的缩放方差作为统计误差估计。
/// 需要通过 `report_cross_validation_scores()` 报告 CV 分数。
pub struct CrossValidationErrorEvaluator;

/// 交叉验证分数的系统属性键名。
const CROSS_VALIDATION_SCORES_KEY: &str = "terminator:cv_scores";

impl CrossValidationErrorEvaluator {
    pub fn new() -> Self {
        Self
    }
}

impl Default for CrossValidationErrorEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

impl ErrorEvaluator for CrossValidationErrorEvaluator {
    fn evaluate(&self, trials: &[FrozenTrial], study_direction: StudyDirection) -> f64 {
        // 过滤出完成的试验
        let completed: Vec<&FrozenTrial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();
        if completed.is_empty() {
            return f64::MAX; // 无完成试验时返回最大值（不触发终止）
        }
        // 找到最佳试验
        let best_trial = if study_direction == StudyDirection::Maximize {
            completed.iter().max_by(|a, b| {
                let va = a.values.as_ref().and_then(|v| v.first()).copied().unwrap_or(f64::NEG_INFINITY);
                let vb = b.values.as_ref().and_then(|v| v.first()).copied().unwrap_or(f64::NEG_INFINITY);
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            })
        } else {
            completed.iter().min_by(|a, b| {
                let va = a.values.as_ref().and_then(|v| v.first()).copied().unwrap_or(f64::INFINITY);
                let vb = b.values.as_ref().and_then(|v| v.first()).copied().unwrap_or(f64::INFINITY);
                va.partial_cmp(&vb).unwrap_or(std::cmp::Ordering::Equal)
            })
        };
        let Some(best) = best_trial else {
            return f64::MAX;
        };
        // 从系统属性中读取 CV 分数
        let cv_scores = match best.system_attrs.get(CROSS_VALIDATION_SCORES_KEY) {
            Some(serde_json::Value::Array(arr)) => {
                // 将 JSON 数组解析为 f64 向量
                arr.iter()
                    .filter_map(|v| v.as_f64())
                    .collect::<Vec<f64>>()
            }
            _ => return f64::MAX, // 无 CV 分数时返回最大值
        };
        let k = cv_scores.len();
        if k <= 1 {
            return f64::MAX; // 需要至少 2 个 CV 分数
        }
        // 计算缩放方差: scale = 1/k + 1/(k-1)
        let scale = 1.0 / k as f64 + 1.0 / (k - 1) as f64;
        // 计算方差
        let mean = cv_scores.iter().sum::<f64>() / k as f64;
        let var = cv_scores.iter().map(|&s| (s - mean) * (s - mean)).sum::<f64>() / k as f64;
        let std = (scale * var).sqrt();
        std
    }
}

/// 向试验报告交叉验证分数。
///
/// 对应 Python `optuna.terminator.report_cross_validation_scores()`。
///
/// 在目标函数中调用此函数以报告 CV 分数，用于 `CrossValidationErrorEvaluator`。
///
/// # 参数
/// * `storage` - 存储后端
/// * `trial_id` - 试验 ID
/// * `scores` - 交叉验证分数列表（长度必须 > 1）
pub fn report_cross_validation_scores(
    storage: &dyn crate::storage::Storage,
    trial_id: i64,
    scores: &[f64],
) -> crate::error::Result<()> {
    if scores.len() <= 1 {
        return Err(crate::error::OptunaError::ValueError(
            "`scores` 的长度必须大于 1".into(),
        ));
    }
    let json_scores = serde_json::Value::Array(
        scores.iter().map(|&s| serde_json::json!(s)).collect(),
    );
    storage.set_trial_system_attr(trial_id, CROSS_VALIDATION_SCORES_KEY, json_scores)
}

/// 中位数误差评估器。
///
/// 对应 Python `optuna.terminator.MedianErrorEvaluator`。
///
/// 使用初始阶段的改善中位数作为停止阈值的启发式方法。
/// 适合与 `EMMREvaluator` 类的改善评估器配合使用。
pub struct MedianErrorEvaluator {
    /// 配对的改善评估器（用于计算初始改善水平）
    paired_improvement_evaluator: Box<dyn ImprovementEvaluator>,
    /// 热身试验数（跳过前 N 次试验，通常是随机采样阶段）
    warm_up_trials: usize,
    /// 用于计算中位数的初始试验数
    n_initial_trials: usize,
    /// 阈值与初始中位数的比例
    threshold_ratio: f64,
    /// 缓存的阈值（只计算一次）
    cached_threshold: std::sync::Mutex<Option<f64>>,
}

impl MedianErrorEvaluator {
    /// 创建中位数误差评估器。
    ///
    /// # 参数
    /// * `paired_improvement_evaluator` - 配对改善评估器
    /// * `warm_up_trials` - 热身试验数（默认 10）
    /// * `n_initial_trials` - 初始试验数（默认 20）
    /// * `threshold_ratio` - 阈值比例（默认 0.01）
    pub fn new(
        paired_improvement_evaluator: Box<dyn ImprovementEvaluator>,
        warm_up_trials: usize,
        n_initial_trials: usize,
        threshold_ratio: f64,
    ) -> Self {
        assert!(n_initial_trials > 0, "`n_initial_trials` 必须为正整数");
        assert!(threshold_ratio > 0.0 && threshold_ratio.is_finite(), "`threshold_ratio` 必须为正有限数");
        Self {
            paired_improvement_evaluator,
            warm_up_trials,
            n_initial_trials,
            threshold_ratio,
            cached_threshold: std::sync::Mutex::new(None),
        }
    }
}

impl ErrorEvaluator for MedianErrorEvaluator {
    fn evaluate(&self, trials: &[FrozenTrial], study_direction: StudyDirection) -> f64 {
        // 如果已缓存阈值，直接返回
        let cached = self.cached_threshold.lock().unwrap();
        if let Some(threshold) = *cached {
            return threshold;
        }
        drop(cached);

        // 过滤完成的试验并按编号排序
        let mut completed: Vec<&FrozenTrial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();
        completed.sort_by_key(|t| t.number);

        // 需要足够多的试验
        if completed.len() < self.warm_up_trials + self.n_initial_trials {
            return -f64::MIN_POSITIVE; // 不终止（假设改善量始终非负）
        }

        // 计算初始阶段各步的改善量
        let mut criteria = Vec::with_capacity(self.n_initial_trials);
        for i in 1..=self.n_initial_trials {
            // 取热身后的前 i 个试验
            let subset: Vec<FrozenTrial> = completed[self.warm_up_trials..self.warm_up_trials + i]
                .iter()
                .map(|t| (*t).clone())
                .collect();
            let improvement = self.paired_improvement_evaluator.evaluate(&subset, study_direction);
            criteria.push(improvement);
        }

        // 排序后取中位数
        criteria.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = criteria[criteria.len() / 2];

        // 阈值 = 中位数 * 比例
        let threshold = (median * self.threshold_ratio).min(f64::MAX);

        // 缓存结果
        let mut cached = self.cached_threshold.lock().unwrap();
        *cached = Some(threshold);

        threshold
    }
}

// ============================================================================
// 改善评估器实现
// ============================================================================

/// 最佳值停滞评估器。
///
/// 对应 Python `optuna.terminator.BestValueStagnationEvaluator`。
///
/// 评估最佳值的停滞期。返回 `max_stagnation_trials - 当前停滞步数`。
/// 当返回值 <= 0 时，配合 `StaticErrorEvaluator(0)` 将触发终止。
pub struct BestValueStagnationEvaluator {
    /// 最大容忍停滞试验数
    max_stagnation_trials: usize,
}

impl BestValueStagnationEvaluator {
    /// 创建新的停滞评估器。
    ///
    /// # 参数
    /// * `max_stagnation_trials` - 最大容忍停滞试验数（默认 30）
    pub fn new(max_stagnation_trials: usize) -> Self {
        Self {
            max_stagnation_trials,
        }
    }
}

impl ImprovementEvaluator for BestValueStagnationEvaluator {
    fn evaluate(&self, trials: &[FrozenTrial], study_direction: StudyDirection) -> f64 {
        // 过滤完成的试验
        let completed: Vec<&FrozenTrial> = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .collect();
        if completed.is_empty() {
            return f64::MAX; // 无完成试验时返回最大值（不触发终止）
        }
        let is_maximize = study_direction == StudyDirection::Maximize;
        let current_step = completed.len() - 1;
        // 找到最佳值的步数
        let mut best_step = 0;
        for (i, trial) in completed.iter().enumerate() {
            let best_value = completed[best_step]
                .values.as_ref().and_then(|v| v.first()).copied();
            let current_value = trial
                .values.as_ref().and_then(|v| v.first()).copied();
            match (best_value, current_value) {
                (Some(bv), Some(cv)) => {
                    if (is_maximize && bv < cv) || (!is_maximize && bv > cv) {
                        best_step = i;
                    }
                }
                _ => {}
            }
        }
        // 返回剩余容忍步数（对应 Python: max_stagnation_trials - (current_step - best_step)）
        self.max_stagnation_trials as f64 - (current_step - best_step) as f64
    }
}

// ============================================================================
// GP-based 改善评估器实现
// ============================================================================

/// 准备 GP 训练数据：从试验中提取归一化参数和标准化目标值。
///
/// 返回 (x_train, y_values, is_categorical, param_names)
/// y_values 未标准化（raw sign-adjusted values），调用者负责标准化。
fn prepare_gp_data(
    trials: &[FrozenTrial],
    search_space: &std::collections::HashMap<String, Distribution>,
    study_direction: StudyDirection,
) -> (Vec<Vec<f64>>, Vec<f64>, Vec<bool>, Vec<String>) {
    let param_names: Vec<String> = search_space.keys().cloned().collect();
    let is_categorical: Vec<bool> = param_names.iter().map(|name| {
        matches!(search_space[name], Distribution::CategoricalDistribution(_))
    }).collect();

    // GP 模块假设最大化：最小化时取反
    let sign = if study_direction == StudyDirection::Minimize { -1.0 } else { 1.0 };

    let mut x_train = Vec::new();
    let mut y_values = Vec::new();

    for trial in trials.iter().filter(|t| t.state == TrialState::Complete) {
        if let Some(vals) = &trial.values {
            if vals.is_empty() { continue; }
            let mut row = Vec::with_capacity(param_names.len());
            let mut complete = true;
            for name in &param_names {
                if let Some(pv) = trial.params.get(name) {
                    let dist = &search_space[name];
                    if let Ok(internal) = dist.to_internal_repr(pv) {
                        row.push(normalize_param(internal, dist));
                    } else {
                        complete = false;
                        break;
                    }
                } else {
                    complete = false;
                    break;
                }
            }
            if complete {
                x_train.push(row);
                y_values.push(vals[0] * sign);
            }
        }
    }
    (x_train, y_values, is_categorical, param_names)
}

/// Regret Bound 改善评估器。
///
/// 对应 Python `optuna.terminator.RegretBoundEvaluator`。
///
/// 使用高斯过程拟合试验数据，通过 UCB / LCB 估算剩余优化空间（regret bound）。
/// 当 regret bound 足够小时，表示剩余改善空间已不显著。
pub struct RegretBoundEvaluator {
    /// Top 试验比例（用于 GP 拟合）
    top_trials_ratio: f64,
    /// 最少试验数
    min_n_trials: usize,
    /// 随机种子
    seed: Option<u64>,
}

impl RegretBoundEvaluator {
    /// 创建 RegretBound 评估器。
    ///
    /// # 参数
    /// * `top_trials_ratio` - 使用前 N% 的试验拟合 GP（默认 0.5）
    /// * `min_n_trials` - 最少使用试验数（默认 20）
    /// * `seed` - 随机种子
    pub fn new(
        top_trials_ratio: Option<f64>,
        min_n_trials: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        Self {
            top_trials_ratio: top_trials_ratio.unwrap_or(0.5),
            min_n_trials: min_n_trials.unwrap_or(20),
            seed,
        }
    }
}

impl Default for RegretBoundEvaluator {
    fn default() -> Self {
        Self::new(None, None, None)
    }
}

/// 计算标准化 regret bound: UCB_max - LCB_max。
///
/// β = 2 * ln(d * n² * π² / 6 / δ) / 5，δ = 0.1
/// UCB(x) = μ(x) + √β * σ(x)
/// LCB(x) = μ(x) - √β * σ(x)
fn compute_standardized_regret_bound(
    gpr: &GPRegressor,
    top_params: &[Vec<f64>],
    n_trials: usize,
    n_params: usize,
    n_optimize_samples: usize,
    rng: &mut ChaCha8Rng,
    is_categorical: &[bool],
    search_space: &std::collections::HashMap<String, Distribution>,
    param_names: &[String],
) -> f64 {
    let delta = 0.1_f64;
    let n = n_trials as f64;
    let d = n_params as f64;

    // β = (2/5) * ln(d * n² * π² / (6δ))
    let beta_arg = d * n * n * std::f64::consts::PI * std::f64::consts::PI / (6.0 * delta);
    let beta = if beta_arg > 0.0 { 2.0 * beta_arg.ln() / 5.0 } else { 1.0 };
    let sqrt_beta = beta.sqrt();

    // UCB 需同时搜索训练点和随机候选点
    let mut best_ucb = f64::NEG_INFINITY;
    let mut best_lcb = f64::NEG_INFINITY;

    // 在训练点上计算 UCB 和 LCB
    for params in top_params {
        let (mean, var) = gpr.posterior(params);
        let std = var.max(0.0).sqrt();
        let ucb = mean + sqrt_beta * std;
        let lcb = mean - sqrt_beta * std;
        if ucb > best_ucb { best_ucb = ucb; }
        if lcb > best_lcb { best_lcb = lcb; }
    }

    // 随机候选点搜索 UCB
    for _ in 0..n_optimize_samples {
        let candidate: Vec<f64> = (0..n_params).map(|d_idx| {
            if is_categorical[d_idx] {
                match &search_space[&param_names[d_idx]] {
                    Distribution::CategoricalDistribution(c) => {
                        (rng.gen_range(0.0_f64..1.0) * c.choices.len() as f64).floor()
                    }
                    _ => rng.gen_range(0.0_f64..1.0),
                }
            } else {
                rng.gen_range(0.0_f64..1.0)
            }
        }).collect();

        let (mean, var) = gpr.posterior(&candidate);
        let std = var.max(0.0).sqrt();
        let ucb = mean + sqrt_beta * std;
        if ucb > best_ucb { best_ucb = ucb; }
    }

    // regret bound = UCB_max - LCB_max（非负）
    (best_ucb - best_lcb).max(0.0)
}

impl ImprovementEvaluator for RegretBoundEvaluator {
    fn evaluate(&self, trials: &[FrozenTrial], study_direction: StudyDirection) -> f64 {
        // 计算搜索空间
        let completed: Vec<FrozenTrial> = trials.iter()
            .filter(|t| t.state == TrialState::Complete)
            .cloned()
            .collect();
        if completed.is_empty() { return f64::MAX; }

        let mut ss = IntersectionSearchSpace::new(false);
        let search_space = ss.calculate(&completed);
        if search_space.is_empty() { return f64::MAX; }

        // 准备 GP 训练数据
        let (x_train, y_values, is_categorical, param_names) =
            prepare_gp_data(trials, &search_space, study_direction);
        let n = x_train.len();
        if n == 0 { return f64::MAX; }

        // 选择 top-N 试验
        // 对齐 Python: int() 截断（非四舍五入），即 floor 正数
        let top_n = (n as f64 * self.top_trials_ratio) as usize;
        let top_n = top_n.max(self.min_n_trials).min(n);

        // 按值排序取 top_n（降序，最大化指标）
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| y_values[b].partial_cmp(&y_values[a]).unwrap_or(std::cmp::Ordering::Equal));
        let top_indices: Vec<usize> = indices[..top_n].to_vec();

        let top_x: Vec<Vec<f64>> = top_indices.iter().map(|&i| x_train[i].clone()).collect();
        let top_y: Vec<f64> = top_indices.iter().map(|&i| y_values[i]).collect();

        // 标准化 top-N 的目标值
        let mean_y = top_y.iter().sum::<f64>() / top_y.len() as f64;
        let std_y = {
            let var = top_y.iter().map(|v| (v - mean_y).powi(2)).sum::<f64>() / top_y.len() as f64;
            var.sqrt().max(1e-10)
        };
        let standardized_y: Vec<f64> = top_y.iter().map(|v| (v - mean_y) / std_y).collect();

        // 拟合 GP
        let seed = self.seed.unwrap_or(42);
        let gpr = fit_kernel_params(&top_x, &standardized_y, &is_categorical, seed, None);

        // 计算标准化 regret bound
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let regret = compute_standardized_regret_bound(
            &gpr,
            &top_x,
            top_n,
            param_names.len(),
            2048, // n_optimize_samples
            &mut rng,
            &is_categorical,
            &search_space,
            &param_names,
        );

        // 返回未标准化的 regret bound
        regret * std_y
    }
}

/// EMMR (Expected Minimum Model Regret) 改善评估器。
///
/// 对应 Python `optuna.terminator.EMMREvaluator`。
///
/// 使用两个 GP 模型（一个用前 t-1 个试验，一个用全部 t 个试验），
/// 通过比较后验分布的差异估算每个新试验带来的期望模型遗憾减少量。
pub struct EMMREvaluator {
    /// 目标函数是否确定性
    _deterministic_objective: bool,
    /// 置信度参数（默认 0.1）
    _delta: f64,
    /// 最少试验数（必须 > 1）
    min_n_trials: usize,
    /// 随机种子
    seed: Option<u64>,
}

impl EMMREvaluator {
    /// 创建 EMMR 评估器。
    pub fn new(
        deterministic_objective: Option<bool>,
        delta: Option<f64>,
        min_n_trials: Option<usize>,
        seed: Option<u64>,
    ) -> Self {
        let min_n = min_n_trials.unwrap_or(2);
        assert!(min_n > 1, "`min_n_trials` 必须 > 1");
        Self {
            _deterministic_objective: deterministic_objective.unwrap_or(false),
            _delta: delta.unwrap_or(0.1),
            min_n_trials: min_n,
            seed,
        }
    }
}

impl Default for EMMREvaluator {
    fn default() -> Self {
        Self::new(None, None, None, None)
    }
}

/// 数值稳定性边距
const EMMR_MARGIN: f64 = 0.1;

/// 计算两点联合后验协方差 cov_t(θ1, θ2)
fn compute_gp_posterior_cov(
    gpr: &GPRegressor,
    params1: &[f64],
    params2: &[f64],
) -> f64 {
    let chol_l = match &gpr.chol_l {
        Some(l) => l,
        None => return 0.0,
    };

    // k(θ1, θ2) = kernel_matrix([θ1], [θ2])[0][0]
    let k_12_mat = gpr.kernel_matrix(&[params1.to_vec()], &[params2.to_vec()]);
    let k_12 = k_12_mat[0][0];

    // k_star_1 = K(θ1, X_train), k_star_2 = K(θ2, X_train)
    let ks1_mat = gpr.kernel_matrix(&[params1.to_vec()], &gpr.x_train);
    let ks1 = &ks1_mat[0];
    let ks2_mat = gpr.kernel_matrix(&[params2.to_vec()], &gpr.x_train);
    let ks2 = &ks2_mat[0];

    // v1 = L^{-1} k_star_1, v2 = L^{-1} k_star_2
    use crate::samplers::gp::solve_lower;
    let v1 = solve_lower(chol_l, ks1);
    let v2 = solve_lower(chol_l, ks2);

    // cov = k_12 - v1^T v2
    let dot: f64 = v1.iter().zip(v2.iter()).map(|(a, b)| a * b).sum();
    k_12 - dot
}

impl ImprovementEvaluator for EMMREvaluator {
    fn evaluate(&self, trials: &[FrozenTrial], study_direction: StudyDirection) -> f64 {
        // 计算搜索空间
        let completed: Vec<FrozenTrial> = trials.iter()
            .filter(|t| t.state == TrialState::Complete)
            .cloned()
            .collect();
        if completed.len() < self.min_n_trials { return f64::MAX * 0.1; }

        let mut ss = IntersectionSearchSpace::new(false);
        let search_space = ss.calculate(&completed);
        if search_space.is_empty() { return f64::MAX * 0.1; }

        // 准备 GP 训练数据
        let (x_train, y_values, is_categorical, param_names) =
            prepare_gp_data(trials, &search_space, study_direction);
        let n = x_train.len();
        if n < self.min_n_trials { return f64::MAX * 0.1; }

        // 标准化所有目标值
        let y_mean = y_values.iter().sum::<f64>() / n as f64;
        let y_std = {
            let var = y_values.iter().map(|v| (v - y_mean).powi(2)).sum::<f64>() / n as f64;
            var.sqrt().max(1e-10)
        };
        let std_y: Vec<f64> = y_values.iter().map(|v| (v - y_mean) / y_std).collect();

        let seed = self.seed.unwrap_or(42);

        // 用全 t 个观测拟合 gpr_t
        let gpr_t = fit_kernel_params(&x_train, &std_y, &is_categorical, seed, None);

        // 用前 t-1 个观测拟合 gpr_{t-1}
        let x_t_minus_1: Vec<Vec<f64>> = x_train[..n - 1].to_vec();
        let y_t_minus_1: Vec<f64> = std_y[..n - 1].to_vec();
        let gpr_t1 = fit_kernel_params(&x_t_minus_1, &y_t_minus_1, &is_categorical, seed, None);

        // θ_t* = argmax(std_y 全部)，θ_{t-1}* = argmax(std_y 前 t-1)
        let idx_t_star = std_y.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap();
        let idx_t1_star = y_t_minus_1.iter().enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i).unwrap();

        let theta_t_star = &x_train[idx_t_star];
        let theta_t1_star = &x_train[idx_t1_star];

        // 最后一个观测 (x_t, y_t)
        let x_t = &x_train[n - 1];
        let y_t = std_y[n - 1];

        // 用 gpr_t 计算后验
        let (mu_t_theta_t_star, var_t_theta_t_star) = gpr_t.posterior(theta_t_star);
        let (_, var_t_theta_t1_star) = gpr_t.posterior(theta_t1_star);
        let cov_t = compute_gp_posterior_cov(&gpr_t, theta_t_star, theta_t1_star);

        // 对齐 Python: 用 gpr_t（而非 gpr_{t-1}）在第 t 个观测 (x_t) 处计算后验
        // Python 注释: "Use gpr_t instead of gpr_t1 because KL Div. requires the same prior for both posterior."
        let (mu_t1_x_t, var_t1_x_t) = gpr_t.posterior(x_t);
        let (mu_t1_theta_t1_star, _) = gpr_t1.posterior(theta_t1_star);

        // === 计算四项 ===

        // term1: Δμ = μ_{t-1}(θ_{t-1}*) - μ_t(θ_t*)
        let delta_mu = mu_t1_theta_t1_star - mu_t_theta_t_star;

        // v = sqrt(max(1e-10, var_t(θ_t*) - 2*cov_t + var_t(θ_{t-1}*)))
        let v_sq = var_t_theta_t_star - 2.0 * cov_t + var_t_theta_t1_star;
        let v = v_sq.max(1e-10).sqrt();

        // g = (μ_t(θ_t*) - μ_{t-1}(θ_{t-1}*)) / v
        let g = (mu_t_theta_t_star - mu_t1_theta_t1_star) / v;

        // term2 = v * φ(g)
        let term2 = v * normal_pdf(g);

        // term3 = v * g * Φ(g)
        let term3 = v * g * normal_cdf(g);

        // term4 = κ_{t-1} * sqrt(0.5 * KL_bound)
        // 计算 κ_{t-1} 使用前 t-1 个试验的 regret bound
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let kappa_t1 = compute_standardized_regret_bound(
            &gpr_t1,
            &x_t_minus_1,
            n - 1,
            param_names.len(),
            2048,
            &mut rng,
            &is_categorical,
            &search_space,
            &param_names,
        );

        // KL 散度项
        let lambda_inv = DEFAULT_MINIMUM_NOISE_VAR;
        let lambda = 1.0 / lambda_inv;

        let rhs1 = 0.5 * (1.0 + lambda * var_t1_x_t).ln();
        let rhs2 = -0.5 * var_t1_x_t / (var_t1_x_t + lambda_inv);
        let rhs3 = 0.5 * var_t1_x_t * (y_t - mu_t1_x_t).powi(2)
            / (var_t1_x_t + lambda_inv).powi(2);

        let kl_bound = rhs1 + rhs2 + rhs3;
        let term4 = kappa_t1 * (0.5 * kl_bound.max(0.0)).sqrt();

        // EMMR = term1 + term2 + term3 + term4（对齐 Python，不加额外 margin）
        let emmr = delta_mu + term2 + term3 + term4;

        // 乘以 y_std 转换回原始尺度，限制上界
        (emmr * y_std).min(f64::MAX * 0.5)
    }
}

// ============================================================================
// 基础终止器实现（保留原有简单终止器）
// ============================================================================

/// 最大试验数终止器：在固定数量的完成试验后停止。
///
/// 对应 Python 通过 `MaxTrialsCallback` 实现的功能。
pub struct MaxTrialsTerminator {
    /// 最大完成试验数
    max_trials: usize,
}

impl MaxTrialsTerminator {
    /// 创建终止器，在 `max_trials` 次完成试验后停止。
    pub fn new(max_trials: usize) -> Self {
        Self { max_trials }
    }
}

impl Terminator for MaxTrialsTerminator {
    fn should_terminate(&self, study: &Study) -> bool {
        // 获取完成试验数
        let n = study
            .get_trials(Some(&[TrialState::Complete]))
            .map(|t| t.len())
            .unwrap_or(0);
        // 达到上限则终止
        n >= self.max_trials
    }
}

/// 无改善终止器：连续 N 次试验无改善则停止。
///
/// Rust 特有接口（Python 通过 BestValueStagnationEvaluator + Terminator 组合实现）。
pub struct NoImprovementTerminator {
    /// 耐心值：允许的连续无改善试验数
    patience: usize,
}

impl NoImprovementTerminator {
    /// 创建终止器，`patience` 次无改善后停止。
    pub fn new(patience: usize) -> Self {
        Self { patience }
    }
}

impl Terminator for NoImprovementTerminator {
    fn should_terminate(&self, study: &Study) -> bool {
        // 获取完成的试验
        let Ok(trials) = study.get_trials(Some(&[TrialState::Complete])) else {
            return false;
        };
        // 不够进行判断则不终止
        if trials.len() < self.patience + 1 {
            return false;
        }
        // 获取优化方向
        let Ok(direction) = study.direction() else {
            return false;
        };
        // 初始化最佳值
        let mut best_value = match direction {
            StudyDirection::Minimize | StudyDirection::NotSet => f64::INFINITY,
            StudyDirection::Maximize => f64::NEG_INFINITY,
        };
        let mut best_idx = 0;
        // 遍历找到最后一次改善的位置
        for (i, trial) in trials.iter().enumerate() {
            if let Some(values) = &trial.values
                && !values.is_empty()
            {
                let v = values[0];
                let is_better = match direction {
                    StudyDirection::Minimize | StudyDirection::NotSet => v < best_value,
                    StudyDirection::Maximize => v > best_value,
                };
                if is_better {
                    best_value = v;
                    best_idx = i;
                }
            }
        }
        // 判断是否超出耐心
        trials.len() - best_idx > self.patience
    }
}

/// 目标值终止器：达到目标值时停止。
///
/// Rust 特有接口。
pub struct TargetValueTerminator {
    /// 目标值
    target: f64,
    /// 优化方向
    direction: StudyDirection,
}

impl TargetValueTerminator {
    /// 创建终止器，达到 `target` 值时停止。
    ///
    /// 最小化：best <= target 时终止。
    /// 最大化：best >= target 时终止。
    pub fn new(target: f64, direction: StudyDirection) -> Self {
        Self { target, direction }
    }
}

impl Terminator for TargetValueTerminator {
    fn should_terminate(&self, study: &Study) -> bool {
        // 获取当前最佳值
        let Ok(best) = study.best_value() else {
            return false;
        };
        // 根据方向判断是否达标
        match self.direction {
            StudyDirection::Minimize | StudyDirection::NotSet => best <= self.target,
            StudyDirection::Maximize => best >= self.target,
        }
    }
}

/// 最佳值停滞终止器（便捷包装）。
///
/// 对应 Python `Terminator(BestValueStagnationEvaluator(N), StaticErrorEvaluator(0))`。
///
/// 内部使用 `EvaluatorTerminator` + `BestValueStagnationEvaluator` + `StaticErrorEvaluator(0)`。
pub struct BestValueStagnationTerminator {
    /// 内部组合终止器
    inner: EvaluatorTerminator,
}

impl BestValueStagnationTerminator {
    /// 创建停滞终止器。
    ///
    /// # 参数
    /// * `max_stagnation_trials` - 最大容忍停滞试验数（默认 30）
    pub fn new(max_stagnation_trials: usize) -> Self {
        Self {
            inner: EvaluatorTerminator::with_stagnation(max_stagnation_trials),
        }
    }
}

impl Terminator for BestValueStagnationTerminator {
    fn should_terminate(&self, study: &Study) -> bool {
        self.inner.should_terminate(study)
    }
}

/// 改善-误差比终止器（便捷包装）。
///
/// 使用静态误差阈值和滚动最佳值窗口估算改善量。
pub struct ImprovementTerminator {
    /// 静态误差阈值
    error_threshold: f64,
    /// 最小试验数
    min_n_trials: usize,
    /// 改善窗口大小
    improvement_window: usize,
}

impl ImprovementTerminator {
    /// 创建改善终止器。
    ///
    /// # 参数
    /// * `error_threshold` - 误差阈值，改善量低于此值时终止
    /// * `min_n_trials` - 最小试验数（默认 20）
    /// * `improvement_window` - 改善窗口大小（默认 10）
    pub fn new(error_threshold: f64, min_n_trials: usize, improvement_window: usize) -> Self {
        Self {
            error_threshold,
            min_n_trials,
            improvement_window,
        }
    }
}

impl Terminator for ImprovementTerminator {
    fn should_terminate(&self, study: &Study) -> bool {
        // 获取完成的试验
        let Ok(trials) = study.get_trials(Some(&[TrialState::Complete])) else {
            return false;
        };
        // 不够开始评估则不终止
        if trials.len() < self.min_n_trials {
            return false;
        }
        // 获取优化方向
        let Ok(direction) = study.direction() else {
            return false;
        };
        // 收集所有目标值
        let values: Vec<f64> = trials
            .iter()
            .filter_map(|t| t.values.as_ref().and_then(|v| v.first().copied()))
            .collect();
        // 窗口数据不足则不终止
        if values.len() < self.improvement_window + 1 {
            return false;
        }
        // 计算滚动最佳值序列
        let mut running_best = Vec::with_capacity(values.len());
        let mut best = match direction {
            StudyDirection::Minimize | StudyDirection::NotSet => f64::INFINITY,
            StudyDirection::Maximize => f64::NEG_INFINITY,
        };
        for &v in &values {
            let is_better = match direction {
                StudyDirection::Minimize | StudyDirection::NotSet => v < best,
                StudyDirection::Maximize => v > best,
            };
            if is_better {
                best = v;
            }
            running_best.push(best);
        }
        // 计算最近窗口内的改善量
        let n = running_best.len();
        let recent_improvement =
            (running_best[n - 1] - running_best[n - self.improvement_window]).abs();
        // 改善量低于阈值时终止
        recent_improvement < self.error_threshold
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::samplers::RandomSampler;
    use crate::study::create_study;
    use std::sync::Arc;

    /// 辅助函数：创建带随机采样器的研究
    fn make_study(direction: StudyDirection) -> Study {
        let sampler: Arc<dyn crate::samplers::Sampler> =
            Arc::new(RandomSampler::new(Some(42)));
        create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(direction),
            None,
            false,
        )
        .unwrap()
    }

    // ── MaxTrialsTerminator 测试 ──

    #[test]
    fn test_max_trials_terminator() {
        let term = MaxTrialsTerminator::new(5);
        let study = make_study(StudyDirection::Minimize);
        // 无试验时不终止
        assert!(!term.should_terminate(&study));
        // 运行 5 个试验
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x)
                },
                Some(5),
                None,
                None,
            )
            .unwrap();
        // 达到 5 个试验时终止
        assert!(term.should_terminate(&study));
    }

    // ── NoImprovementTerminator 测试 ──

    #[test]
    fn test_no_improvement_terminator() {
        let term = NoImprovementTerminator::new(3);
        let study = make_study(StudyDirection::Minimize);
        // 无试验时不终止
        assert!(!term.should_terminate(&study));
        // 运行 20 个试验
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x)
                },
                Some(20),
                None,
                None,
            )
            .unwrap();
        // 至少能调用
        let _ = term.should_terminate(&study);
    }

    // ── TargetValueTerminator 测试 ──

    #[test]
    fn test_target_value_terminator_minimize() {
        let term = TargetValueTerminator::new(0.5, StudyDirection::Minimize);
        let study = make_study(StudyDirection::Minimize);
        assert!(!term.should_terminate(&study));
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x * x)
                },
                Some(50),
                None,
                None,
            )
            .unwrap();
        assert!(
            term.should_terminate(&study),
            "应找到 <= 0.5 的值"
        );
    }

    #[test]
    fn test_target_value_terminator_maximize() {
        let term = TargetValueTerminator::new(0.8, StudyDirection::Maximize);
        let study = make_study(StudyDirection::Maximize);
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x)
                },
                Some(50),
                None,
                None,
            )
            .unwrap();
        assert!(
            term.should_terminate(&study),
            "应找到 >= 0.8 的值"
        );
    }

    // ── EvaluatorTerminator + 评估器组合测试 ──

    #[test]
    fn test_static_error_evaluator() {
        let eval = StaticErrorEvaluator::new(0.5);
        let result = eval.evaluate(&[], StudyDirection::Minimize);
        assert!((result - 0.5).abs() < 1e-14);
    }

    #[test]
    fn test_best_value_stagnation_evaluator_minimize() {
        // 构造手动试验：值递减然后停滞
        let mut trials = Vec::new();
        for i in 0..10 {
            trials.push(FrozenTrial {
                number: i,
                trial_id: i,
                state: TrialState::Complete,
                values: Some(vec![10.0 - i as f64]), // 10, 9, 8, ..., 1
                datetime_start: None,
                datetime_complete: None,
                params: std::collections::HashMap::new(),
                distributions: std::collections::HashMap::new(),
                user_attrs: std::collections::HashMap::new(),
                system_attrs: std::collections::HashMap::new(),
                intermediate_values: std::collections::HashMap::new(),
            });
        }
        // 添加 5 个停滞试验（值都是 5.0，比最佳值 1.0 差）
        for i in 10..15 {
            trials.push(FrozenTrial {
                number: i,
                trial_id: i,
                state: TrialState::Complete,
                values: Some(vec![5.0]),
                datetime_start: None,
                datetime_complete: None,
                params: std::collections::HashMap::new(),
                distributions: std::collections::HashMap::new(),
                user_attrs: std::collections::HashMap::new(),
                system_attrs: std::collections::HashMap::new(),
                intermediate_values: std::collections::HashMap::new(),
            });
        }

        let eval = BestValueStagnationEvaluator::new(3);
        let result = eval.evaluate(&trials, StudyDirection::Minimize);
        // best_step = 9 (值=1.0), current_step = 14, stagnation = 5
        // room_left = 3 - 5 = -2
        assert!(result < 0.0, "停滞 5 步，容忍 3 步，应返回负值: {result}");
    }

    #[test]
    fn test_best_value_stagnation_evaluator_maximize() {
        let mut trials = Vec::new();
        for i in 0..5 {
            trials.push(FrozenTrial {
                number: i,
                trial_id: i,
                state: TrialState::Complete,
                values: Some(vec![i as f64]), // 0, 1, 2, 3, 4
                datetime_start: None,
                datetime_complete: None,
                params: std::collections::HashMap::new(),
                distributions: std::collections::HashMap::new(),
                user_attrs: std::collections::HashMap::new(),
                system_attrs: std::collections::HashMap::new(),
                intermediate_values: std::collections::HashMap::new(),
            });
        }
        // 在最大值后继续但无改善
        for i in 5..8 {
            trials.push(FrozenTrial {
                number: i,
                trial_id: i,
                state: TrialState::Complete,
                values: Some(vec![2.0]),
                datetime_start: None,
                datetime_complete: None,
                params: std::collections::HashMap::new(),
                distributions: std::collections::HashMap::new(),
                user_attrs: std::collections::HashMap::new(),
                system_attrs: std::collections::HashMap::new(),
                intermediate_values: std::collections::HashMap::new(),
            });
        }

        let eval = BestValueStagnationEvaluator::new(5);
        let result = eval.evaluate(&trials, StudyDirection::Maximize);
        // best_step=4 (值=4.0), current_step=7, stagnation=3
        // room_left = 5 - 3 = 2
        assert!(result > 0.0, "停滞 3 步，容忍 5 步，应返回正值: {result}");
    }

    #[test]
    fn test_evaluator_terminator_stagnation() {
        let term = EvaluatorTerminator::with_stagnation(3);
        let study = make_study(StudyDirection::Minimize);
        assert!(!term.should_terminate(&study));
    }

    #[test]
    fn test_cross_validation_error_evaluator() {
        let eval = CrossValidationErrorEvaluator::new();
        // 无试验时返回 MAX
        let result = eval.evaluate(&[], StudyDirection::Minimize);
        assert_eq!(result, f64::MAX);
    }

    #[test]
    fn test_cross_validation_error_with_scores() {
        let mut trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![0.9]),
            datetime_start: None,
            datetime_complete: None,
            params: std::collections::HashMap::new(),
            distributions: std::collections::HashMap::new(),
            user_attrs: std::collections::HashMap::new(),
            system_attrs: std::collections::HashMap::new(),
            intermediate_values: std::collections::HashMap::new(),
        };
        // 模拟 CV 分数: [0.8, 0.9, 1.0]
        trial.system_attrs.insert(
            CROSS_VALIDATION_SCORES_KEY.to_string(),
            serde_json::json!([0.8, 0.9, 1.0]),
        );
        let eval = CrossValidationErrorEvaluator::new();
        let result = eval.evaluate(&[trial], StudyDirection::Minimize);
        // k=3, scale = 1/3 + 1/2 = 5/6
        // var = ((0.8-0.9)^2 + (0.9-0.9)^2 + (1.0-0.9)^2) / 3 = 0.02/3
        // std = sqrt(5/6 * 0.02/3)
        assert!(result > 0.0 && result < 1.0, "CV 误差应为正有限值: {result}");
    }

    // ── BestValueStagnationTerminator 便捷包装测试 ──

    #[test]
    fn test_best_value_stagnation_terminator() {
        let term = BestValueStagnationTerminator::new(3);
        let study = make_study(StudyDirection::Minimize);
        assert!(!term.should_terminate(&study));
    }

    // ── 终止器在 optimize 中的集成测试 ──

    #[test]
    fn test_terminators_in_optimize() {
        let study = make_study(StudyDirection::Minimize);
        let terminators: Vec<Arc<dyn Terminator>> =
            vec![Arc::new(MaxTrialsTerminator::new(10))];
        study
            .optimize_with_terminators(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x * x)
                },
                Some(1000),
                None,
                None,
                Some(&terminators),
            )
            .unwrap();
        let n = study.trials().unwrap().len();
        assert!(
            n <= 11,
            "MaxTrialsTerminator(10) 约束下应有 ~10 个试验，实际 {n}"
        );
    }

    // ── RegretBoundEvaluator 测试 ──

    #[test]
    fn test_regret_bound_evaluator_creation() {
        let _eval = RegretBoundEvaluator::default();
        let _eval2 = RegretBoundEvaluator::new(Some(0.3), Some(10), Some(42));
    }

    #[test]
    fn test_regret_bound_evaluator_empty_trials() {
        let eval = RegretBoundEvaluator::default();
        let result = eval.evaluate(&[], StudyDirection::Minimize);
        assert_eq!(result, f64::MAX);
    }

    #[test]
    fn test_regret_bound_evaluator_with_trials() {
        let eval = RegretBoundEvaluator::new(Some(0.5), Some(2), Some(42));
        let study = make_study(StudyDirection::Minimize);

        // 运行 20 个试验产生足够的数据
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(x * x)
            },
            Some(20),
            None,
            None,
        ).unwrap();

        let trials = study.trials().unwrap();
        let result = eval.evaluate(&trials, StudyDirection::Minimize);
        // regret bound 应为有限正值
        assert!(result > 0.0, "regret bound 应为正: {result}");
        assert!(result.is_finite(), "regret bound 应为有限值: {result}");
    }

    // ── EMMREvaluator 测试 ──

    #[test]
    fn test_emmr_evaluator_creation() {
        let _eval = EMMREvaluator::default();
        let _eval2 = EMMREvaluator::new(Some(true), Some(0.05), Some(5), Some(42));
    }

    #[test]
    fn test_emmr_evaluator_insufficient_trials() {
        let eval = EMMREvaluator::default();
        // 空试验
        let result = eval.evaluate(&[], StudyDirection::Minimize);
        assert!(result > 0.0, "不足试验应返回大值: {result}");
    }

    #[test]
    fn test_emmr_evaluator_with_trials() {
        let eval = EMMREvaluator::new(None, None, Some(2), Some(42));
        let study = make_study(StudyDirection::Minimize);

        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(x * x)
            },
            Some(20),
            None,
            None,
        ).unwrap();

        let trials = study.trials().unwrap();
        let result = eval.evaluate(&trials, StudyDirection::Minimize);
        // EMMR 应为有限值
        assert!(result.is_finite(), "EMMR 应为有限值: {result}");
    }

    #[test]
    fn test_regret_bound_with_evaluator_terminator() {
        // 测试 RegretBound + StaticError 组合
        let term = EvaluatorTerminator::new(
            Box::new(RegretBoundEvaluator::new(Some(0.5), Some(2), Some(42))),
            Box::new(StaticErrorEvaluator::new(100.0)), // 大阈值，确保终止
            2,
        );
        let study = make_study(StudyDirection::Minimize);
        study.optimize(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                Ok(x * x)
            },
            Some(30),
            None,
            None,
        ).unwrap();
        // 应该能评估（不 panic）
        let _ = term.should_terminate(&study);
    }
}
