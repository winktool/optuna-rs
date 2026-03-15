//! 进度条模块。
//!
//! 对应 Python `optuna.progress_bar`。
//! 基于 [`indicatif`](https://docs.rs/indicatif) crate 实现优化过程的进度可视化。
//!
//! # 功能
//! - 按试验数显示进度
//! - 按超时时间显示进度（秒级精度）
//! - 动态更新最佳值
//!
//! # 使用方式
//! 需要启用 `progress` feature:
//! ```toml
//! optuna-rs = { version = "0.1", features = ["progress"] }
//! ```

#[cfg(feature = "progress")]
use indicatif::{ProgressBar, ProgressStyle};

#[cfg(feature = "progress")]
use std::time::Duration;

/// 优化进度条包装器。
///
/// 对应 Python `optuna.progress_bar._ProgressBar`。
/// 在 `study.optimize()` 执行期间显示实时进度。
#[cfg(feature = "progress")]
pub struct OptimizationProgressBar {
    /// 内部进度条实例
    bar: ProgressBar,
    /// 是否按时间模式（而非试验数模式）
    _is_timeout_mode: bool,
}

#[cfg(feature = "progress")]
impl OptimizationProgressBar {
    /// 创建按试验数计数的进度条。
    ///
    /// # 参数
    /// * `n_trials` - 总试验数
    pub fn new_with_trials(n_trials: u64) -> Self {
        let bar = ProgressBar::new(n_trials);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} trials ({eta} 剩余)")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏  "),
        );
        Self {
            bar,
            _is_timeout_mode: false,
        }
    }

    /// 创建按超时时间计数的进度条。
    ///
    /// # 参数
    /// * `timeout_secs` - 超时秒数
    pub fn new_with_timeout(timeout_secs: f64) -> Self {
        let total = timeout_secs.ceil() as u64;
        let bar = ProgressBar::new(total);
        bar.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len}s ({msg})")
                .unwrap()
                .progress_chars("█▉▊▋▌▍▎▏  "),
        );
        Self {
            bar,
            _is_timeout_mode: true,
        }
    }

    /// 更新进度（增加 1）。
    pub fn update(&self, best_value: Option<f64>) {
        self.bar.inc(1);
        if let Some(v) = best_value {
            self.bar.set_message(format!("best: {v:.6}"));
        }
    }

    /// 设置当前进度位置（用于超时模式）。
    pub fn set_position(&self, pos: u64) {
        self.bar.set_position(pos);
    }

    /// 设置进度条消息。
    pub fn set_message(&self, msg: impl Into<std::borrow::Cow<'static, str>>) {
        self.bar.set_message(msg);
    }

    /// 完成进度条。
    pub fn finish(&self) {
        self.bar.finish_with_message("优化完成 ✓");
    }

    /// 完成并清除进度条。
    pub fn finish_and_clear(&self) {
        self.bar.finish_and_clear();
    }

    /// 启用稳定输出模式（防止进度条干扰日志）。
    pub fn enable_steady_tick(&self, millis: u64) {
        self.bar.enable_steady_tick(Duration::from_millis(millis));
    }
}

#[cfg(feature = "progress")]
impl Drop for OptimizationProgressBar {
    fn drop(&mut self) {
        if !self.bar.is_finished() {
            self.bar.abandon();
        }
    }
}

// ── 无 progress feature 时的空实现 ──

/// 空操作进度条（未启用 `progress` feature 时使用）。
#[cfg(not(feature = "progress"))]
pub struct OptimizationProgressBar;

#[cfg(not(feature = "progress"))]
impl OptimizationProgressBar {
    /// 创建空进度条（无操作）。
    pub fn new_with_trials(_n_trials: u64) -> Self {
        Self
    }

    /// 创建空进度条（无操作）。
    pub fn new_with_timeout(_timeout_secs: f64) -> Self {
        Self
    }

    /// 更新（无操作）。
    pub fn update(&self, _best_value: Option<f64>) {}

    /// 设置位置（无操作）。
    pub fn set_position(&self, _pos: u64) {}

    /// 设置消息（无操作）。
    pub fn set_message(&self, _msg: impl Into<String>) {}

    /// 完成（无操作）。
    pub fn finish(&self) {}

    /// 完成并清除（无操作）。
    pub fn finish_and_clear(&self) {}

    /// 启用稳定刷新（无操作）。
    pub fn enable_steady_tick(&self, _millis: u64) {}
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_progress_bar_noop() {
        // 验证无 feature 时不会 panic
        let bar = OptimizationProgressBar::new_with_trials(100);
        bar.update(Some(0.5));
        bar.set_position(50);
        bar.finish();
    }
}
