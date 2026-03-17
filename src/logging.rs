//! 结构化日志模块。
//!
//! 对应 Python `optuna.logging`。
//! 基于 [`tracing`](https://docs.rs/tracing) crate 提供结构化日志支持。
//!
//! Python 版本使用 `colorlog` + `logging` 标准库，
//! Rust 版本使用 `tracing` + `tracing-subscriber`，提供更丰富的结构化输出。
//!
//! # 使用方式
//! 需要启用 `logging` feature:
//! ```toml
//! optuna-rs = { version = "0.1", features = ["logging"] }
//! ```
//!
//! # 日志级别
//! - `DEBUG` — 采样细节、中间值
//! - `INFO` — 试验完成、最佳值更新
//! - `WARN` — 搜索空间问题、废弃 API
//! - `ERROR` — 存储错误、致命异常

use std::sync::atomic::{AtomicU8, Ordering};

/// 全局日志级别存储。
/// 对齐 Python: 允许通过 set_verbosity/get_verbosity 动态查询当前级别。
static VERBOSITY: AtomicU8 = AtomicU8::new(20); // 默认 Info=20

/// 日志级别枚举。
///
/// 对应 Python `optuna.logging` 的日志级别常量。
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    /// 调试信息
    Debug = 10,
    /// 一般信息
    Info = 20,
    /// 警告
    Warning = 30,
    /// 错误
    Error = 40,
    /// 严重错误
    Critical = 50,
}

/// 初始化 Optuna 日志系统。
///
/// 对应 Python `optuna.logging._configure_library_root_logger()`。
/// 应在程序启动时调用一次。
///
/// # 参数
/// * `level` - 日志级别
///
/// # 示例
/// ```ignore
/// optuna_rs::logging::init(optuna_rs::logging::LogLevel::Info);
/// ```
#[cfg(feature = "logging")]
pub fn init(level: LogLevel) {
    use tracing_subscriber::EnvFilter;
    let filter = match level {
        LogLevel::Debug => "optuna_rs=debug",
        LogLevel::Info => "optuna_rs=info",
        LogLevel::Warning => "optuna_rs=warn",
        LogLevel::Error | LogLevel::Critical => "optuna_rs=error",
    };
    let _ = tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new(filter)),
        )
        .with_target(true)
        .with_thread_ids(false)
        .with_file(false)
        .with_line_number(false)
        .try_init();
}

/// 初始化日志（无 logging feature 时为空操作）。
#[cfg(not(feature = "logging"))]
pub fn init(_level: LogLevel) {}

/// 设置日志级别。
///
/// 对应 Python `optuna.logging.set_verbosity()`。
///
/// 注意：由于 `tracing-subscriber` 的全局过滤器在初始化后无法修改，
/// 建议使用 `RUST_LOG` 环境变量动态调整级别。
/// 但 get_verbosity() 将正确返回最后一次设置的级别。
pub fn set_verbosity(level: LogLevel) {
    VERBOSITY.store(level as u8, Ordering::Relaxed);
    // 重新初始化（tracing-subscriber 的 try_init 会静默忽略重复调用）
    init(level);
}

/// 获取当前日志级别。
///
/// 对应 Python `optuna.logging.get_verbosity()`。
/// 返回最后一次通过 `set_verbosity()` 设置的级别。
pub fn get_verbosity() -> LogLevel {
    match VERBOSITY.load(Ordering::Relaxed) {
        10 => LogLevel::Debug,
        30 => LogLevel::Warning,
        40 => LogLevel::Error,
        50 => LogLevel::Critical,
        _ => LogLevel::Info, // 默认 20 = Info
    }
}

/// 禁用默认日志处理器。
///
/// 对应 Python `optuna.logging.disable_default_handler()`。
pub fn disable_default_handler() {
    // tracing-subscriber 的全局 subscriber 无法在初始化后移除
    // 建议通过 RUST_LOG=off 环境变量禁用
}

/// 启用 Optuna 传播日志。
///
/// 对应 Python `optuna.logging.enable_propagation()`。
pub fn enable_propagation() {
    // tracing 天然支持 span 传播，无需额外配置
}

/// 禁用 Optuna 传播日志。
///
/// 对应 Python `optuna.logging.disable_propagation()`。
pub fn disable_propagation() {
    // 空操作 — tracing 的传播由 subscriber 配置控制
}

// ── 便捷宏 ──

/// 记录调试日志。
///
/// 在启用 `logging` feature 时使用 `tracing::debug!`，
/// 否则为空操作。
#[macro_export]
#[cfg(feature = "logging")]
macro_rules! optuna_debug {
    ($($arg:tt)*) => { tracing::debug!($($arg)*) }
}

/// 记录调试日志（无 feature 时空操作）。
#[macro_export]
#[cfg(not(feature = "logging"))]
macro_rules! optuna_debug {
    ($($arg:tt)*) => {};
}

/// 记录信息日志。
#[macro_export]
#[cfg(feature = "logging")]
macro_rules! optuna_info {
    ($($arg:tt)*) => { tracing::info!($($arg)*) }
}

/// 记录信息日志（无 feature 时空操作）。
#[macro_export]
#[cfg(not(feature = "logging"))]
macro_rules! optuna_info {
    ($($arg:tt)*) => {};
}

/// 记录警告日志。
#[macro_export]
#[cfg(feature = "logging")]
macro_rules! optuna_warn {
    ($($arg:tt)*) => { tracing::warn!($($arg)*) }
}

/// 记录警告日志（无 feature 时空操作）。
#[macro_export]
#[cfg(not(feature = "logging"))]
macro_rules! optuna_warn {
    ($($arg:tt)*) => {};
}

/// 记录错误日志。
#[macro_export]
#[cfg(feature = "logging")]
macro_rules! optuna_error {
    ($($arg:tt)*) => { tracing::error!($($arg)*) }
}

/// 记录错误日志（无 feature 时空操作）。
#[macro_export]
#[cfg(not(feature = "logging"))]
macro_rules! optuna_error {
    ($($arg:tt)*) => {};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_ordering() {
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warning);
        assert!(LogLevel::Warning < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Critical);
    }

    #[test]
    fn test_init_noop() {
        // 无 feature 时不应 panic
        init(LogLevel::Info);
        set_verbosity(LogLevel::Debug);
        assert_eq!(get_verbosity(), LogLevel::Debug);
        // 恢复默认
        set_verbosity(LogLevel::Info);
    }

    #[test]
    fn test_macros_noop() {
        optuna_debug!("test debug");
        optuna_info!("test info");
        optuna_warn!("test warn");
        optuna_error!("test error");
    }

    /// 对齐 Python: 日志级别数值应与 Python logging 标准一致
    #[test]
    fn test_log_level_values() {
        assert_eq!(LogLevel::Debug as i32, 10);
        assert_eq!(LogLevel::Info as i32, 20);
        assert_eq!(LogLevel::Warning as i32, 30);
        assert_eq!(LogLevel::Error as i32, 40);
        assert_eq!(LogLevel::Critical as i32, 50);
    }

    /// 对齐 Python: 所有日志级别的顺序关系
    #[test]
    fn test_log_level_all_relations() {
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warning);
        assert!(LogLevel::Warning < LogLevel::Error);
        assert!(LogLevel::Error < LogLevel::Critical);
        // 反向
        assert!(LogLevel::Critical > LogLevel::Debug);
        assert!(LogLevel::Warning > LogLevel::Info);
    }

    /// 对齐 Python: set_verbosity 多次调用不应 panic
    #[test]
    fn test_set_verbosity_idempotent() {
        set_verbosity(LogLevel::Debug);
        set_verbosity(LogLevel::Warning);
        set_verbosity(LogLevel::Info);
        set_verbosity(LogLevel::Critical);
        // 不应 panic，即使多次调用
    }

    /// 对齐 Python: disable/enable handler 空操作不 panic
    #[test]
    fn test_disable_enable_handlers() {
        disable_default_handler();
        enable_propagation();
        disable_propagation();
    }

    /// 对齐 Python: LogLevel 相等性
    #[test]
    fn test_log_level_equality() {
        assert_eq!(LogLevel::Debug, LogLevel::Debug);
        assert_ne!(LogLevel::Debug, LogLevel::Info);
        assert_ne!(LogLevel::Warning, LogLevel::Error);
    }

    /// 对齐 Python: get_verbosity 返回最后设置的级别
    #[test]
    fn test_get_verbosity_tracks_set() {
        set_verbosity(LogLevel::Debug);
        assert_eq!(get_verbosity(), LogLevel::Debug);
        set_verbosity(LogLevel::Warning);
        assert_eq!(get_verbosity(), LogLevel::Warning);
        set_verbosity(LogLevel::Error);
        assert_eq!(get_verbosity(), LogLevel::Error);
        set_verbosity(LogLevel::Critical);
        assert_eq!(get_verbosity(), LogLevel::Critical);
        // 恢复默认
        set_verbosity(LogLevel::Info);
        assert_eq!(get_verbosity(), LogLevel::Info);
    }
}
