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
pub fn set_verbosity(level: LogLevel) {
    // 重新初始化（tracing-subscriber 的 try_init 会静默忽略重复调用）
    init(level);
}

/// 获取当前日志级别。
///
/// 对应 Python `optuna.logging.get_verbosity()`。
/// 默认返回 Info 级别（实际级别由 RUST_LOG 环境变量控制）。
pub fn get_verbosity() -> LogLevel {
    // tracing 没有直接暴露当前级别的 API，返回默认值
    LogLevel::Info
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
        assert_eq!(get_verbosity(), LogLevel::Info);
    }

    #[test]
    fn test_macros_noop() {
        optuna_debug!("test debug");
        optuna_info!("test info");
        optuna_warn!("test warn");
        optuna_error!("test error");
    }
}
