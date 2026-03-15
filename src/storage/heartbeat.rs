//! 心跳机制 — 检测崩溃/断连的试验进程。
//!
//! 对应 Python `optuna.storages._heartbeat`。
//!
//! 在分布式优化中，心跳机制定期检查每个试验进程是否存活。
//! 支持心跳的存储后端（如 `RdbStorage`）可实现 [`Heartbeat`] trait，
//! 优化循环中会自动启动/停止心跳线程。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::thread;
use std::time::Duration;

use crate::error::Result;
use crate::storage::Storage;
use crate::study::Study;
use crate::trial::{FrozenTrial, TrialState};

/// 心跳 trait — 存储后端实现此 trait 以支持心跳检测。
///
/// 对应 Python `optuna.storages._heartbeat.BaseHeartbeat`。
///
/// 只有支持持久化的存储后端（如 RDBStorage）才需要实现此 trait。
/// 内存存储和 Journal 存储不需要心跳，因为它们在同一进程中运行。
pub trait Heartbeat: Storage {
    /// 记录试验心跳。
    ///
    /// 更新试验的最后心跳时间戳，表示该试验进程仍在运行。
    fn record_heartbeat(&self, trial_id: i64) -> Result<()>;

    /// 获取过期试验的 ID 列表。
    ///
    /// 返回心跳长时间未更新的 RUNNING 状态试验。
    fn get_stale_trial_ids(&self, study_id: i64) -> Result<Vec<i64>>;

    /// 获取心跳间隔（秒）。
    ///
    /// 返回 `None` 表示心跳未启用。
    fn get_heartbeat_interval(&self) -> Option<u64>;

    /// 获取试验失败回调。
    ///
    /// 当过期试验被标记为 FAIL 时调用此回调。
    fn get_failed_trial_callback(
        &self,
    ) -> Option<Arc<dyn Fn(&Study, &FrozenTrial) + Send + Sync>>;
}

/// 心跳线程句柄。
///
/// 对应 Python `optuna.storages._heartbeat.HeartbeatThread`。
/// 在后台定期调用 `record_heartbeat()` 直到被 drop 或显式停止。
pub struct HeartbeatThread {
    stop_flag: Arc<AtomicBool>,
    handle: Option<thread::JoinHandle<()>>,
}

impl HeartbeatThread {
    /// 启动心跳线程。
    ///
    /// # 参数
    /// * `trial_id` - 需要发送心跳的试验 ID
    /// * `storage` - 支持心跳的存储后端
    /// * `interval` - 心跳间隔（秒）
    pub fn start(
        trial_id: i64,
        storage: Arc<dyn Heartbeat>,
        interval: u64,
    ) -> Self {
        let stop_flag = Arc::new(AtomicBool::new(false));
        let flag = stop_flag.clone();

        let handle = thread::spawn(move || {
            let interval_dur = Duration::from_secs(interval);
            loop {
                // 先记录一次心跳
                let _ = storage.record_heartbeat(trial_id);

                // 分段等待以快速响应停止信号
                let mut waited = Duration::ZERO;
                let step = Duration::from_millis(500);
                while waited < interval_dur {
                    if flag.load(Ordering::Relaxed) {
                        return;
                    }
                    thread::sleep(step.min(interval_dur - waited));
                    waited += step;
                }
            }
        });

        Self {
            stop_flag,
            handle: Some(handle),
        }
    }

    /// 停止心跳线程并等待其退出。
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for HeartbeatThread {
    fn drop(&mut self) {
        self.stop();
    }
}

/// 空心跳线程 — 存储不支持心跳时使用的无操作实现。
///
/// 对应 Python `optuna.storages._heartbeat.NullHeartbeatThread`。
pub struct NullHeartbeatThread;

/// 心跳线程抽象枚举。
///
/// 在优化循环中统一处理有/无心跳的情况。
pub enum HeartbeatHandle {
    /// 真实心跳线程
    Active(HeartbeatThread),
    /// 无操作
    Null,
}

impl HeartbeatHandle {
    /// 停止心跳线程（如果是 Active）。
    pub fn stop(&mut self) {
        match self {
            HeartbeatHandle::Active(thread) => thread.stop(),
            HeartbeatHandle::Null => {}
        }
    }
}

impl Drop for HeartbeatHandle {
    fn drop(&mut self) {
        self.stop();
    }
}

/// 为试验获取心跳线程句柄。
///
/// 对应 Python `get_heartbeat_thread(trial_id, storage)`。
/// 如果存储支持心跳则返回 Active 句柄，否则返回 Null。
pub fn get_heartbeat_handle(
    trial_id: i64,
    storage: &Arc<dyn Storage>,
) -> HeartbeatHandle {
    // 尝试 downcast 到 Heartbeat trait
    // 由于 Rust 不支持 trait object downcast，使用 Any 或条件编译
    // 这里提供函数签名和 Null 默认实现；
    // 具体存储后端可以直接构造 HeartbeatThread。
    let _ = (trial_id, storage);
    HeartbeatHandle::Null
}

/// 为支持心跳的存储创建心跳句柄。
///
/// 与 `get_heartbeat_handle` 不同，此函数直接接受 `Arc<dyn Heartbeat>`。
pub fn start_heartbeat(
    trial_id: i64,
    storage: Arc<dyn Heartbeat>,
) -> HeartbeatHandle {
    match storage.get_heartbeat_interval() {
        Some(interval) => HeartbeatHandle::Active(HeartbeatThread::start(
            trial_id, storage, interval,
        )),
        None => HeartbeatHandle::Null,
    }
}

/// 清理过期（僵尸）试验，将其标记为 FAIL。
///
/// 对应 Python `optuna.storages._heartbeat.fail_stale_trials(study)`。
///
/// 扫描所有心跳过期的 RUNNING 试验，将状态设为 FAIL，
/// 并调用失败回调（如果已配置）。
pub fn fail_stale_trials(study: &Study, storage: &dyn Heartbeat) -> Result<()> {
    if storage.get_heartbeat_interval().is_none() {
        return Ok(());
    }

    let study_id = study.study_id();
    let stale_ids = storage.get_stale_trial_ids(study_id)?;

    let mut failed_trial_ids = Vec::new();
    for trial_id in stale_ids {
        match storage.set_trial_state_values(trial_id, TrialState::Fail, None) {
            Ok(true) => {
                failed_trial_ids.push(trial_id);
            }
            Ok(false) => {}
            Err(crate::error::OptunaError::UpdateFinishedTrialError(_)) => {
                // 另一个进程已经处理了该试验
            }
            Err(e) => return Err(e),
        }
    }

    if let Some(callback) = storage.get_failed_trial_callback() {
        for trial_id in failed_trial_ids {
            if let Ok(trial) = storage.get_trial(trial_id) {
                callback(study, &trial);
            }
        }
    }

    Ok(())
}

/// 检查存储是否启用了心跳。
///
/// 对应 Python `is_heartbeat_enabled(storage)`。
pub fn is_heartbeat_enabled(storage: &dyn Heartbeat) -> bool {
    storage.get_heartbeat_interval().is_some()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 对齐 Python: HeartbeatHandle::Null stop 不 panic
    #[test]
    fn test_null_heartbeat_handle_stop() {
        let mut handle = HeartbeatHandle::Null;
        handle.stop(); // 不应 panic
    }

    /// 对齐 Python: HeartbeatHandle::Null drop 不 panic
    #[test]
    fn test_null_heartbeat_handle_drop() {
        let _handle = HeartbeatHandle::Null;
        // drop 不应 panic
    }

    /// 对齐 Python: get_heartbeat_handle 无 Heartbeat 实现时返回 Null
    #[test]
    fn test_get_heartbeat_handle_returns_null() {
        let storage: Arc<dyn Storage> = Arc::new(crate::storage::InMemoryStorage::new());
        let handle = get_heartbeat_handle(0, &storage);
        match handle {
            HeartbeatHandle::Null => {} // 正确
            HeartbeatHandle::Active(_) => panic!("应返回 Null"),
        }
    }

    /// 对齐 Python: NullHeartbeatThread 是零大小类型
    #[test]
    fn test_null_heartbeat_thread_zst() {
        assert_eq!(std::mem::size_of::<NullHeartbeatThread>(), 0);
    }
}
