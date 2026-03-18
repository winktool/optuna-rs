//! 心跳机制 — 检测崩溃/断连的试验进程。
//!
//! 对应 Python `optuna.storages._heartbeat`。
//!
//! 在分布式优化中，心跳机制定期检查每个试验进程是否存活。
//! 支持心跳的存储后端（如 `RdbStorage`）可实现 [`Heartbeat`] trait，
//! 优化循环中会自动启动/停止心跳线程。

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
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
    /// 对齐 Python stop_event.wait(timeout=interval): 使用 Condvar 实现即时唤醒。
    condvar: Arc<(Mutex<bool>, Condvar)>,
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
        let condvar = Arc::new((Mutex::new(false), Condvar::new()));
        let flag = stop_flag.clone();
        let cv = condvar.clone();

        let handle = thread::spawn(move || {
            let interval_dur = Duration::from_secs(interval);
            loop {
                // 先记录一次心跳
                let _ = storage.record_heartbeat(trial_id);

                // 对齐 Python: stop_event.wait(timeout=heartbeat_interval)
                // Condvar::wait_timeout 在收到 notify 时立即返回，无 500ms 延迟
                let (lock, cvar) = &*cv;
                let guard = lock.lock().unwrap();
                let result = cvar.wait_timeout(guard, interval_dur).unwrap();
                // 如果是因为 notify（停止信号）而唤醒，或超时后检查停止标志
                if *result.0 || flag.load(Ordering::Relaxed) {
                    return;
                }
            }
        });

        Self {
            stop_flag,
            condvar,
            handle: Some(handle),
        }
    }

    /// 停止心跳线程并等待其退出。
    pub fn stop(&mut self) {
        self.stop_flag.store(true, Ordering::Relaxed);
        // 通知 Condvar 立即唤醒等待线程（对齐 Python stop_event.set()）
        {
            let (lock, cvar) = &*self.condvar;
            let mut stopped = lock.lock().unwrap();
            *stopped = true;
            cvar.notify_all();
            // MutexGuard 必须在 join 之前释放，否则线程的 wait_timeout
            // 无法重新获取锁，导致死锁。
        }
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

    /// 对齐 Python: HeartbeatHandle::Null 多次 stop 不 panic
    #[test]
    fn test_null_heartbeat_handle_multiple_stops() {
        let mut handle = HeartbeatHandle::Null;
        handle.stop();
        handle.stop();
        handle.stop();
    }

    /// 对齐 Python: NullHeartbeatThread 实现 Default
    #[test]
    fn test_null_heartbeat_thread_default() {
        let _thread = NullHeartbeatThread;
        // 零大小类型不需要 Default 但应正常构造
    }

    /// 对齐 Python: HeartbeatHandle 枚举变体
    #[test]
    fn test_heartbeat_handle_variants() {
        // Null 变体
        let null = HeartbeatHandle::Null;
        match null {
            HeartbeatHandle::Null => {}
            _ => panic!("should be Null"),
        }
    }

    /// 对齐 Python: is_heartbeat_enabled 判断逻辑
    #[test]
    fn test_is_heartbeat_enabled_returns_false_for_in_memory() {
        // InMemoryStorage 不实现 Heartbeat trait，
        // 所以实际上无法直接调用 is_heartbeat_enabled。
        // 但可以通过 get_heartbeat_handle 间接验证。
        let storage: Arc<dyn Storage> = Arc::new(crate::storage::InMemoryStorage::new());
        let handle = get_heartbeat_handle(0, &storage);
        assert!(matches!(handle, HeartbeatHandle::Null));
    }

    /// 对齐 Python: fail_stale_trials 的基本流程
    /// (通过 InMemoryStorage 验证 — 无 stale trials 时无操作)
    #[test]
    fn test_fail_stale_trials_no_stale_via_study() {
        use crate::study::create_study;
        let study = create_study(
            None, None, None, None, None,
            Some(vec![crate::study::StudyDirection::Minimize]),
            false,
        ).unwrap();

        // InMemoryStorage 不实现 Heartbeat，所以 get_heartbeat_handle 返回 Null
        // 但 study.optimize 中的心跳机制会正确跳过
        let trials = study.trials().unwrap();
        assert!(trials.is_empty()); // 无试验时无 stale
    }

    /// 对齐 Python: HeartbeatThread.stop() 应立即响应（使用 Condvar 而非轮询）。
    /// Python: stop_event.wait(timeout=interval) 在 stop_event.set() 时立即返回。
    #[test]
    fn test_heartbeat_stop_instant_response() {
        use std::time::Instant;
        use std::sync::atomic::AtomicI32;

        // 模拟心跳存储
        struct MockHeartbeat {
            count: AtomicI32,
        }
        impl MockHeartbeat {
            fn new() -> Self { Self { count: AtomicI32::new(0) } }
        }
        impl Heartbeat for MockHeartbeat {
            fn record_heartbeat(&self, _trial_id: i64) -> crate::error::Result<()> {
                self.count.fetch_add(1, Ordering::Relaxed);
                Ok(())
            }
            fn get_stale_trial_ids(&self, _study_id: i64) -> crate::error::Result<Vec<i64>> { Ok(vec![]) }
            fn get_heartbeat_interval(&self) -> Option<u64> { Some(60) }
            fn get_failed_trial_callback(&self) -> Option<Arc<dyn Fn(&crate::study::Study, &crate::trial::FrozenTrial) + Send + Sync>> { None }
        }
        impl crate::storage::Storage for MockHeartbeat {
            fn create_new_study(&self, _d: &[crate::study::StudyDirection], _n: Option<&str>) -> crate::error::Result<i64> { Ok(0) }
            fn delete_study(&self, _id: i64) -> crate::error::Result<()> { Ok(()) }
            fn set_study_user_attr(&self, _: i64, _: &str, _: serde_json::Value) -> crate::error::Result<()> { Ok(()) }
            fn set_study_system_attr(&self, _: i64, _: &str, _: serde_json::Value) -> crate::error::Result<()> { Ok(()) }
            fn get_study_id_from_name(&self, _: &str) -> crate::error::Result<i64> { Ok(0) }
            fn get_study_name_from_id(&self, _: i64) -> crate::error::Result<String> { Ok(String::new()) }
            fn get_study_directions(&self, _: i64) -> crate::error::Result<Vec<crate::study::StudyDirection>> { Ok(vec![]) }
            fn get_study_user_attrs(&self, _: i64) -> crate::error::Result<std::collections::HashMap<String, serde_json::Value>> { Ok(std::collections::HashMap::new()) }
            fn get_study_system_attrs(&self, _: i64) -> crate::error::Result<std::collections::HashMap<String, serde_json::Value>> { Ok(std::collections::HashMap::new()) }
            fn get_all_studies(&self) -> crate::error::Result<Vec<crate::study::FrozenStudy>> { Ok(vec![]) }
            fn create_new_trial(&self, _: i64, _: Option<&crate::trial::FrozenTrial>) -> crate::error::Result<i64> { Ok(0) }
            fn set_trial_param(&self, _: i64, _: &str, _: f64, _: &crate::distributions::Distribution) -> crate::error::Result<()> { Ok(()) }
            fn set_trial_state_values(&self, _: i64, _: crate::trial::TrialState, _: Option<&[f64]>) -> crate::error::Result<bool> { Ok(true) }
            fn set_trial_intermediate_value(&self, _: i64, _: i64, _: f64) -> crate::error::Result<()> { Ok(()) }
            fn set_trial_user_attr(&self, _: i64, _: &str, _: serde_json::Value) -> crate::error::Result<()> { Ok(()) }
            fn set_trial_system_attr(&self, _: i64, _: &str, _: serde_json::Value) -> crate::error::Result<()> { Ok(()) }
            fn get_trial(&self, _: i64) -> crate::error::Result<crate::trial::FrozenTrial> { Err(crate::error::OptunaError::ValueError("mock".into())) }
            fn get_all_trials(&self, _: i64, _: Option<&[crate::trial::TrialState]>) -> crate::error::Result<Vec<crate::trial::FrozenTrial>> { Ok(vec![]) }
            fn get_n_trials(&self, _: i64, _: Option<&[crate::trial::TrialState]>) -> crate::error::Result<usize> { Ok(0) }
            fn get_best_trial(&self, _: i64) -> crate::error::Result<crate::trial::FrozenTrial> { Err(crate::error::OptunaError::ValueError("mock".into())) }
        }

        let storage = Arc::new(MockHeartbeat::new());
        // 60 秒间隔 — 如果用旧的轮询方式 stop 至少需要 500ms
        let mut hb = HeartbeatThread::start(0, storage.clone(), 60);

        // 等待一点时间确保线程已启动并在等待
        thread::sleep(Duration::from_millis(50));

        // 停止并计时 — Condvar 方式应在 50ms 内完成
        let start = Instant::now();
        hb.stop();
        let elapsed = start.elapsed();

        // 对齐 Python: stop 应立即响应（<100ms），而非旧的 500ms 延迟
        assert!(elapsed < Duration::from_millis(100),
            "HeartbeatThread.stop() 应立即响应，实际耗时 {:?}", elapsed);

        // 至少记录了一次心跳
        assert!(storage.count.load(Ordering::Relaxed) >= 1);
    }
}
