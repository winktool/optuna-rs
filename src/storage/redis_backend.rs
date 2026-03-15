//! Redis Journal 后端。
//!
//! 对应 Python `optuna.storages.journal.JournalRedisBackend`。
//! 基于 [`redis`](https://docs.rs/redis) crate 实现日志持久化到 Redis。
//!
//! # 功能
//! - 使用 Redis List 存储日志条目
//! - 支持 `RPUSH` 追加 + `LRANGE` 批量读取
//! - 原子操作保证多进程安全
//!
//! # 使用方式
//! 需要启用 `redis-storage` feature:
//! ```toml
//! optuna-rs = { version = "0.1", features = ["redis-storage"] }
//! ```

#[cfg(feature = "redis-storage")]
use redis::{Client, Commands};

#[cfg(feature = "redis-storage")]
use parking_lot::Mutex;

#[cfg(feature = "redis-storage")]
use crate::error::{OptunaError, Result};

#[cfg(feature = "redis-storage")]
use crate::storage::journal::{JournalBackend, JournalLogEntry};

/// Redis Journal 后端。
///
/// 对应 Python `optuna.storages.journal.JournalRedisBackend`。
/// 将日志条目序列化为 JSON 字符串，存储在 Redis List 中。
///
/// # Redis 数据结构
/// - Key: `{key_prefix}` (默认 `"optuna:journal"`)
/// - Type: List (RPUSH/LRANGE)
/// - Value: JSON 序列化的 `JournalLogEntry`
#[cfg(feature = "redis-storage")]
pub struct JournalRedisBackend {
    /// Redis 客户端
    client: Client,
    /// Redis 连接（复用）
    connection: Mutex<redis::Connection>,
    /// Redis Key 前缀
    key: String,
}

#[cfg(feature = "redis-storage")]
impl JournalRedisBackend {
    /// 创建 Redis Journal 后端。
    ///
    /// # 参数
    /// * `url` - Redis 连接 URL (如 `"redis://127.0.0.1:6379"`)
    /// * `key` - Redis List 的 key 名称（可选，默认 `"optuna:journal"`）
    ///
    /// # 示例
    /// ```ignore
    /// use optuna_rs::storage::JournalRedisBackend;
    ///
    /// let backend = JournalRedisBackend::new("redis://127.0.0.1:6379", None).unwrap();
    /// ```
    pub fn new(url: &str, key: Option<&str>) -> Result<Self> {
        let client = Client::open(url)
            .map_err(|e| OptunaError::StorageInternalError(
                format!("Redis 连接失败: {e}")))?;
        let connection = client.get_connection()
            .map_err(|e| OptunaError::StorageInternalError(
                format!("Redis 获取连接失败: {e}")))?;
        Ok(Self {
            client,
            connection: Mutex::new(connection),
            key: key.unwrap_or("optuna:journal").to_string(),
        })
    }

    /// 获取 Redis 客户端引用（用于高级操作）。
    pub fn client(&self) -> &Client {
        &self.client
    }

    /// 获取当前使用的 Redis key。
    pub fn key(&self) -> &str {
        &self.key
    }
}

#[cfg(feature = "redis-storage")]
impl JournalBackend for JournalRedisBackend {
    /// 追加日志到 Redis List (RPUSH)。
    fn append_log(&self, entry: &JournalLogEntry) -> Result<()> {
        let json = serde_json::to_string(entry)
            .map_err(|e| OptunaError::StorageInternalError(
                format!("JSON 序列化失败: {e}")))?;
        let mut conn = self.connection.lock();
        conn.rpush::<_, _, ()>(&self.key, &json)
            .map_err(|e| OptunaError::StorageInternalError(
                format!("Redis RPUSH 失败: {e}")))?;
        Ok(())
    }

    /// 从 Redis List 读取日志 (LRANGE)。
    ///
    /// # 参数
    /// * `log_number_from` - 起始日志编号（0-based 索引）
    fn read_logs(&self, log_number_from: usize) -> Result<Vec<JournalLogEntry>> {
        let mut conn = self.connection.lock();
        let raw: Vec<String> = conn.lrange(&self.key, log_number_from as isize, -1)
            .map_err(|e| OptunaError::StorageInternalError(
                format!("Redis LRANGE 失败: {e}")))?;

        let mut entries = Vec::with_capacity(raw.len());
        for json_str in &raw {
            match serde_json::from_str::<JournalLogEntry>(json_str) {
                Ok(entry) => entries.push(entry),
                Err(e) => {
                    // 跳过损坏的日志条目（与 Python 行为一致）
                    eprintln!("警告: 跳过损坏的 Redis 日志条目: {e}");
                }
            }
        }
        Ok(entries)
    }
}

#[cfg(test)]
#[cfg(feature = "redis-storage")]
mod tests {
    // Redis 测试需要运行中的 Redis 实例，标记为 ignore
    use super::*;

    #[test]
    #[ignore = "需要运行中的 Redis 实例"]
    fn test_redis_backend_roundtrip() {
        let backend = JournalRedisBackend::new(
            "redis://127.0.0.1:6379",
            Some("optuna:test:journal"),
        ).unwrap();

        // 清理测试数据
        let mut conn = backend.connection.lock();
        let _: () = redis::cmd("DEL")
            .arg("optuna:test:journal")
            .query(&mut *conn)
            .unwrap();
        drop(conn);

        // 写入日志
        let entry = JournalLogEntry {
            op_code: 0,
            study_id: None,
            trial_id: None,
            data: serde_json::json!({"study_name": "test", "directions": [1]}),
        };
        backend.append_log(&entry).unwrap();

        // 读取日志
        let logs = backend.read_logs(0).unwrap();
        assert_eq!(logs.len(), 1);
        assert_eq!(logs[0].op_code, 0);
    }

    #[test]
    #[ignore = "需要运行中的 Redis 实例"]
    fn test_redis_journal_storage_integration() {
        use crate::storage::journal::JournalStorage;
        use crate::storage::Storage;
        use crate::study::StudyDirection;

        // 使用唯一 key 避免冲突
        let key = format!("optuna:test:{}", uuid::Uuid::new_v4());
        let backend = JournalRedisBackend::new(
            "redis://127.0.0.1:6379",
            Some(&key),
        ).unwrap();

        let storage = JournalStorage::new(Box::new(backend)).unwrap();
        let study_id = storage
            .create_new_study(&[StudyDirection::Minimize], Some("redis_test"))
            .unwrap();

        let trial_id = storage.create_new_trial(study_id, None).unwrap();
        storage
            .set_trial_state_values(trial_id, crate::trial::TrialState::Complete, Some(&[0.5]))
            .unwrap();

        let trials = storage.get_all_trials(study_id, None).unwrap();
        assert_eq!(trials.len(), 1);
        assert_eq!(trials[0].values.as_ref().unwrap()[0], 0.5);

        // 清理
        let client = redis::Client::open("redis://127.0.0.1:6379").unwrap();
        let mut conn = client.get_connection().unwrap();
        let _: () = redis::cmd("DEL").arg(&key).query(&mut conn).unwrap();
    }
}
