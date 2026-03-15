//! Artifact 存储与管理模块。
//!
//! 对应 Python `optuna.artifacts`。
//!
//! 提供文件存储协议 (`ArtifactStore` trait)、元数据管理、文件上传/下载功能。
//! 默认实现为本地文件系统存储 (`FileSystemArtifactStore`)。

#[cfg(feature = "s3")]
pub mod s3;
#[cfg(feature = "gcs")]
pub mod gcs;

use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::error::{OptunaError, Result};
use crate::storage::Storage;

// ============================================================================
// 错误类型
// ============================================================================

/// Artifact 未找到错误。
///
/// 对应 Python `optuna.artifacts.exceptions.ArtifactNotFound`。
#[derive(Debug)]
pub struct ArtifactNotFound(pub String);

impl std::fmt::Display for ArtifactNotFound {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "artifact not found: {}", self.0)
    }
}

impl std::error::Error for ArtifactNotFound {}

// ============================================================================
// Artifact 元数据
// ============================================================================

/// Artifact 元数据。
///
/// 对应 Python `optuna.artifacts.ArtifactMeta`。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtifactMeta {
    /// 唯一标识符 (UUID)
    pub artifact_id: String,
    /// 文件名
    pub filename: String,
    /// MIME 类型
    pub mimetype: String,
    /// 内容编码（如 gzip）
    pub encoding: Option<String>,
}

// ============================================================================
// ArtifactStore trait
// ============================================================================

/// Artifact 存储协议。
///
/// 对应 Python `optuna.artifacts.ArtifactStore` 协议。
pub trait ArtifactStore: Send + Sync {
    /// 打开读取流。artifact 不存在时返回 `Err(ArtifactNotFound)`。
    fn open_reader(&self, artifact_id: &str) -> std::result::Result<Box<dyn Read>, ArtifactNotFound>;

    /// 写入 artifact 内容。
    fn write(&self, artifact_id: &str, content: &mut dyn Read) -> io::Result<()>;

    /// 删除 artifact。不存在时静默忽略。
    fn remove(&self, artifact_id: &str) -> io::Result<()>;
}

// ============================================================================
// 本地文件系统存储
// ============================================================================

/// 本地文件系统 Artifact 存储。
///
/// 对应 Python `optuna.artifacts.FileSystemArtifactStore`。
pub struct FileSystemArtifactStore {
    /// 根目录
    base_path: PathBuf,
}

impl FileSystemArtifactStore {
    /// 创建文件系统存储。
    ///
    /// # 参数
    /// * `base_path` - 存储根目录（自动创建）
    pub fn new(base_path: impl AsRef<Path>) -> io::Result<Self> {
        let base = base_path.as_ref().to_path_buf();
        std::fs::create_dir_all(&base)?;
        Ok(Self { base_path: base })
    }

    /// 安全路径拼接 — 防止路径穿越攻击。
    fn safe_path(&self, artifact_id: &str) -> std::result::Result<PathBuf, ArtifactNotFound> {
        // 拒绝绝对路径和 ".." 路径组件
        let p = Path::new(artifact_id);
        if p.is_absolute()
            || p.components().any(|c| matches!(c, std::path::Component::ParentDir))
        {
            return Err(ArtifactNotFound(format!(
                "invalid artifact_id (path traversal): {artifact_id}"
            )));
        }
        Ok(self.base_path.join(artifact_id))
    }
}

impl ArtifactStore for FileSystemArtifactStore {
    fn open_reader(&self, artifact_id: &str) -> std::result::Result<Box<dyn Read>, ArtifactNotFound> {
        let path = self.safe_path(artifact_id)?;
        let file = std::fs::File::open(&path)
            .map_err(|_| ArtifactNotFound(artifact_id.to_string()))?;
        Ok(Box::new(file))
    }

    fn write(&self, artifact_id: &str, content: &mut dyn Read) -> io::Result<()> {
        let path = self.safe_path(artifact_id)
            .map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.to_string()))?;
        // 确保父目录存在
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let mut file = std::fs::File::create(&path)?;
        io::copy(content, &mut file)?;
        file.flush()?;
        Ok(())
    }

    fn remove(&self, artifact_id: &str) -> io::Result<()> {
        let path = match self.safe_path(artifact_id) {
            Ok(p) => p,
            Err(_) => return Ok(()), // 无效 ID 静默忽略
        };
        if path.exists() {
            std::fs::remove_file(&path)?;
        }
        Ok(())
    }
}

// ============================================================================
// 指数退避中间件
// ============================================================================

/// 带有指数退避重试的 ArtifactStore 包装器。
///
/// 对应 Python `optuna.artifacts.Backoff`。
pub struct BackoffArtifactStore {
    /// 底层存储
    backend: Box<dyn ArtifactStore>,
    /// 最大重试次数
    max_retries: usize,
    /// 退避乘数
    multiplier: f64,
    /// 最小延迟（秒）
    min_delay: f64,
    /// 最大延迟（秒）
    max_delay: f64,
}

impl BackoffArtifactStore {
    /// 创建退避存储包装器。
    ///
    /// # 默认参数
    /// * `max_retries` = 10
    /// * `multiplier` = 2.0
    /// * `min_delay` = 0.1s
    /// * `max_delay` = 30.0s
    pub fn new(
        backend: Box<dyn ArtifactStore>,
        max_retries: Option<usize>,
        multiplier: Option<f64>,
        min_delay: Option<f64>,
        max_delay: Option<f64>,
    ) -> Self {
        Self {
            backend,
            max_retries: max_retries.unwrap_or(10),
            multiplier: multiplier.unwrap_or(2.0),
            min_delay: min_delay.unwrap_or(0.1),
            max_delay: max_delay.unwrap_or(30.0),
        }
    }

    /// 计算第 n 次重试的延迟（秒）
    fn delay(&self, retry: usize) -> f64 {
        (self.min_delay * self.multiplier.powi(retry as i32)).min(self.max_delay)
    }
}

impl ArtifactStore for BackoffArtifactStore {
    fn open_reader(&self, artifact_id: &str) -> std::result::Result<Box<dyn Read>, ArtifactNotFound> {
        // ArtifactNotFound 不重试
        self.backend.open_reader(artifact_id)
    }

    fn write(&self, artifact_id: &str, content: &mut dyn Read) -> io::Result<()> {
        let mut last_err = None;
        for retry in 0..=self.max_retries {
            match self.backend.write(artifact_id, content) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    last_err = Some(e);
                    if retry < self.max_retries {
                        let delay = self.delay(retry);
                        std::thread::sleep(std::time::Duration::from_secs_f64(delay));
                    }
                }
            }
        }
        Err(last_err.unwrap())
    }

    fn remove(&self, artifact_id: &str) -> io::Result<()> {
        let mut last_err = None;
        for retry in 0..=self.max_retries {
            match self.backend.remove(artifact_id) {
                Ok(()) => return Ok(()),
                Err(e) => {
                    last_err = Some(e);
                    if retry < self.max_retries {
                        let delay = self.delay(retry);
                        std::thread::sleep(std::time::Duration::from_secs_f64(delay));
                    }
                }
            }
        }
        Err(last_err.unwrap())
    }
}

// ============================================================================
// Artifact MIME 类型猜测
// ============================================================================

/// 简单 MIME 类型猜测（按文件扩展名）。
fn guess_mimetype(filename: &str) -> String {
    let ext = Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();
    match ext.as_str() {
        "txt" => "text/plain",
        "csv" => "text/csv",
        "json" => "application/json",
        "yaml" | "yml" => "application/x-yaml",
        "html" | "htm" => "text/html",
        "xml" => "application/xml",
        "pdf" => "application/pdf",
        "png" => "image/png",
        "jpg" | "jpeg" => "image/jpeg",
        "gif" => "image/gif",
        "svg" => "image/svg+xml",
        "zip" => "application/zip",
        "gz" | "gzip" => "application/gzip",
        "tar" => "application/x-tar",
        "bin" => "application/octet-stream",
        "pkl" | "pickle" => "application/octet-stream",
        "pt" | "pth" => "application/octet-stream",
        "onnx" => "application/octet-stream",
        "parquet" => "application/octet-stream",
        _ => "application/octet-stream",
    }
    .to_string()
}

// ============================================================================
// Artifact 系统属性键前缀
// ============================================================================

const ARTIFACT_PREFIX: &str = "artifacts:";

// ============================================================================
// 上传 / 下载 / 列出 API
// ============================================================================

/// 上传 artifact 文件到存储。
///
/// 对应 Python `optuna.artifacts.upload_artifact()`。
///
/// # 参数
/// * `artifact_store` - Artifact 存储后端
/// * `file_path` - 要上传的文件路径
/// * `storage` - Trial/Study 元数据存储
/// * `trial_id` - 关联的 trial ID（如果是 trial 级别的 artifact）
/// * `study_id` - 关联的 study ID（如果是 study 级别的 artifact）
/// * `mimetype` - 自定义 MIME 类型（默认自动猜测）
/// * `encoding` - 内容编码
///
/// # 返回
/// artifact ID (UUID 字符串)
pub fn upload_artifact(
    artifact_store: &dyn ArtifactStore,
    file_path: impl AsRef<Path>,
    storage: &dyn Storage,
    trial_id: Option<i64>,
    study_id: Option<i64>,
    mimetype: Option<&str>,
    encoding: Option<&str>,
) -> Result<String> {
    let path = file_path.as_ref();
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or("unknown")
        .to_string();

    let artifact_id = Uuid::new_v4().to_string();

    let mime = mimetype
        .map(|s| s.to_string())
        .unwrap_or_else(|| guess_mimetype(&filename));

    let meta = ArtifactMeta {
        artifact_id: artifact_id.clone(),
        filename,
        mimetype: mime,
        encoding: encoding.map(|s| s.to_string()),
    };

    // 序列化元数据为 JSON
    let meta_json = serde_json::to_value(&meta)
        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
    let attr_key = format!("{ARTIFACT_PREFIX}{artifact_id}");

    // 写入系统属性
    if let Some(tid) = trial_id {
        storage.set_trial_system_attr(tid, &attr_key, meta_json)?;
    } else if let Some(sid) = study_id {
        storage.set_study_system_attr(sid, &attr_key, meta_json)?;
    } else {
        return Err(OptunaError::ValueError(
            "must specify either trial_id or study_id".into(),
        ));
    }

    // 读取文件并写入 artifact 存储
    let mut file = std::fs::File::open(path)
        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
    artifact_store
        .write(&artifact_id, &mut file)
        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;

    Ok(artifact_id)
}

/// 下载 artifact 到本地文件。
///
/// 对应 Python `optuna.artifacts.download_artifact()`。
pub fn download_artifact(
    artifact_store: &dyn ArtifactStore,
    artifact_id: &str,
    file_path: impl AsRef<Path>,
) -> Result<()> {
    let path = file_path.as_ref();
    if path.exists() {
        return Err(OptunaError::ValueError(format!(
            "file already exists: {}",
            path.display()
        )));
    }
    let mut reader = artifact_store
        .open_reader(artifact_id)
        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
    let mut file = std::fs::File::create(path)
        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
    io::copy(&mut reader, &mut file)
        .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
    Ok(())
}

/// 获取 trial 的所有 artifact 元数据。
///
/// 对应 Python `optuna.artifacts.get_all_artifact_meta()`。
pub fn get_all_artifact_meta_for_trial(
    storage: &dyn Storage,
    trial_id: i64,
) -> Result<Vec<ArtifactMeta>> {
    let trial = storage.get_trial(trial_id)?;
    extract_artifact_meta(&trial.system_attrs)
}

/// 获取 study 的所有 artifact 元数据。
pub fn get_all_artifact_meta_for_study(
    storage: &dyn Storage,
    study_id: i64,
) -> Result<Vec<ArtifactMeta>> {
    let attrs = storage.get_study_system_attrs(study_id)?;
    extract_artifact_meta(&attrs)
}

/// 从系统属性中提取 artifact 元数据。
fn extract_artifact_meta(
    attrs: &HashMap<String, serde_json::Value>,
) -> Result<Vec<ArtifactMeta>> {
    let mut metas = Vec::new();
    for (key, value) in attrs {
        if let Some(_id) = key.strip_prefix(ARTIFACT_PREFIX) {
            let meta: ArtifactMeta = serde_json::from_value(value.clone())
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
            metas.push(meta);
        }
    }
    Ok(metas)
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_guess_mimetype() {
        assert_eq!(guess_mimetype("test.txt"), "text/plain");
        assert_eq!(guess_mimetype("data.json"), "application/json");
        assert_eq!(guess_mimetype("model.onnx"), "application/octet-stream");
        assert_eq!(guess_mimetype("image.png"), "image/png");
    }

    #[test]
    fn test_filesystem_artifact_store() {
        let tmp = std::env::temp_dir().join(format!("optuna_artifact_test_{}", Uuid::new_v4()));
        let store = FileSystemArtifactStore::new(&tmp).unwrap();

        // 写入
        let data = b"hello artifact world";
        let mut cursor = io::Cursor::new(data.to_vec());
        store.write("test-artifact", &mut cursor).unwrap();

        // 读取
        let mut reader = store.open_reader("test-artifact").unwrap();
        let mut buf = Vec::new();
        reader.read_to_end(&mut buf).unwrap();
        assert_eq!(buf, data);

        // 删除
        store.remove("test-artifact").unwrap();

        // 读取应失败
        assert!(store.open_reader("test-artifact").is_err());

        // 清理
        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_filesystem_path_traversal_prevention() {
        let tmp = std::env::temp_dir().join(format!("optuna_artifact_safe_{}", Uuid::new_v4()));
        let store = FileSystemArtifactStore::new(&tmp).unwrap();

        // 绝对路径应被拒绝
        assert!(store.open_reader("/etc/passwd").is_err());

        // 路径穿越应被拒绝
        assert!(store.open_reader("../../../etc/passwd").is_err());

        let _ = std::fs::remove_dir_all(&tmp);
    }

    #[test]
    fn test_artifact_meta_serialization() {
        let meta = ArtifactMeta {
            artifact_id: "abc-123".to_string(),
            filename: "model.pt".to_string(),
            mimetype: "application/octet-stream".to_string(),
            encoding: None,
        };
        let json = serde_json::to_value(&meta).unwrap();
        let deserialized: ArtifactMeta = serde_json::from_value(json).unwrap();
        assert_eq!(deserialized.artifact_id, "abc-123");
        assert_eq!(deserialized.filename, "model.pt");
    }

    #[test]
    fn test_upload_and_download_artifact() {
        let artifact_dir = std::env::temp_dir().join(format!("optuna_art_{}", Uuid::new_v4()));
        let store = FileSystemArtifactStore::new(&artifact_dir).unwrap();
        let mem_storage = crate::storage::InMemoryStorage::new();

        // 创建测试文件
        let src_file = std::env::temp_dir().join(format!("optuna_src_{}.txt", Uuid::new_v4()));
        std::fs::write(&src_file, b"test data for artifact").unwrap();

        // 创建 study + trial
        let study_id = mem_storage.create_new_study(
            &[crate::study::StudyDirection::Minimize],
            Some("art_test"),
        ).unwrap();
        let trial_id = mem_storage.create_new_trial(study_id, None).unwrap();

        // 上传
        let artifact_id = upload_artifact(
            &store,
            &src_file,
            &mem_storage,
            Some(trial_id),
            None,
            None,
            None,
        ).unwrap();
        assert!(!artifact_id.is_empty());

        // 列出元数据
        let metas = get_all_artifact_meta_for_trial(&mem_storage, trial_id).unwrap();
        assert_eq!(metas.len(), 1);
        assert_eq!(metas[0].artifact_id, artifact_id);
        assert_eq!(metas[0].mimetype, "text/plain");

        // 下载
        let dst_file = std::env::temp_dir().join(format!("optuna_dst_{}.txt", Uuid::new_v4()));
        download_artifact(&store, &artifact_id, &dst_file).unwrap();
        let content = std::fs::read(&dst_file).unwrap();
        assert_eq!(content, b"test data for artifact");

        // 下载到已存在文件应失败
        assert!(download_artifact(&store, &artifact_id, &dst_file).is_err());

        // 清理
        let _ = std::fs::remove_file(&src_file);
        let _ = std::fs::remove_file(&dst_file);
        let _ = std::fs::remove_dir_all(&artifact_dir);
    }

    #[test]
    fn test_backoff_delay_calculation() {
        let store = BackoffArtifactStore::new(
            Box::new(FileSystemArtifactStore::new(std::env::temp_dir().join("_bo")).unwrap()),
            Some(10), Some(2.0), Some(0.1), Some(30.0),
        );
        assert!((store.delay(0) - 0.1).abs() < 1e-10);
        assert!((store.delay(1) - 0.2).abs() < 1e-10);
        assert!((store.delay(2) - 0.4).abs() < 1e-10);
        assert!((store.delay(8) - 25.6).abs() < 1e-10);
        assert!((store.delay(9) - 30.0).abs() < 1e-10); // capped
        let _ = std::fs::remove_dir_all(std::env::temp_dir().join("_bo"));
    }
}
