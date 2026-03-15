//! Google Cloud Storage 工件存储后端。
//!
//! 对应 Python `optuna.artifacts.GCSArtifactStore`。
//! 基于 [`cloud-storage`](https://docs.rs/cloud-storage) crate 实现 GCS 对象存储。
//!
//! # 使用方式
//! 需要启用 `gcs` feature:
//! ```toml
//! optuna-rs = { version = "0.1", features = ["gcs"] }
//! ```

#[cfg(feature = "gcs")]
use std::io::{self, Read};

#[cfg(feature = "gcs")]
use cloud_storage::Client as GcsClient;

#[cfg(feature = "gcs")]
use crate::artifacts::{ArtifactNotFound, ArtifactStore};

#[cfg(feature = "gcs")]
use crate::error::{OptunaError, Result};

/// Google Cloud Storage 工件存储后端。
///
/// 对应 Python `optuna.artifacts.GCSArtifactStore`。
/// 使用 `cloud-storage` crate 进行 GCS 对象操作。
///
/// # 功能
/// - 上传工件到 GCS bucket + prefix
/// - 下载工件为字节流
/// - 删除单个工件
///
/// # 凭证
/// 使用标准 GCP 凭证链（GOOGLE_APPLICATION_CREDENTIALS 环境变量、
/// gcloud CLI 默认凭证、GCE 元数据服务等）。
#[cfg(feature = "gcs")]
pub struct GcsArtifactStore {
    /// GCS 客户端
    client: GcsClient,
    /// GCS Bucket 名称
    bucket: String,
    /// 对象 key 前缀
    prefix: String,
}

#[cfg(feature = "gcs")]
impl GcsArtifactStore {
    /// 创建 GCS 工件存储。
    ///
    /// # 参数
    /// * `bucket` - GCS Bucket 名称
    /// * `prefix` - 对象 key 前缀（默认 `"optuna-artifacts/"`）
    pub fn new(bucket: &str, prefix: Option<&str>) -> Result<Self> {
        let client = GcsClient::default();
        Ok(Self {
            client,
            bucket: bucket.to_string(),
            prefix: prefix.unwrap_or("optuna-artifacts/").to_string(),
        })
    }

    /// 构建完整的 GCS 对象路径。
    fn full_path(&self, artifact_id: &str) -> String {
        format!("{}{}", self.prefix, artifact_id)
    }

    /// 同步辅助: 获取或创建 tokio Runtime。
    fn runtime() -> Result<tokio::runtime::Runtime> {
        tokio::runtime::Runtime::new()
            .map_err(|e| OptunaError::StorageInternalError(
                format!("创建 tokio runtime 失败: {e}")))
    }
}

#[cfg(feature = "gcs")]
impl ArtifactStore for GcsArtifactStore {
    /// 从 GCS 读取工件内容。
    fn open_reader(&self, artifact_id: &str) -> std::result::Result<Box<dyn Read>, ArtifactNotFound> {
        let path = self.full_path(artifact_id);
        let rt = Self::runtime()
            .map_err(|e| ArtifactNotFound(format!("runtime error: {e}")))?;
        let data = rt.block_on(async {
            self.client
                .object()
                .download(&self.bucket, &path)
                .await
        }).map_err(|e| ArtifactNotFound(
            format!("GCS 下载失败 (path={}): {e}", path)))?;
        Ok(Box::new(std::io::Cursor::new(data)))
    }

    /// 将内容写入 GCS。
    fn write(&self, artifact_id: &str, content: &mut dyn Read) -> io::Result<()> {
        let path = self.full_path(artifact_id);
        let mut buf = Vec::new();
        content.read_to_end(&mut buf)?;

        let rt = Self::runtime()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        rt.block_on(async {
            self.client
                .object()
                .create(&self.bucket, buf, &path, "application/octet-stream")
                .await
        }).map_err(|e| io::Error::new(io::ErrorKind::Other,
            format!("GCS 上传失败 (path={}): {e}", path)))?;
        Ok(())
    }

    /// 从 GCS 删除工件。
    fn remove(&self, artifact_id: &str) -> io::Result<()> {
        let path = self.full_path(artifact_id);
        let rt = Self::runtime()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        rt.block_on(async {
            self.client
                .object()
                .delete(&self.bucket, &path)
                .await
        }).map_err(|e| io::Error::new(io::ErrorKind::Other,
            format!("GCS 删除失败 (path={}): {e}", path)))?;
        Ok(())
    }
}
