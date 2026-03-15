//! S3 工件存储后端。
//!
//! 对应 Python `optuna.artifacts.Boto3ArtifactStore`。
//! 基于 [`aws-sdk-s3`](https://docs.rs/aws-sdk-s3) crate 实现 S3 对象存储。
//!
//! # 使用方式
//! 需要启用 `s3` feature:
//! ```toml
//! optuna-rs = { version = "0.1", features = ["s3"] }
//! ```

#[cfg(feature = "s3")]
use std::io::{self, Read};

#[cfg(feature = "s3")]
use aws_sdk_s3::Client as S3Client;

#[cfg(feature = "s3")]
use crate::artifacts::{ArtifactNotFound, ArtifactStore};

#[cfg(feature = "s3")]
use crate::error::{OptunaError, Result};

/// S3 工件存储后端。
///
/// 对应 Python `optuna.artifacts.Boto3ArtifactStore`。
/// 使用 AWS SDK for Rust 进行 S3 对象操作。
///
/// # 功能
/// - 上传工件到 S3 bucket + prefix
/// - 下载工件为字节流
/// - 删除单个工件
///
/// # 凭证
/// 使用标准 AWS 凭证链（环境变量、~/.aws/credentials、IAM Role 等）。
#[cfg(feature = "s3")]
pub struct S3ArtifactStore {
    /// S3 客户端
    client: S3Client,
    /// S3 Bucket 名称
    bucket: String,
    /// 对象 key 前缀（用于隔离不同 study）
    prefix: String,
}

#[cfg(feature = "s3")]
impl S3ArtifactStore {
    /// 创建 S3 工件存储。
    ///
    /// # 参数
    /// * `bucket` - S3 Bucket 名称
    /// * `prefix` - 对象 key 前缀（默认 `"optuna-artifacts/"`）
    ///
    /// # 示例
    /// ```ignore
    /// use optuna_rs::artifacts::S3ArtifactStore;
    ///
    /// let rt = tokio::runtime::Runtime::new().unwrap();
    /// let store = rt.block_on(S3ArtifactStore::new("my-bucket", None));
    /// ```
    pub async fn new(bucket: &str, prefix: Option<&str>) -> Result<Self> {
        let config = aws_config::load_defaults(aws_config::BehaviorVersion::latest()).await;
        let client = S3Client::new(&config);
        Ok(Self {
            client,
            bucket: bucket.to_string(),
            prefix: prefix.unwrap_or("optuna-artifacts/").to_string(),
        })
    }

    /// 从自定义 AWS 配置创建。
    pub fn with_client(client: S3Client, bucket: &str, prefix: Option<&str>) -> Self {
        Self {
            client,
            bucket: bucket.to_string(),
            prefix: prefix.unwrap_or("optuna-artifacts/").to_string(),
        }
    }

    /// 构建完整的 S3 对象 key。
    fn full_key(&self, artifact_id: &str) -> String {
        format!("{}{}", self.prefix, artifact_id)
    }

    /// 同步辅助：获取或创建 tokio Runtime。
    fn runtime() -> Result<tokio::runtime::Runtime> {
        tokio::runtime::Runtime::new()
            .map_err(|e| OptunaError::StorageInternalError(
                format!("创建 tokio runtime 失败: {e}")))
    }
}

#[cfg(feature = "s3")]
impl ArtifactStore for S3ArtifactStore {
    /// 从 S3 读取工件内容。
    fn open_reader(&self, artifact_id: &str) -> std::result::Result<Box<dyn Read>, ArtifactNotFound> {
        let key = self.full_key(artifact_id);
        let rt = Self::runtime()
            .map_err(|e| ArtifactNotFound(format!("runtime error: {e}")))?;
        let result = rt.block_on(async {
            self.client
                .get_object()
                .bucket(&self.bucket)
                .key(&key)
                .send()
                .await
        });
        match result {
            Ok(output) => {
                let body = rt.block_on(async {
                    output.body.collect().await
                }).map_err(|e| ArtifactNotFound(
                    format!("S3 读取 body 失败: {e}")))?;
                Ok(Box::new(std::io::Cursor::new(body.into_bytes().to_vec())))
            }
            Err(e) => Err(ArtifactNotFound(
                format!("S3 GetObject 失败 (key={}): {e}", key))),
        }
    }

    /// 将内容写入 S3。
    fn write(&self, artifact_id: &str, content: &mut dyn Read) -> io::Result<()> {
        let key = self.full_key(artifact_id);
        let mut buf = Vec::new();
        content.read_to_end(&mut buf)?;

        let rt = Self::runtime()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        rt.block_on(async {
            self.client
                .put_object()
                .bucket(&self.bucket)
                .key(&key)
                .body(buf.into())
                .send()
                .await
        }).map_err(|e| io::Error::new(io::ErrorKind::Other,
            format!("S3 PutObject 失败 (key={}): {e}", key)))?;
        Ok(())
    }

    /// 从 S3 删除工件。
    fn remove(&self, artifact_id: &str) -> io::Result<()> {
        let key = self.full_key(artifact_id);
        let rt = Self::runtime()
            .map_err(|e| io::Error::new(io::ErrorKind::Other, e.to_string()))?;
        rt.block_on(async {
            self.client
                .delete_object()
                .bucket(&self.bucket)
                .key(&key)
                .send()
                .await
        }).map_err(|e| io::Error::new(io::ErrorKind::Other,
            format!("S3 DeleteObject 失败 (key={}): {e}", key)))?;
        Ok(())
    }
}
