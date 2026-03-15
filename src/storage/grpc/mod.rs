//! gRPC 存储代理模块。
//!
//! 对应 Python `optuna.storages._grpc`。
//! 使用 [`tonic`](https://docs.rs/tonic) 实现 gRPC 服务端和客户端。
//!
//! # 使用方式
//! 需要启用 `grpc` feature:
//! ```toml
//! optuna-rs = { version = "0.1", features = ["grpc"] }
//! ```

#[cfg(feature = "grpc")]
pub mod server;
#[cfg(feature = "grpc")]
pub mod client;

#[cfg(feature = "grpc")]
pub mod proto {
    tonic::include_proto!("optuna.storage");
}

#[cfg(feature = "grpc")]
pub use client::GrpcStorageProxy;
#[cfg(feature = "grpc")]
pub use server::run_grpc_proxy_server;
