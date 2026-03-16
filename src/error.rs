use thiserror::Error;

/// All errors that can occur in optuna.
#[derive(Debug, Error)]
pub enum OptunaError {
    /// Raised when a trial is pruned.
    #[error("trial was pruned")]
    TrialPruned,

    /// Raised when a study with the same name already exists.
    #[error("duplicated study: {0}")]
    DuplicatedStudyError(String),

    /// Raised when a storage operation fails.
    #[error("storage internal error: {0}")]
    StorageInternalError(String),

    /// Raised when attempting to update a finished trial.
    #[error("cannot update a finished trial: {0}")]
    UpdateFinishedTrialError(String),

    /// Raised for general value/argument errors.
    #[error("{0}")]
    ValueError(String),

    /// Raised when a distribution is invalid.
    #[error("invalid distribution: {0}")]
    InvalidDistribution(String),

    /// Raised when a feature is not implemented (e.g., multi-objective pruning).
    /// 对应 Python `NotImplementedError`。
    #[error("not implemented: {0}")]
    NotImplemented(String),

    /// 对应 Python `RuntimeError` — 通用运行时错误。
    #[error("runtime error: {0}")]
    RuntimeError(String),
}

/// Convenience alias used throughout the crate.
pub type Result<T> = std::result::Result<T, OptunaError>;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_not_implemented_display() {
        let e = OptunaError::NotImplemented("multi-obj pruning".into());
        assert!(e.to_string().contains("multi-obj pruning"));
    }

    #[test]
    fn test_error_variants() {
        // 测试所有 error variant 的 Display
        assert!(OptunaError::TrialPruned.to_string().contains("pruned"));
        assert!(OptunaError::ValueError("bad".into()).to_string().contains("bad"));
        assert!(OptunaError::InvalidDistribution("x".into()).to_string().contains("x"));
    }

    /// 对齐 Python: DuplicatedStudyError 包含研究名
    #[test]
    fn test_duplicated_study_error() {
        let e = OptunaError::DuplicatedStudyError("my_study".into());
        assert!(e.to_string().contains("my_study"));
    }

    /// 对齐 Python: StorageInternalError 信息传递
    #[test]
    fn test_storage_internal_error() {
        let e = OptunaError::StorageInternalError("db crash".into());
        assert!(e.to_string().contains("db crash"));
    }

    /// 对齐 Python: UpdateFinishedTrialError 信息传递
    #[test]
    fn test_update_finished_trial_error() {
        let e = OptunaError::UpdateFinishedTrialError("trial #5".into());
        assert!(e.to_string().contains("trial #5"));
    }

    /// 确保 Result 类型别名工作正常
    #[test]
    fn test_result_alias() {
        let ok: Result<i32> = Ok(42);
        assert_eq!(ok.unwrap(), 42);
        let err: Result<i32> = Err(OptunaError::TrialPruned);
        assert!(err.is_err());
    }
}
