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
}
