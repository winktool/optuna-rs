use serde::{Deserialize, Serialize};

/// The state of a trial.
///
/// Corresponds to Python `optuna.trial.TrialState`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[repr(u8)]
pub enum TrialState {
    /// The trial is currently executing.
    Running = 0,
    /// The trial finished without error and has values.
    Complete = 1,
    /// The trial was pruned via `TrialPruned`.
    Pruned = 2,
    /// The trial failed due to an uncaught error.
    Fail = 3,
    /// The trial is queued but not yet started.
    Waiting = 4,
}

impl TrialState {
    /// Returns `true` for terminal states: `Complete`, `Pruned`, `Fail`.
    pub fn is_finished(self) -> bool {
        matches!(self, Self::Complete | Self::Pruned | Self::Fail)
    }
}

impl std::fmt::Display for TrialState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Running => write!(f, "RUNNING"),
            Self::Complete => write!(f, "COMPLETE"),
            Self::Pruned => write!(f, "PRUNED"),
            Self::Fail => write!(f, "FAIL"),
            Self::Waiting => write!(f, "WAITING"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_is_finished() {
        assert!(!TrialState::Running.is_finished());
        assert!(TrialState::Complete.is_finished());
        assert!(TrialState::Pruned.is_finished());
        assert!(TrialState::Fail.is_finished());
        assert!(!TrialState::Waiting.is_finished());
    }

    #[test]
    fn test_display() {
        assert_eq!(format!("{}", TrialState::Running), "RUNNING");
        assert_eq!(format!("{}", TrialState::Complete), "COMPLETE");
        assert_eq!(format!("{}", TrialState::Pruned), "PRUNED");
        assert_eq!(format!("{}", TrialState::Fail), "FAIL");
        assert_eq!(format!("{}", TrialState::Waiting), "WAITING");
    }

    /// 对齐 Python: TrialState repr 值
    #[test]
    fn test_repr_values() {
        assert_eq!(TrialState::Running as u8, 0);
        assert_eq!(TrialState::Complete as u8, 1);
        assert_eq!(TrialState::Pruned as u8, 2);
        assert_eq!(TrialState::Fail as u8, 3);
        assert_eq!(TrialState::Waiting as u8, 4);
    }

    /// 对齐 Python: TrialState 可以 Hash
    #[test]
    fn test_hash() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(TrialState::Running);
        set.insert(TrialState::Complete);
        set.insert(TrialState::Running);  // 重复
        assert_eq!(set.len(), 2);
    }

    /// 对齐 Python: TrialState 可以 Clone/Copy
    #[test]
    fn test_clone_copy() {
        let s = TrialState::Running;
        let s2 = s;  // Copy
        let s3 = s.clone();  // Clone
        assert_eq!(s, s2);
        assert_eq!(s, s3);
    }
}
