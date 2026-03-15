use crate::error::Result;
use crate::pruners::Pruner;
use crate::trial::FrozenTrial;

/// A pruner that never prunes.
///
/// Corresponds to Python `optuna.pruners.NopPruner`.
#[derive(Debug, Default)]
pub struct NopPruner;

impl NopPruner {
    pub fn new() -> Self {
        Self
    }
}

impl Pruner for NopPruner {
    fn prune(&self, _study_trials: &[FrozenTrial], _trial: &FrozenTrial, _storage: Option<&dyn crate::storage::Storage>) -> Result<bool> {
        Ok(false)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trial::TrialState;
    use std::collections::HashMap;

    fn make_trial() -> FrozenTrial {
        FrozenTrial {
            number: 0,
            state: TrialState::Running,
            values: None,
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        }
    }

    /// 对齐 Python: NopPruner 永远不剪枝
    #[test]
    fn test_nop_pruner_never_prunes() {
        let pruner = NopPruner::new();
        let trial = make_trial();
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }

    /// 对齐 Python: 即使有已完成试验也不剪枝
    #[test]
    fn test_nop_pruner_with_completed_trials() {
        let pruner = NopPruner::new();
        let completed = FrozenTrial {
            number: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(chrono::Utc::now()),
            datetime_complete: Some(chrono::Utc::now()),
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        let trial = make_trial();
        assert!(!pruner.prune(&[completed], &trial, None).unwrap());
    }

    /// 对齐 Python: Default trait
    #[test]
    fn test_nop_pruner_default() {
        let pruner = NopPruner::default();
        let trial = make_trial();
        assert!(!pruner.prune(&[], &trial, None).unwrap());
    }
}
