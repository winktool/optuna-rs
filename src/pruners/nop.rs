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
