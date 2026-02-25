use crate::trial::FrozenTrial;

/// A callback invoked after each trial completes.
///
/// Corresponds to Python `optuna.study.Study.optimize(..., callbacks=...)`.
pub trait Callback: Send + Sync {
    /// Called after each trial finishes.
    ///
    /// `n_complete` is the total number of complete trials so far.
    /// `trial` is the just-finished trial.
    fn on_trial_complete(&self, n_complete: usize, trial: &FrozenTrial);
}

/// A callback that stops the study after a maximum number of completed trials.
pub struct MaxTrialsCallback {
    pub max_trials: usize,
}

impl MaxTrialsCallback {
    pub fn new(max_trials: usize) -> Self {
        Self { max_trials }
    }
}

impl Callback for MaxTrialsCallback {
    fn on_trial_complete(&self, _n_complete: usize, _trial: &FrozenTrial) {
        // The actual stopping logic is checked in the optimize loop
        // by inspecting the total trial count.
    }
}
