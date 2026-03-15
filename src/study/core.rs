use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use crate::callbacks::Callback;
use crate::distributions::Distribution;
use crate::error::{OptunaError, Result};
use crate::pruners::Pruner;
use crate::samplers::Sampler;
use crate::storage::Storage;
use crate::study::StudyDirection;
use crate::terminators::Terminator;
use crate::trial::{FrozenTrial, Trial, TrialState};

/// 全局 SIGINT 标志，用于优雅终止优化循环。
/// 对应 Python `_optimize.py` 中的信号处理器。
static SIGINT_RECEIVED: AtomicBool = AtomicBool::new(false);

/// 安装 Ctrl+C (SIGINT) 信号处理器。
/// 第一次 Ctrl+C 设置标志，第二次 Ctrl+C 强制退出进程。
fn install_signal_handler(stop_flag: &Arc<AtomicBool>) {
    let flag = Arc::clone(stop_flag);
    let _ = ctrlc::set_handler(move || {
        if SIGINT_RECEIVED.load(Ordering::SeqCst) {
            // 第二次 Ctrl+C → 直接退出
            std::process::exit(130);
        }
        SIGINT_RECEIVED.store(true, Ordering::SeqCst);
        flag.store(true, Ordering::Release);
        eprintln!("\nOptimization stopped by Ctrl+C. Finishing current trial...");
    });
}

/// 判断错误是否应被 catch 列表捕获。
/// 对应 Python `isinstance(func_err, catch)`。
/// Rust 中通过错误变体名称字符串匹配:
/// - "*" — 等同 Python 的 catch=(Exception,)，捕获所有异常
/// - "ValueError" — 匹配 OptunaError::ValueError
/// - "StorageInternalError" — 匹配 OptunaError::StorageInternalError
/// - "InvalidDistribution" — 匹配 OptunaError::InvalidDistribution
fn error_matches_catch(err: &OptunaError, catch: &[&str]) -> bool {
    if catch.is_empty() {
        return false;
    }
    if catch.contains(&"*") {
        return true;
    }
    let variant = match err {
        OptunaError::TrialPruned => "TrialPruned",
        OptunaError::ValueError(_) => "ValueError",
        OptunaError::StorageInternalError(_) => "StorageInternalError",
        OptunaError::DuplicatedStudyError(_) => "DuplicatedStudyError",
        OptunaError::UpdateFinishedTrialError(_) => "UpdateFinishedTrialError",
        OptunaError::InvalidDistribution(_) => "InvalidDistribution",
        OptunaError::NotImplemented(_) => "NotImplemented",
    };
    catch.contains(&variant)
}

/// An optimization study.
///
/// Corresponds to Python `optuna.study.Study`.
pub struct Study {
    study_name: String,
    study_id: i64,
    storage: Arc<dyn Storage>,
    directions: Vec<StudyDirection>,
    sampler: Arc<dyn Sampler>,
    pruner: Arc<dyn Pruner>,
    stop_flag: Arc<AtomicBool>,
}

impl Study {
    /// Create a new Study. Prefer using `create_study()` instead.
    pub(crate) fn new(
        study_name: String,
        study_id: i64,
        storage: Arc<dyn Storage>,
        directions: Vec<StudyDirection>,
        sampler: Arc<dyn Sampler>,
        pruner: Arc<dyn Pruner>,
    ) -> Self {
        Self {
            study_name,
            study_id,
            storage,
            directions,
            sampler,
            pruner,
            stop_flag: Arc::new(AtomicBool::new(false)),
        }
    }

    // ── Properties ──────────────────────────────────────────────────────

    pub fn study_name(&self) -> &str {
        &self.study_name
    }

    pub fn study_id(&self) -> i64 {
        self.study_id
    }

    pub fn directions(&self) -> &[StudyDirection] {
        &self.directions
    }

    /// Single-objective direction (errors if multi-objective).
    pub fn direction(&self) -> Result<StudyDirection> {
        if self.directions.len() != 1 {
            return Err(OptunaError::ValueError(
                "direction is not supported for multi-objective studies".into(),
            ));
        }
        Ok(self.directions[0])
    }

    pub fn storage(&self) -> &Arc<dyn Storage> {
        &self.storage
    }

    /// Get the best trial for single-objective studies.
    ///
    /// 对应 Python `Study.best_trial`。
    /// 支持约束优化：如果试验带有约束信息，只考虑可行试验。
    pub fn best_trial(&self) -> Result<FrozenTrial> {
        let trials = self
            .storage
            .get_all_trials(self.study_id, Some(&[TrialState::Complete]))?;
        if trials.is_empty() {
            return Err(OptunaError::ValueError("No completed trials.".into()));
        }
        let direction = self.direction()?;

        // 检查是否有约束信息
        let has_constraints = trials.iter().any(|t| {
            t.system_attrs.contains_key("constraints")
        });

        // 筛选可行试验（如果存在约束）
        // 对齐 Python: 无 constraints key 的试验视为不可行（unwrap_or(false)）
        let feasible: Vec<&FrozenTrial> = if has_constraints {
            trials.iter().filter(|t| {
                t.system_attrs.get("constraints")
                    .and_then(|v| serde_json::from_value::<Vec<f64>>(v.clone()).ok())
                    .map(|cs| cs.iter().all(|c| *c <= 0.0))
                    .unwrap_or(false)
            }).collect()
        } else {
            trials.iter().collect()
        };

        if feasible.is_empty() {
            return Err(OptunaError::ValueError(
                "No feasible (constraint-satisfying) completed trials.".into(),
            ));
        }

        let best = feasible
            .iter()
            .filter_map(|t| t.value().ok().flatten().map(|v| (t, v)))
            .min_by(|(_, a), (_, b)| {
                let (a, b) = match direction {
                    StudyDirection::Minimize => (*a, *b),
                    _ => (-*a, -*b),
                };
                a.partial_cmp(&b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(t, _)| (*t).clone())
            .ok_or_else(|| OptunaError::ValueError("No trial with a finite value.".into()))?;

        Ok(best)
    }

    /// Get the Pareto-optimal trials for multi-objective studies.
    ///
    /// 对应 Python `Study.best_trials`。
    /// 支持约束优化：只考虑可行试验。
    pub fn best_trials(&self) -> Result<Vec<FrozenTrial>> {
        let trials = self
            .storage
            .get_all_trials(self.study_id, Some(&[TrialState::Complete]))?;

        // 检查是否有约束信息，过滤不可行试验
        let has_constraints = trials.iter().any(|t| {
            t.system_attrs.contains_key("constraints")
        });

        // 对齐 Python: 无 constraints key 的试验视为不可行（unwrap_or(false)）
        let feasible: Vec<FrozenTrial> = if has_constraints {
            trials.into_iter().filter(|t| {
                t.system_attrs.get("constraints")
                    .and_then(|v| serde_json::from_value::<Vec<f64>>(v.clone()).ok())
                    .map(|cs| cs.iter().all(|c| *c <= 0.0))
                    .unwrap_or(false)
            }).collect()
        } else {
            trials
        };

        Ok(crate::multi_objective::get_pareto_front_trials(
            &feasible,
            &self.directions,
        ))
    }

    /// Get the best value for single-objective studies.
    pub fn best_value(&self) -> Result<f64> {
        self.best_trial()?
            .value()?
            .ok_or_else(|| OptunaError::ValueError("best trial has no value".into()))
    }

    /// Get the best params for single-objective studies.
    pub fn best_params(&self) -> Result<HashMap<String, crate::distributions::ParamValue>> {
        Ok(self.best_trial()?.params)
    }

    /// Get all trials.
    pub fn trials(&self) -> Result<Vec<FrozenTrial>> {
        self.storage.get_all_trials(self.study_id, None)
    }

    /// Get trials filtered by state.
    pub fn get_trials(&self, states: Option<&[TrialState]>) -> Result<Vec<FrozenTrial>> {
        self.storage.get_all_trials(self.study_id, states)
    }

    /// 将试验数据转换为 polars DataFrame。
    ///
    /// 对应 Python `Study.trials_dataframe(attrs, multi_index)`。
    /// 需要启用 `dataframe` feature。
    ///
    /// # 参数
    /// * `metric_names` - 目标名（多目标场景）
    /// * `attrs` - 要包含的列名。`None` 使用默认列。
    /// * `multi_index` - 是否使用 MultiIndex 风格列名
    #[cfg(feature = "dataframe")]
    pub fn trials_dataframe(
        &self,
        metric_names: Option<&[String]>,
        attrs: Option<&[&str]>,
        multi_index: bool,
    ) -> Result<polars::prelude::DataFrame> {
        let trials = self.trials()?;
        let multi_objective = self.directions().len() > 1;
        super::dataframe::trials_to_dataframe(&trials, multi_objective, metric_names, attrs, multi_index)
    }

    /// Get user attributes.
    pub fn user_attrs(&self) -> Result<HashMap<String, serde_json::Value>> {
        self.storage.get_study_user_attrs(self.study_id)
    }

    /// Set a user attribute.
    pub fn set_user_attr(&self, key: &str, value: serde_json::Value) -> Result<()> {
        self.storage
            .set_study_user_attr(self.study_id, key, value)
    }

    /// 获取系统属性。
    ///
    /// 对应 Python `Study.system_attrs`。
    pub fn system_attrs(&self) -> Result<HashMap<String, serde_json::Value>> {
        self.storage.get_study_system_attrs(self.study_id)
    }

    /// 设置系统属性。
    ///
    /// 对应 Python `Study.set_system_attr()`。
    pub fn set_system_attr(&self, key: &str, value: serde_json::Value) -> Result<()> {
        self.storage
            .set_study_system_attr(self.study_id, key, value)
    }

    /// 设置度量指标名称。
    ///
    /// 对应 Python `Study.set_metric_names()`。
    /// 将度量名称存储在系统属性 `study:metric_names` 中。
    pub fn set_metric_names(&self, metric_names: &[&str]) -> Result<()> {
        let names: Vec<String> = metric_names.iter().map(|s| s.to_string()).collect();
        if names.len() != self.directions.len() {
            return Err(OptunaError::ValueError(format!(
                "metric_names 长度 ({}) 必须与 directions 长度 ({}) 一致",
                names.len(),
                self.directions.len()
            )));
        }
        self.storage.set_study_system_attr(
            self.study_id,
            "study:metric_names",
            serde_json::json!(names),
        )
    }

    /// 获取度量指标名称。
    ///
    /// 对应 Python `Study.metric_names`。
    pub fn metric_names(&self) -> Result<Option<Vec<String>>> {
        let attrs = self.storage.get_study_system_attrs(self.study_id)?;
        match attrs.get("study:metric_names") {
            Some(v) => {
                let names: Vec<String> = serde_json::from_value(v.clone())
                    .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?;
                Ok(Some(names))
            }
            None => Ok(None),
        }
    }

    /// Signal the optimize loop to stop after the current trial.
    pub fn stop(&self) {
        self.stop_flag.store(true, Ordering::Release);
    }

    // ── Ask / Tell ──────────────────────────────────────────────────────

    /// Create a new trial and return a mutable handle.
    ///
    /// If `fixed_distributions` is provided, the trial's parameters will
    /// be pre-sampled for those distributions.
    pub fn ask(
        &self,
        fixed_distributions: Option<&HashMap<String, Distribution>>,
    ) -> Result<Trial> {
        // Try to pop a WAITING trial first
        let trial_id = match self.pop_waiting_trial()? {
            Some(tid) => tid,
            None => self.storage.create_new_trial(self.study_id, None)?,
        };

        let trial = self.storage.get_trial(trial_id)?;
        let all_trials = self.storage.get_all_trials(self.study_id, None)?;

        // Run sampler hooks
        self.sampler.before_trial(&all_trials);

        // Determine search space: fixed_distributions override sampler's relative space
        let search_space = if let Some(fixed) = fixed_distributions {
            fixed.clone()
        } else {
            self.sampler.infer_relative_search_space(&all_trials)
        };

        // Sample relative params
        let fixed_params = trial
            .system_attrs
            .get("fixed_params")
            .map(|fp| serde_json::from_value::<HashMap<String, crate::distributions::ParamValue>>(fp.clone()))
            .transpose()
            .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?
            .unwrap_or_default();

        let mut relative_params = if !search_space.is_empty() {
            self.sampler
                .sample_relative(&all_trials, &search_space)?
        } else {
            HashMap::new()
        };

        // For any distributions not covered by relative sampling,
        // do independent sampling
        if let Some(fixed) = fixed_distributions {
            for (name, dist) in fixed {
                if !relative_params.contains_key(name) {
                    let all_trials = self.storage.get_all_trials(self.study_id, None)?;
                    let v = self.sampler.sample_independent(&all_trials, &trial, name, dist)?;
                    relative_params.insert(name.clone(), v);
                }
            }
        }

        Ok(Trial::new(
            trial_id,
            self.study_id,
            trial.number,
            Arc::clone(&self.storage),
            Arc::clone(&self.sampler),
            Arc::clone(&self.pruner),
            self.directions.clone(),
            relative_params,
            search_space,
            fixed_params,
        ))
    }

    /// Finalize a trial with a value/values and state.
    ///
    /// 对应 Python `Study.tell()`。
    /// `skip_if_finished` 为 true 时，若试验已结束则跳过。
    pub fn tell(
        &self,
        trial_id: i64,
        state: TrialState,
        values: Option<&[f64]>,
    ) -> Result<FrozenTrial> {
        self.tell_with_options(trial_id, state, values, false)
    }

    /// 带 skip_if_finished 选项的 tell。
    ///
    /// 对应 Python `Study.tell(skip_if_finished=True)`。
    /// 完整对齐 Python `_tell_with_warning` 的验证逻辑：
    /// 1. skip_if_finished 检查
    /// 2. 非 RUNNING 试验不允许 tell
    /// 3. state/values 一致性校验 (_check_state_and_values)
    /// 4. values 可行性校验: NaN 检查、数量匹配 (_check_values_are_feasible)
    /// 5. PRUNED 状态自动使用最后中间值
    /// 6. after_trial 在 set_trial_state_values 之前执行
    pub fn tell_with_options(
        &self,
        trial_id: i64,
        state: TrialState,
        values: Option<&[f64]>,
        skip_if_finished: bool,
    ) -> Result<FrozenTrial> {
        let frozen = self.storage.get_trial(trial_id)?;

        // 1. skip_if_finished 检查
        if frozen.state.is_finished() && skip_if_finished {
            return Ok(frozen);
        }

        // 2. 非 RUNNING 试验不允许 tell（对齐 Python）
        if frozen.state != TrialState::Running {
            return Err(OptunaError::ValueError(format!(
                "Cannot tell a {} trial.",
                frozen.state
            )));
        }

        // 3. state/values 一致性校验（对齐 Python _check_state_and_values）
        match state {
            TrialState::Complete => {
                if values.is_none() {
                    return Err(OptunaError::ValueError(
                        "No values were told. Values are required when state is TrialState.Complete.".into(),
                    ));
                }
            }
            TrialState::Pruned | TrialState::Fail => {
                // 对齐 Python: Pruned/Fail 时忽略 values（如果传了就丢弃）
                // Python 会发出 UserWarning 但不报错
            }
            TrialState::Running | TrialState::Waiting => {
                return Err(OptunaError::ValueError(format!(
                    "Cannot tell with state {}.",
                    state
                )));
            }
        }

        // 4. values 可行性校验（对齐 Python _check_values_are_feasible）
        let final_values: Option<Vec<f64>>;
        if state == TrialState::Complete {
            let vals = values.unwrap(); // 已确保 Some
            // NaN 检查
            for v in vals {
                if v.is_nan() {
                    return Err(OptunaError::ValueError(format!(
                        "The value {} is not acceptable",
                        v
                    )));
                }
            }
            // values 数量与 directions 数量匹配
            if vals.len() != self.directions.len() {
                return Err(OptunaError::ValueError(format!(
                    "The number of the values {} did not match the number of the objectives {}",
                    vals.len(),
                    self.directions.len()
                )));
            }
            final_values = Some(vals.to_vec());
        } else if state == TrialState::Pruned {
            // 对齐 Python: PRUNED 状态自动使用最后中间值（如果可行）
            if let Some(last_step) = frozen.last_step() {
                let last_val = frozen.intermediate_values[&last_step];
                if !last_val.is_nan() && self.directions.len() == 1 {
                    final_values = Some(vec![last_val]);
                } else {
                    final_values = None;
                }
            } else {
                final_values = None;
            }
        } else {
            final_values = None;
        }

        // 5. after_trial 在 set_trial_state_values 之前执行（对齐 Python try/finally 顺序）
        let all_trials = self.storage.get_all_trials(self.study_id, None)?;
        self.sampler.after_trial(
            &all_trials,
            &frozen,
            state,
            final_values.as_deref(),
        );

        // 6. set_trial_state_values 放在 finally 等价位置
        self.storage
            .set_trial_state_values(trial_id, state, final_values.as_deref())?;

        let result = self.storage.get_trial(trial_id)?;
        Ok(result)
    }

    // ── Optimize loop ───────────────────────────────────────────────────

    /// 运行优化循环（简化版）。
    ///
    /// 对应 Python `optuna.Study.optimize()` 的最常见用法。
    /// 如需 `n_jobs`/`catch`/`show_progress_bar`，请使用 `optimize_with_options()`。
    /// 支持 Ctrl+C 优雅终止。
    pub fn optimize<F>(
        &self,
        func: F,
        n_trials: Option<usize>,
        timeout: Option<Duration>,
        callbacks: Option<&[&dyn Callback]>,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<f64>,
    {
        self.stop_flag.store(false, Ordering::Release);
        SIGINT_RECEIVED.store(false, Ordering::SeqCst);
        install_signal_handler(&self.stop_flag);
        let start = Instant::now();
        let mut i_trial: usize = 0;
        loop {
            if self.stop_flag.load(Ordering::Acquire) { break; }
            if n_trials.is_some_and(|n| i_trial >= n) { break; }
            if timeout.is_some_and(|t| start.elapsed() >= t) { break; }
            self.run_trial(&func, callbacks)?;
            i_trial += 1;
        }
        self.storage.remove_session();
        Ok(())
    }

    /// 带完整选项的优化循环（对应 Python `optimize` 的所有参数）。
    /// 支持 Ctrl+C 优雅终止。
    pub fn optimize_with_options<F>(
        &self,
        func: F,
        n_trials: Option<usize>,
        timeout: Option<Duration>,
        n_jobs: i32,
        catch: &[&str],
        callbacks: Option<&[&dyn Callback]>,
        show_progress_bar: bool,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<f64> + Send + Sync,
    {
        self.stop_flag.store(false, Ordering::Release);
        SIGINT_RECEIVED.store(false, Ordering::SeqCst);
        install_signal_handler(&self.stop_flag);
        let start = Instant::now();

        // 进度条支持
        #[cfg(feature = "progress")]
        let progress: Option<Arc<indicatif::ProgressBar>> = if show_progress_bar {
            let pb = indicatif::ProgressBar::new(n_trials.unwrap_or(0) as u64);
            pb.set_style(
                indicatif::ProgressStyle::default_bar()
                    .template("{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} trials ({eta})")
                    .unwrap_or_else(|_| indicatif::ProgressStyle::default_bar())
                    .progress_chars("#>-"),
            );
            if n_trials.is_none() {
                pb.set_length(0);
                pb.set_style(
                    indicatif::ProgressStyle::default_spinner()
                        .template("{spinner:.green} [{elapsed_precise}] {pos} trials")
                        .unwrap_or_else(|_| indicatif::ProgressStyle::default_spinner()),
                );
            }
            Some(Arc::new(pb))
        } else {
            None
        };
        #[cfg(not(feature = "progress"))]
        let _ = show_progress_bar;

        // 确定实际线程数
        let actual_jobs = if n_jobs == -1 {
            std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(1)
        } else {
            n_jobs.max(1)
        };

        if actual_jobs <= 1 {
            // ── 串行模式 ──
            let mut i_trial: usize = 0;
            loop {
                if self.stop_flag.load(Ordering::Acquire) {
                    break;
                }
                if n_trials.is_some_and(|n| i_trial >= n) {
                    break;
                }
                if timeout.is_some_and(|t| start.elapsed() >= t) {
                    break;
                }

                match self.run_trial_with_catch(&func, callbacks, catch) {
                    Ok(()) => {}
                    Err(e) => {
                        // catch 未匹配的异常向上传播
                        self.storage.remove_session();
                        return Err(e);
                    }
                }
                i_trial += 1;

                #[cfg(feature = "progress")]
                if let Some(ref pb) = progress {
                    pb.inc(1);
                }
            }
        } else {
            // ── 并行模式 (类似 Python ThreadPoolExecutor) ──
            use std::sync::atomic::AtomicUsize;
            let counter = Arc::new(AtomicUsize::new(0));
            let func_ref = &func;
            let catch_ref = catch;

            std::thread::scope(|scope| -> Result<()> {
                let mut handles = Vec::new();

                // 启动 actual_jobs 个工作线程
                for _ in 0..actual_jobs {
                    let counter = counter.clone();
                    #[cfg(feature = "progress")]
                    let progress = progress.clone();
                    let handle = scope.spawn(move || -> Result<()> {
                        loop {
                            // 原子递增计数器，检查是否超过 n_trials
                            let idx = counter.fetch_add(1, Ordering::SeqCst);
                            if n_trials.is_some_and(|n| idx >= n) {
                                break;
                            }
                            if self.stop_flag.load(Ordering::Acquire) {
                                break;
                            }
                            if timeout.is_some_and(|t| start.elapsed() >= t) {
                                break;
                            }

                            match self.run_trial_with_catch(func_ref, callbacks, catch_ref) {
                                Ok(()) => {}
                                Err(e) => return Err(e),
                            }

                            #[cfg(feature = "progress")]
                            if let Some(ref pb) = progress {
                                pb.inc(1);
                            }
                        }
                        Ok(())
                    });
                    handles.push(handle);
                }

                // 等待所有线程完成
                for handle in handles {
                    handle.join().map_err(|_| {
                        OptunaError::StorageInternalError("Worker thread panicked".into())
                    })??;
                }
                Ok(())
            })?;
        }

        #[cfg(feature = "progress")]
        if let Some(pb) = progress {
            pb.finish_with_message("optimization complete");
        }

        self.storage.remove_session();
        Ok(())
    }

    /// 运行多目标优化循环（简化版）。
    pub fn optimize_multi<F>(
        &self,
        func: F,
        n_trials: Option<usize>,
        timeout: Option<Duration>,
        callbacks: Option<&[&dyn Callback]>,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<Vec<f64>>,
    {
        self.stop_flag.store(false, Ordering::Release);
        let start = Instant::now();
        let mut i_trial: usize = 0;
        loop {
            if self.stop_flag.load(Ordering::Acquire) { break; }
            if n_trials.is_some_and(|n| i_trial >= n) { break; }
            if timeout.is_some_and(|t| start.elapsed() >= t) { break; }
            self.run_trial_multi(&func, callbacks)?;
            i_trial += 1;
        }
        self.storage.remove_session();
        Ok(())
    }

    /// 带完整选项的多目标优化循环。
    pub fn optimize_multi_with_options<F>(
        &self,
        func: F,
        n_trials: Option<usize>,
        timeout: Option<Duration>,
        n_jobs: i32,
        catch: &[&str],
        callbacks: Option<&[&dyn Callback]>,
        show_progress_bar: bool,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<Vec<f64>> + Send + Sync,
    {
        self.stop_flag.store(false, Ordering::Release);
        let start = Instant::now();

        #[cfg(feature = "progress")]
        let progress: Option<Arc<indicatif::ProgressBar>> = if show_progress_bar {
            let pb = indicatif::ProgressBar::new(n_trials.unwrap_or(0) as u64);
            if n_trials.is_none() {
                pb.set_style(
                    indicatif::ProgressStyle::default_spinner()
                        .template("{spinner:.green} [{elapsed_precise}] {pos} trials")
                        .unwrap_or_else(|_| indicatif::ProgressStyle::default_spinner()),
                );
            }
            Some(Arc::new(pb))
        } else {
            None
        };
        #[cfg(not(feature = "progress"))]
        let _ = show_progress_bar;

        let actual_jobs = if n_jobs == -1 {
            std::thread::available_parallelism()
                .map(|n| n.get() as i32)
                .unwrap_or(1)
        } else {
            n_jobs.max(1)
        };

        if actual_jobs <= 1 {
            let mut i_trial: usize = 0;
            loop {
                if self.stop_flag.load(Ordering::Acquire) { break; }
                if n_trials.is_some_and(|n| i_trial >= n) { break; }
                if timeout.is_some_and(|t| start.elapsed() >= t) { break; }

                match self.run_trial_multi_with_catch(&func, callbacks, catch) {
                    Ok(()) => {}
                    Err(e) => {
                        self.storage.remove_session();
                        return Err(e);
                    }
                }
                i_trial += 1;

                #[cfg(feature = "progress")]
                if let Some(ref pb) = progress {
                    pb.inc(1);
                }
            }
        } else {
            use std::sync::atomic::AtomicUsize;
            let counter = Arc::new(AtomicUsize::new(0));
            let func_ref = &func;
            let catch_ref = catch;

            std::thread::scope(|scope| -> Result<()> {
                let mut handles = Vec::new();
                for _ in 0..actual_jobs {
                    let counter = counter.clone();
                    #[cfg(feature = "progress")]
                    let progress = progress.clone();
                    let handle = scope.spawn(move || -> Result<()> {
                        loop {
                            let idx = counter.fetch_add(1, Ordering::SeqCst);
                            if n_trials.is_some_and(|n| idx >= n) { break; }
                            if self.stop_flag.load(Ordering::Acquire) { break; }
                            if timeout.is_some_and(|t| start.elapsed() >= t) { break; }

                            match self.run_trial_multi_with_catch(func_ref, callbacks, catch_ref) {
                                Ok(()) => {}
                                Err(e) => return Err(e),
                            }

                            #[cfg(feature = "progress")]
                            if let Some(ref pb) = progress {
                                pb.inc(1);
                            }
                        }
                        Ok(())
                    });
                    handles.push(handle);
                }
                for handle in handles {
                    handle.join().map_err(|_| {
                        OptunaError::StorageInternalError("Worker thread panicked".into())
                    })??;
                }
                Ok(())
            })?;
        }

        #[cfg(feature = "progress")]
        if let Some(pb) = progress {
            pb.finish_with_message("optimization complete");
        }

        self.storage.remove_session();
        Ok(())
    }

    /// Run the optimization loop with terminator support.
    ///
    /// Like [`optimize`](Self::optimize), but also checks terminators after
    /// each trial. If any terminator returns `true`, optimization stops.
    pub fn optimize_with_terminators<F>(
        &self,
        func: F,
        n_trials: Option<usize>,
        timeout: Option<Duration>,
        callbacks: Option<&[&dyn Callback]>,
        terminators: Option<&[Arc<dyn Terminator>]>,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<f64>,
    {
        self.stop_flag.store(false, Ordering::Release);
        let start = Instant::now();

        let mut i_trial: usize = 0;
        loop {
            if self.stop_flag.load(Ordering::Acquire) {
                break;
            }
            if n_trials.is_some_and(|n| i_trial >= n) {
                break;
            }
            if timeout.is_some_and(|t| start.elapsed() >= t) {
                break;
            }
            if let Some(terms) = terminators
                && terms.iter().any(|t| t.should_terminate(self))
            {
                break;
            }

            self.run_trial(&func, callbacks)?;
            i_trial += 1;
        }

        self.storage.remove_session();
        Ok(())
    }

    /// Run the multi-objective optimization loop with terminator support.
    pub fn optimize_multi_with_terminators<F>(
        &self,
        func: F,
        n_trials: Option<usize>,
        timeout: Option<Duration>,
        callbacks: Option<&[&dyn Callback]>,
        terminators: Option<&[Arc<dyn Terminator>]>,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<Vec<f64>>,
    {
        self.stop_flag.store(false, Ordering::Release);
        let start = Instant::now();

        let mut i_trial: usize = 0;
        loop {
            if self.stop_flag.load(Ordering::Acquire) {
                break;
            }
            if n_trials.is_some_and(|n| i_trial >= n) {
                break;
            }
            if timeout.is_some_and(|t| start.elapsed() >= t) {
                break;
            }
            if let Some(terms) = terminators
                && terms.iter().any(|t| t.should_terminate(self))
            {
                break;
            }

            self.run_trial_multi(&func, callbacks)?;
            i_trial += 1;
        }

        self.storage.remove_session();
        Ok(())
    }

    /// 执行单次试验（单目标），支持 catch 异常捕获。
    ///
    /// 对应 Python `_run_trial()`。
    /// `catch` 中列出的异常类型不会中断优化（试验标记为 Fail)。
    /// 不在 `catch` 中的异常会向上传播。
    fn run_trial_with_catch<F>(
        &self,
        func: &F,
        callbacks: Option<&[&dyn Callback]>,
        catch: &[&str],
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<f64>,
    {
        let mut trial = self.ask(None)?;
        let trial_id = trial.trial_id();

        // 保存异常信息用于后续 catch 判断
        let mut func_err: Option<OptunaError> = None;

        let (state, values) = match func(&mut trial) {
            Ok(value) => {
                if value.is_nan() {
                    // NaN → FAIL（对齐 Python _tell_with_warning suppress_warning=True）
                    (TrialState::Fail, None)
                } else {
                    (TrialState::Complete, Some(vec![value]))
                }
            }
            Err(OptunaError::TrialPruned) => {
                // Pruned: tell() 会自动提取最后中间值，不传 values
                (TrialState::Pruned, None)
            }
            Err(e) => {
                func_err = Some(e);
                (TrialState::Fail, None)
            }
        };

        let frozen = self.tell(trial_id, state, values.as_deref())?;

        // 运行回调
        if let Some(cbs) = callbacks {
            for cb in cbs {
                cb.on_trial_complete(self, &frozen);
            }
        }

        // catch 判断: 如果异常不在 catch 列表中，则重新抛出
        if state == TrialState::Fail {
            if let Some(err) = func_err {
                if !error_matches_catch(&err, catch) {
                    return Err(err);
                }
                // 在 catch 列表中 → 静默吞掉，继续优化
            }
        }

        Ok(())
    }

    /// 执行单次多目标试验，支持 catch 异常捕获。
    fn run_trial_multi_with_catch<F>(
        &self,
        func: &F,
        callbacks: Option<&[&dyn Callback]>,
        catch: &[&str],
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<Vec<f64>>,
    {
        let mut trial = self.ask(None)?;
        let trial_id = trial.trial_id();

        let mut func_err: Option<OptunaError> = None;

        let (state, values) = match func(&mut trial) {
            Ok(vals) => {
                if vals.iter().any(|v| v.is_nan())
                    || vals.len() != self.directions.len()
                {
                    // NaN 或数量不匹配 → FAIL
                    (TrialState::Fail, None)
                } else {
                    (TrialState::Complete, Some(vals))
                }
            }
            Err(OptunaError::TrialPruned) => {
                // Pruned: tell() 会自动提取最后中间值
                (TrialState::Pruned, None)
            }
            Err(e) => {
                func_err = Some(e);
                (TrialState::Fail, None)
            }
        };

        let frozen = self.tell(trial_id, state, values.as_deref())?;

        if let Some(cbs) = callbacks {
            for cb in cbs {
                cb.on_trial_complete(self, &frozen);
            }
        }

        if state == TrialState::Fail {
            if let Some(err) = func_err {
                if !error_matches_catch(&err, catch) {
                    return Err(err);
                }
            }
        }

        Ok(())
    }

    /// 执行单次试验 (向后兼容的简化版本)。
    fn run_trial<F>(
        &self,
        func: &F,
        callbacks: Option<&[&dyn Callback]>,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<f64>,
    {
        self.run_trial_with_catch(func, callbacks, &[])
    }

    /// 执行单次多目标试验 (向后兼容的简化版本)。
    fn run_trial_multi<F>(
        &self,
        func: &F,
        callbacks: Option<&[&dyn Callback]>,
    ) -> Result<()>
    where
        F: Fn(&mut Trial) -> Result<Vec<f64>>,
    {
        self.run_trial_multi_with_catch(func, callbacks, &[])
    }

    /// 尝试弹出一个 WAITING 试验并设为 RUNNING。
    /// 尝试弹出一个 WAITING 试验并设为 RUNNING。
    ///
    /// 对齐 Python `_pop_waiting_trial_id`:
    /// 遍历所有 WAITING 试验，逐一尝试 CAS 设为 RUNNING。
    /// 如果某个已被其他线程占用（返回 false）或已完成，继续尝试下一个。
    fn pop_waiting_trial(&self) -> Result<Option<i64>> {
        let waiting = self
            .storage
            .get_all_trials(self.study_id, Some(&[TrialState::Waiting]))?;
        for t in &waiting {
            match self.storage.set_trial_state_values(
                t.trial_id,
                TrialState::Running,
                None,
            ) {
                Ok(true) => return Ok(Some(t.trial_id)),
                Ok(false) => continue, // 已被其他线程占用
                Err(OptunaError::UpdateFinishedTrialError(_)) => continue, // 已完成
                Err(e) => return Err(e),
            }
        }
        Ok(None)
    }
}

/// Add a trial to a study directly.
impl Study {
    /// Add a trial to this study.
    ///
    /// 对齐 Python `Study.add_trial()`:
    /// 1. 调用 trial.validate() 验证不变量
    /// 2. 检查 values 数量与 directions 一致
    pub fn add_trial(&self, trial: &FrozenTrial) -> Result<()> {
        trial.validate()?;
        if let Some(vals) = &trial.values {
            if self.directions.len() != vals.len() {
                return Err(OptunaError::ValueError(format!(
                    "The number of the values {} did not match the number of the objectives {}",
                    vals.len(),
                    self.directions.len()
                )));
            }
        }
        self.storage
            .create_new_trial(self.study_id, Some(trial))?;
        Ok(())
    }

    pub fn add_trials(&self, trials: &[FrozenTrial]) -> Result<()> {
        for trial in trials {
            self.add_trial(trial)?;
        }
        Ok(())
    }

    pub fn enqueue_trial(
        &self,
        params: HashMap<String, crate::distributions::ParamValue>,
        user_attrs: Option<HashMap<String, serde_json::Value>>,
        skip_if_exists: bool,
    ) -> Result<()> {
        // skip_if_exists: 对齐 Python — 检查是否已有相同参数的试验（任意状态）
        if skip_if_exists {
            let all_trials = self.storage.get_all_trials(
                self.study_id,
                None,  // 所有状态
            )?;
            for t in &all_trials {
                if let Some(fp) = t.system_attrs.get("fixed_params") {
                    if let Ok(existing) = serde_json::from_value::<HashMap<String, crate::distributions::ParamValue>>(fp.clone()) {
                        if existing == params {
                            return Ok(()); // 已存在相同参数的试验
                        }
                    }
                }
            }
        }

        let mut template = FrozenTrial {
            number: 0,
            state: TrialState::Waiting,
            values: None,
            datetime_start: None,
            datetime_complete: None,
            params,
            distributions: HashMap::new(),
            user_attrs: user_attrs.unwrap_or_default(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        };
        // Distributions will be set when the trial is actually run
        // Clear params for WAITING trials (they'll be suggested later)
        let enqueued_params = template.params.clone();
        template.params = HashMap::new();
        template.system_attrs.insert(
            "fixed_params".to_string(),
            serde_json::to_value(enqueued_params)
                .map_err(|e| OptunaError::StorageInternalError(e.to_string()))?,
        );
        self.storage
            .create_new_trial(self.study_id, Some(&template))?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::InMemoryStorage;
    use crate::study::create_study;

    #[test]
    fn test_create_study_default() {
        let study = create_study(None, None, None, None, None, None, false).unwrap();
        assert_eq!(study.directions(), &[StudyDirection::Minimize]);
    }

    #[test]
    fn test_create_study_named() {
        let study =
            create_study(None, None, None, Some("my-study"), None, None, false).unwrap();
        assert_eq!(study.study_name(), "my-study");
    }

    #[test]
    fn test_create_study_maximize() {
        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Maximize),
            None,
            false,
        )
        .unwrap();
        assert_eq!(study.direction().unwrap(), StudyDirection::Maximize);
    }

    #[test]
    fn test_create_study_both_directions_errors() {
        assert!(create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Minimize),
            Some(vec![StudyDirection::Maximize]),
            false,
        )
        .is_err());
    }

    #[test]
    fn test_create_study_load_if_exists() {
        let storage: Arc<dyn Storage> = Arc::new(InMemoryStorage::new());
        let _s1 = create_study(
            Some(Arc::clone(&storage)),
            None,
            None,
            Some("dup"),
            None,
            None,
            false,
        )
        .unwrap();
        // Without load_if_exists, duplicate errors
        assert!(create_study(
            Some(Arc::clone(&storage)),
            None,
            None,
            Some("dup"),
            None,
            None,
            false,
        )
        .is_err());
        // With load_if_exists, succeeds
        let s2 = create_study(
            Some(Arc::clone(&storage)),
            None,
            None,
            Some("dup"),
            None,
            None,
            true,
        )
        .unwrap();
        assert_eq!(s2.study_name(), "dup");
    }

    #[test]
    fn test_optimize_quadratic() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                    let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
                    Ok(x * x + y * y)
                },
                Some(50),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 50);
        assert!(trials.iter().all(|t| t.state == TrialState::Complete));

        let best = study.best_value().unwrap();
        // With 50 random trials in [-10, 10]^2, best should be reasonably small
        assert!(best < 50.0, "best value {best} is too large");
    }

    #[test]
    fn test_optimize_maximize() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Maximize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x)
                },
                Some(20),
                None,
                None,
            )
            .unwrap();

        let best = study.best_value().unwrap();
        assert!(best > 0.5, "best value {best} should be > 0.5 for maximize");
    }

    #[test]
    fn test_optimize_with_pruning() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                    trial.report(x * x, 0)?;
                    if x.abs() > 5.0 {
                        return Err(OptunaError::TrialPruned);
                    }
                    Ok(x * x)
                },
                Some(30),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 30);
        let n_pruned = trials
            .iter()
            .filter(|t| t.state == TrialState::Pruned)
            .count();
        let n_complete = trials
            .iter()
            .filter(|t| t.state == TrialState::Complete)
            .count();
        assert!(n_pruned > 0, "expected some pruned trials");
        assert!(n_complete > 0, "expected some complete trials");
        assert_eq!(n_pruned + n_complete, 30);
    }

    #[test]
    fn test_ask_tell_lifecycle() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let mut trial = study.ask(None).unwrap();
        let x = trial.suggest_float("x", 0.0, 1.0, false, None).unwrap();
        let value = x * x;

        let frozen = study
            .tell(trial.trial_id(), TrialState::Complete, Some(&[value]))
            .unwrap();
        assert_eq!(frozen.state, TrialState::Complete);
        assert_eq!(frozen.values, Some(vec![value]));
    }

    #[test]
    fn test_study_stop() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let study = Arc::new(
            create_study(
                None,
                Some(sampler),
                None,
                None,
                Some(StudyDirection::Minimize),
                None,
                false,
            )
            .unwrap(),
        );

        // Stop after first trial via callback-like logic
        // We'll test the stop flag directly
        let study_ref = Arc::clone(&study);
        std::thread::spawn(move || {
            // Wait a tiny bit then stop
            std::thread::sleep(std::time::Duration::from_millis(10));
            study_ref.stop();
        });

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x * x)
                },
                Some(10000), // Would take a long time without stop
                None,
                None,
            )
            .unwrap();

        // Should have been stopped early
        let n_trials = study.trials().unwrap().len();
        assert!(n_trials < 10000, "study should have stopped early, got {n_trials} trials");
    }

    #[test]
    fn test_optimize_with_int_param() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let n = trial.suggest_int("n_layers", 1, 5, false, 1)?;
                    Ok((n as f64 - 3.0).powi(2))
                },
                Some(20),
                None,
                None,
            )
            .unwrap();

        let best = study.best_trial().unwrap();
        assert_eq!(best.state, TrialState::Complete);
    }

    #[test]
    fn test_optimize_timeout() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let start = std::time::Instant::now();
        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
                    Ok(x * x)
                },
                None, // unlimited trials
                Some(std::time::Duration::from_millis(100)),
                None,
            )
            .unwrap();

        let elapsed = start.elapsed();
        assert!(elapsed < std::time::Duration::from_secs(2));
        assert!(!study.trials().unwrap().is_empty());
    }

    #[test]
    fn test_best_params() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    Ok(x * x)
                },
                Some(30),
                None,
                None,
            )
            .unwrap();

        let params = study.best_params().unwrap();
        assert!(params.contains_key("x"));
    }

    #[test]
    fn test_study_user_attrs() {
        let study = create_study(None, None, None, None, None, None, false).unwrap();
        study
            .set_user_attr("key", serde_json::json!("value"))
            .unwrap();
        let attrs = study.user_attrs().unwrap();
        assert_eq!(attrs.get("key"), Some(&serde_json::json!("value")));
    }

    #[test]
    fn test_optimize_with_median_pruner() {
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(42)));
        let pruner: Arc<dyn crate::pruners::Pruner> = Arc::new(
            crate::pruners::MedianPruner::new(3, 0, 1, 1, StudyDirection::Minimize),
        );
        let study = create_study(
            None,
            Some(sampler),
            Some(pruner),
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
                    for step in 0..5 {
                        let v = x * x + step as f64;
                        trial.report(v, step)?;
                        if trial.should_prune()? {
                            return Err(OptunaError::TrialPruned);
                        }
                    }
                    Ok(x * x)
                },
                Some(50),
                None,
                None,
            )
            .unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 50);
        let n_pruned = trials.iter().filter(|t| t.state == TrialState::Pruned).count();
        let n_complete = trials.iter().filter(|t| t.state == TrialState::Complete).count();
        assert!(n_complete > 0, "expected some complete trials");
        // With median pruner active after 3 startup trials, some should be pruned
        assert!(
            n_pruned > 0,
            "expected some pruned trials with median pruner, got {n_pruned} pruned / {n_complete} complete"
        );
    }

    #[test]
    fn test_end_to_end_with_search_space_transform() {
        // Verify the full pipeline: RandomSampler -> SearchSpaceTransform -> optimize
        // with mixed param types
        let sampler: Arc<dyn Sampler> = Arc::new(crate::samplers::RandomSampler::new(Some(99)));
        let study = create_study(
            None,
            Some(sampler),
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        study
            .optimize(
                |trial| {
                    let x = trial.suggest_float("x", -5.0, 5.0, false, None)?;
                    let n = trial.suggest_int("n", 1, 10, false, 1)?;
                    Ok(x * x + (n as f64 - 5.0).powi(2))
                },
                Some(40),
                None,
                None,
            )
            .unwrap();

        let best = study.best_trial().unwrap();
        assert_eq!(best.state, TrialState::Complete);
        assert!(best.params.contains_key("x"));
        assert!(best.params.contains_key("n"));
        // With 40 random trials, should find something reasonable
        let best_val = study.best_value().unwrap();
        assert!(best_val < 30.0, "best value {best_val} should be < 30");
    }

    #[test]
    fn test_study_system_attrs() {
        let study = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        // 设置系统属性
        study.set_system_attr("test_key", serde_json::json!("test_value")).unwrap();
        let attrs = study.system_attrs().unwrap();
        assert_eq!(attrs.get("test_key").unwrap(), &serde_json::json!("test_value"));

        // 覆盖已有属性
        study.set_system_attr("test_key", serde_json::json!(42)).unwrap();
        let attrs = study.system_attrs().unwrap();
        assert_eq!(attrs.get("test_key").unwrap(), &serde_json::json!(42));
    }

    #[test]
    fn test_study_metric_names() {
        let study = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        // 初始无 metric_names
        assert!(study.metric_names().unwrap().is_none());

        // 设置 metric_names（长度必须与 directions 一致）
        study.set_metric_names(&["loss"]).unwrap();
        let names = study.metric_names().unwrap().unwrap();
        assert_eq!(names, vec!["loss"]);
    }

    #[test]
    fn test_trial_attribute_accessors() {
        let study = create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        let mut trial = study.ask(None).unwrap();
        let x = trial.suggest_float("x", 0.0, 1.0, false, None).unwrap();

        // params 访问器
        let params = trial.params().unwrap();
        assert!(params.contains_key("x"));

        // distributions 访问器
        let dists = trial.distributions().unwrap();
        assert!(dists.contains_key("x"));

        // user_attrs 访问器
        trial.set_user_attr("note", serde_json::json!("hello")).unwrap();
        let attrs = trial.user_attrs().unwrap();
        assert_eq!(attrs.get("note").unwrap(), &serde_json::json!("hello"));

        // datetime_start 访问器
        let start = trial.datetime_start().unwrap();
        assert!(start.is_some());

        // Tell 完成
        study.tell(trial.trial_id(), TrialState::Complete, Some(&[x])).unwrap();
    }

    #[test]
    fn test_create_trial_function() {
        use crate::trial::create_trial;
        use crate::distributions::{FloatDistribution, ParamValue};

        let mut params = HashMap::new();
        params.insert("x".to_string(), ParamValue::Float(0.5));

        let mut dists = HashMap::new();
        dists.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap()),
        );

        let trial = create_trial(
            Some(TrialState::Complete),
            Some(0.25),
            None,
            Some(params),
            Some(dists),
            None,
            None,
            None,
        );

        assert_eq!(trial.state, TrialState::Complete);
        assert_eq!(trial.values.as_ref().unwrap()[0], 0.25);
        assert!(trial.params.contains_key("x"));
        assert!(trial.distributions.contains_key("x"));
    }

    #[test]
    fn test_tell_skip_if_finished_returns_existing_trial() {
        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let mut trial = study.ask(None).unwrap();
        let x = trial.suggest_float_default("x", 0.0, 1.0).unwrap();
        let first = study
            .tell(trial.trial_id(), TrialState::Complete, Some(&[x]))
            .unwrap();

        let second = study
            .tell_with_options(trial.trial_id(), TrialState::Complete, Some(&[x]), true)
            .unwrap();
        assert_eq!(first.trial_id, second.trial_id);
        assert_eq!(second.state, TrialState::Complete);
    }

    #[test]
    fn test_ask_with_fixed_distributions_prefills_relative_param() {
        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let mut fixed = HashMap::new();
        fixed.insert(
            "x".to_string(),
            Distribution::FloatDistribution(
                crate::distributions::FloatDistribution::new(0.0, 1.0, false, None).unwrap(),
            ),
        );

        let mut trial = study.ask(Some(&fixed)).unwrap();
        let rel = trial.relative_params_internal();
        assert!(rel.contains_key("x"));

        let x = trial.suggest_float_default("x", 0.0, 1.0).unwrap();
        assert!((0.0..=1.0).contains(&x));
    }

    #[test]
    fn test_enqueue_trial_applies_fixed_params_on_ask() {
        let study = create_study(
            None,
            None,
            None,
            None,
            Some(StudyDirection::Minimize),
            None,
            false,
        )
        .unwrap();

        let mut params = HashMap::new();
        params.insert("x".to_string(), crate::distributions::ParamValue::Float(0.33));
        study.enqueue_trial(params, None, false).unwrap();

        let mut trial = study.ask(None).unwrap();
        let x = trial.suggest_float_default("x", 0.0, 1.0).unwrap();
        assert!((x - 0.33).abs() < 1e-12);
    }
}
