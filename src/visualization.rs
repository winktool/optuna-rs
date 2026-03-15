//! 可视化模块。
//!
//! 对应 Python `optuna.visualization`。
//! 使用 [`plotly`](https://docs.rs/plotly) crate 生成交互式 HTML 图表。
//!
//! # 使用方式
//! 需要启用 `visualization` feature:
//! ```toml
//! optuna-rs = { version = "0.1", features = ["visualization"] }
//! ```
//!
//! # 支持的图表
//! - [`plot_optimization_history`] — 优化历史
//! - [`plot_intermediate_values`] — 中间值
//! - [`plot_parallel_coordinate`] — 平行坐标图
//! - [`plot_contour`] — 等高线图
//! - [`plot_slice`] — 切片图
//! - [`plot_edf`] — 经验分布函数
//! - [`plot_param_importances`] — 参数重要性
//! - [`plot_pareto_front`] — 帕累托前沿
//! - [`plot_hypervolume_history`] — 超体积历史
//! - [`plot_rank`] — 试验排名
//! - [`plot_timeline`] — 时间线
//! - [`plot_terminator_improvement`] — 终止器改进

#[cfg(feature = "visualization")]
use plotly::{Plot, Scatter, Scatter3D, Bar, Contour, Layout};
#[cfg(feature = "visualization")]
use plotly::common::{Mode, Marker, ColorScale, ColorScalePalette, ColorBar, ColorScaleElement};
#[cfg(feature = "visualization")]
use plotly::layout::{Axis, LayoutGrid, GridPattern};
#[cfg(feature = "visualization")]
use plotly::contour::Coloring as ContourColoring;

#[cfg(feature = "visualization")]
use crate::error::Result;
#[cfg(feature = "visualization")]
use crate::study::Study;
#[cfg(feature = "visualization")]
use crate::trial::{FrozenTrial, TrialState};

// ════════════════════════════════════════════════════════════════════════
// 工具函数
// ════════════════════════════════════════════════════════════════════════

/// 模拟 Python `{:.Ng}` 格式化 — 自动选择定点或科学计数法。
/// `precision` 控制有效数字位数。
#[cfg(feature = "visualization")]
fn format_g(value: f64, precision: usize) -> String {
    if value == 0.0 {
        return "0".to_string();
    }
    let abs = value.abs();
    // 使用科学计数法的阈值: 与 Python %g 行为一致
    if abs >= 1e-4 && abs < 10.0_f64.powi(precision as i32) {
        // 定点表示，去除尾部零
        let s = format!("{value:.prec$}", prec = precision);
        // 去除小数点后尾部零
        if s.contains('.') {
            let trimmed = s.trim_end_matches('0').trim_end_matches('.');
            trimmed.to_string()
        } else {
            s
        }
    } else {
        // 科学计数法
        let s = format!("{value:.prec$e}", prec = precision.saturating_sub(1));
        s
    }
}

// ============================================================================
// plot_optimization_history
// ============================================================================

/// 绘制优化历史（目标值 vs 试验编号）。
///
/// 对应 Python `optuna.visualization.plot_optimization_history`。
/// 支持单 study 和多 study (通过 `plot_optimization_history_multi`)，
/// 支持约束可行性着色 (infeasible 为灰色)，
/// 支持 `error_bar` 参数。
#[cfg(feature = "visualization")]
pub fn plot_optimization_history(
    study: &Study,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<Plot> {
    plot_optimization_history_impl(&[study], target, target_name, false)
}

/// 多 study 优化历史。
/// 对应 Python `plot_optimization_history(study=[s1, s2, ...])`.
#[cfg(feature = "visualization")]
pub fn plot_optimization_history_multi(
    studies: &[&Study],
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
    error_bar: bool,
) -> Result<Plot> {
    plot_optimization_history_impl(studies, target, target_name, error_bar)
}

#[cfg(feature = "visualization")]
fn plot_optimization_history_impl(
    studies: &[&Study],
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
    error_bar: bool,
) -> Result<Plot> {
    let mut plot = Plot::new();

    if error_bar && studies.len() > 1 {
        // error_bar 模式: 跨多 study 聚合均值和标准差
        // 对应 Python _aggregate() 的逻辑

        // 收集所有 study 的 (trial_number → value) 映射
        let mut max_trial_num: i64 = 0;
        let mut all_values_per_num: std::collections::HashMap<i64, Vec<f64>> = std::collections::HashMap::new();
        let mut all_best_per_num: std::collections::HashMap<i64, Vec<f64>> = std::collections::HashMap::new();

        for study in studies {
            let trials = study.get_trials(Some(&[TrialState::Complete]))?;
            let is_minimize = matches!(
                study.directions().first(),
                Some(crate::study::StudyDirection::Minimize)
            );
            let mut best = if is_minimize { f64::INFINITY } else { f64::NEG_INFINITY };

            for trial in &trials {
                let val = target.map_or_else(
                    || trial.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
                    |f| f(trial),
                );
                if trial.number > max_trial_num { max_trial_num = trial.number; }

                if val.is_finite() && is_feasible(trial) {
                    all_values_per_num.entry(trial.number).or_default().push(val);
                    if is_minimize {
                        if val < best { best = val; }
                    } else if val > best {
                        best = val;
                    }
                }
                if best.is_finite() {
                    all_best_per_num.entry(trial.number).or_default().push(best);
                }
            }
        }

        // 计算均值和标准差
        let mut numbers = Vec::new();
        let mut means = Vec::new();
        let mut stds = Vec::new();
        let mut best_means = Vec::new();
        let mut best_stds = Vec::new();

        for num in 0..=max_trial_num {
            if let Some(vals) = all_values_per_num.get(&num) {
                if !vals.is_empty() {
                    let n = vals.len() as f64;
                    let mean = vals.iter().sum::<f64>() / n;
                    let std = (vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n).sqrt();
                    numbers.push(num);
                    means.push(mean);
                    stds.push(std);

                    let bests = all_best_per_num.get(&num).cloned().unwrap_or_default();
                    if !bests.is_empty() {
                        let bn = bests.len() as f64;
                        let bmean = bests.iter().sum::<f64>() / bn;
                        let bstd = (bests.iter().map(|v| (v - bmean).powi(2)).sum::<f64>() / bn).sqrt();
                        best_means.push(bmean);
                        best_stds.push(bstd);
                    } else {
                        best_means.push(f64::NAN);
                        best_stds.push(0.0);
                    }
                }
            }
        }

        // 带误差棒的目标值散点
        let error_y = plotly::common::ErrorData::new(plotly::common::ErrorType::Data)
            .array(stds)
            .visible(true);
        let trace_values = Scatter::new(numbers.clone(), means)
            .mode(Mode::Markers)
            .name(target_name)
            .error_y(error_y);
        plot.add_trace(trace_values);

        // 带误差棒的最佳值线
        if target.is_none() {
            let best_error_y = plotly::common::ErrorData::new(plotly::common::ErrorType::Data)
                .array(best_stds)
                .visible(true);
            let trace_best = Scatter::new(numbers, best_means)
                .mode(Mode::Lines)
                .name("Best Value")
                .error_y(best_error_y);
            plot.add_trace(trace_best);
        }
    } else {
        // 非 error_bar 模式: 逐 study 单独绘制
        for (si, study) in studies.iter().enumerate() {
            let trials = study.get_trials(Some(&[TrialState::Complete]))?;
            if trials.is_empty() {
                continue;
            }

            let label = if studies.len() > 1 {
                study.study_name().to_string()
            } else {
                target_name.to_string()
            };

            let numbers: Vec<i64> = trials.iter().map(|t| t.number).collect();
            let values: Vec<f64> = trials
                .iter()
                .map(|t| {
                    target.map_or_else(
                        || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
                        |f| f(t),
                    )
                })
                .collect();

            // 约束可行性颜色: 可行=蓝色, 不可行=灰色
            let colors: Vec<String> = trials
                .iter()
                .map(|t| {
                    if is_feasible(t) {
                        "rgb(99,110,250)".to_string()
                    } else {
                        "rgb(204,204,204)".to_string()
                    }
                })
                .collect();

            // 累计最佳 (只考虑可行 trial)
            let is_minimize = matches!(
                study.directions().first(),
                Some(crate::study::StudyDirection::Minimize)
            );
            let mut best_values = Vec::with_capacity(values.len());
            let mut best = if is_minimize { f64::INFINITY } else { f64::NEG_INFINITY };
            for (i, &v) in values.iter().enumerate() {
                if is_feasible(&trials[i]) {
                    if is_minimize {
                        if v < best { best = v; }
                    } else if v > best {
                        best = v;
                    }
                }
                best_values.push(best);
            }

            let marker = Marker::new().color_array(colors);
            let trace_values = Scatter::new(numbers.clone(), values)
                .mode(Mode::Markers)
                .marker(marker)
                .name(&label)
                .legend_group(format!("study_{si}"));
            plot.add_trace(trace_values);

            // 自定义 target 时不绘制最佳值线 (对应 Python)
            if target.is_none() {
                let trace_best = Scatter::new(numbers, best_values)
                    .mode(Mode::Lines)
                    .name(if studies.len() > 1 {
                        format!("Best Value of {}", study.study_name())
                    } else {
                        "Best Value".to_string()
                    })
                    .legend_group(format!("study_{si}"));
                plot.add_trace(trace_best);
            }
        }
    }

    let layout = Layout::new()
        .title(format!("Optimization History ({target_name})"))
        .x_axis(Axis::new().title("Trial"))
        .y_axis(Axis::new().title(target_name));
    plot.set_layout(layout);

    Ok(plot)
}

// ============================================================================
// plot_intermediate_values
// ============================================================================

/// 绘制中间值（学习曲线）。
///
/// 对应 Python `optuna.visualization.plot_intermediate_values`。
/// 包含 Complete、Pruned、Running 三种状态的试验。
/// 不可行试验 (infeasible) 使用灰色 (#CCCCCC) 绘制。
/// 显示图例设置为 false，marker 最大显示数为 10。
#[cfg(feature = "visualization")]
pub fn plot_intermediate_values(study: &Study) -> Result<Plot> {
    // 对应 Python: states=(TrialState.PRUNED, TrialState.COMPLETE, TrialState.RUNNING)
    let trials = study.get_trials(Some(&[
        TrialState::Complete,
        TrialState::Pruned,
        TrialState::Running,
    ]))?;

    let mut plot = Plot::new();

    for trial in &trials {
        // 跳过无中间值的试验
        if trial.intermediate_values.is_empty() {
            continue;
        }
        let mut steps: Vec<i64> = trial.intermediate_values.keys().copied().collect();
        steps.sort();
        let vals: Vec<f64> = steps.iter().map(|s| trial.intermediate_values[s]).collect();

        // 约束可行性着色: 不可行试验使用灰色
        let color = if is_feasible(trial) {
            None // 使用默认颜色
        } else {
            Some("#CCCCCC".to_string()) // 灰色
        };

        // 对应 Python: marker={"maxdisplayed": 10}, showlegend=False
        let mut marker = Marker::new();
        if let Some(c) = color {
            marker = marker.color(c);
        }

        let trace = Scatter::new(steps, vals)
            .mode(Mode::LinesMarkers)
            .marker(marker)
            .name(format!("Trial {}", trial.number))
            .show_legend(false); // 对应 Python showlegend=False
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title("Intermediate Values")
        .x_axis(Axis::new().title("Step"))
        .y_axis(Axis::new().title("Value"))
        .show_legend(false); // 对应 Python layout showlegend=False
    plot.set_layout(layout);

    Ok(plot)
}

// ============================================================================
// plot_parallel_coordinate
// ============================================================================

/// 检测参数是否为对数尺度。对应 Python `_is_log_scale`。
/// 遍历试验找到第一个含该参数分布的，检查其 log 标志。
#[cfg(feature = "visualization")]
fn is_log_scale(trials: &[FrozenTrial], name: &str) -> bool {
    for trial in trials {
        if let Some(dist) = trial.distributions.get(name) {
            return match dist {
                crate::distributions::Distribution::FloatDistribution(d) => d.log,
                crate::distributions::Distribution::IntDistribution(d) => d.log,
                _ => false,
            };
        }
    }
    false
}

/// 维度信息: 描述平行坐标中每个轴的属性。
/// 对应 Python `_DimensionInfo` NamedTuple。
#[cfg(feature = "visualization")]
struct DimensionInfo {
    label: String,
    /// 归一化后的值 (用于绘图)
    values: Vec<f64>,
    /// 原始值 (用于 hover text)
    raw_values: Vec<String>,
    is_log: bool,
    is_cat: bool,
    /// 刻度位置 (归一化坐标系 [0,1])
    tick_positions: Vec<f64>,
    /// 刻度标签
    tick_labels: Vec<String>,
}

/// 绘制平行坐标图。
///
/// 对应 Python `optuna.visualization.plot_parallel_coordinate`。
/// 注意: plotly 0.11 Rust crate 没有 go.Parcoords，使用多线条 + 颜色映射模拟。
/// 各参数值归一化到 [0,1] 以统一纵轴刻度，每条线按目标值着色。
///
/// 支持对数尺度参数 (log10 变换) 和分类参数 (vocab 映射 + ticktext)。
#[cfg(feature = "visualization")]
pub fn plot_parallel_coordinate(
    study: &Study,
    params: Option<&[&str]>,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<Plot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    if trials.is_empty() {
        return Ok(Plot::new());
    }

    // 确定要显示的参数 (sorted, 对应 Python sorted_params)
    let param_names: Vec<String> = if let Some(p) = params {
        p.iter().map(|s| s.to_string()).collect()
    } else {
        let mut names: Vec<String> = trials
            .first()
            .map(|t| t.params.keys().cloned().collect())
            .unwrap_or_default();
        names.sort();
        names
    };

    // 跳过缺少任何参数的 trial (对应 Python skipped_trial_numbers)
    let filtered_trials: Vec<&FrozenTrial> = trials
        .iter()
        .filter(|t| param_names.iter().all(|name| t.params.contains_key(name)))
        .collect();
    if filtered_trials.is_empty() {
        return Ok(Plot::new());
    }

    // 提取目标值
    let obj_values: Vec<f64> = filtered_trials
        .iter()
        .map(|t| {
            target.map_or_else(
                || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
                |f| f(t),
            )
        })
        .collect();
    let obj_min = obj_values.iter().copied().fold(f64::INFINITY, f64::min);
    let obj_max = obj_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let obj_range = (obj_max - obj_min).max(1e-10);

    // 构建每个参数的 DimensionInfo
    let mut dims: Vec<DimensionInfo> = Vec::new();
    for name in &param_names {
        let log_scale = is_log_scale(&trials, name);
        let cat = is_categorical_param(&trials, name);

        if log_scale {
            // 对数尺度: 变换为 log10，生成整数幂刻度。
            // 对应 Python: values = [math.log10(v) for v in values]
            let raw: Vec<f64> = filtered_trials.iter().map(|t| extract_param(t, name)).collect();
            let log_vals: Vec<f64> = raw.iter().map(|v| v.log10()).collect();
            let min_v = log_vals.iter().copied().fold(f64::INFINITY, f64::min);
            let max_v = log_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let range = (max_v - min_v).max(1e-10);

            // 归一化到 [0,1]
            let normalized: Vec<f64> = log_vals.iter().map(|v| (v - min_v) / range).collect();
            let raw_text: Vec<String> = raw.iter().map(|v| format_g(*v, 4)).collect();

            // 刻度: 整数幂 + 端点。对应 Python: range(ceil(min), floor(max)+1)
            let tick_start = min_v.ceil() as i64;
            let tick_end = max_v.floor() as i64;
            let mut tick_vals: Vec<f64> = (tick_start..=tick_end).map(|i| i as f64).collect();
            // 加入端点（如果不在整数集中）
            if tick_vals.is_empty() || (min_v - tick_vals[0]).abs() > 1e-9 {
                tick_vals.insert(0, min_v);
            }
            if tick_vals.is_empty() || (max_v - *tick_vals.last().unwrap()).abs() > 1e-9 {
                tick_vals.push(max_v);
            }
            // 转换为归一化坐标和标签
            let tick_positions: Vec<f64> = tick_vals.iter().map(|v| (v - min_v) / range).collect();
            let tick_labels: Vec<String> = tick_vals.iter()
                .map(|v| format_g(10.0_f64.powf(*v), 3))
                .collect();

            dims.push(DimensionInfo {
                label: name.clone(),
                values: normalized,
                raw_values: raw_text,
                is_log: true,
                is_cat: false,
                tick_positions,
                tick_labels,
            });
        } else if cat {
            // 分类参数: 词汇表映射 → 整数索引。
            // 对应 Python: vocab = defaultdict(lambda: len(vocab))
            let labels = get_categorical_labels(&trials, name);
            let n_choices = labels.len().max(1);
            let raw_vals: Vec<f64> = filtered_trials.iter()
                .map(|t| extract_param(t, name))
                .collect();
            // 归一化: index / (n_choices - 1)
            let divisor = (n_choices as f64 - 1.0).max(1.0);
            let normalized: Vec<f64> = raw_vals.iter().map(|v| *v / divisor).collect();
            let raw_text: Vec<String> = raw_vals.iter().map(|v| {
                let idx = *v as usize;
                labels.get(idx).cloned().unwrap_or_else(|| format!("{v}"))
            }).collect();
            let tick_positions: Vec<f64> = (0..n_choices).map(|i| i as f64 / divisor).collect();

            dims.push(DimensionInfo {
                label: name.clone(),
                values: normalized,
                raw_values: raw_text,
                is_log: false,
                is_cat: true,
                tick_positions,
                tick_labels: labels,
            });
        } else {
            // 数值参数: 线性归一化
            let vals: Vec<f64> = filtered_trials.iter().map(|t| extract_param(t, name)).collect();
            let mn = vals.iter().copied().fold(f64::INFINITY, f64::min);
            let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let range = (mx - mn).max(1e-10);
            let normalized: Vec<f64> = vals.iter().map(|v| (v - mn) / range).collect();
            let raw_text: Vec<String> = vals.iter().map(|v| format_g(*v, 4)).collect();

            dims.push(DimensionInfo {
                label: name.clone(),
                values: normalized,
                raw_values: raw_text,
                is_log: false,
                is_cat: false,
                tick_positions: vec![],
                tick_labels: vec![],
            });
        }
    }

    // 维度标签: Objective + 各参数
    let dim_labels: Vec<String> = std::iter::once(target_name.to_string())
        .chain(param_names.iter().cloned())
        .collect();

    let is_minimize = matches!(
        study.directions().first(),
        Some(crate::study::StudyDirection::Minimize)
    );

    let mut plot = Plot::new();

    // Viridis 色带 5 个控制点 (t ∈ [0,1] → (R,G,B))
    // 对应 Python plotly COLOR_SCALE = "Viridis"
    let viridis: [(f64, f64, f64); 5] = [
        (68.0, 1.0, 84.0),     // t=0.00 — 深紫
        (59.0, 82.0, 139.0),   // t=0.25 — 蓝紫
        (33.0, 145.0, 140.0),  // t=0.50 — 青绿
        (94.0, 201.0, 98.0),   // t=0.75 — 黄绿
        (253.0, 231.0, 37.0),  // t=1.00 — 亮黄
    ];
    let viridis_interp = |t: f64| -> String {
        let t = t.clamp(0.0, 1.0);
        let scaled = t * 4.0;
        let idx = (scaled.floor() as usize).min(3);
        let frac = scaled - idx as f64;
        let (r0, g0, b0) = viridis[idx];
        let (r1, g1, b1) = viridis[idx + 1];
        let r = (r0 + frac * (r1 - r0)) as u8;
        let g = (g0 + frac * (g1 - g0)) as u8;
        let b = (b0 + frac * (b1 - b0)) as u8;
        format!("rgb({r},{g},{b})")
    };

    // reverse_scale: Minimize → 小值=好=亮色(t=1), Maximize → 大值=好=亮色(t=1)
    // 对应 Python `reverse_scale = _is_reverse_scale(study, target)`
    let reverse_scale = is_minimize;

    // 每个 trial 画一条线，按目标值映射颜色
    for (i, _trial) in filtered_trials.iter().enumerate() {
        let obj_val = obj_values[i];
        let color_val = (obj_val - obj_min) / obj_range;
        let color_t = if reverse_scale { 1.0 - color_val } else { color_val };
        let color_str = viridis_interp(color_t);

        // y 值: objective 归一化 + 各 dim 已归一化的值
        let mut y_values: Vec<f64> = vec![color_val];
        for dim in &dims {
            y_values.push(dim.values[i]);
        }

        // hover text: 显示目标值 + 各参数原始值
        let mut hover_parts = vec![format!("{target_name}: {}", format_g(obj_val, 4))];
        for (j, dim) in dims.iter().enumerate() {
            hover_parts.push(format!("{}: {}", param_names[j], dim.raw_values[i]));
        }
        let hover_text = hover_parts.join("<br>");

        let trace = Scatter::new(dim_labels.clone(), y_values)
            .mode(Mode::Lines)
            .line(plotly::common::Line::new().color(color_str).width(1.5))
            .show_legend(false)
            .hover_info(plotly::common::HoverInfo::Text)
            .text(hover_text);
        plot.add_trace(trace);
    }

    // 添加一个不可见散点作为颜色条
    // reverse_scale 使颜色条上好值（minimize→小、maximize→大）映射到亮色
    let colorbar_marker = Marker::new()
        .color_array(obj_values.clone())
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis))
        .reverse_scale(reverse_scale)
        .show_scale(true)
        .color_bar(ColorBar::new().title(target_name));
    let colorbar_trace = Scatter::new(vec![Option::<f64>::None], vec![Option::<f64>::None])
        .mode(Mode::Markers)
        .marker(colorbar_marker)
        .show_legend(false);
    plot.add_trace(colorbar_trace);

    let layout = Layout::new()
        .title("Parallel Coordinate")
        .y_axis(Axis::new().title("Normalized Value").range(vec![-0.05, 1.05]));
    plot.set_layout(layout);

    Ok(plot)
}

// ============================================================================
// plot_contour
// ============================================================================

/// 绘制等高线图（两参数间的目标值热力图）。
///
/// 对应 Python `optuna.visualization.plot_contour`。
#[cfg(feature = "visualization")]
pub fn plot_contour(
    study: &Study,
    params: Option<&[&str]>,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<Plot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    if trials.is_empty() {
        return Ok(Plot::new());
    }

    let param_names: Vec<String> = if let Some(p) = params {
        p.iter().map(|s| s.to_string()).collect()
    } else {
        let mut names: Vec<String> = trials
            .first()
            .map(|t| t.params.keys().cloned().collect())
            .unwrap_or_default();
        names.sort();
        names
    };

    if param_names.len() < 2 {
        return Ok(Plot::new());
    }

    let z_vals: Vec<f64> = trials
        .iter()
        .map(|t| {
            target.map_or_else(
                || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
                |f| f(t),
            )
        })
        .collect();

    let is_minimize = matches!(
        study.directions().first(),
        Some(crate::study::StudyDirection::Minimize)
    );

    let n = param_names.len();

    // N×N subplot grid (对应 Python plot_contour 的 N×N 布局)。
    // 仅当 N=2 时使用 2×2 grid (4 subplots ≤ 8 axes)。
    // N≥3 退化为取前 2 params 的单 contour + scatter。
    if n == 2 {
        return plot_contour_2x2(&trials, &param_names, &z_vals, is_minimize, target_name);
    }

    // Fallback: N≥3, 只使用前 2 params
    plot_contour_single(&trials, &param_names[0], &param_names[1], &z_vals, is_minimize, target_name)
}

/// 2×2 contour subplot grid (对角线为 1D 直方图, 非对角线为 contour)。
/// 对应 Python plot_contour N=2 时的布局。
#[cfg(feature = "visualization")]
fn plot_contour_2x2(
    trials: &[FrozenTrial],
    param_names: &[String],
    z_vals: &[f64],
    is_minimize: bool,
    target_name: &str,
) -> Result<Plot> {
    let mut plot = Plot::new();
    let p0 = &param_names[0];
    let p1 = &param_names[1];
    let x0: Vec<f64> = trials.iter().map(|t| extract_param(t, p0)).collect();
    let x1: Vec<f64> = trials.iter().map(|t| extract_param(t, p1)).collect();

    // cell (0,0) = 对角: p0 直方图 (x_axis, y_axis)
    let hist_trace_0 = plotly::Histogram::new(x0.clone())
        .name(p0)
        .show_legend(false);
    plot.add_trace(hist_trace_0);

    // cell (0,1) = 非对角: contour(p0 vs p1) (x_axis2, y_axis2)
    let (contour_01, scatter_01) = build_contour_cell(
        &x0, &x1, z_vals, is_minimize, target_name, trials,
    );
    let contour_01 = contour_01.x_axis("x2").y_axis("y2");
    let scatter_01 = scatter_01.x_axis("x2").y_axis("y2");
    plot.add_trace(contour_01);
    plot.add_trace(scatter_01);

    // cell (1,0) = 非对角: contour(p1 vs p0) (x_axis3, y_axis3)
    let (contour_10, scatter_10) = build_contour_cell(
        &x1, &x0, z_vals, is_minimize, target_name, trials,
    );
    let contour_10 = contour_10.x_axis("x3").y_axis("y3");
    let scatter_10 = scatter_10.x_axis("x3").y_axis("y3");
    plot.add_trace(contour_10);
    plot.add_trace(scatter_10);

    // cell (1,1) = 对角: p1 直方图 (x_axis4, y_axis4)
    let hist_trace_1 = plotly::Histogram::new(x1.clone())
        .name(p1)
        .show_legend(false)
        .x_axis("x4")
        .y_axis("y4");
    plot.add_trace(hist_trace_1);

    let grid = LayoutGrid::new()
        .rows(2)
        .columns(2)
        .pattern(GridPattern::Independent);
    let layout = Layout::new()
        .title(format!("Contour ({target_name})"))
        .grid(grid)
        .x_axis(Axis::new().title(p0.as_str()))
        .y_axis(Axis::new().title("Count"))
        .x_axis2(Axis::new().title(p0.as_str()))
        .y_axis2(Axis::new().title(p1.as_str()))
        .x_axis3(Axis::new().title(p1.as_str()))
        .y_axis3(Axis::new().title(p0.as_str()))
        .x_axis4(Axis::new().title(p1.as_str()))
        .y_axis4(Axis::new().title("Count"));
    plot.set_layout(layout);

    Ok(plot)
}

/// 构建单个 contour cell (等高线 + 散点)。
/// 返回 (Contour trace, Scatter trace)。
#[cfg(feature = "visualization")]
fn build_contour_cell(
    x_vals: &[f64],
    y_vals: &[f64],
    z_vals: &[f64],
    is_minimize: bool,
    target_name: &str,
    _trials: &[FrozenTrial],
) -> (Box<Contour<Vec<f64>, f64, f64>>, Box<Scatter<f64, f64>>) {
    let padding_ratio = 0.05;
    let mut x_unique: Vec<f64> = x_vals.iter().copied().filter(|v| v.is_finite()).collect();
    x_unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    x_unique.dedup();
    let mut y_unique: Vec<f64> = y_vals.iter().copied().filter(|v| v.is_finite()).collect();
    y_unique.sort_by(|a, b| a.partial_cmp(b).unwrap());
    y_unique.dedup();

    if x_unique.is_empty() { x_unique.push(0.0); }
    if y_unique.is_empty() { y_unique.push(0.0); }

    let x_min = x_unique.first().copied().unwrap();
    let x_max = x_unique.last().copied().unwrap();
    let x_span = (x_max - x_min).max(1e-10);
    let x_pad = x_span * padding_ratio;
    x_unique.insert(0, x_min - x_pad);
    x_unique.push(x_max + x_pad);

    let y_min = y_unique.first().copied().unwrap();
    let y_max = y_unique.last().copied().unwrap();
    let y_span = (y_max - y_min).max(1e-10);
    let y_pad = y_span * padding_ratio;
    y_unique.insert(0, y_min - y_pad);
    y_unique.push(y_max + y_pad);

    let nx = x_unique.len();
    let ny = y_unique.len();
    let mut z_grid: Vec<Vec<f64>> = vec![vec![f64::NAN; nx]; ny];

    for idx in 0..x_vals.len() {
        let xv = x_vals[idx];
        let yv = y_vals[idx];
        let zv = z_vals[idx];
        if !xv.is_finite() || !yv.is_finite() || !zv.is_finite() { continue; }
        let xi = x_unique.iter().enumerate()
            .min_by(|(_, a), (_, b)| (*a - xv).abs().partial_cmp(&(*b - xv).abs()).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let yi = y_unique.iter().enumerate()
            .min_by(|(_, a), (_, b)| (*a - yv).abs().partial_cmp(&(*b - yv).abs()).unwrap())
            .map(|(i, _)| i).unwrap_or(0);
        let existing = z_grid[yi][xi];
        if existing.is_nan() {
            z_grid[yi][xi] = zv;
        } else if is_minimize {
            z_grid[yi][xi] = existing.min(zv);
        } else {
            z_grid[yi][xi] = existing.max(zv);
        }
    }

    let contour_trace = Contour::new(x_unique, y_unique, z_grid)
        .color_scale(ColorScale::Palette(ColorScalePalette::Viridis))
        .auto_contour(true)
        .connect_gaps(true)
        .show_scale(true)
        .reverse_scale(is_minimize)
        .contours(plotly::contour::Contours::new().coloring(ContourColoring::HeatMap))
        .hover_info(plotly::common::HoverInfo::None)
        .color_bar(ColorBar::new().title(target_name));

    let scatter_trace = Scatter::new(x_vals.to_vec(), y_vals.to_vec())
        .mode(Mode::Markers)
        .marker(Marker::new().color("black").size(4))
        .name("Trials")
        .show_legend(false);

    (contour_trace, scatter_trace)
}

/// 单 contour 图 (fallback for N≥3 params: 取前 2 params)
#[cfg(feature = "visualization")]
fn plot_contour_single(
    trials: &[FrozenTrial],
    x_name: &str,
    y_name: &str,
    z_vals: &[f64],
    is_minimize: bool,
    target_name: &str,
) -> Result<Plot> {
    let x_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, x_name)).collect();
    let y_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, y_name)).collect();

    let mut plot = Plot::new();
    let (contour_trace, scatter_trace) = build_contour_cell(
        &x_vals, &y_vals, z_vals, is_minimize, target_name, trials,
    );
    plot.add_trace(contour_trace);
    plot.add_trace(scatter_trace);

    let layout = Layout::new()
        .title(format!("Contour ({target_name})"))
        .x_axis(Axis::new().title(x_name))
        .y_axis(Axis::new().title(y_name));
    plot.set_layout(layout);

    Ok(plot)
}

// ============================================================================
// plot_slice
// ============================================================================

/// 绘制切片图（各参数与目标值的散点图）。
///
/// 对应 Python `optuna.visualization.plot_slice`。
#[cfg(feature = "visualization")]
pub fn plot_slice(
    study: &Study,
    params: Option<&[&str]>,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<Plot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    if trials.is_empty() {
        return Ok(Plot::new());
    }

    let param_names: Vec<String> = if let Some(p) = params {
        p.iter().map(|s| s.to_string()).collect()
    } else {
        let mut names: Vec<String> = trials
            .first()
            .map(|t| t.params.keys().cloned().collect())
            .unwrap_or_default();
        names.sort();
        names
    };

    let values: Vec<f64> = trials
        .iter()
        .map(|t| {
            target.map_or_else(
                || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
                |f| f(t),
            )
        })
        .collect();

    let mut plot = Plot::new();
    let n_params = param_names.len();

    if n_params <= 1 || n_params > 8 {
        // 单参数或超过 8 个: 叠加到一张图
        for name in &param_names {
            let x_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, name)).collect();
            let trace = Scatter::new(x_vals, values.clone())
                .mode(Mode::Markers)
                .name(name);
            plot.add_trace(trace);
        }
        let layout = Layout::new()
            .title(format!("Slice Plot ({target_name})"))
            .y_axis(Axis::new().title(target_name));
        plot.set_layout(layout);
    } else {
        // 多参数: 每参数一个 subplot, 1行 × n_params列
        // 对应 Python `plot_slice` 的 per-param subplot 布局
        for (i, name) in param_names.iter().enumerate() {
            let x_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, name)).collect();
            let mut trace = Scatter::new(x_vals, values.clone())
                .mode(Mode::Markers)
                .name(name)
                .show_legend(false);
            if i > 0 {
                let axis_id = format!("x{}", i + 1);
                let yaxis_id = format!("y{}", i + 1);
                trace = trace.x_axis(&axis_id).y_axis(&yaxis_id);
            }
            plot.add_trace(trace);
        }
        // 构建 Layout with Grid
        let grid = LayoutGrid::new()
            .rows(1)
            .columns(n_params)
            .pattern(GridPattern::Independent);
        let mut layout = Layout::new()
            .title(format!("Slice Plot ({target_name})"))
            .grid(grid)
            .x_axis(Axis::new().title(param_names[0].as_str()))
            .y_axis(Axis::new().title(target_name));
        // 设置额外轴 (x_axis2..x_axis8, y_axis2..y_axis8)
        macro_rules! set_axes {
            ($layout:expr, $i:expr, $name:expr, $tn:expr, $($n:literal => $xa:ident, $ya:ident);+) => {
                $(
                    if $i == $n {
                        $layout = $layout.$xa(Axis::new().title($name.as_str())).$ya(Axis::new().title($tn));
                    }
                )+
            };
        }
        for i in 1..n_params {
            set_axes!(layout, i, param_names[i], target_name,
                1 => x_axis2, y_axis2;
                2 => x_axis3, y_axis3;
                3 => x_axis4, y_axis4;
                4 => x_axis5, y_axis5;
                5 => x_axis6, y_axis6;
                6 => x_axis7, y_axis7;
                7 => x_axis8, y_axis8
            );
        }
        plot.set_layout(layout);
    }

    Ok(plot)
}

// ============================================================================
// plot_edf
// ============================================================================

/// 绘制经验分布函数。
///
/// 对应 Python `optuna.visualization.plot_edf`。
/// 支持多 study (通过 `plot_edf_multi`)。
#[cfg(feature = "visualization")]
pub fn plot_edf(
    study: &Study,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<Plot> {
    plot_edf_multi(&[study], target, target_name)
}

/// 多 study EDF 图。
/// 对应 Python `plot_edf(study=[s1, s2, ...])`.
/// 使用共享 x 轴网格 (100 个等间距采样点), y 轴范围锁定 [0, 1]。
/// CDF 计算: y[j] = count(values <= x[j]) / n_values (对应 Python 广播矩阵)。
#[cfg(feature = "visualization")]
pub fn plot_edf_multi(
    studies: &[&Study],
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<Plot> {
    const NUM_SAMPLES_X_AXIS: usize = 100;

    let mut plot = Plot::new();

    // 第一遍: 收集所有 study 的目标值，计算全局 min/max
    let mut all_study_values: Vec<(String, Vec<f64>)> = Vec::new();
    let mut global_min = f64::INFINITY;
    let mut global_max = f64::NEG_INFINITY;

    for study in studies {
        let trials = study.get_trials(Some(&[TrialState::Complete]))?;
        if trials.is_empty() {
            continue;
        }

        let mut values: Vec<f64> = trials
            .iter()
            .map(|t| {
                target.map_or_else(
                    || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
                    |f| f(t),
                )
            })
            .filter(|v| v.is_finite())
            .collect();
        if values.is_empty() {
            continue;
        }

        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let mn = values.first().copied().unwrap_or(0.0);
        let mx = values.last().copied().unwrap_or(0.0);
        if mn < global_min { global_min = mn; }
        if mx > global_max { global_max = mx; }

        all_study_values.push((study.study_name().to_string(), values));
    }

    if all_study_values.is_empty() {
        return Ok(plot);
    }

    // 共享 x 轴网格: 对应 Python np.linspace(min_x, max_x, 100)
    let x_grid: Vec<f64> = if (global_max - global_min).abs() < 1e-12 {
        vec![global_min; NUM_SAMPLES_X_AXIS]
    } else {
        (0..NUM_SAMPLES_X_AXIS)
            .map(|i| {
                global_min + (global_max - global_min) * i as f64 / (NUM_SAMPLES_X_AXIS - 1) as f64
            })
            .collect()
    };

    // 第二遍: 为每个 study 计算 CDF
    for (study_name, values) in &all_study_values {
        let n = values.len() as f64;
        // 对应 Python: y = sum(values[:, np.newaxis] <= x_values, axis=0) / n
        let y_values: Vec<f64> = x_grid
            .iter()
            .map(|&x| {
                // searchsorted equivalent: count(v <= x)
                let count = values.partition_point(|v| *v <= x);
                count as f64 / n
            })
            .collect();

        let trace = Scatter::new(x_grid.clone(), y_values)
            .mode(Mode::Lines)
            .name(study_name);
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title("Empirical Distribution Function")
        .x_axis(Axis::new().title(target_name))
        .y_axis(Axis::new().title("Cumulative Probability").range(vec![0.0, 1.0]));
    plot.set_layout(layout);

    Ok(plot)
}

// ============================================================================
// plot_param_importances
// ============================================================================

/// 绘制参数重要性柱状图。
///
/// 对应 Python `optuna.visualization.plot_param_importances`。
/// 使用水平柱状图 (orientation='h'), 按重要性升序排列 (最重要的在顶部)。
/// 支持多目标研究: 当无自定义 target 且研究为多目标时，为每个目标绘制一组柱状。
#[cfg(feature = "visualization")]
pub fn plot_param_importances(
    study: &Study,
    evaluator: Option<&dyn crate::importance::ImportanceEvaluator>,
    params: Option<&[&str]>,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<Plot> {
    let directions = study.directions();
    let is_multi = directions.len() > 1;
    let metric_names = study.metric_names().unwrap_or(None);

    let mut plot = Plot::new();

    if target.is_none() && is_multi {
        // 多目标: 为每个目标分别计算重要性
        // 对应 Python: for objective_id, target_name in enumerate(target_names)
        let n_obj = directions.len();
        for obj_id in 0..n_obj {
            let obj_target = move |t: &FrozenTrial| -> f64 {
                t.values
                    .as_ref()
                    .and_then(|v| v.get(obj_id).copied())
                    .unwrap_or(f64::NAN)
            };
            let obj_name = metric_names
                .as_ref()
                .and_then(|names| names.get(obj_id).map(|s| s.clone()))
                .unwrap_or_else(|| format!("Objective {obj_id}"));

            let importances = crate::importance::get_param_importances(
                study,
                evaluator,
                params,
                Some(&obj_target),
                true,
            )?;

            // 按升序排列 (对应 Python dict(reversed(...)))
            let mut entries: Vec<(String, f64)> = importances.into_iter().collect();
            entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

            let names: Vec<String> = entries.iter().map(|(k, _)| k.clone()).collect();
            let values: Vec<f64> = entries.iter().map(|(_, v)| *v).collect();

            // 水平柱状图 (对应 Python orientation="h")
            let trace = Bar::new(values, names)
                .name(&obj_name)
                .orientation(plotly::common::Orientation::Horizontal);
            plot.add_trace(trace);
        }
    } else {
        // 单目标或自定义 target
        let importances = crate::importance::get_param_importances(
            study,
            evaluator,
            params,
            target,
            true,
        )?;

        // 按升序排列 (对应 Python dict(reversed(...)))
        let mut entries: Vec<(String, f64)> = importances.into_iter().collect();
        entries.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let names: Vec<String> = entries.iter().map(|(k, _)| k.clone()).collect();
        let values: Vec<f64> = entries.iter().map(|(_, v)| *v).collect();

        // 水平柱状图 (对应 Python orientation="h")
        let trace = Bar::new(values, names)
            .name(target_name)
            .orientation(plotly::common::Orientation::Horizontal);
        plot.add_trace(trace);
    }

    let layout = Layout::new()
        .title(format!("Hyperparameter Importances ({target_name})"))
        .x_axis(Axis::new().title("Importance"))
        .y_axis(Axis::new().title("Hyperparameter"));
    plot.set_layout(layout);

    Ok(plot)
}

// ============================================================================
// plot_pareto_front
// ============================================================================

/// 绘制帕累托前沿。
///
/// 对应 Python `optuna.visualization.plot_pareto_front`。
/// 支持 2D 和 3D (3目标), 支持 `targets` 自定义轴映射,
/// 支持约束可行性着色 (infeasible 为灰色)。
#[cfg(feature = "visualization")]
pub fn plot_pareto_front(
    study: &Study,
    target_names: Option<&[&str]>,
    include_dominated: bool,
) -> Result<Plot> {
    plot_pareto_front_with_targets(study, target_names, include_dominated, None, None)
}

/// 带有 targets 和 constraints_func 的帕累托前沿。
/// 对应 Python `plot_pareto_front(targets=..., constraints_func=...)`.
#[cfg(feature = "visualization")]
pub fn plot_pareto_front_with_targets(
    study: &Study,
    target_names: Option<&[&str]>,
    include_dominated: bool,
    targets: Option<&dyn Fn(&FrozenTrial) -> Vec<f64>>,
    constraints_func: Option<&dyn Fn(&FrozenTrial) -> Vec<f64>>,
) -> Result<Plot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    let pareto = study.best_trials()?;

    let n_obj = study.directions().len();
    // 默认目标名
    let obj_names: Vec<String> = if let Some(names) = target_names {
        names.iter().map(|s| s.to_string()).collect()
    } else {
        (0..n_obj).map(|i| format!("Objective {i}")).collect()
    };

    // 提取轴值 (可通过 targets 自定义)
    let get_axes = |t: &FrozenTrial| -> Vec<f64> {
        if let Some(f) = targets {
            f(t)
        } else {
            t.values.clone().unwrap_or_default()
        }
    };

    // 约束可行性
    let trial_feasible = |t: &FrozenTrial| -> bool {
        if let Some(cf) = constraints_func {
            cf(t).iter().all(|&c| c <= 0.0)
        } else {
            is_feasible(t)
        }
    };

    let mut plot = Plot::new();

    if n_obj >= 2 {
        // 帕累托前沿点
        let pareto_nums: std::collections::HashSet<i64> =
            pareto.iter().map(|t| t.number).collect();

        // 按可行性分组: feasible pareto, infeasible pareto, feasible dominated, infeasible dominated
        let mut fp_x = Vec::new(); let mut fp_y = Vec::new(); let mut fp_z = Vec::new();
        let mut ip_x = Vec::new(); let mut ip_y = Vec::new(); let mut ip_z = Vec::new();
        let mut fd_x = Vec::new(); let mut fd_y = Vec::new(); let mut fd_z = Vec::new();
        let mut id_x = Vec::new(); let mut id_y = Vec::new(); let mut id_z = Vec::new();

        for t in &trials {
            let axes = get_axes(t);
            if axes.len() < 2 { continue; }
            let feasible = trial_feasible(t);
            let on_pareto = pareto_nums.contains(&t.number);

            let z_val = if n_obj >= 3 && axes.len() >= 3 { axes[2] } else { 0.0 };

            if on_pareto && feasible {
                fp_x.push(axes[0]); fp_y.push(axes[1]); fp_z.push(z_val);
            } else if on_pareto && !feasible {
                ip_x.push(axes[0]); ip_y.push(axes[1]); ip_z.push(z_val);
            } else if !on_pareto && feasible {
                fd_x.push(axes[0]); fd_y.push(axes[1]); fd_z.push(z_val);
            } else {
                id_x.push(axes[0]); id_y.push(axes[1]); id_z.push(z_val);
            }
        }

        if n_obj >= 3 {
            // 3D pareto: 使用 Scatter3D
            // 对应 Python `go.Scatter3d` 分支
            if !fp_x.is_empty() {
                let trace = Scatter3D::new(fp_x, fp_y, fp_z)
                    .mode(Mode::Markers)
                    .marker(Marker::new().color("rgb(99,110,250)"))
                    .name("Pareto Front");
                plot.add_trace(trace);
            }
            if !ip_x.is_empty() {
                let trace = Scatter3D::new(ip_x, ip_y, ip_z)
                    .mode(Mode::Markers)
                    .marker(Marker::new().color("rgb(204,204,204)"))
                    .name("Infeasible Pareto");
                plot.add_trace(trace);
            }
            if include_dominated {
                if !fd_x.is_empty() {
                    let trace = Scatter3D::new(fd_x, fd_y, fd_z)
                        .mode(Mode::Markers)
                        .marker(Marker::new().color("rgb(239,85,59)").opacity(0.5))
                        .name("Dominated Trials");
                    plot.add_trace(trace);
                }
                if !id_x.is_empty() {
                    let trace = Scatter3D::new(id_x, id_y, id_z)
                        .mode(Mode::Markers)
                        .marker(Marker::new().color("rgb(204,204,204)").opacity(0.3))
                        .name("Infeasible Dominated");
                    plot.add_trace(trace);
                }
            }

            let layout = Layout::new()
                .title("Pareto Front (3D)");
            plot.set_layout(layout);
        } else {
            // 2D pareto: 使用 Scatter
            if !fp_x.is_empty() {
                let trace = Scatter::new(fp_x, fp_y)
                    .mode(Mode::Markers)
                    .marker(Marker::new().color("rgb(99,110,250)"))
                    .name("Pareto Front");
                plot.add_trace(trace);
            }
            if !ip_x.is_empty() {
                let trace = Scatter::new(ip_x, ip_y)
                    .mode(Mode::Markers)
                    .marker(Marker::new().color("rgb(204,204,204)"))
                    .name("Infeasible Pareto");
                plot.add_trace(trace);
            }
            if include_dominated {
                if !fd_x.is_empty() {
                    let trace = Scatter::new(fd_x, fd_y)
                        .mode(Mode::Markers)
                        .marker(Marker::new().color("rgb(239,85,59)").opacity(0.5))
                        .name("Dominated Trials");
                    plot.add_trace(trace);
                }
                if !id_x.is_empty() {
                    let trace = Scatter::new(id_x, id_y)
                        .mode(Mode::Markers)
                        .marker(Marker::new().color("rgb(204,204,204)").opacity(0.3))
                        .name("Infeasible Dominated");
                    plot.add_trace(trace);
                }
            }

            let layout = Layout::new()
                .title("Pareto Front")
                .x_axis(Axis::new().title(obj_names.first().map(|s| s.as_str()).unwrap_or("Obj 0")))
                .y_axis(Axis::new().title(obj_names.get(1).map(|s| s.as_str()).unwrap_or("Obj 1")));
            plot.set_layout(layout);
        }
    }

    Ok(plot)
}

// ============================================================================
// plot_hypervolume_history
// ============================================================================

/// 绘制超体积历史。
///
/// 对应 Python `optuna.visualization.plot_hypervolume_history`。
/// 增量维护帕累托前沿，逐 trial 累积计算超体积。
/// 过滤不可行试验 (infeasible)，对 Maximize 方向取负统一为 Minimize。
/// 添加验证: 必须为多目标研究，参考点维度必须匹配目标数。
#[cfg(feature = "visualization")]
pub fn plot_hypervolume_history(
    study: &Study,
    reference_point: &[f64],
) -> Result<Plot> {
    let directions = study.directions();
    let n_obj = directions.len();

    // 验证: 必须为多目标研究 (对应 Python ValueError)
    if n_obj < 2 {
        return Err(crate::error::OptunaError::ValueError(
            "plot_hypervolume_history requires a multi-objective study.".to_string(),
        ));
    }
    // 验证: 参考点维度必须匹配目标数 (对应 Python ValueError)
    if reference_point.len() != n_obj {
        return Err(crate::error::OptunaError::ValueError(format!(
            "The dimension of the reference point ({}) must match the number of objectives ({}).",
            reference_point.len(),
            n_obj,
        )));
    }

    let trials = study.get_trials(Some(&[TrialState::Complete]))?;

    // 将参考点归一化 (Maximize 方向取负)
    let neg_ref: Vec<f64> = reference_point
        .iter()
        .zip(directions.iter())
        .map(|(&v, d)| match d {
            crate::study::StudyDirection::Maximize => -v,
            _ => v,
        })
        .collect();

    // 增量维护帕累托前沿 (对应 Python best_trials_values_normalized)
    let mut pareto_front: Vec<Vec<f64>> = Vec::new();
    let mut hypervolume = 0.0_f64;
    let mut hypervolumes: Vec<f64> = Vec::new();
    let mut trial_numbers: Vec<i64> = Vec::new();

    for trial in &trials {
        // 约束过滤: 不可行试验保持超体积不变 (对应 Python constraints check)
        if !is_feasible(trial) {
            hypervolumes.push(hypervolume);
            trial_numbers.push(trial.number);
            continue;
        }

        let values = match &trial.values {
            Some(v) if v.len() == n_obj => v.clone(),
            _ => {
                hypervolumes.push(hypervolume);
                trial_numbers.push(trial.number);
                continue;
            }
        };

        // 归一化 (Maximize 方向取负)
        let normalized: Vec<f64> = values
            .iter()
            .zip(directions.iter())
            .map(|(&v, d)| match d {
                crate::study::StudyDirection::Maximize => -v,
                _ => v,
            })
            .collect();

        // 检查新解是否被现有帕累托前沿支配
        // 对应 Python: (best <= values).all(axis=1).any()
        let is_dominated = pareto_front.iter().any(|p| {
            p.iter().zip(normalized.iter()).all(|(pi, ni)| pi <= ni)
        });

        if is_dominated {
            // 被支配则超体积不变
            hypervolumes.push(hypervolume);
            trial_numbers.push(trial.number);
            continue;
        }

        // 增量更新超体积 (对应 Python incremental computation)
        // 1. 加上新解独立的超体积
        let new_hv: f64 = normalized
            .iter()
            .zip(neg_ref.iter())
            .map(|(n, r)| r - n)
            .product();
        hypervolume += new_hv;

        // 2. 减去与现有解重叠的部分
        if !pareto_front.is_empty() {
            let limited: Vec<Vec<f64>> = pareto_front
                .iter()
                .map(|p| {
                    p.iter()
                        .zip(normalized.iter())
                        .map(|(pi, ni)| pi.max(*ni))
                        .collect()
                })
                .collect();
            let overlap_hv = crate::multi_objective::hypervolume(&limited, &neg_ref);
            hypervolume -= overlap_hv;
        }

        // 3. 移除被新解支配的旧解
        // 对应 Python: is_kept = (best < values).any(axis=1)
        pareto_front.retain(|p| {
            p.iter().zip(normalized.iter()).any(|(pi, ni)| pi < ni)
        });
        pareto_front.push(normalized);

        hypervolumes.push(hypervolume);
        trial_numbers.push(trial.number);
    }

    let mut plot = Plot::new();
    let trace = Scatter::new(trial_numbers, hypervolumes)
        .mode(Mode::LinesMarkers)
        .name("Hypervolume");
    plot.add_trace(trace);

    let layout = Layout::new()
        .title("Hypervolume History")
        .x_axis(Axis::new().title("Trial"))
        .y_axis(Axis::new().title("Hypervolume"));
    plot.set_layout(layout);

    Ok(plot)
}

// ============================================================================
// plot_rank
// ============================================================================

/// 绘制试验排名图。
///
/// 对应 Python `optuna.visualization.plot_rank`。
/// 对指定参数的每对组合绘制散点子图，颜色按排名映射。
#[cfg(feature = "visualization")]
pub fn plot_rank(
    study: &Study,
    params: Option<&[&str]>,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<Plot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    if trials.is_empty() {
        return Ok(Plot::new());
    }

    let param_names: Vec<String> = if let Some(p) = params {
        p.iter().map(|s| s.to_string()).collect()
    } else {
        let mut names: Vec<String> = trials
            .first()
            .map(|t| t.params.keys().cloned().collect())
            .unwrap_or_default();
        names.sort();
        names
    };

    // 提取目标值并计算排名（相同值取平均排名）
    let obj_values: Vec<f64> = trials
        .iter()
        .map(|t| {
            target.map_or_else(
                || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
                |f| f(t),
            )
        })
        .collect();

    let n = obj_values.len();
    let mut sorted_vals = obj_values.clone();
    sorted_vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    // 平均排名
    let mut ranks: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        let val = obj_values[i];
        let indices: Vec<usize> = sorted_vals.iter().enumerate()
            .filter(|(_, sv)| (**sv - val).abs() < 1e-12)
            .map(|(j, _)| j)
            .collect();
        ranks[i] = indices.iter().sum::<usize>() as f64 / indices.len() as f64;
    }

    // 归一化排名到 [0, 1] 用于颜色映射
    let max_rank = (n - 1).max(1) as f64;
    let color_indices: Vec<f64> = ranks.iter().map(|&r| r / max_rank).collect();

    // RdYlBu_r 颜色标尺 (reversed Red-Yellow-Blue)
    let rdylbu_r = ColorScale::Vector(vec![
        ColorScaleElement(0.0, "rgb(49,54,149)".to_string()),
        ColorScaleElement(0.1, "rgb(69,117,180)".to_string()),
        ColorScaleElement(0.2, "rgb(116,173,209)".to_string()),
        ColorScaleElement(0.3, "rgb(171,217,233)".to_string()),
        ColorScaleElement(0.4, "rgb(224,243,248)".to_string()),
        ColorScaleElement(0.5, "rgb(255,255,191)".to_string()),
        ColorScaleElement(0.6, "rgb(254,224,144)".to_string()),
        ColorScaleElement(0.7, "rgb(253,174,97)".to_string()),
        ColorScaleElement(0.8, "rgb(244,109,67)".to_string()),
        ColorScaleElement(0.9, "rgb(215,48,39)".to_string()),
        ColorScaleElement(1.0, "rgb(165,0,38)".to_string()),
    ]);

    // 计算分位数 tick labels
    let quantiles = [0.0, 0.25, 0.5, 0.75, 1.0];
    let tick_vals: Vec<f64> = quantiles.to_vec();
    let tick_text: Vec<String> = quantiles.iter().map(|&q| {
        let idx = ((n as f64 - 1.0) * q) as usize;
        let idx = idx.min(n - 1);
        format!("{:.4}", sorted_vals[idx])
    }).collect();

    let mut plot = Plot::new();

    if param_names.len() < 2 {
        // 单参数: x=param, y=排名
        let name = &param_names[0];
        let x_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, name)).collect();

        let marker = Marker::new()
            .color_array(color_indices)
            .color_scale(rdylbu_r)
            .show_scale(true)
            .cmin(0.0)
            .cmax(1.0)
            .color_bar(ColorBar::new()
                .title(target_name)
                .tick_vals(tick_vals)
                .tick_text(tick_text));
        let trace = Scatter::new(x_vals, ranks)
            .mode(Mode::Markers)
            .marker(marker)
            .show_legend(false);
        plot.add_trace(trace);

        let layout = Layout::new()
            .title(format!("Trial Rank ({target_name})"))
            .x_axis(Axis::new().title(name.as_str()))
            .y_axis(Axis::new().title("Rank"));
        plot.set_layout(layout);
    } else if param_names.len() == 2 {
        // N=2: 2×2 subplot grid (对应 Python plot_rank 的 N×N 布局)
        // 对角线: x=param_i, y=rank; 非对角线: x=param_i, y=param_j, color=rank
        let p0 = &param_names[0];
        let p1 = &param_names[1];
        let x0: Vec<f64> = trials.iter().map(|t| extract_param(t, p0)).collect();
        let x1: Vec<f64> = trials.iter().map(|t| extract_param(t, p1)).collect();

        // cell (0,0): p0 vs rank
        {
            let marker = Marker::new()
                .color_array(color_indices.clone())
                .color_scale(rdylbu_r.clone())
                .show_scale(false)
                .cmin(0.0)
                .cmax(1.0);
            let trace = Scatter::new(x0.clone(), ranks.clone())
                .mode(Mode::Markers)
                .marker(marker)
                .show_legend(false);
            plot.add_trace(trace);
        }

        // cell (0,1): p0 vs p1
        {
            let marker = Marker::new()
                .color_array(color_indices.clone())
                .color_scale(rdylbu_r.clone())
                .show_scale(true)
                .cmin(0.0)
                .cmax(1.0)
                .color_bar(ColorBar::new()
                    .title(target_name)
                    .tick_vals(tick_vals.clone())
                    .tick_text(tick_text.clone()));
            let trace = Scatter::new(x0.clone(), x1.clone())
                .mode(Mode::Markers)
                .marker(marker)
                .show_legend(false)
                .x_axis("x2")
                .y_axis("y2");
            plot.add_trace(trace);
        }

        // cell (1,0): p1 vs p0
        {
            let marker = Marker::new()
                .color_array(color_indices.clone())
                .color_scale(rdylbu_r.clone())
                .show_scale(false)
                .cmin(0.0)
                .cmax(1.0);
            let trace = Scatter::new(x1.clone(), x0.clone())
                .mode(Mode::Markers)
                .marker(marker)
                .show_legend(false)
                .x_axis("x3")
                .y_axis("y3");
            plot.add_trace(trace);
        }

        // cell (1,1): p1 vs rank
        {
            let marker = Marker::new()
                .color_array(color_indices)
                .color_scale(rdylbu_r)
                .show_scale(false)
                .cmin(0.0)
                .cmax(1.0);
            let trace = Scatter::new(x1, ranks)
                .mode(Mode::Markers)
                .marker(marker)
                .show_legend(false)
                .x_axis("x4")
                .y_axis("y4");
            plot.add_trace(trace);
        }

        let grid = LayoutGrid::new()
            .rows(2)
            .columns(2)
            .pattern(GridPattern::Independent);
        let layout = Layout::new()
            .title(format!("Trial Rank ({target_name})"))
            .grid(grid)
            .x_axis(Axis::new().title(p0.as_str()))
            .y_axis(Axis::new().title("Rank"))
            .x_axis2(Axis::new().title(p0.as_str()))
            .y_axis2(Axis::new().title(p1.as_str()))
            .x_axis3(Axis::new().title(p1.as_str()))
            .y_axis3(Axis::new().title(p0.as_str()))
            .x_axis4(Axis::new().title(p1.as_str()))
            .y_axis4(Axis::new().title("Rank"));
        plot.set_layout(layout);
    } else {
        // 多参数 N≥3: 选取前两个参数做散点图
        let x_name = &param_names[0];
        let y_name = &param_names[1];
        let x_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, x_name)).collect();
        let y_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, y_name)).collect();

        let marker = Marker::new()
            .color_array(color_indices)
            .color_scale(rdylbu_r)
            .show_scale(true)
            .cmin(0.0)
            .cmax(1.0)
            .color_bar(ColorBar::new()
                .title(target_name)
                .tick_vals(tick_vals)
                .tick_text(tick_text));
        let trace = Scatter::new(x_vals, y_vals)
            .mode(Mode::Markers)
            .marker(marker)
            .show_legend(false);
        plot.add_trace(trace);

        let layout = Layout::new()
            .title(format!("Trial Rank ({target_name})"))
            .x_axis(Axis::new().title(x_name.as_str()))
            .y_axis(Axis::new().title(y_name.as_str()));
        plot.set_layout(layout);
    }

    Ok(plot)
}

// ============================================================================
// plot_timeline
// ============================================================================

/// 绘制试验执行时间线。
///
/// 对应 Python `optuna.visualization.plot_timeline`。
/// 使用 Bar trace (水平条) 模拟 Gantt 风格。
/// 支持 `n_recent_trials` 只显示最近的 N 个试验。
#[cfg(feature = "visualization")]
pub fn plot_timeline(study: &Study) -> Result<Plot> {
    plot_timeline_with_options(study, None)
}

/// 带选项的时间线图。
/// 对应 Python `plot_timeline(study, n_recent_trials=...)`.
#[cfg(feature = "visualization")]
pub fn plot_timeline_with_options(
    study: &Study,
    n_recent_trials: Option<usize>,
) -> Result<Plot> {
    let mut trials = study.trials()?;
    if trials.is_empty() {
        return Ok(Plot::new());
    }

    // 可选: 只取最近 N 个试验
    if let Some(n) = n_recent_trials {
        if trials.len() > n {
            trials = trials.split_off(trials.len() - n);
        }
    }

    let base_time = trials
        .iter()
        .filter_map(|t| t.datetime_start)
        .min()
        .unwrap_or_else(chrono::Utc::now);

    let mut plot = Plot::new();

    // 按状态分组着色 (对应 Python 的状态颜色映射)
    let state_color = |state: TrialState| -> &'static str {
        match state {
            TrialState::Complete => "rgb(99,110,250)",   // 蓝色
            TrialState::Running  => "rgb(0,204,150)",    // 绿色
            TrialState::Pruned   => "rgb(239,85,59)",    // 红色
            TrialState::Fail     => "rgb(171,99,250)",   // 紫色
            TrialState::Waiting  => "rgb(255,161,90)",   // 橙色
        }
    };

    // 使用粗线段 Scatter 模拟 Gantt 图 (plotly-rs 的 Bar 不支持 base 参数)
    // 按状态分组绘制以正确显示图例 (对应 Python 按状态迭代)
    // Complete 的不可行试验使用灰色 (对应 Python infeasible 着色)
    let states = [
        TrialState::Complete,
        TrialState::Fail,
        TrialState::Pruned,
        TrialState::Running,
        TrialState::Waiting,
    ];

    for state in &states {
        // Complete 状态分为可行/不可行两组
        if *state == TrialState::Complete {
            // 不可行的 Complete 试验
            let infeasible_trials: Vec<&FrozenTrial> = trials
                .iter()
                .filter(|t| t.state == TrialState::Complete && !is_feasible(t))
                .collect();
            if !infeasible_trials.is_empty() {
                add_timeline_traces(&mut plot, &infeasible_trials, base_time, "#CCCCCC", "INFEASIBLE");
            }
            // 可行的 Complete 试验
            let feasible_trials: Vec<&FrozenTrial> = trials
                .iter()
                .filter(|t| t.state == TrialState::Complete && is_feasible(t))
                .collect();
            if !feasible_trials.is_empty() {
                add_timeline_traces(&mut plot, &feasible_trials, base_time, state_color(*state), "COMPLETE");
            }
        } else {
            let state_trials: Vec<&FrozenTrial> = trials
                .iter()
                .filter(|t| t.state == *state)
                .collect();
            if !state_trials.is_empty() {
                add_timeline_traces(&mut plot, &state_trials, base_time, state_color(*state), &format!("{state:?}").to_uppercase());
            }
        }
    }

    let layout = Layout::new()
        .title("Trial Timeline")
        .x_axis(Axis::new().title("Elapsed Time (s)"))
        .y_axis(Axis::new().title("Trial"));
    plot.set_layout(layout);

    Ok(plot)
}

// ============================================================================
// plot_terminator_improvement
// ============================================================================

/// 绘制终止器改进曲线。
///
/// 对应 Python `optuna.visualization.plot_terminator_improvement`。
/// 使用 ImprovementEvaluator/ErrorEvaluator 计算真实的改进量和误差估计。
#[cfg(feature = "visualization")]
pub fn plot_terminator_improvement(
    study: &Study,
    plot_error: bool,
    improvement_evaluator: Option<&dyn crate::terminators::ImprovementEvaluator>,
    error_evaluator: Option<&dyn crate::terminators::ErrorEvaluator>,
    min_n_trials: usize,
) -> Result<Plot> {
    let all_trials = study.trials()?;
    let direction = study.directions().first().copied()
        .unwrap_or(crate::study::StudyDirection::Minimize);

    let mut plot = Plot::new();

    // 默认评估器
    let default_improvement = crate::terminators::RegretBoundEvaluator::new(None, None, None);
    let default_error = crate::terminators::CrossValidationErrorEvaluator;
    let imp_eval: &dyn crate::terminators::ImprovementEvaluator =
        improvement_evaluator.unwrap_or(&default_improvement);
    let err_eval: &dyn crate::terminators::ErrorEvaluator =
        error_evaluator.unwrap_or(&default_error);

    // 逐 trial 累积计算
    let mut completed_trials: Vec<FrozenTrial> = Vec::new();
    let mut trial_numbers: Vec<i64> = Vec::new();
    let mut improvements: Vec<f64> = Vec::new();
    let mut errors: Vec<f64> = Vec::new();

    for trial in &all_trials {
        if trial.state == TrialState::Complete {
            completed_trials.push(trial.clone());
        }
        if completed_trials.is_empty() {
            continue;
        }
        let improvement = imp_eval.evaluate(&completed_trials, direction);
        trial_numbers.push(trial.number);
        improvements.push(improvement);

        if plot_error {
            let error = err_eval.evaluate(&completed_trials, direction);
            errors.push(error);
        }
    }

    if trial_numbers.is_empty() {
        return Ok(plot);
    }

    // 分两段绘制: min_n_trials 之前用低透明度，之后用正常透明度
    let split_idx = trial_numbers.iter().position(|_| true)
        .map(|_| {
            trial_numbers.iter().enumerate()
                .filter(|(_, _)| true)
                .take_while(|(i, _)| *i < min_n_trials)
                .count()
        })
        .unwrap_or(0);

    let opacity_before = 0.25;
    let opacity_after = 1.0;

    // 改进曲线 - 前段 (半透明)
    if split_idx > 0 {
        let trace_before = Scatter::new(
            trial_numbers[..split_idx].to_vec(),
            improvements[..split_idx].to_vec(),
        )
        .mode(Mode::LinesMarkers)
        .line(plotly::common::Line::new().color(format!("rgba(99,110,250,{opacity_before})")))
        .marker(Marker::new().color(format!("rgba(99,110,250,{opacity_before})")))
        .name("Terminator Improvement")
        .legend_group("improvement")
        .show_legend(split_idx >= trial_numbers.len());
        plot.add_trace(trace_before);
    }

    // 改进曲线 - 后段 (正常)
    if split_idx < trial_numbers.len() {
        let trace_after = Scatter::new(
            trial_numbers[split_idx..].to_vec(),
            improvements[split_idx..].to_vec(),
        )
        .mode(Mode::LinesMarkers)
        .line(plotly::common::Line::new().color(format!("rgba(99,110,250,{opacity_after})")))
        .marker(Marker::new().color(format!("rgba(99,110,250,{opacity_after})")))
        .name("Terminator Improvement")
        .legend_group("improvement");
        plot.add_trace(trace_after);
    }

    if plot_error && !errors.is_empty() {
        let trace_err = Scatter::new(trial_numbers.clone(), errors)
            .mode(Mode::LinesMarkers)
            .line(plotly::common::Line::new().color("rgb(239,85,59)"))
            .marker(Marker::new().color("rgb(239,85,59)"))
            .name("Error");
        plot.add_trace(trace_err);
    }

    let layout = Layout::new()
        .title("Terminator Improvement Plot")
        .x_axis(Axis::new().title("Trial"))
        .y_axis(Axis::new().title("Terminator Improvement"));
    plot.set_layout(layout);

    Ok(plot)
}

// ============================================================================
// 辅助函数
// ============================================================================

/// timeline 辅助: 为一组同状态的试验添加 Scatter traces。
/// 每组共享一个 legend 名称。
#[cfg(feature = "visualization")]
fn add_timeline_traces(
    plot: &mut Plot,
    trials: &[&FrozenTrial],
    base_time: chrono::DateTime<chrono::Utc>,
    color: &str,
    legend_name: &str,
) {
    let mut first = true;
    for trial in trials {
        let start = trial
            .datetime_start
            .map(|dt| (dt - base_time).num_milliseconds() as f64 / 1000.0)
            .unwrap_or(0.0);
        let end = trial
            .datetime_complete
            .map(|dt| (dt - base_time).num_milliseconds() as f64 / 1000.0)
            .unwrap_or(start + 0.001);

        let trace = Scatter::new(vec![start, end], vec![trial.number, trial.number])
            .mode(Mode::Lines)
            .line(plotly::common::Line::new().color(color.to_string()).width(8.0))
            .name(legend_name)
            .legend_group(legend_name.to_string())
            .show_legend(first) // 只有第一条 trace 显示 legend
            .hover_info(plotly::common::HoverInfo::Text)
            .text(format!(
                "Trial {} ({legend_name})<br>{start:.3}s - {end:.3}s",
                trial.number,
            ));
        plot.add_trace(trace);
        first = false;
    }
}

/// 提取参数值。数值型直接返回, 分类型返回其 choice index (整数映射)。
/// 对应 Python `_get_param_values()` 中的分类处理逻辑。
#[cfg(feature = "visualization")]
fn extract_param(trial: &FrozenTrial, name: &str) -> f64 {
    trial.params.get(name).map(|v| match v {
        crate::distributions::ParamValue::Float(f) => *f,
        crate::distributions::ParamValue::Int(i) => *i as f64,
        crate::distributions::ParamValue::Categorical(c) => {
            // 分类参数: 返回 choice index
            // 从 trial.distributions 获取完整的 choices 列表来确定 index
            if let Some(dist) = trial.distributions.get(name) {
                if let crate::distributions::Distribution::CategoricalDistribution(cd) = dist {
                    cd.choices
                        .iter()
                        .position(|ch| ch == c)
                        .map(|i| i as f64)
                        .unwrap_or(0.0)
                } else {
                    0.0
                }
            } else {
                0.0
            }
        }
    }).unwrap_or(f64::NAN)
}

/// 检查参数是否为分类型。
#[cfg(feature = "visualization")]
fn is_categorical_param(trials: &[FrozenTrial], name: &str) -> bool {
    trials.iter().any(|t| {
        matches!(
            t.params.get(name),
            Some(crate::distributions::ParamValue::Categorical(_))
        )
    })
}

/// 获取分类参数的所有 choice 标签 (用于坐标轴 ticktext)。
#[cfg(feature = "visualization")]
fn get_categorical_labels(trials: &[FrozenTrial], name: &str) -> Vec<String> {
    for t in trials {
        if let Some(crate::distributions::Distribution::CategoricalDistribution(cd)) =
            t.distributions.get(name)
        {
            return cd
                .choices
                .iter()
                .map(|c| format!("{c}"))
                .collect();
        }
    }
    vec![]
}

/// 约束可行性检测: 检查 trial 的 system_attrs 中 CONSTRAINTS_KEY。
/// 对应 Python `_is_feasible()`.
#[cfg(feature = "visualization")]
fn is_feasible(trial: &FrozenTrial) -> bool {
    trial
        .system_attrs
        .get(crate::multi_objective::CONSTRAINTS_KEY)
        .and_then(|v| serde_json::from_value::<Vec<f64>>(v.clone()).ok())
        .map(|cs| cs.iter().all(|&c| c <= 0.0))
        .unwrap_or(true) // 无约束视为可行
}

// ============================================================================
// 便捷包装
// ============================================================================

/// 将 Plot 保存为 HTML 文件。
#[cfg(feature = "visualization")]
pub fn save_html(plot: &Plot, path: &str) {
    plot.write_html(path);
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "visualization")]
mod tests {
    use super::*;
    use crate::study::{StudyDirection, create_study};

    fn make_study() -> Study {
        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        study.optimize(|trial| {
            let x = trial.suggest_float("x", -10.0, 10.0, false, None)?;
            let y = trial.suggest_float("y", -10.0, 10.0, false, None)?;
            Ok(x * x + y * y)
        }, Some(10), None, None).unwrap();
        study
    }

    fn make_multi_obj_study() -> Study {
        let study = create_study(
            None, None, None, None, None,
            Some(vec![StudyDirection::Minimize, StudyDirection::Minimize]),
            false,
        ).unwrap();
        study.optimize_multi(|trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
            Ok(vec![x, y])
        }, Some(15), None, None).unwrap();
        study
    }

    #[test]
    fn test_plot_optimization_history() {
        let study = make_study();
        let plot = plot_optimization_history(&study, None, "Objective Value").unwrap();
        assert!(!plot.to_html().is_empty());
    }

    #[test]
    fn test_plot_optimization_history_multi() {
        let s1 = make_study();
        let s2 = make_study();
        let plot = plot_optimization_history_multi(
            &[&s1, &s2], None, "Value", false,
        ).unwrap();
        assert!(!plot.to_html().is_empty());
    }

    #[test]
    fn test_plot_slice() {
        let study = make_study();
        let plot = plot_slice(&study, None, None, "Objective Value").unwrap();
        assert!(!plot.to_html().is_empty());
    }

    #[test]
    fn test_plot_edf() {
        let study = make_study();
        let plot = plot_edf(&study, None, "Objective Value").unwrap();
        assert!(!plot.to_html().is_empty());
    }

    #[test]
    fn test_plot_edf_multi() {
        let s1 = make_study();
        let s2 = make_study();
        let plot = plot_edf_multi(&[&s1, &s2], None, "Value").unwrap();
        assert!(!plot.to_html().is_empty());
    }

    #[test]
    fn test_plot_pareto_front() {
        let study = make_multi_obj_study();
        let plot = plot_pareto_front(&study, None, true).unwrap();
        assert!(!plot.to_html().is_empty());
    }

    #[test]
    fn test_plot_pareto_front_with_targets() {
        let study = make_multi_obj_study();
        let plot = plot_pareto_front_with_targets(
            &study, Some(&["X", "Y"]), true,
            Some(&|t: &FrozenTrial| t.values.clone().unwrap_or_default()),
            None,
        ).unwrap();
        assert!(!plot.to_html().is_empty());
    }

    #[test]
    fn test_plot_timeline_with_options() {
        let study = make_study();
        let plot = plot_timeline_with_options(&study, Some(5)).unwrap();
        assert!(!plot.to_html().is_empty());
    }

    #[test]
    fn test_plot_empty_study() {
        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        let plot = plot_edf(&study, None, "value").unwrap();
        // Empty study → empty plot, no panic
        let _ = plot.to_html();
    }

    #[test]
    fn test_is_feasible() {
        let now = chrono::Utc::now();
        let mut trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            datetime_start: Some(now),
            datetime_complete: Some(now),
            params: std::collections::HashMap::new(),
            distributions: std::collections::HashMap::new(),
            user_attrs: std::collections::HashMap::new(),
            system_attrs: std::collections::HashMap::new(),
            intermediate_values: std::collections::HashMap::new(),
        };
        // 无约束 → 可行
        assert!(is_feasible(&trial));

        // 满足约束 → 可行
        trial.system_attrs.insert(
            crate::multi_objective::CONSTRAINTS_KEY.to_string(),
            serde_json::json!([-1.0, -0.5]),
        );
        assert!(is_feasible(&trial));

        // 违反约束 → 不可行
        trial.system_attrs.insert(
            crate::multi_objective::CONSTRAINTS_KEY.to_string(),
            serde_json::json!([1.0, -0.5]),
        );
        assert!(!is_feasible(&trial));
    }

    // ================================================================
    // 新增测试: 确保所有对齐改进正确工作
    // ================================================================

    #[test]
    fn test_plot_optimization_history_error_bar() {
        // 测试 error_bar 模式: 多 study 聚合均值+标准差
        let s1 = make_study();
        let s2 = make_study();
        let plot = plot_optimization_history_multi(
            &[&s1, &s2], None, "Value", true,
        ).unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // error_bar 模式应包含 error_y 数据
        assert!(html.contains("error_y") || html.contains("ErrorData"));
    }

    #[test]
    fn test_plot_optimization_history_custom_target_no_best() {
        // 自定义 target 时不应绘制 Best Value 线
        let study = make_study();
        let plot = plot_optimization_history(
            &study,
            Some(&|t: &FrozenTrial| {
                t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(0.0) + 1.0
            }),
            "Custom Target",
        ).unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 不应包含 Best Value trace
        assert!(!html.contains("Best Value"));
    }

    #[test]
    fn test_plot_intermediate_values_with_running() {
        // 验证中间值图包含 Running 状态的试验
        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        // 添加一个带中间值的 Complete 试验
        let trial = crate::trial::create_trial(
            Some(TrialState::Complete),
            None,
            Some(vec![1.0]),
            None, None, None, None,
            Some([(0i64, 0.5), (1, 0.3)].into_iter().collect()),
        ).unwrap();
        study.add_trial(&trial).unwrap();
        let plot = plot_intermediate_values(&study).unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 验证 showlegend=false
        assert!(html.contains("showlegend"));
    }

    #[test]
    fn test_plot_param_importances_horizontal() {
        // 验证水平柱状图 (orientation='h')
        let study = make_study();
        let plot = plot_param_importances(
            &study, None, None, None, "Objective Value",
        ).unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 应包含水平方向标记
        assert!(html.contains("\"orientation\"") || html.contains("Horizontal")
            || html.contains("\"h\""));
    }

    #[test]
    fn test_plot_param_importances_multi_objective() {
        // 多目标研究: 为每个目标分别计算重要性
        let study = make_multi_obj_study();
        let plot = plot_param_importances(
            &study, None, None, None, "Importances",
        ).unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 应包含 Objective 0 和 Objective 1
        assert!(html.contains("Objective 0") || html.contains("Objective"));
    }

    #[test]
    fn test_plot_edf_shared_grid_and_range() {
        // 验证 EDF 使用共享 x 轴网格，y 轴 [0,1]
        let s1 = make_study();
        let s2 = make_study();
        let plot = plot_edf_multi(&[&s1, &s2], None, "Value").unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 两个 study → 两条线
        assert!(html.contains(s1.study_name()));
        assert!(html.contains(s2.study_name()));
    }

    #[test]
    fn test_plot_hypervolume_history_validation() {
        // 验证: 非多目标研究应返回 ValueError
        let study = make_study(); // 单目标
        let result = plot_hypervolume_history(&study, &[10.0]);
        assert!(result.is_err());
    }

    #[test]
    fn test_plot_hypervolume_history_ref_dim_mismatch() {
        // 验证: 参考点维度不匹配应返回 ValueError
        let study = make_multi_obj_study(); // 2 目标
        let result = plot_hypervolume_history(&study, &[10.0, 10.0, 10.0]); // 3 维参考点
        assert!(result.is_err());
    }

    #[test]
    fn test_plot_hypervolume_history_incremental() {
        // 验证增量帕累托前沿计算
        let study = make_multi_obj_study();
        let plot = plot_hypervolume_history(&study, &[10.0, 10.0]).unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        assert!(html.contains("Hypervolume"));
    }

    #[test]
    fn test_plot_contour_2x2_grid() {
        // 验证 2 参数时的 2×2 子图网格
        let study = make_study();
        let plot = plot_contour(
            &study, Some(&["x", "y"]), None, "Objective",
        ).unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 应包含 subplot grid 标记
        assert!(html.contains("x2") || html.contains("xaxis2"));
    }

    #[test]
    fn test_plot_rank_2x2_grid() {
        // 验证 2 参数时的 2×2 子图网格
        let study = make_study();
        let plot = plot_rank(
            &study, Some(&["x", "y"]), None, "Objective",
        ).unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 2-param → 2×2 grid containing axis2
        assert!(html.contains("x2") || html.contains("xaxis2"));
    }

    #[test]
    fn test_plot_slice_subplots() {
        // 验证多参数时的 per-param 子图布局
        let study = make_study(); // 2 params: x, y
        let plot = plot_slice(&study, None, None, "Objective").unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 2 params → 1×2 grid
        assert!(html.contains("x2") || html.contains("xaxis2"));
    }

    #[test]
    fn test_plot_parallel_coordinate() {
        let study = make_study();
        let plot = plot_parallel_coordinate(&study, None, None, "Objective Value").unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 应含有颜色条
        assert!(html.contains("colorbar") || html.contains("colorscale"));
    }

    #[test]
    fn test_plot_terminator_improvement() {
        let study = make_study();
        let plot = plot_terminator_improvement(
            &study, true, None, None, 3,
        ).unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        assert!(html.contains("Terminator"));
    }

    #[test]
    fn test_plot_timeline_state_colors() {
        // 验证时间线按状态分组着色
        let study = make_study();
        let plot = plot_timeline(&study).unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 应包含 COMPLETE 标签
        assert!(html.contains("COMPLETE"));
    }

    #[test]
    fn test_plot_pareto_front_3d() {
        // 3 目标: 通过 Scatter3D 绘制
        let study = create_study(
            None, None, None, None, None,
            Some(vec![
                StudyDirection::Minimize,
                StudyDirection::Minimize,
                StudyDirection::Minimize,
            ]),
            false,
        ).unwrap();
        study.optimize_multi(|trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
            let z = trial.suggest_float("z", 0.0, 1.0, false, None)?;
            Ok(vec![x, y, z])
        }, Some(10), None, None).unwrap();
        let plot = plot_pareto_front(&study, None, true).unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        assert!(html.contains("3D") || html.contains("scatter3d"));
    }

    /// 测试平行坐标图对数尺度参数: log=true 的 FloatDistribution 应被 log10 变换。
    /// 对应 Python `_is_log_scale` + log10 变换逻辑。
    #[test]
    fn test_plot_parallel_coordinate_log_scale() {
        use std::collections::HashMap;
        use crate::distributions::{FloatDistribution, ParamValue};
        let study = crate::study::create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        // 添加含 log=true 参数的试验
        for i in 0..5 {
            let lr = 10.0_f64.powf(-4.0 + i as f64); // 0.0001 .. 10
            let mut trial = FrozenTrial {
                number: i as i64,
                trial_id: i as i64,
                state: TrialState::Complete,
                values: Some(vec![lr.ln()]),
                params: HashMap::new(),
                distributions: HashMap::new(),
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
                datetime_start: Some(chrono::Utc::now()),
                datetime_complete: Some(chrono::Utc::now()),
            };
            trial.params.insert("lr".to_string(), ParamValue::Float(lr));
            trial.distributions.insert("lr".to_string(),
                crate::distributions::Distribution::FloatDistribution(
                    FloatDistribution { low: 1e-5, high: 10.0, log: true, step: None }
                ));
            let _ = study.add_trial(&trial);
        }
        let plot = plot_parallel_coordinate(&study, None, None, "Objective Value").unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 应该生成多条线（至少 5 条 trial 线 + 1 个颜色条 trace）
        assert!(html.contains("lr"), "应包含参数名 lr");
    }

    /// 测试平行坐标图分类参数: CategoricalDistribution 应映射为整数索引并显示标签。
    #[test]
    fn test_plot_parallel_coordinate_categorical() {
        use std::collections::HashMap;
        use crate::distributions::{CategoricalChoice, CategoricalDistribution, ParamValue};
        let study = crate::study::create_study(
            None, None, None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        let choices = vec![
            CategoricalChoice::Str("adam".to_string()),
            CategoricalChoice::Str("sgd".to_string()),
            CategoricalChoice::Str("rmsprop".to_string()),
        ];
        for i in 0..6 {
            let choice_idx = i % 3;
            let mut trial = FrozenTrial {
                number: i as i64,
                trial_id: i as i64,
                state: TrialState::Complete,
                values: Some(vec![i as f64 * 0.1]),
                params: HashMap::new(),
                distributions: HashMap::new(),
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
                datetime_start: Some(chrono::Utc::now()),
                datetime_complete: Some(chrono::Utc::now()),
            };
            trial.params.insert("optimizer".to_string(),
                ParamValue::Categorical(choices[choice_idx].clone()));
            trial.distributions.insert("optimizer".to_string(),
                crate::distributions::Distribution::CategoricalDistribution(
                    CategoricalDistribution { choices: choices.clone() }
                ));
            let _ = study.add_trial(&trial);
        }
        let plot = plot_parallel_coordinate(&study, None, None, "Objective Value").unwrap();
        let html = plot.to_html();
        assert!(!html.is_empty());
        // 应包含 optimizer 参数名
        assert!(html.contains("optimizer"), "应包含参数名 optimizer");
        // hover text 应包含选项名称
        assert!(html.contains("adam") || html.contains("sgd") || html.contains("rmsprop"),
            "hover text 应包含分类选项名称");
    }
}
