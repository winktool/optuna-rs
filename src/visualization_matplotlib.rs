//! Matplotlib 风格可视化后端（基于 plotters）。
//!
//! 对应 Python `optuna.visualization.matplotlib` 模块。
//! 使用 [`plotters`](https://docs.rs/plotters) crate 生成 SVG/PNG 静态图表，
//! 不需要浏览器即可查看。
//!
//! # 使用方式
//! ```toml
//! optuna-rs = { version = "0.1", features = ["visualization-matplotlib"] }
//! ```
//!
//! # 与 plotly 后端的区别
//! | | plotly (visualization) | plotters (visualization-matplotlib) |
//! |---|---|---|
//! | 输出格式 | HTML (交互式) | SVG / PNG (静态) |
//! | 依赖 | plotly 0.11 | plotters 0.3 |
//! | 查看方式 | 浏览器 | 任何图片查看器 |
//! | 文件大小 | 较大 (含 JS) | 较小 |
//!
//! # 支持的图表（12 个，与 plotly 后端 1:1 对应）
//! - [`plot_optimization_history`] — 优化历史
//! - [`plot_intermediate_values`] — 中间值
//! - [`plot_parallel_coordinate`] — 平行坐标图
//! - [`plot_contour`] — 等高线图（热力图近似)
//! - [`plot_slice`] — 切片图
//! - [`plot_edf`] — 经验分布函数
//! - [`plot_param_importances`] — 参数重要性
//! - [`plot_pareto_front`] — 帕累托前沿
//! - [`plot_hypervolume_history`] — 超体积历史
//! - [`plot_rank`] — 试验排名
//! - [`plot_timeline`] — 时间线
//! - [`plot_terminator_improvement`] — 终止器改进
//! - [`save_svg`] — 直接保存为 SVG 文件

#[cfg(feature = "visualization-matplotlib")]
use plotters::prelude::*;

#[cfg(feature = "visualization-matplotlib")]
use crate::error::{OptunaError, Result};
#[cfg(feature = "visualization-matplotlib")]
use crate::study::Study;
#[cfg(feature = "visualization-matplotlib")]
use crate::trial::{FrozenTrial, TrialState};

/// SVG 字符串包装——matplotlib 后端的输出类型。
///
/// 可直接写入文件或转换为字符串。
#[cfg(feature = "visualization-matplotlib")]
#[derive(Debug, Clone)]
pub struct SvgPlot {
    /// SVG 内容
    pub svg: String,
}

#[cfg(feature = "visualization-matplotlib")]
impl SvgPlot {
    /// 获取 SVG 内容。
    pub fn to_svg(&self) -> &str {
        &self.svg
    }

    /// 保存为 SVG 文件。
    pub fn save(&self, path: &str) -> Result<()> {
        std::fs::write(path, &self.svg).map_err(|e| {
            OptunaError::StorageInternalError(format!("写入 SVG 失败: {e}"))
        })
    }
}

// 默认图表尺寸
#[cfg(feature = "visualization-matplotlib")]
const W: u32 = 800;
#[cfg(feature = "visualization-matplotlib")]
const H: u32 = 600;

// 调色板（matplotlib tab10 前 10 色）
#[cfg(feature = "visualization-matplotlib")]
const TAB10: [RGBColor; 10] = [
    RGBColor(31, 119, 180),
    RGBColor(255, 127, 14),
    RGBColor(44, 160, 44),
    RGBColor(214, 39, 40),
    RGBColor(148, 103, 189),
    RGBColor(140, 86, 75),
    RGBColor(227, 119, 194),
    RGBColor(127, 127, 127),
    RGBColor(188, 189, 34),
    RGBColor(23, 190, 207),
];

#[cfg(feature = "visualization-matplotlib")]
fn tab10(i: usize) -> RGBColor {
    TAB10[i % TAB10.len()]
}

/// Viridis 色到 RGB（简化 5 段插值）
#[cfg(feature = "visualization-matplotlib")]
fn viridis(t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    // 5 个关键点
    let stops: [(f64, f64, f64); 5] = [
        (68.0, 1.0, 84.0),     // 0.00
        (59.0, 82.0, 139.0),   // 0.25
        (33.0, 145.0, 140.0),  // 0.50
        (94.0, 201.0, 98.0),   // 0.75
        (253.0, 231.0, 37.0),  // 1.00
    ];
    let idx = (t * 4.0).min(3.999);
    let i = idx as usize;
    let frac = idx - i as f64;
    let (r0, g0, b0) = stops[i];
    let (r1, g1, b1) = stops[i + 1];
    RGBColor(
        (r0 + frac * (r1 - r0)) as u8,
        (g0 + frac * (g1 - g0)) as u8,
        (b0 + frac * (b1 - b0)) as u8,
    )
}

/// RdYlBu reversed（排名图用）
#[cfg(feature = "visualization-matplotlib")]
fn rdylbu_r(t: f64) -> RGBColor {
    let t = t.clamp(0.0, 1.0);
    let stops: [(f64, f64, f64); 5] = [
        (49.0, 54.0, 149.0),
        (116.0, 173.0, 209.0),
        (255.0, 255.0, 191.0),
        (244.0, 109.0, 67.0),
        (165.0, 0.0, 38.0),
    ];
    let idx = (t * 4.0).min(3.999);
    let i = idx as usize;
    let frac = idx - i as f64;
    let (r0, g0, b0) = stops[i];
    let (r1, g1, b1) = stops[i + 1];
    RGBColor(
        (r0 + frac * (r1 - r0)) as u8,
        (g0 + frac * (g1 - g0)) as u8,
        (b0 + frac * (b1 - b0)) as u8,
    )
}

#[cfg(feature = "visualization-matplotlib")]
fn extract_param(trial: &FrozenTrial, name: &str) -> f64 {
    trial.params.get(name).map(|v| match v {
        crate::distributions::ParamValue::Float(f) => *f,
        crate::distributions::ParamValue::Int(i) => *i as f64,
        crate::distributions::ParamValue::Categorical(c) => {
            // 分类参数: 返回 choice index
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

/// 约束可行性检测。
#[cfg(feature = "visualization-matplotlib")]
fn is_feasible(trial: &FrozenTrial) -> bool {
    trial
        .system_attrs
        .get(crate::multi_objective::CONSTRAINTS_KEY)
        .and_then(|v| serde_json::from_value::<Vec<f64>>(v.clone()).ok())
        .map(|cs| cs.iter().all(|&c| c <= 0.0))
        .unwrap_or(true)
}

/// 计算数据范围并加 5% padding
#[cfg(feature = "visualization-matplotlib")]
fn padded_range(vals: &[f64]) -> (f64, f64) {
    let finite: Vec<f64> = vals.iter().copied().filter(|v| v.is_finite()).collect();
    if finite.is_empty() {
        return (0.0, 1.0);
    }
    let mn = finite.iter().copied().fold(f64::INFINITY, f64::min);
    let mx = finite.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let span = (mx - mn).max(1e-10);
    let pad = span * 0.05;
    (mn - pad, mx + pad)
}

#[cfg(feature = "visualization-matplotlib")]
macro_rules! svg_plot {
    ($body:expr) => {{
        let mut buf = String::new();
        {
            let root = SVGBackend::with_string(&mut buf, (W, H)).into_drawing_area();
            root.fill(&WHITE).map_err(|e| OptunaError::StorageInternalError(format!("SVG fill: {e}")))?;
            let result: std::result::Result<(), Box<dyn std::error::Error>> = $body(&root);
            result.map_err(|e| OptunaError::StorageInternalError(format!("SVG plot: {e}")))?;
            root.present().map_err(|e| OptunaError::StorageInternalError(format!("SVG present: {e}")))?;
        }
        Ok(SvgPlot { svg: buf })
    }};
}

// ============================================================================
// 1. plot_optimization_history
// ============================================================================

/// 绘制优化历史（SVG）。
///
/// 对应 Python `optuna.visualization.matplotlib.plot_optimization_history`。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_optimization_history(
    study: &Study,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<SvgPlot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    if trials.is_empty() {
        return empty_svg("Optimization History (no trials)");
    }

    let numbers: Vec<f64> = trials.iter().map(|t| t.number as f64).collect();
    let values: Vec<f64> = trials.iter().map(|t| {
        target.map_or_else(
            || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
            |f| f(t),
        )
    }).collect();

    let is_minimize = matches!(
        study.directions().first(),
        Some(crate::study::StudyDirection::Minimize)
    );
    let mut best_values = Vec::with_capacity(values.len());
    let mut best = if is_minimize { f64::INFINITY } else { f64::NEG_INFINITY };
    for &v in &values {
        if is_minimize { if v < best { best = v; } } else if v > best { best = v; }
        best_values.push(best);
    }

    let x_range = padded_range(&numbers);
    let all_y: Vec<f64> = values.iter().chain(best_values.iter()).copied().collect();
    let y_range = padded_range(&all_y);
    let target_name_owned = target_name.to_string();

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption(format!("Optimization History ({target_name_owned})"), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        chart.configure_mesh()
            .x_desc("Trial")
            .y_desc(&target_name_owned)
            .draw()?;

        // 散点
        chart.draw_series(
            numbers.iter().zip(values.iter())
                .filter(|(_, y)| y.is_finite())
                .map(|(&x, &y)| Circle::new((x, y), 3, tab10(0).filled()))
        )?.label(&target_name_owned).legend(|(x, y)| Circle::new((x, y), 3, tab10(0).filled()));

        // 最佳值折线
        let best_pts: Vec<(f64, f64)> = numbers.iter().zip(best_values.iter())
            .filter(|(_, y)| y.is_finite())
            .map(|(&x, &y)| (x, y)).collect();
        chart.draw_series(LineSeries::new(best_pts, tab10(1).stroke_width(2)))?
            .label("Best Value").legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], tab10(1).stroke_width(2)));

        chart.configure_series_labels().border_style(BLACK).draw()?;
        Ok(())
    })
}

// ============================================================================
// 2. plot_intermediate_values
// ============================================================================

/// 绘制中间值（SVG）。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_intermediate_values(study: &Study) -> Result<SvgPlot> {
    let trials = study.get_trials(Some(&[TrialState::Complete, TrialState::Pruned]))?;
    let trial_curves: Vec<(i64, Vec<(f64, f64)>)> = trials.iter()
        .filter(|t| !t.intermediate_values.is_empty())
        .map(|t| {
            let mut steps: Vec<i64> = t.intermediate_values.keys().copied().collect();
            steps.sort();
            let pts: Vec<(f64, f64)> = steps.iter().map(|&s| (s as f64, t.intermediate_values[&s])).collect();
            (t.number, pts)
        })
        .collect();

    if trial_curves.is_empty() {
        return empty_svg("Intermediate Values (no data)");
    }

    let all_x: Vec<f64> = trial_curves.iter().flat_map(|(_, pts)| pts.iter().map(|p| p.0)).collect();
    let all_y: Vec<f64> = trial_curves.iter().flat_map(|(_, pts)| pts.iter().map(|p| p.1)).collect();
    let x_range = padded_range(&all_x);
    let y_range = padded_range(&all_y);

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption("Intermediate Values", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        chart.configure_mesh().x_desc("Step").y_desc("Value").draw()?;

        for (i, (num, pts)) in trial_curves.iter().enumerate() {
            let color = tab10(i);
            chart.draw_series(LineSeries::new(pts.iter().copied(), color.stroke_width(1)))?
                .label(format!("Trial {num}"))
                .legend(move |(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], color));
        }

        if trial_curves.len() <= 20 {
            chart.configure_series_labels().border_style(BLACK).draw()?;
        }
        Ok(())
    })
}

// ============================================================================
// 3. plot_parallel_coordinate
// ============================================================================

/// 绘制平行坐标图（SVG）。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_parallel_coordinate(
    study: &Study,
    params: Option<&[&str]>,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<SvgPlot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    if trials.is_empty() {
        return empty_svg("Parallel Coordinate (no trials)");
    }

    let param_names: Vec<String> = if let Some(p) = params {
        p.iter().map(|s| s.to_string()).collect()
    } else {
        let mut names: Vec<String> = trials.first()
            .map(|t| t.params.keys().cloned().collect())
            .unwrap_or_default();
        names.sort();
        names
    };

    let filtered: Vec<&FrozenTrial> = trials.iter()
        .filter(|t| param_names.iter().all(|n| t.params.contains_key(n)))
        .collect();
    if filtered.is_empty() {
        return empty_svg("Parallel Coordinate (no matching trials)");
    }

    let obj_values: Vec<f64> = filtered.iter().map(|t| {
        target.map_or_else(
            || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
            |f| f(t),
        )
    }).collect();
    let obj_min = obj_values.iter().copied().fold(f64::INFINITY, f64::min);
    let obj_max = obj_values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let obj_range = (obj_max - obj_min).max(1e-10);

    let n_dims = param_names.len() + 1; // objective + params
    let target_name_owned = target_name.to_string();

    // 计算归一化参数
    let mut param_mins = Vec::new();
    let mut param_ranges = Vec::new();
    for name in &param_names {
        let vals: Vec<f64> = filtered.iter().map(|t| extract_param(t, name)).collect();
        let mn = vals.iter().copied().fold(f64::INFINITY, f64::min);
        let mx = vals.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        param_mins.push(mn);
        param_ranges.push((mx - mn).max(1e-10));
    }

    // 预计算各试验的归一化坐标
    let all_lines: Vec<(f64, Vec<(f64, f64)>)> = filtered.iter().enumerate().map(|(i, trial)| {
        let obj_val = obj_values[i];
        let color_t = (obj_val - obj_min) / obj_range;
        let mut points = Vec::with_capacity(n_dims);
        // first dimension: objective
        points.push((0.0, color_t));
        for (j, name) in param_names.iter().enumerate() {
            let raw = extract_param(trial, name);
            let norm = (raw - param_mins[j]) / param_ranges[j];
            points.push(((j + 1) as f64, norm));
        }
        (color_t, points)
    }).collect();

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption(format!("Parallel Coordinate ({target_name_owned})"), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(50)
            .build_cartesian_2d(
                -0.2..(n_dims as f64 - 0.8),
                -0.05..1.05,
            )?;

        chart.configure_mesh()
            .disable_x_mesh()
            .y_desc("Normalized Value")
            .draw()?;

        for (color_t, pts) in &all_lines {
            let color = viridis(*color_t);
            chart.draw_series(LineSeries::new(pts.iter().copied(), color.stroke_width(1)))?;
        }

        // 轴标签
        for (j, name) in std::iter::once(&target_name_owned).chain(param_names.iter()).enumerate() {
            chart.draw_series(std::iter::once(
                plotters::element::Text::new(name.clone(), (j as f64, -0.03), ("sans-serif", 10).into_font())
            ))?;
        }

        Ok(())
    })
}

// ============================================================================
// 4. plot_contour
// ============================================================================

/// 绘制等高线图（热力图近似，SVG）。
///
/// plotters 没有原生等高线支持，使用 矩形色块 热力图模拟。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_contour(
    study: &Study,
    params: Option<&[&str]>,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<SvgPlot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    if trials.is_empty() {
        return empty_svg("Contour (no trials)");
    }

    let param_names: Vec<String> = if let Some(p) = params {
        p.iter().map(|s| s.to_string()).collect()
    } else {
        let mut names: Vec<String> = trials.first()
            .map(|t| t.params.keys().cloned().collect())
            .unwrap_or_default();
        names.sort();
        names.into_iter().take(2).collect()
    };

    if param_names.len() < 2 {
        return empty_svg("Contour (need >= 2 params)");
    }

    let x_name = param_names[0].clone();
    let y_name = param_names[1].clone();

    let x_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, &x_name)).collect();
    let y_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, &y_name)).collect();
    let z_vals: Vec<f64> = trials.iter().map(|t| {
        target.map_or_else(
            || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
            |f| f(t),
        )
    }).collect();

    let x_range = padded_range(&x_vals);
    let y_range = padded_range(&y_vals);
    let z_min = z_vals.iter().copied().filter(|v| v.is_finite()).fold(f64::INFINITY, f64::min);
    let z_max = z_vals.iter().copied().filter(|v| v.is_finite()).fold(f64::NEG_INFINITY, f64::max);
    let z_span = (z_max - z_min).max(1e-10);

    let is_minimize = matches!(
        study.directions().first(),
        Some(crate::study::StudyDirection::Minimize)
    );

    // 构建网格
    let grid_n = 30usize;
    let dx = (x_range.1 - x_range.0) / grid_n as f64;
    let dy = (y_range.1 - y_range.0) / grid_n as f64;

    // 简单最近邻插值
    let mut grid = vec![vec![f64::NAN; grid_n]; grid_n];
    for gy in 0..grid_n {
        for gx in 0..grid_n {
            let cx = x_range.0 + (gx as f64 + 0.5) * dx;
            let cy = y_range.0 + (gy as f64 + 0.5) * dy;
            // 高斯权重最近邻
            let sigma2 = (dx * dx + dy * dy) * 4.0;
            let mut wsum = 0.0f64;
            let mut vsum = 0.0f64;
            for k in 0..x_vals.len() {
                if !x_vals[k].is_finite() || !y_vals[k].is_finite() || !z_vals[k].is_finite() {
                    continue;
                }
                let d2 = (x_vals[k] - cx).powi(2) + (y_vals[k] - cy).powi(2);
                let w = (-d2 / sigma2).exp();
                wsum += w;
                vsum += w * z_vals[k];
            }
            if wsum > 1e-12 {
                grid[gy][gx] = vsum / wsum;
            }
        }
    }

    let target_name_owned = target_name.to_string();

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption(format!("Contour ({target_name_owned})"), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        chart.configure_mesh()
            .x_desc(&x_name)
            .y_desc(&y_name)
            .draw()?;

        // 绘制热力图色块
        for gy in 0..grid_n {
            for gx in 0..grid_n {
                let val = grid[gy][gx];
                if !val.is_finite() { continue; }
                let t = (val - z_min) / z_span;
                let t = if is_minimize { t } else { 1.0 - t };
                let color = viridis(t);
                let x0 = x_range.0 + gx as f64 * dx;
                let y0 = y_range.0 + gy as f64 * dy;
                chart.draw_series(std::iter::once(
                    Rectangle::new([(x0, y0), (x0 + dx, y0 + dy)], color.filled())
                ))?;
            }
        }

        // 散点叠加
        chart.draw_series(
            x_vals.iter().zip(y_vals.iter())
                .filter(|(x, y)| x.is_finite() && y.is_finite())
                .map(|(&x, &y)| Circle::new((x, y), 3, BLACK.filled()))
        )?;

        Ok(())
    })
}

// ============================================================================
// 5. plot_slice
// ============================================================================

/// 绘制切片图（SVG）。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_slice(
    study: &Study,
    params: Option<&[&str]>,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<SvgPlot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    if trials.is_empty() {
        return empty_svg("Slice Plot (no trials)");
    }

    let param_names: Vec<String> = if let Some(p) = params {
        p.iter().map(|s| s.to_string()).collect()
    } else {
        let mut names: Vec<String> = trials.first()
            .map(|t| t.params.keys().cloned().collect())
            .unwrap_or_default();
        names.sort();
        names
    };

    let values: Vec<f64> = trials.iter().map(|t| {
        target.map_or_else(
            || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
            |f| f(t),
        )
    }).collect();

    let all_params_x: Vec<Vec<f64>> = param_names.iter()
        .map(|name| trials.iter().map(|t| extract_param(t, name)).collect())
        .collect();

    // 用第一个参数范围做 x 轴（多参数共享 y 轴）
    let all_x: Vec<f64> = all_params_x.iter().flat_map(|v| v.iter().copied()).collect();
    let x_range = padded_range(&all_x);
    let y_range = padded_range(&values);
    let target_name_owned = target_name.to_string();

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption(format!("Slice Plot ({target_name_owned})"), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        chart.configure_mesh()
            .x_desc("Parameter Value")
            .y_desc(&target_name_owned)
            .draw()?;

        for (i, (name, x_vals)) in param_names.iter().zip(all_params_x.iter()).enumerate() {
            let color = tab10(i);
            let name_clone = name.clone();
            chart.draw_series(
                x_vals.iter().zip(values.iter())
                    .filter(|(x, y)| x.is_finite() && y.is_finite())
                    .map(move |(&x, &y)| Circle::new((x, y), 3, color.filled()))
            )?.label(name_clone).legend(move |(x, y)| Circle::new((x, y), 3, color.filled()));
        }

        chart.configure_series_labels().border_style(BLACK).draw()?;
        Ok(())
    })
}

// ============================================================================
// 6. plot_edf
// ============================================================================

/// 绘制经验分布函数（SVG）。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_edf(
    study: &Study,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<SvgPlot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    if trials.is_empty() {
        return empty_svg("EDF (no trials)");
    }

    let mut values: Vec<f64> = trials.iter().map(|t| {
        target.map_or_else(
            || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
            |f| f(t),
        )
    }).collect();
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = values.len() as f64;
    let probs: Vec<f64> = (1..=values.len()).map(|i| i as f64 / n).collect();

    let x_range = padded_range(&values);
    let target_name_owned = target_name.to_string();
    let study_name = study.study_name().to_string();

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption("Empirical Distribution Function", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, -0.05..1.05)?;

        chart.configure_mesh()
            .x_desc(&target_name_owned)
            .y_desc("Cumulative Probability")
            .draw()?;

        let pts: Vec<(f64, f64)> = values.iter().zip(probs.iter()).map(|(&x, &y)| (x, y)).collect();
        chart.draw_series(LineSeries::new(pts, tab10(0).stroke_width(2)))?
            .label(&study_name)
            .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], tab10(0).stroke_width(2)));

        chart.configure_series_labels().border_style(BLACK).draw()?;
        Ok(())
    })
}

// ============================================================================
// 7. plot_param_importances
// ============================================================================

/// 绘制参数重要性柱状图（SVG）。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_param_importances(
    study: &Study,
    evaluator: Option<&dyn crate::importance::ImportanceEvaluator>,
    params: Option<&[&str]>,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<SvgPlot> {
    let importances = crate::importance::get_param_importances(
        study, evaluator, params, target, true,
    )?;

    let mut entries: Vec<(String, f64)> = importances.into_iter().collect();
    entries.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

    if entries.is_empty() {
        return empty_svg("Parameter Importances (no data)");
    }

    let names: Vec<String> = entries.iter().map(|(k, _)| k.clone()).collect();
    let values: Vec<f64> = entries.iter().map(|(_, v)| *v).collect();
    let max_val = values.iter().copied().fold(0.0f64, f64::max).max(0.01);
    let n = names.len();
    let target_name_owned = target_name.to_string();

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption(format!("Hyperparameter Importances ({target_name_owned})"), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(120)
            .build_cartesian_2d(0.0..max_val * 1.1, 0..n)?;

        chart.configure_mesh()
            .x_desc("Importance")
            .y_label_formatter(&|y| {
                names.get(*y as usize).cloned().unwrap_or_default()
            })
            .draw()?;

        chart.draw_series(
            (0..n).map(|i| {
                let color = tab10(0);
                Rectangle::new([(0.0, i), (values[i], i + 1)], color.filled())
            })
        )?;

        Ok(())
    })
}

// ============================================================================
// 8. plot_pareto_front
// ============================================================================

/// 绘制帕累托前沿（SVG）。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_pareto_front(
    study: &Study,
    target_names: Option<&[&str]>,
    include_dominated: bool,
) -> Result<SvgPlot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    let pareto = study.best_trials()?;
    let n_obj = study.directions().len();

    if n_obj < 2 || trials.is_empty() {
        return empty_svg("Pareto Front (need >= 2 objectives)");
    }

    let obj_names: Vec<String> = if let Some(names) = target_names {
        names.iter().map(|s| s.to_string()).collect()
    } else {
        (0..n_obj).map(|i| format!("Objective {i}")).collect()
    };

    let px: Vec<f64> = pareto.iter().filter_map(|t| t.values.as_ref().map(|v| v[0])).collect();
    let py: Vec<f64> = pareto.iter().filter_map(|t| t.values.as_ref().map(|v| v[1])).collect();

    let mut all_x = px.clone();
    let mut all_y = py.clone();

    let pareto_nums: std::collections::HashSet<i64> = pareto.iter().map(|t| t.number).collect();
    let dominated: Vec<&FrozenTrial> = trials.iter().filter(|t| !pareto_nums.contains(&t.number)).collect();
    let dx: Vec<f64> = dominated.iter().filter_map(|t| t.values.as_ref().map(|v| v[0])).collect();
    let dy: Vec<f64> = dominated.iter().filter_map(|t| t.values.as_ref().map(|v| v[1])).collect();

    if include_dominated {
        all_x.extend(&dx);
        all_y.extend(&dy);
    }

    let x_range = padded_range(&all_x);
    let y_range = padded_range(&all_y);

    let obj0 = obj_names.first().cloned().unwrap_or_else(|| "Obj 0".into());
    let obj1 = obj_names.get(1).cloned().unwrap_or_else(|| "Obj 1".into());

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption("Pareto Front", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        chart.configure_mesh()
            .x_desc(&obj0)
            .y_desc(&obj1)
            .draw()?;

        // 帕累托前沿
        chart.draw_series(
            px.iter().zip(py.iter()).map(|(&x, &y)| Circle::new((x, y), 4, tab10(0).filled()))
        )?.label("Pareto Front").legend(|(x, y)| Circle::new((x, y), 4, tab10(0).filled()));

        // 被支配点
        if include_dominated && !dx.is_empty() {
            chart.draw_series(
                dx.iter().zip(dy.iter()).map(|(&x, &y)| Circle::new((x, y), 3, tab10(1).filled()))
            )?.label("Dominated").legend(|(x, y)| Circle::new((x, y), 3, tab10(1).filled()));
        }

        chart.configure_series_labels().border_style(BLACK).draw()?;
        Ok(())
    })
}

// ============================================================================
// 9. plot_hypervolume_history
// ============================================================================

/// 绘制超体积历史（SVG）。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_hypervolume_history(
    study: &Study,
    reference_point: &[f64],
) -> Result<SvgPlot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    let directions = study.directions();

    let mut hvs = Vec::new();
    let mut nums = Vec::new();

    for i in 0..trials.len() {
        let points: Vec<Vec<f64>> = trials[..=i].iter()
            .filter_map(|t| t.values.clone())
            .collect();
        if points.is_empty() { continue; }

        let normalized: Vec<Vec<f64>> = points.iter().map(|p| {
            p.iter().zip(directions.iter()).map(|(&v, d)| match d {
                crate::study::StudyDirection::Maximize => -v,
                _ => v,
            }).collect()
        }).collect();

        let neg_ref: Vec<f64> = reference_point.iter().zip(directions.iter()).map(|(&v, d)| match d {
            crate::study::StudyDirection::Maximize => -v,
            _ => v,
        }).collect();

        hvs.push(crate::multi_objective::hypervolume(&normalized, &neg_ref));
        nums.push(trials[i].number as f64);
    }

    if hvs.is_empty() {
        return empty_svg("Hypervolume History (no data)");
    }

    let x_range = padded_range(&nums);
    let y_range = padded_range(&hvs);

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption("Hypervolume History", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        chart.configure_mesh().x_desc("Trial").y_desc("Hypervolume").draw()?;

        let pts: Vec<(f64, f64)> = nums.iter().zip(hvs.iter()).map(|(&x, &y)| (x, y)).collect();
        chart.draw_series(LineSeries::new(pts.clone(), tab10(0).stroke_width(2)))?;
        chart.draw_series(pts.iter().map(|&(x, y)| Circle::new((x, y), 3, tab10(0).filled())))?;

        Ok(())
    })
}

// ============================================================================
// 10. plot_rank
// ============================================================================

/// 绘制试验排名图（SVG）。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_rank(
    study: &Study,
    params: Option<&[&str]>,
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<SvgPlot> {
    let trials = study.get_trials(Some(&[TrialState::Complete]))?;
    if trials.is_empty() {
        return empty_svg("Trial Rank (no trials)");
    }

    let param_names: Vec<String> = if let Some(p) = params {
        p.iter().map(|s| s.to_string()).collect()
    } else {
        let mut names: Vec<String> = trials.first()
            .map(|t| t.params.keys().cloned().collect())
            .unwrap_or_default();
        names.sort();
        names
    };

    let obj_values: Vec<f64> = trials.iter().map(|t| {
        target.map_or_else(
            || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
            |f| f(t),
        )
    }).collect();

    let n = obj_values.len();
    let mut sorted = obj_values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let mut ranks: Vec<f64> = vec![0.0; n];
    for i in 0..n {
        let val = obj_values[i];
        let indices: Vec<usize> = sorted.iter().enumerate()
            .filter(|(_, sv)| (**sv - val).abs() < 1e-12)
            .map(|(j, _)| j).collect();
        ranks[i] = indices.iter().sum::<usize>() as f64 / indices.len() as f64;
    }
    let max_rank = (n - 1).max(1) as f64;

    let target_name_owned = target_name.to_string();

    if param_names.len() < 2 {
        let name = param_names.first().map(|s| s.as_str()).unwrap_or("param");
        let x_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, name)).collect();
        let x_range = padded_range(&x_vals);
        let y_range = padded_range(&ranks);
        let name_owned = name.to_string();

        svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
            let mut chart = ChartBuilder::on(root)
                .caption(format!("Trial Rank ({target_name_owned})"), ("sans-serif", 20))
                .margin(10)
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

            chart.configure_mesh().x_desc(&name_owned).y_desc("Rank").draw()?;

            chart.draw_series(
                x_vals.iter().zip(ranks.iter()).enumerate()
                    .filter(|(_, (x, _))| x.is_finite())
                    .map(|(i, (&x, &r))| {
                        let t = ranks[i] / max_rank;
                        Circle::new((x, r), 4, rdylbu_r(t).filled())
                    })
            )?;
            Ok(())
        })
    } else {
        let x_name = &param_names[0];
        let y_name = &param_names[1];
        let x_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, x_name)).collect();
        let y_vals: Vec<f64> = trials.iter().map(|t| extract_param(t, y_name)).collect();
        let x_range = padded_range(&x_vals);
        let y_range = padded_range(&y_vals);
        let xn = x_name.clone();
        let yn = y_name.clone();

        svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
            let mut chart = ChartBuilder::on(root)
                .caption(format!("Trial Rank ({target_name_owned})"), ("sans-serif", 20))
                .margin(10)
                .x_label_area_size(40)
                .y_label_area_size(60)
                .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

            chart.configure_mesh().x_desc(&xn).y_desc(&yn).draw()?;

            chart.draw_series(
                x_vals.iter().zip(y_vals.iter()).enumerate()
                    .filter(|(_, (x, y))| x.is_finite() && y.is_finite())
                    .map(|(i, (&x, &y))| {
                        let t = ranks[i] / max_rank;
                        Circle::new((x, y), 4, rdylbu_r(t).filled())
                    })
            )?;
            Ok(())
        })
    }
}

// ============================================================================
// 11. plot_timeline
// ============================================================================

/// 绘制试验执行时间线（SVG）。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_timeline(study: &Study) -> Result<SvgPlot> {
    let trials = study.trials()?;
    if trials.is_empty() {
        return empty_svg("Timeline (no trials)");
    }

    let base_time = trials.iter()
        .filter_map(|t| t.datetime_start)
        .min()
        .unwrap_or_else(chrono::Utc::now);

    let mut segments: Vec<(f64, f64, i64, TrialState)> = Vec::new();
    for trial in &trials {
        let start = trial.datetime_start
            .map(|dt| (dt - base_time).num_milliseconds() as f64 / 1000.0)
            .unwrap_or(0.0);
        let end = trial.datetime_complete
            .map(|dt| (dt - base_time).num_milliseconds() as f64 / 1000.0)
            .unwrap_or(start);
        segments.push((start, end, trial.number, trial.state));
    }

    let all_t: Vec<f64> = segments.iter().flat_map(|(s, e, _, _)| vec![*s, *e]).collect();
    let all_n: Vec<f64> = segments.iter().map(|(_, _, n, _)| *n as f64).collect();
    let x_range = padded_range(&all_t);
    let y_range = padded_range(&all_n);

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption("Trial Timeline", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        chart.configure_mesh().x_desc("Time (s)").y_desc("Trial Number").draw()?;

        for (i, (start, end, num, state)) in segments.iter().enumerate() {
            let color = match state {
                TrialState::Complete => tab10(0),
                TrialState::Pruned => tab10(1),
                TrialState::Fail => tab10(3),
                _ => tab10(7),
            };
            let _ = i; // suppress warning
            let y = *num as f64;
            chart.draw_series(LineSeries::new(
                vec![(*start, y), (*end, y)],
                color.stroke_width(3),
            ))?;
        }

        Ok(())
    })
}

// ============================================================================
// 12. plot_terminator_improvement
// ============================================================================

/// 绘制终止器改进曲线（SVG）。
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_terminator_improvement(
    study: &Study,
    plot_error: bool,
    improvement_evaluator: Option<&dyn crate::terminators::ImprovementEvaluator>,
    error_evaluator: Option<&dyn crate::terminators::ErrorEvaluator>,
    min_n_trials: usize,
) -> Result<SvgPlot> {
    let all_trials = study.trials()?;
    let direction = study.directions().first().copied()
        .unwrap_or(crate::study::StudyDirection::Minimize);

    let default_improvement = crate::terminators::RegretBoundEvaluator::new(None, None, None);
    let default_error = crate::terminators::CrossValidationErrorEvaluator;
    let imp_eval: &dyn crate::terminators::ImprovementEvaluator =
        improvement_evaluator.unwrap_or(&default_improvement);
    let err_eval: &dyn crate::terminators::ErrorEvaluator =
        error_evaluator.unwrap_or(&default_error);

    let mut completed: Vec<FrozenTrial> = Vec::new();
    let mut nums = Vec::new();
    let mut imps = Vec::new();
    let mut errs = Vec::new();

    for trial in &all_trials {
        if trial.state == TrialState::Complete {
            completed.push(trial.clone());
        }
        if completed.is_empty() { continue; }
        let improvement = imp_eval.evaluate(&completed, direction);
        nums.push(trial.number as f64);
        imps.push(improvement);
        if plot_error {
            errs.push(err_eval.evaluate(&completed, direction));
        }
    }

    if nums.is_empty() {
        return empty_svg("Terminator Improvement (no data)");
    }

    let x_range = padded_range(&nums);
    let all_y: Vec<f64> = imps.iter().chain(errs.iter()).copied().collect();
    let y_range = padded_range(&all_y);

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption("Terminator Improvement Plot", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        chart.configure_mesh()
            .x_desc("Trial")
            .y_desc("Terminator Improvement")
            .draw()?;

        // 分两段：min_n_trials 前用浅色，后用深色
        let split = min_n_trials.min(nums.len());
        let color_light = RGBColor(173, 181, 255);
        let color_full = tab10(0);

        if split > 0 {
            let pts: Vec<(f64, f64)> = nums[..split].iter().zip(imps[..split].iter())
                .map(|(&x, &y)| (x, y)).collect();
            chart.draw_series(LineSeries::new(pts.clone(), color_light.stroke_width(2)))?;
            chart.draw_series(pts.iter().map(|&(x, y)| Circle::new((x, y), 2, color_light.filled())))?;
        }
        if split < nums.len() {
            let pts: Vec<(f64, f64)> = nums[split..].iter().zip(imps[split..].iter())
                .map(|(&x, &y)| (x, y)).collect();
            chart.draw_series(LineSeries::new(pts.clone(), tab10(0).stroke_width(2)))?
                .label("Improvement")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], tab10(0).stroke_width(2)));
            chart.draw_series(pts.iter().map(|&(x, y)| Circle::new((x, y), 2, tab10(0).filled())))?;
        }

        if plot_error && !errs.is_empty() {
            let err_pts: Vec<(f64, f64)> = nums.iter().zip(errs.iter())
                .map(|(&x, &y)| (x, y)).collect();
            chart.draw_series(LineSeries::new(err_pts, tab10(3).stroke_width(2)))?
                .label("Error")
                .legend(|(x, y)| PathElement::new(vec![(x, y), (x + 20, y)], tab10(3).stroke_width(2)));
        }

        chart.configure_series_labels().border_style(BLACK).draw()?;
        Ok(())
    })
}

// ============================================================================
// save_svg helper
// ============================================================================

/// 将 SVG 图表保存到文件。
#[cfg(feature = "visualization-matplotlib")]
pub fn save_svg(plot: &SvgPlot, path: &str) -> Result<()> {
    plot.save(path)
}

/// 生成空白 SVG 图（无数据时使用）
#[cfg(feature = "visualization-matplotlib")]
fn empty_svg(title: &str) -> Result<SvgPlot> {
    let mut buf = String::new();
    {
        let root = SVGBackend::with_string(&mut buf, (W, H)).into_drawing_area();
        root.fill(&WHITE).map_err(|e| OptunaError::StorageInternalError(format!("SVG error: {e}")))?;
        root.draw_text(
            title,
            &("sans-serif", 20).into_text_style(&root),
            (W as i32 / 4, H as i32 / 2),
        ).map_err(|e| OptunaError::StorageInternalError(format!("SVG error: {e}")))?;
        root.present().map_err(|e| OptunaError::StorageInternalError(format!("SVG error: {e}")))?;
    }
    Ok(SvgPlot { svg: buf })
}

// ============================================================================
// Multi-study 和增强版函数 (对齐 plotly 后端)
// ============================================================================

/// 多 study 优化历史（SVG）。
/// 对应 Python `plot_optimization_history(study=[s1, s2, ...])`.
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_optimization_history_multi(
    studies: &[&Study],
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<SvgPlot> {
    // 收集所有 study 的数据
    let mut all_numbers = Vec::new();
    let mut all_values = Vec::new();
    let mut study_names = Vec::new();

    for study in studies {
        let trials = study.get_trials(Some(&[TrialState::Complete]))?;
        let numbers: Vec<f64> = trials.iter().map(|t| t.number as f64).collect();
        let values: Vec<f64> = trials.iter().map(|t| {
            target.map_or_else(
                || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
                |f| f(t),
            )
        }).collect();
        all_numbers.push(numbers);
        all_values.push(values);
        study_names.push(study.study_name().to_string());
    }

    if all_numbers.iter().all(|ns| ns.is_empty()) {
        return empty_svg("Optimization History (no trials)");
    }

    let flat_x: Vec<f64> = all_numbers.iter().flatten().copied().collect();
    let flat_y: Vec<f64> = all_values.iter().flatten().copied().collect();
    let x_range = padded_range(&flat_x);
    let y_range = padded_range(&flat_y);
    let target_name_owned = target_name.to_string();

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption(format!("Optimization History ({target_name_owned})"), ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        chart.configure_mesh()
            .x_desc("Trial")
            .y_desc(&target_name_owned)
            .draw()?;

        for (si, (numbers, values)) in all_numbers.iter().zip(all_values.iter()).enumerate() {
            let color = tab10(si);
            chart.draw_series(
                numbers.iter().zip(values.iter())
                    .filter(|(_, y)| y.is_finite())
                    .map(|(&x, &y)| Circle::new((x, y), 3, color.filled()))
            )?.label(&study_names[si]);
        }

        chart.configure_series_labels().border_style(BLACK).draw()?;
        Ok(())
    })
}

/// 多 study EDF 图（SVG）。
/// 对应 Python `plot_edf(study=[s1, s2, ...])`.
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_edf_multi(
    studies: &[&Study],
    target: Option<&dyn Fn(&FrozenTrial) -> f64>,
    target_name: &str,
) -> Result<SvgPlot> {
    let mut all_curves: Vec<(String, Vec<f64>, Vec<f64>)> = Vec::new();

    for study in studies {
        let trials = study.get_trials(Some(&[TrialState::Complete]))?;
        if trials.is_empty() { continue; }
        let mut values: Vec<f64> = trials.iter().map(|t| {
            target.map_or_else(
                || t.values.as_ref().and_then(|v| v.first().copied()).unwrap_or(f64::NAN),
                |f| f(t),
            )
        }).collect();
        values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n = values.len() as f64;
        let probs: Vec<f64> = (1..=values.len()).map(|i| i as f64 / n).collect();
        all_curves.push((study.study_name().to_string(), values, probs));
    }

    if all_curves.is_empty() {
        return empty_svg("EDF (no data)");
    }

    let flat_x: Vec<f64> = all_curves.iter().flat_map(|(_, vs, _)| vs.iter().copied()).collect();
    let x_range = padded_range(&flat_x);
    let target_name_owned = target_name.to_string();

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption("Empirical Distribution Function", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, -0.05..1.05)?;

        chart.configure_mesh()
            .x_desc(&target_name_owned)
            .y_desc("Cumulative Probability")
            .draw()?;

        for (ci, (name, values, probs)) in all_curves.iter().enumerate() {
            let pts: Vec<(f64, f64)> = values.iter().zip(probs.iter())
                .map(|(&x, &y)| (x, y)).collect();
            chart.draw_series(LineSeries::new(pts, tab10(ci).stroke_width(2)))?
                .label(name);
        }

        chart.configure_series_labels().border_style(BLACK).draw()?;
        Ok(())
    })
}

/// 带选项的时间线图（SVG）。
/// 对应 Python `plot_timeline(study, n_recent_trials=...)`.
#[cfg(feature = "visualization-matplotlib")]
pub fn plot_timeline_with_options(
    study: &Study,
    n_recent_trials: Option<usize>,
) -> Result<SvgPlot> {
    let mut trials = study.trials()?;
    if trials.is_empty() {
        return empty_svg("Timeline (no trials)");
    }

    if let Some(n) = n_recent_trials {
        if trials.len() > n {
            trials = trials.split_off(trials.len() - n);
        }
    }

    let base_time = trials.iter()
        .filter_map(|t| t.datetime_start)
        .min()
        .unwrap_or_else(chrono::Utc::now);

    let pts: Vec<(f64, f64, f64, TrialState)> = trials.iter().map(|t| {
        let start = t.datetime_start
            .map(|dt| (dt - base_time).num_milliseconds() as f64 / 1000.0)
            .unwrap_or(0.0);
        let end = t.datetime_complete
            .map(|dt| (dt - base_time).num_milliseconds() as f64 / 1000.0)
            .unwrap_or(start + 0.001);
        (t.number as f64, start, end, t.state)
    }).collect();

    let xs: Vec<f64> = pts.iter().flat_map(|(_, s, e, _)| vec![*s, *e]).collect();
    let ys: Vec<f64> = pts.iter().map(|(n, _, _, _)| *n).collect();
    let x_range = padded_range(&xs);
    let y_range = padded_range(&ys);

    svg_plot!(|root: &DrawingArea<SVGBackend, _>| -> std::result::Result<(), Box<dyn std::error::Error>> {
        let mut chart = ChartBuilder::on(root)
            .caption("Trial Timeline", ("sans-serif", 20))
            .margin(10)
            .x_label_area_size(40)
            .y_label_area_size(60)
            .build_cartesian_2d(x_range.0..x_range.1, y_range.0..y_range.1)?;

        chart.configure_mesh()
            .x_desc("Elapsed Time (s)")
            .y_desc("Trial")
            .draw()?;

        for &(num, start, end, state) in &pts {
            let color = match state {
                TrialState::Complete => tab10(0),
                TrialState::Pruned   => tab10(3),
                TrialState::Fail     => tab10(4),
                _                    => tab10(7),
            };
            chart.draw_series(LineSeries::new(
                vec![(start, num), (end, num)],
                color.stroke_width(4),
            ))?;
        }

        Ok(())
    })
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
#[cfg(feature = "visualization-matplotlib")]
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
        study.optimize_multi(
            |trial| {
                let x = trial.suggest_float("x", 0.0, 5.0, false, None)?;
                let y = trial.suggest_float("y", 0.0, 5.0, false, None)?;
                Ok(vec![x * x + y, x + y * y])
            },
            Some(15),
            None,
            None,
        ).unwrap();
        study
    }

    #[test]
    fn test_mpl_optimization_history() {
        let study = make_study();
        let plot = plot_optimization_history(&study, None, "Objective").unwrap();
        assert!(plot.svg.contains("<svg"));
        assert!(plot.svg.contains("Optimization History"));
    }

    #[test]
    fn test_mpl_intermediate_values() {
        // 没有中间值 → 空图
        let study = make_study();
        let plot = plot_intermediate_values(&study).unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_parallel_coordinate() {
        let study = make_study();
        let plot = plot_parallel_coordinate(&study, None, None, "Objective").unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_contour() {
        let study = make_study();
        let plot = plot_contour(&study, None, None, "Objective").unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_slice() {
        let study = make_study();
        let plot = plot_slice(&study, None, None, "Objective").unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_edf() {
        let study = make_study();
        let plot = plot_edf(&study, None, "Objective").unwrap();
        assert!(plot.svg.contains("<svg"));
        assert!(plot.svg.contains("Empirical Distribution"));
    }

    #[test]
    fn test_mpl_param_importances() {
        let study = make_study();
        let plot = plot_param_importances(&study, None, None, None, "Objective").unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_pareto_front() {
        let study = make_multi_obj_study();
        let plot = plot_pareto_front(&study, None, true).unwrap();
        assert!(plot.svg.contains("<svg"));
        assert!(plot.svg.contains("Pareto Front"));
    }

    #[test]
    fn test_mpl_hypervolume_history() {
        let study = make_multi_obj_study();
        let plot = plot_hypervolume_history(&study, &[100.0, 100.0]).unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_rank() {
        let study = make_study();
        let plot = plot_rank(&study, None, None, "Objective").unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_timeline() {
        let study = make_study();
        let plot = plot_timeline(&study).unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_terminator_improvement() {
        let study = make_study();
        // 使用 BestValueStagnationEvaluator（快速）+ StaticErrorEvaluator
        let imp = crate::terminators::BestValueStagnationEvaluator::new(3);
        let err = crate::terminators::StaticErrorEvaluator::new(0.1);
        let plot = plot_terminator_improvement(
            &study, true, Some(&imp), Some(&err), 2,
        ).unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_empty_study() {
        let study = create_study(
            None, None, None, None, Some(StudyDirection::Minimize), None, false,
        ).unwrap();
        let plot = plot_edf(&study, None, "value").unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_save_svg() {
        let study = make_study();
        let plot = plot_optimization_history(&study, None, "Obj").unwrap();
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.svg");
        save_svg(&plot, path.to_str().unwrap()).unwrap();
        let content = std::fs::read_to_string(&path).unwrap();
        assert!(content.contains("<svg"));
    }

    #[test]
    fn test_mpl_optimization_history_multi() {
        let s1 = make_study();
        let s2 = make_study();
        let plot = plot_optimization_history_multi(&[&s1, &s2], None, "Value").unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_edf_multi() {
        let s1 = make_study();
        let s2 = make_study();
        let plot = plot_edf_multi(&[&s1, &s2], None, "Value").unwrap();
        assert!(plot.svg.contains("<svg"));
    }

    #[test]
    fn test_mpl_timeline_with_options() {
        let study = make_study();
        let plot = plot_timeline_with_options(&study, Some(5)).unwrap();
        assert!(plot.svg.contains("<svg"));
    }
}
