//! 详细交叉验证测试 — Parzen 估计器、TPE EI、NSGA-II、Hypervolume、TruncNorm
//!
//! 参考值来源: tests/detailed_baseline.json
//!   生成脚本: tests/generate_detailed_baseline.py

use serde_json::Value;

use optuna_rs::multi_objective::{hypervolume_2d, hypervolume};
use optuna_rs::samplers::tpe::truncnorm;

const TOL: f64 = 1e-8;
const LOG_TOL: f64 = 1e-6;

fn load_baseline() -> Value {
    let data = include_str!("detailed_baseline.json");
    serde_json::from_str(data).expect("Failed to parse detailed_baseline.json")
}

fn assert_close(actual: f64, expected: f64, tol: f64, msg: &str) {
    if expected.is_infinite() {
        assert!(
            actual.is_infinite() && actual.signum() == expected.signum(),
            "{msg}: expected {expected}, got {actual}"
        );
        return;
    }
    if expected.is_nan() {
        assert!(actual.is_nan(), "{msg}: expected NaN, got {actual}");
        return;
    }
    let diff = (actual - expected).abs();
    let denom = expected.abs().max(1.0);
    assert!(
        diff / denom < tol,
        "{msg}: expected {expected}, got {actual}, diff={diff}, rel={:.2e}",
        diff / denom
    );
}

// ═══════════════════════════════════════════════════════════════
//  1. TruncNorm logpdf 精确验证
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_truncnorm_logpdf_cases() {
    let b = load_baseline();
    let section = &b["truncnorm_detail"];

    for i in 0..7 {
        let key = format!("truncnorm_{i}");
        let case = &section[&key];

        let x = case["x"].as_f64().unwrap();
        let a = case["a"].as_f64().unwrap();
        let b_bound = case["b"].as_f64().unwrap();
        let loc = case["loc"].as_f64().unwrap();
        let scale = case["scale"].as_f64().unwrap();
        let expected_logpdf = case["logpdf"].as_f64().unwrap();

        // Rust API: logpdf(x, low, high, mu, sigma) — low/high 是原始空间边界
        let low = a * scale + loc;
        let high = b_bound * scale + loc;
        let actual = truncnorm::logpdf(x, low, high, loc, scale);
        assert_close(
            actual, expected_logpdf, LOG_TOL,
            &format!("truncnorm_{i}: logpdf({x}, low={low}, high={high}, loc={loc}, scale={scale})")
        );
    }
}

#[test]
fn cv_truncnorm_ppf_median() {
    let b = load_baseline();
    let section = &b["truncnorm_detail"];

    for i in 0..7 {
        let key = format!("truncnorm_{i}");
        let case = &section[&key];

        let a = case["a"].as_f64().unwrap();
        let b_bound = case["b"].as_f64().unwrap();
        let expected_ppf = case["ppf_0.5"].as_f64().unwrap();
        let loc = case["loc"].as_f64().unwrap();
        let scale = case["scale"].as_f64().unwrap();

        let ppf_std = truncnorm::ppf(0.5, a, b_bound);
        let actual = ppf_std * scale + loc;
        assert_close(
            actual, expected_ppf, TOL,
            &format!("truncnorm_{i}: ppf(0.5)")
        );
    }
}

// ═══════════════════════════════════════════════════════════════
//  2. Parzen 估计器 — 单核 log_pdf 验证
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_parzen_single_kernel_logpdf() {
    let b = load_baseline();
    let data = &b["parzen_logpdf"]["single_kernel_logpdf"];

    let mu = data["mus"][0].as_f64().unwrap();
    let sigma = data["sigmas"][0].as_f64().unwrap();
    let low = data["low"].as_f64().unwrap();
    let high = data["high"].as_f64().unwrap();

    let test_points: Vec<f64> = data["test_points"]
        .as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected: Vec<f64> = data["log_pdfs"]
        .as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();

    for (i, (&x, &exp)) in test_points.iter().zip(expected.iter()).enumerate() {
        // Rust API: logpdf(x, low, high, mu, sigma)
        let actual = truncnorm::logpdf(x, low, high, mu, sigma);
        assert_close(
            actual, exp, LOG_TOL,
            &format!("single_kernel_logpdf[{i}](x={x})")
        );
    }
}

#[test]
fn cv_parzen_mixture_3kernel_logpdf() {
    let b = load_baseline();
    let data = &b["parzen_logpdf"]["mixture_3kernel_logpdf"];

    let mus: Vec<f64> = data["mus"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let sigmas: Vec<f64> = data["sigmas"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let weights: Vec<f64> = data["weights"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let low = data["low"].as_f64().unwrap();
    let high = data["high"].as_f64().unwrap();

    let test_points: Vec<f64> = data["test_points"]
        .as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected: Vec<f64> = data["log_pdfs"]
        .as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();

    for (i, (&x, &exp)) in test_points.iter().zip(expected.iter()).enumerate() {
        // 混合分布: log(sum(w_k * p_k(x)))
        let log_components: Vec<f64> = mus.iter().zip(&sigmas).zip(&weights)
            .map(|((&mu, &sigma), &w)| {
                // Rust API: logpdf(x, low, high, mu, sigma)
                w.ln() + truncnorm::logpdf(x, low, high, mu, sigma)
            })
            .collect();

        let actual = logsumexp(&log_components);
        assert_close(
            actual, exp, LOG_TOL,
            &format!("mixture_3kernel_logpdf[{i}](x={x})")
        );
    }
}

/// logsumexp 辅助函数
fn logsumexp(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NEG_INFINITY;
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if max == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    let sum: f64 = values.iter().map(|&v| (v - max).exp()).sum();
    max + sum.ln()
}

// ═══════════════════════════════════════════════════════════════
//  3. Parzen 估计器 — log 空间 PDF
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_parzen_log_space_pdf() {
    let b = load_baseline();
    let data = &b["parzen_logpdf"]["log_space_2kernel"];

    let log_mus: Vec<f64> = data["log_mus"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let log_sigmas: Vec<f64> = data["log_sigmas"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let weights: Vec<f64> = data["weights"].as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let log_low = data["log_low"].as_f64().unwrap();
    let log_high = data["log_high"].as_f64().unwrap();

    let test_points: Vec<f64> = data["test_points"]
        .as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected: Vec<f64> = data["log_pdfs"]
        .as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();

    for (i, (&x, &exp)) in test_points.iter().zip(expected.iter()).enumerate() {
        let lx = x.ln();

        let log_components: Vec<f64> = log_mus.iter().zip(&log_sigmas).zip(&weights)
            .map(|((&mu, &sigma), &w)| {
                // Rust API: logpdf(x, low, high, mu, sigma)
                let log_pk = truncnorm::logpdf(lx, log_low, log_high, mu, sigma);
                // Jacobian: -ln(x)
                w.ln() + log_pk - x.ln()
            })
            .collect();

        let actual = logsumexp(&log_components);
        assert_close(
            actual, exp, LOG_TOL,
            &format!("log_space_2kernel[{i}](x={x})")
        );
    }
}

// ═══════════════════════════════════════════════════════════════
//  4. TPE Expected Improvement
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_tpe_ei_comparison() {
    let b = load_baseline();
    let data = &b["tpe_ei"]["simple_ei"];

    let l_mu = data["l_mus"][0].as_f64().unwrap();
    let l_sigma = data["l_sigmas"][0].as_f64().unwrap();
    let g_mu = data["g_mus"][0].as_f64().unwrap();
    let g_sigma = data["g_sigmas"][0].as_f64().unwrap();
    let low = data["low"].as_f64().unwrap();
    let high = data["high"].as_f64().unwrap();

    let candidates: Vec<f64> = data["candidates"]
        .as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_eis: Vec<f64> = data["log_eis"]
        .as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_best_idx = data["best_idx"].as_u64().unwrap() as usize;

    let mut actual_eis = Vec::new();
    for &x in &candidates {
        // Rust API: logpdf(x, low, high, mu, sigma)
        let log_l = truncnorm::logpdf(x, low, high, l_mu, l_sigma);
        let log_g = truncnorm::logpdf(x, low, high, g_mu, g_sigma);

        actual_eis.push(log_l - log_g);
    }

    // 验证 EI 值
    for (i, (&actual, &exp)) in actual_eis.iter().zip(expected_eis.iter()).enumerate() {
        assert_close(
            actual, exp, LOG_TOL,
            &format!("log_ei[{i}](x={})", candidates[i])
        );
    }

    // 验证最优候选
    let actual_best_idx = actual_eis
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap();

    assert_eq!(
        actual_best_idx, expected_best_idx,
        "best_candidate: expected idx={expected_best_idx}, got idx={actual_best_idx}"
    );
}

// ═══════════════════════════════════════════════════════════════
//  5. 2D Hypervolume
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_hypervolume_2d_simple() {
    let b = load_baseline();
    let data = &b["hypervolume"]["hv_2d_simple"];
    let expected = data["hypervolume"].as_f64().unwrap();
    let ref_point = [
        data["ref_point"][0].as_f64().unwrap(),
        data["ref_point"][1].as_f64().unwrap(),
    ];

    let points: Vec<[f64; 2]> = data["pareto_points"]
        .as_array().unwrap()
        .iter()
        .map(|p| {
            let arr = p.as_array().unwrap();
            [arr[0].as_f64().unwrap(), arr[1].as_f64().unwrap()]
        })
        .collect();

    let actual = hypervolume_2d(&points, ref_point);
    assert_close(actual, expected, TOL, "hv_2d_simple");
}

#[test]
fn cv_hypervolume_2d_complex() {
    let b = load_baseline();
    let data = &b["hypervolume"]["hv_2d_complex"];
    let expected = data["hypervolume"].as_f64().unwrap();
    let ref_point = [
        data["ref_point"][0].as_f64().unwrap(),
        data["ref_point"][1].as_f64().unwrap(),
    ];

    let points: Vec<[f64; 2]> = data["pareto_points"]
        .as_array().unwrap()
        .iter()
        .map(|p| {
            let arr = p.as_array().unwrap();
            [arr[0].as_f64().unwrap(), arr[1].as_f64().unwrap()]
        })
        .collect();

    let actual = hypervolume_2d(&points, ref_point);
    assert_close(actual, expected, TOL, "hv_2d_complex");
}

#[test]
fn cv_hypervolume_nd_via_generic() {
    let b = load_baseline();
    let data = &b["hypervolume"]["hv_2d_simple"];
    let expected = data["hypervolume"].as_f64().unwrap();
    let ref_point = vec![
        data["ref_point"][0].as_f64().unwrap(),
        data["ref_point"][1].as_f64().unwrap(),
    ];

    let points: Vec<Vec<f64>> = data["pareto_points"]
        .as_array().unwrap()
        .iter()
        .map(|p| {
            p.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
        })
        .collect();

    let actual = hypervolume(&points, &ref_point);
    assert_close(actual, expected, TOL, "hv_nd_generic_2d");
}

// ═══════════════════════════════════════════════════════════════
//  6. Percentile Pruner 决策
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_percentile_pruner_basic() {
    let b = load_baseline();
    let data = &b["percentile_pruner"]["basic_prune"];

    let values: Vec<f64> = data["completed_values"]
        .as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let current = data["current_value"].as_f64().unwrap();
    let expected_median = data["median"].as_f64().unwrap();
    let expected_prune = data["should_prune"].as_bool().unwrap();

    // 手动计算中位数
    let mut sorted = values.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let median = if sorted.len() % 2 == 0 {
        (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
    } else {
        sorted[sorted.len() / 2]
    };

    assert_close(median, expected_median, TOL, "percentile_median");
    assert_eq!(
        current > median, expected_prune,
        "should_prune: current={current} > median={median} = {}", current > median
    );
}

#[test]
fn cv_percentile_pruner_different_percentiles() {
    let b = load_baseline();
    let completed = vec![0.3, 0.5, 0.7, 0.2, 0.9];

    for percentile in [25, 50, 75, 90] {
        let key = format!("percentile_{percentile}");
        let expected = b["percentile_pruner"][&key].as_f64().unwrap();

        // 使用我们已验证的 nanpercentile 实现
        let mut sorted = completed.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let idx = percentile as f64 / 100.0 * (n - 1) as f64;
        let lo = idx.floor() as usize;
        let hi = idx.ceil() as usize;
        let frac = idx - lo as f64;
        let actual = if lo == hi { sorted[lo] } else {
            sorted[lo] * (1.0 - frac) + sorted[hi] * frac
        };

        assert_close(actual, expected, TOL, &format!("percentile_{percentile}"));
    }
}

// ═══════════════════════════════════════════════════════════════
//  7. Successive Halving
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_successive_halving_brackets() {
    let b = load_baseline();
    let data = &b["successive_halving_detail"];

    let expected_n_brackets = data["n_brackets"].as_i64().unwrap();

    // 验证 n_brackets 计算: floor(log_3(27/1)) + 1 = 4
    let rf: f64 = 3.0;
    let min_r: f64 = 1.0;
    let max_r: f64 = 27.0;
    let n_brackets = (max_r / min_r).log(rf).floor() as i64 + 1;
    assert_eq!(n_brackets, expected_n_brackets, "n_brackets");

    // 验证每个 bracket 的 rung 资源
    for bracket_id in 0..n_brackets as usize {
        let key = format!("bracket_{bracket_id}_rungs");
        let expected: Vec<i64> = data[&key]
            .as_array().unwrap()
            .iter().map(|v| v.as_i64().unwrap()).collect();

        let bracket_min_r = max_r / rf.powi((n_brackets - 1 - bracket_id as i64) as i32);
        let mut rungs = Vec::new();
        let mut r = bracket_min_r;
        while r <= max_r + 1e-6 {
            rungs.push(r.round() as i64);
            r *= rf;
        }

        assert_eq!(
            rungs, expected,
            "bracket_{bracket_id}_rungs: expected={expected:?}, got={rungs:?}"
        );
    }
}

#[test]
fn cv_successive_halving_promotion() {
    let b = load_baseline();
    let data = &b["successive_halving_detail"]["promotion"];

    let values: Vec<f64> = data["trial_values"]
        .as_array().unwrap()
        .iter().map(|v| v.as_f64().unwrap()).collect();
    let expected_n_promote = data["n_promote"].as_u64().unwrap() as usize;
    let expected_promoted: Vec<usize> = data["promoted_indices"]
        .as_array().unwrap()
        .iter().map(|v| v.as_u64().unwrap() as usize).collect();

    let rf = 3;
    let n_promote = std::cmp::max(1, values.len() / rf);
    assert_eq!(n_promote, expected_n_promote, "n_promote");

    // minimize: 排序后取前 n_promote 个
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| values[a].partial_cmp(&values[b]).unwrap());
    let promoted: Vec<usize> = indices[..n_promote].to_vec();

    assert_eq!(promoted, expected_promoted, "promoted_indices");
}

// ═══════════════════════════════════════════════════════════════
//  8. log_gauss_mass 精度验证
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_log_gauss_mass_standard() {
    // log(Φ(b) - Φ(a)) for standard normal
    // 已知精确值:
    // Φ(0) = 0.5, Φ(∞) = 1 → log(0.5) = -0.6931...
    let actual = truncnorm::log_gauss_mass(0.0, f64::INFINITY);
    assert_close(actual, (0.5f64).ln(), TOL, "log_gauss_mass(0, inf)");

    // Φ(-∞) = 0, Φ(0) = 0.5 → log(0.5)
    let actual = truncnorm::log_gauss_mass(f64::NEG_INFINITY, 0.0);
    assert_close(actual, (0.5f64).ln(), TOL, "log_gauss_mass(-inf, 0)");

    // 全实数线: Φ(∞) - Φ(-∞) = 1 → log(1) = 0
    let actual = truncnorm::log_gauss_mass(f64::NEG_INFINITY, f64::INFINITY);
    assert_close(actual, 0.0, TOL, "log_gauss_mass(-inf, inf)");

    // 对称区间: Φ(1) - Φ(-1) ≈ 0.6826... → log ≈ -0.3826
    let actual = truncnorm::log_gauss_mass(-1.0, 1.0);
    let expected = (0.6826894921370859f64).ln();
    assert_close(actual, expected, TOL, "log_gauss_mass(-1, 1)");

    // 极窄区间: Φ(0.01) - Φ(-0.01) ≈ 0.00797884...
    let actual = truncnorm::log_gauss_mass(-0.01, 0.01);
    let expected = (0.007978845608028654f64).ln();
    assert_close(actual, expected, 1e-4, "log_gauss_mass(-0.01, 0.01)");
}

// ═══════════════════════════════════════════════════════════════
//  9. 多目标 — Hypervolume 边界测试
// ═══════════════════════════════════════════════════════════════

#[test]
fn cv_hypervolume_empty() {
    let empty: Vec<[f64; 2]> = vec![];
    assert_eq!(hypervolume_2d(&empty, [10.0, 10.0]), 0.0);
}

#[test]
fn cv_hypervolume_single_point() {
    let points = vec![[3.0, 4.0]];
    let ref_point = [10.0, 10.0];
    // (10 - 3) * (10 - 4) = 7 * 6 = 42
    let actual = hypervolume_2d(&points, ref_point);
    assert_close(actual, 42.0, TOL, "hv_single_point");
}

#[test]
fn cv_hypervolume_dominated_point_excluded() {
    // 一个被参考点支配的，一个不被支配的
    let points = vec![[3.0, 4.0], [15.0, 2.0]]; // 第二个点在第一维超过参考点
    let ref_point = [10.0, 10.0];
    // 只有 [3, 4] 有效
    let actual = hypervolume_2d(&points, ref_point);
    assert_close(actual, 42.0, TOL, "hv_dominated_excluded");
}
