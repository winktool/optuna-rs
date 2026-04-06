#!/usr/bin/env python3
"""
终止器模块深度交叉验证黄金值生成器。

生成 Python optuna terminator 模块的精确参考值，
供 Rust 交叉验证测试使用。
"""

import json
import sys
import math
import numpy as np
from scipy import stats as scipy_stats

# ============================================================================
# Group 1: CrossValidationErrorEvaluator 扩展精度验证
# ============================================================================
def gen_cv_error_extended():
    """扩展 CV 误差测试用例（大 k 值、极端方差等）"""
    cases = []
    
    # Case 1: k=10 均匀分布
    scores_10 = list(np.linspace(0.1, 1.0, 10))
    k = len(scores_10)
    scale = 1.0/k + 1.0/(k-1)
    var_ = np.var(scores_10)  # ddof=0
    std_ = float(np.sqrt(scale * var_))
    cases.append({
        "name": "k10_uniform",
        "scores": scores_10,
        "k": k,
        "scale": scale,
        "var": float(var_),
        "std": std_
    })
    
    # Case 2: k=2 最小情况
    scores_2 = [0.3, 0.7]
    k = 2
    scale = 1.0/k + 1.0/(k-1)
    var_ = np.var(scores_2)
    std_ = float(np.sqrt(scale * var_))
    cases.append({
        "name": "k2_minimal",
        "scores": scores_2,
        "k": k,
        "scale": scale,
        "var": float(var_),
        "std": std_
    })
    
    # Case 3: k=7 非均匀
    scores_7 = [0.1, 0.5, 0.5, 0.6, 0.8, 0.9, 0.95]
    k = len(scores_7)
    scale = 1.0/k + 1.0/(k-1)
    var_ = np.var(scores_7)
    std_ = float(np.sqrt(scale * var_))
    cases.append({
        "name": "k7_nonuniform",
        "scores": scores_7,
        "k": k,
        "scale": scale,
        "var": float(var_),
        "std": std_
    })
    
    # Case 4: 极小方差
    scores_small = [0.9999, 1.0000, 1.0001]
    k = len(scores_small)
    scale = 1.0/k + 1.0/(k-1)
    var_ = np.var(scores_small)
    std_ = float(np.sqrt(scale * var_))
    cases.append({
        "name": "k3_tiny_variance",
        "scores": scores_small,
        "k": k,
        "scale": scale,
        "var": float(var_),
        "std": std_
    })
    
    return cases

# ============================================================================
# Group 2: BestValueStagnationEvaluator 边界场景
# ============================================================================
def gen_stagnation_edge_cases():
    """边界场景：负值、浮点数、大范围值"""
    from optuna.terminator.improvement.evaluator import BestValueStagnationEvaluator
    from optuna.study import StudyDirection
    from optuna.trial import create_trial, FrozenTrial, TrialState
    
    cases = []
    
    def make_trials(values):
        return [
            FrozenTrial(
                number=i, trial_id=i, state=TrialState.COMPLETE,
                value=v,
                datetime_start=None, datetime_complete=None,
                params={}, distributions={},
                user_attrs={}, system_attrs={},
                intermediate_values={},
            ) for i, v in enumerate(values)
        ]
    
    # Case 1: 负值递减（最小化）
    vals_neg = [-1.0, -2.0, -3.0, -4.0, -5.0, -3.0, -3.0]
    trials = make_trials(vals_neg)
    eval5 = BestValueStagnationEvaluator(max_stagnation_trials=5)
    result = eval5.evaluate(trials, StudyDirection.MINIMIZE)
    cases.append({"name": "negative_values_min", "values": vals_neg, "patience": 5, "result": float(result)})
    
    # Case 2: 交替改善（最小化）
    vals_alt = [10.0, 5.0, 8.0, 3.0, 7.0, 2.0, 6.0, 1.0]
    trials = make_trials(vals_alt)
    result = eval5.evaluate(trials, StudyDirection.MINIMIZE)
    cases.append({"name": "alternating_min", "values": vals_alt, "patience": 5, "result": float(result)})
    
    # Case 3: 最后一步改善
    vals_last = [5.0, 5.0, 5.0, 5.0, 5.0, 1.0]
    trials = make_trials(vals_last)
    eval3 = BestValueStagnationEvaluator(max_stagnation_trials=3)
    result = eval3.evaluate(trials, StudyDirection.MINIMIZE)
    cases.append({"name": "last_step_improve", "values": vals_last, "patience": 3, "result": float(result)})
    
    # Case 4: 极小浮点差异
    vals_tiny = [1.0, 1.0 - 1e-15, 1.0 - 2e-15, 1.0 - 1e-15, 1.0]
    trials = make_trials(vals_tiny)
    result = eval5.evaluate(trials, StudyDirection.MINIMIZE)
    cases.append({"name": "tiny_float_diff", "values": vals_tiny, "patience": 5, "result": float(result)})
    
    # Case 5: 大跨度值
    vals_large = [1e10, 1e5, 1e0, 1e-5, 1e-10, 1e0, 1e0]
    trials = make_trials(vals_large)
    result = eval5.evaluate(trials, StudyDirection.MINIMIZE)
    cases.append({"name": "large_range", "values": vals_large, "patience": 5, "result": float(result)})
    
    return cases

# ============================================================================
# Group 3: Beta 函数扩展精度
# ============================================================================
def gen_beta_extended():
    """扩展 _get_beta 测试用例"""
    from optuna.terminator.improvement.evaluator import _get_beta
    
    cases = []
    test_params = [
        (1, 5, 0.1),
        (1, 10, 0.1),
        (2, 20, 0.1),
        (3, 30, 0.1),
        (5, 50, 0.1),
        (10, 100, 0.1),
        (20, 200, 0.1),
        (1, 100, 0.05),
        (10, 10, 0.01),
    ]
    
    for n_params, n_trials, delta in test_params:
        beta = _get_beta(n_params, n_trials, delta)
        cases.append({
            "n_params": n_params,
            "n_trials": n_trials,
            "delta": delta,
            "beta": float(beta),
        })
    
    return cases

# ============================================================================
# Group 4: normal_pdf / normal_cdf 扩展精度
# ============================================================================
def gen_normal_pdf_cdf_extended():
    """扩展 normal_pdf/cdf 极端值测试"""
    cases = []
    
    test_g = [-5.0, -3.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 3.0, 5.0]
    
    for g in test_g:
        pdf = float(scipy_stats.norm.pdf(g))
        cdf = float(scipy_stats.norm.cdf(g))
        cases.append({"g": g, "pdf": pdf, "cdf": cdf})
    
    return cases

# ============================================================================
# Group 5: EMMR 四项分解测试（已知 GP 后验值）
# ============================================================================
def gen_emmr_term_decomposition():
    """
    测试 EMMR 的四项分解公式，使用已知的 GP 后验值。
    不依赖 GP 拟合，直接验证数学公式。
    """
    cases = []
    
    # Case 1: 正常值
    mu_t_star = 1.5
    mu_t1_star = 1.2
    var_t_star = 0.3
    var_t1_star = 0.25
    cov_t = 0.1
    kappa_t1 = 2.0
    var_t1_x_t = 0.4
    mu_t1_x_t = 0.8
    y_t = 1.0
    min_noise_var = 1e-4  # DEFAULT_MINIMUM_NOISE_VAR
    
    delta_mu = mu_t1_star - mu_t_star  # term1
    v_sq = var_t_star - 2*cov_t + var_t1_star
    v = math.sqrt(max(1e-10, v_sq))
    g = (mu_t_star - mu_t1_star) / v
    term2 = v * scipy_stats.norm.pdf(g)
    term3 = v * g * scipy_stats.norm.cdf(g)
    
    lambda_inv = min_noise_var
    _lambda = 1.0 / lambda_inv
    rhs1 = 0.5 * math.log(1.0 + _lambda * var_t1_x_t)
    rhs2 = -0.5 * var_t1_x_t / (var_t1_x_t + lambda_inv)
    rhs3 = 0.5 * var_t1_x_t * (y_t - mu_t1_x_t)**2 / (var_t1_x_t + lambda_inv)**2
    kl_bound = rhs1 + rhs2 + rhs3
    term4 = kappa_t1 * math.sqrt(0.5 * max(0, kl_bound))
    
    emmr = delta_mu + term2 + term3 + term4
    
    cases.append({
        "name": "normal_case",
        "mu_t_star": mu_t_star, "mu_t1_star": mu_t1_star,
        "var_t_star": var_t_star, "var_t1_star": var_t1_star,
        "cov_t": cov_t, "kappa_t1": kappa_t1,
        "var_t1_x_t": var_t1_x_t, "mu_t1_x_t": mu_t1_x_t,
        "y_t": y_t, "min_noise_var": min_noise_var,
        "delta_mu": float(delta_mu),
        "v": float(v), "g": float(g),
        "term2": float(term2), "term3": float(term3),
        "rhs1": float(rhs1), "rhs2": float(rhs2), "rhs3": float(rhs3),
        "kl_bound": float(kl_bound),
        "term4": float(term4),
        "emmr": float(emmr),
    })
    
    # Case 2: 零 delta_mu（无改善）
    mu_t_star2 = 1.0
    mu_t1_star2 = 1.0
    var_t_star2 = 0.2
    var_t1_star2 = 0.2
    cov_t2 = 0.05
    kappa_t12 = 1.5
    var_t1_x_t2 = 0.3
    mu_t1_x_t2 = 0.9
    y_t2 = 0.9
    
    delta_mu2 = mu_t1_star2 - mu_t_star2
    v_sq2 = var_t_star2 - 2*cov_t2 + var_t1_star2
    v2 = math.sqrt(max(1e-10, v_sq2))
    g2 = (mu_t_star2 - mu_t1_star2) / v2
    term2_2 = v2 * scipy_stats.norm.pdf(g2)
    term3_2 = v2 * g2 * scipy_stats.norm.cdf(g2)
    
    rhs1_2 = 0.5 * math.log(1.0 + _lambda * var_t1_x_t2)
    rhs2_2 = -0.5 * var_t1_x_t2 / (var_t1_x_t2 + lambda_inv)
    rhs3_2 = 0.5 * var_t1_x_t2 * (y_t2 - mu_t1_x_t2)**2 / (var_t1_x_t2 + lambda_inv)**2
    kl_bound2 = rhs1_2 + rhs2_2 + rhs3_2
    term4_2 = kappa_t12 * math.sqrt(0.5 * max(0, kl_bound2))
    
    emmr2 = delta_mu2 + term2_2 + term3_2 + term4_2
    
    cases.append({
        "name": "zero_delta_mu",
        "mu_t_star": mu_t_star2, "mu_t1_star": mu_t1_star2,
        "var_t_star": var_t_star2, "var_t1_star": var_t1_star2,
        "cov_t": cov_t2, "kappa_t1": kappa_t12,
        "var_t1_x_t": var_t1_x_t2, "mu_t1_x_t": mu_t1_x_t2,
        "y_t": y_t2, "min_noise_var": min_noise_var,
        "delta_mu": float(delta_mu2),
        "v": float(v2), "g": float(g2),
        "term2": float(term2_2), "term3": float(term3_2),
        "rhs1": float(rhs1_2), "rhs2": float(rhs2_2), "rhs3": float(rhs3_2),
        "kl_bound": float(kl_bound2),
        "term4": float(term4_2),
        "emmr": float(emmr2),
    })
    
    # Case 3: 负 v_sq（需要 max(1e-10, ...)）
    mu_t_star3 = 2.0
    mu_t1_star3 = 1.0
    var_t_star3 = 0.05
    var_t1_star3 = 0.05
    cov_t3 = 0.1  # cov > (var_a + var_b)/2 → v_sq < 0
    kappa_t13 = 3.0
    var_t1_x_t3 = 0.5
    mu_t1_x_t3 = 1.5
    y_t3 = 2.0
    
    delta_mu3 = mu_t1_star3 - mu_t_star3
    v_sq3 = var_t_star3 - 2*cov_t3 + var_t1_star3
    v3 = math.sqrt(max(1e-10, v_sq3))  # v_sq3 = -0.1, clamped to 1e-10
    g3 = (mu_t_star3 - mu_t1_star3) / v3
    term2_3 = v3 * scipy_stats.norm.pdf(g3)
    term3_3 = v3 * g3 * scipy_stats.norm.cdf(g3)
    
    rhs1_3 = 0.5 * math.log(1.0 + _lambda * var_t1_x_t3)
    rhs2_3 = -0.5 * var_t1_x_t3 / (var_t1_x_t3 + lambda_inv)
    rhs3_3 = 0.5 * var_t1_x_t3 * (y_t3 - mu_t1_x_t3)**2 / (var_t1_x_t3 + lambda_inv)**2
    kl_bound3 = rhs1_3 + rhs2_3 + rhs3_3
    term4_3 = kappa_t13 * math.sqrt(0.5 * max(0, kl_bound3))
    
    emmr3 = delta_mu3 + term2_3 + term3_3 + term4_3
    
    cases.append({
        "name": "negative_v_sq",
        "mu_t_star": mu_t_star3, "mu_t1_star": mu_t1_star3,
        "var_t_star": var_t_star3, "var_t1_star": var_t1_star3,
        "cov_t": cov_t3, "kappa_t1": kappa_t13,
        "var_t1_x_t": var_t1_x_t3, "mu_t1_x_t": mu_t1_x_t3,
        "y_t": y_t3, "min_noise_var": min_noise_var,
        "delta_mu": float(delta_mu3),
        "v": float(v3), "g": float(g3),
        "v_sq_raw": float(v_sq3),
        "term2": float(term2_3), "term3": float(term3_3),
        "rhs1": float(rhs1_3), "rhs2": float(rhs2_3), "rhs3": float(rhs3_3),
        "kl_bound": float(kl_bound3),
        "term4": float(term4_3),
        "emmr": float(emmr3),
    })
    
    return cases

# ============================================================================
# Group 6: KL 散度项细分
# ============================================================================
def gen_kl_divergence_terms():
    """KL 散度三项独立验证"""
    min_noise_var = 1e-4
    lambda_inv = min_noise_var
    _lambda = 1.0 / lambda_inv
    
    cases = []
    test_params = [
        (0.1, 0.5, 0.8, "small_var"),
        (0.5, 1.0, 1.2, "medium_var"),
        (1.0, 0.0, 0.0, "large_var_zero_residual"),
        (0.01, 2.0, 0.5, "tiny_var_large_residual"),
        (2.0, 1.5, 1.5, "large_var_exact_pred"),
    ]
    
    for var_t1, mu_t1, y_t, name in test_params:
        rhs1 = 0.5 * math.log(1.0 + _lambda * var_t1)
        rhs2 = -0.5 * var_t1 / (var_t1 + lambda_inv)
        rhs3 = 0.5 * var_t1 * (y_t - mu_t1)**2 / (var_t1 + lambda_inv)**2
        kl_bound = rhs1 + rhs2 + rhs3
        
        cases.append({
            "name": name,
            "var_t1": var_t1,
            "mu_t1": mu_t1,
            "y_t": y_t,
            "lambda_inv": lambda_inv,
            "rhs1": float(rhs1),
            "rhs2": float(rhs2),
            "rhs3": float(rhs3),
            "kl_bound": float(kl_bound),
            "sqrt_half_kl": float(math.sqrt(0.5 * max(0, kl_bound))),
        })
    
    return cases

# ============================================================================
# Group 7: Terminator 组合决策逻辑
# ============================================================================
def gen_terminator_decisions():
    """终止决策：improvement vs error 的各种组合"""
    cases = []
    
    # improvement < error → terminate
    combos = [
        (-2.0, 0.0, True, "negative_imp_zero_err"),
        (0.5, 1.0, True, "small_imp_large_err"),
        (0.0, 0.001, True, "zero_imp_small_err"),
        (1.0, 0.5, False, "large_imp_small_err"),
        (0.0, 0.0, False, "both_zero"),  # 0 < 0 is false
        (5.0, 5.0, False, "equal"),  # 5 < 5 is false
        (-1.0, -0.5, True, "both_negative"),  # -1 < -0.5 is true
        (1e-15, 1e-14, True, "tiny_values"),
    ]
    
    for imp, err, should_terminate, name in combos:
        cases.append({
            "name": name,
            "improvement": imp,
            "error": err,
            "should_terminate": should_terminate,
        })
    
    return cases

# ============================================================================
# Group 8: MedianErrorEvaluator 阈值计算
# ============================================================================
def gen_median_threshold():
    """MedianErrorEvaluator 的中位数计算验证"""
    cases = []
    
    # Case 1: 奇数个改善值
    criteria_odd = [5.0, 3.0, 7.0, 1.0, 9.0]
    sorted_odd = sorted(criteria_odd)
    median_odd = sorted_odd[len(sorted_odd) // 2]
    threshold_odd = median_odd * 0.01  # default ratio
    cases.append({
        "name": "odd_count",
        "criteria": criteria_odd,
        "sorted": sorted_odd,
        "median": float(median_odd),
        "threshold_ratio": 0.01,
        "threshold": float(threshold_odd),
    })
    
    # Case 2: 偶数个改善值
    criteria_even = [2.0, 8.0, 4.0, 6.0]
    sorted_even = sorted(criteria_even)
    median_even = sorted_even[len(sorted_even) // 2]
    threshold_even = median_even * 0.01
    cases.append({
        "name": "even_count",
        "criteria": criteria_even,
        "sorted": sorted_even,
        "median": float(median_even),
        "threshold_ratio": 0.01,
        "threshold": float(threshold_even),
    })
    
    # Case 3: 自定义 ratio
    criteria_ratio = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0]
    sorted_ratio = sorted(criteria_ratio)
    median_ratio = sorted_ratio[len(sorted_ratio) // 2]
    custom_ratio = 0.05
    threshold_ratio = median_ratio * custom_ratio
    cases.append({
        "name": "custom_ratio",
        "criteria": criteria_ratio,
        "sorted": sorted_ratio,
        "median": float(median_ratio),
        "threshold_ratio": custom_ratio,
        "threshold": float(threshold_ratio),
    })
    
    # Case 4: 全部相同值
    criteria_same = [3.14] * 5
    sorted_same = sorted(criteria_same)
    median_same = sorted_same[len(sorted_same) // 2]
    threshold_same = median_same * 0.01
    cases.append({
        "name": "all_same",
        "criteria": criteria_same,
        "sorted": sorted_same,
        "median": float(median_same),
        "threshold_ratio": 0.01,
        "threshold": float(threshold_same),
    })
    
    return cases

# ============================================================================
# 主函数
# ============================================================================
def main():
    result = {
        "cv_error_extended": gen_cv_error_extended(),
        "stagnation_edge_cases": gen_stagnation_edge_cases(),
        "beta_extended": gen_beta_extended(),
        "normal_pdf_cdf_extended": gen_normal_pdf_cdf_extended(),
        "emmr_term_decomposition": gen_emmr_term_decomposition(),
        "kl_divergence_terms": gen_kl_divergence_terms(),
        "terminator_decisions": gen_terminator_decisions(),
        "median_threshold": gen_median_threshold(),
    }
    
    output_path = "tests/terminators_deep_golden_values.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)
    
    # 统计
    total = sum(len(v) for v in result.values())
    print(f"Generated {total} golden cases across {len(result)} groups:")
    for group, cases in result.items():
        print(f"  {group}: {len(cases)} cases")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    main()
