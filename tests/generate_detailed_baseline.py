#!/usr/bin/env python3
"""Parzen 估计器 + TPE 采样流程详细交叉验证基线

生成 Rust 无法独立计算的 Python 中间结果，用于逐步验证。
覆盖：
1. Parzen 估计器 log_pdf 计算
2. TPE EI (Expected Improvement) 比较
3. NSGA-II 非支配排序
4. 多目标 Hypervolume
5. Successive Halving rung 计算
6. PercentilePruner 决策
"""

import json
import math
import sys

import numpy as np
from scipy import stats


def gen_parzen_log_pdf():
    """Parzen 估计器 log_pdf 验证数据"""
    results = {}

    # 场景 1：单个高斯核，无截断
    # log_pdf(x) = logpdf_normal(x; mu, sigma) - log(Φ(b) - Φ(a))
    mus = [0.5]
    sigmas = [0.1]
    low, high = 0.0, 1.0
    weights = [1.0]

    test_points = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

    log_pdfs = []
    for x in test_points:
        # 截断正态 log PDF
        a = (low - mus[0]) / sigmas[0]
        b = (high - mus[0]) / sigmas[0]
        log_pdf = stats.truncnorm.logpdf(x, a, b, loc=mus[0], scale=sigmas[0])
        log_pdfs.append(float(log_pdf))

    results["single_kernel_logpdf"] = {
        "mus": mus, "sigmas": sigmas, "low": low, "high": high,
        "weights": weights, "test_points": test_points,
        "log_pdfs": log_pdfs
    }

    # 场景 2：多核混合 (3 个核)
    mus2 = [0.2, 0.5, 0.8]
    sigmas2 = [0.1, 0.15, 0.1]
    weights2 = [0.3, 0.4, 0.3]

    log_pdfs_mix = []
    for x in test_points:
        # 混合分布: log(sum(w_k * p_k(x)))
        log_components = []
        for mu, sigma, w in zip(mus2, sigmas2, weights2):
            a = (low - mu) / sigma
            b = (high - mu) / sigma
            log_pk = stats.truncnorm.logpdf(x, a, b, loc=mu, scale=sigma)
            log_components.append(np.log(w) + log_pk)
        log_pdf = float(np.logaddexp.reduce(log_components))
        log_pdfs_mix.append(log_pdf)

    results["mixture_3kernel_logpdf"] = {
        "mus": mus2, "sigmas": sigmas2, "low": low, "high": high,
        "weights": weights2, "test_points": test_points,
        "log_pdfs": log_pdfs_mix
    }

    # 场景 3：log 空间的 Parzen 估计
    # 在 log 空间中操作：x_log = ln(x), 然后在 [ln(low), ln(high)] 上做截断正态
    log_low, log_high = np.log(1e-3), np.log(1.0)
    log_mus = [np.log(0.01), np.log(0.1)]
    log_sigmas = [0.5, 0.8]
    log_weights = [0.5, 0.5]

    log_test_points = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
    log_space_pdfs = []
    for x in log_test_points:
        lx = np.log(x)
        log_components = []
        for mu, sigma, w in zip(log_mus, log_sigmas, log_weights):
            a = (log_low - mu) / sigma
            b = (log_high - mu) / sigma
            log_pk = stats.truncnorm.logpdf(lx, a, b, loc=mu, scale=sigma)
            # 变量替换的 Jacobian: -ln(x)
            log_pk -= np.log(x)
            log_components.append(np.log(w) + log_pk)
        log_pdf = float(np.logaddexp.reduce(log_components))
        log_space_pdfs.append(log_pdf)

    results["log_space_2kernel"] = {
        "log_mus": [float(m) for m in log_mus],
        "log_sigmas": log_sigmas,
        "log_low": float(log_low),
        "log_high": float(log_high),
        "weights": log_weights,
        "test_points": log_test_points,
        "log_pdfs": log_space_pdfs
    }

    return results


def gen_tpe_ei():
    """TPE Expected Improvement (EI) 验证"""
    results = {}

    # EI = l(x) / g(x)
    # 在 log 空间: log_ei = log_l(x) - log_g(x)
    # TPE 选择 argmax(log_ei) 的候选

    # 简单场景：l(x) 集中在 0.3, g(x) 比较分散
    l_mus = [0.3]
    l_sigmas = [0.05]
    g_mus = [0.5]
    g_sigmas = [0.3]
    low, high = 0.0, 1.0

    candidates = list(np.linspace(0.0, 1.0, 21))
    log_eis = []
    for x in candidates:
        # l(x)
        a_l = (low - l_mus[0]) / l_sigmas[0]
        b_l = (high - l_mus[0]) / l_sigmas[0]
        log_l = stats.truncnorm.logpdf(x, a_l, b_l, loc=l_mus[0], scale=l_sigmas[0])

        # g(x)
        a_g = (low - g_mus[0]) / g_sigmas[0]
        b_g = (high - g_mus[0]) / g_sigmas[0]
        log_g = stats.truncnorm.logpdf(x, a_g, b_g, loc=g_mus[0], scale=g_sigmas[0])

        log_ei = float(log_l - log_g)
        log_eis.append(log_ei)

    best_idx = int(np.argmax(log_eis))

    results["simple_ei"] = {
        "l_mus": l_mus, "l_sigmas": l_sigmas,
        "g_mus": g_mus, "g_sigmas": g_sigmas,
        "low": low, "high": high,
        "candidates": candidates,
        "log_eis": log_eis,
        "best_idx": best_idx,
        "best_candidate": candidates[best_idx]
    }

    return results


def gen_nsga_sorting():
    """NSGA-II 非支配排序验证"""
    results = {}

    # 2 目标 minimize
    # 非支配排序: 逐层提取 Pareto front
    objectives = [
        [1.0, 5.0],   # trial 0
        [2.0, 3.0],   # trial 1
        [3.0, 1.0],   # trial 2
        [4.0, 4.0],   # trial 3
        [5.0, 2.0],   # trial 4
        [1.5, 4.5],   # trial 5
        [2.5, 2.5],   # trial 6
        [3.5, 3.5],   # trial 7
    ]

    # 手动非支配排序
    n = len(objectives)
    rank = [0] * n
    remaining = set(range(n))
    current_rank = 0

    while remaining:
        front = []
        for i in remaining:
            dominated = False
            for j in remaining:
                if i == j:
                    continue
                # j dominates i if all(j <= i) and any(j < i)
                if all(objectives[j][k] <= objectives[i][k] for k in range(2)) and \
                   any(objectives[j][k] < objectives[i][k] for k in range(2)):
                    dominated = True
                    break
            if not dominated:
                front.append(i)
        for i in front:
            rank[i] = current_rank
            remaining.remove(i)
        current_rank += 1

    # 拥挤距离 (对第一个 front)
    front_0 = [i for i in range(n) if rank[i] == 0]
    crowding = [0.0] * len(front_0)

    for obj_idx in range(2):
        sorted_indices = sorted(range(len(front_0)), key=lambda x: objectives[front_0[x]][obj_idx])

        obj_range = objectives[front_0[sorted_indices[-1]]][obj_idx] - \
                    objectives[front_0[sorted_indices[0]]][obj_idx]

        crowding[sorted_indices[0]] = float('inf')
        crowding[sorted_indices[-1]] = float('inf')

        if obj_range > 0:
            for k in range(1, len(sorted_indices) - 1):
                crowding[sorted_indices[k]] += (
                    objectives[front_0[sorted_indices[k + 1]]][obj_idx] -
                    objectives[front_0[sorted_indices[k - 1]]][obj_idx]
                ) / obj_range

    results["sorting_2obj"] = {
        "objectives": objectives,
        "ranks": rank,
        "front_0": front_0,
        "crowding_front_0": crowding
    }

    return results


def gen_hypervolume():
    """2D 超体积计算验证"""
    results = {}

    # 2D 超体积: 参考点 [10, 10]
    ref_point = [10.0, 10.0]

    # Pareto front 点
    pareto_points = [
        [1.0, 5.0],
        [2.0, 3.0],
        [3.0, 1.0],
    ]

    # 按第一个目标排序
    sorted_points = sorted(pareto_points, key=lambda p: p[0])

    # 计算超体积
    hv = 0.0
    prev_x = ref_point[0]
    for p in reversed(sorted_points):
        # 每个点贡献 (prev_x - p[0]) * (ref_point[1] - p[1])
        # 但需要从右到左
        pass

    # 正确的 2D 超体积计算
    sorted_pts = sorted(pareto_points, key=lambda p: p[0])
    hv = 0.0
    for i, (x, y) in enumerate(sorted_pts):
        if i + 1 < len(sorted_pts):
            width = sorted_pts[i + 1][0] - x
        else:
            width = ref_point[0] - x
        height = ref_point[1] - y
        hv += width * height

    results["hv_2d_simple"] = {
        "ref_point": ref_point,
        "pareto_points": pareto_points,
        "hypervolume": hv
    }

    # 更复杂：4 个点
    pareto_points_4 = [
        [1.0, 8.0],
        [2.0, 5.0],
        [4.0, 3.0],
        [7.0, 1.0],
    ]
    sorted_pts_4 = sorted(pareto_points_4, key=lambda p: p[0])
    hv_4 = 0.0
    for i, (x, y) in enumerate(sorted_pts_4):
        if i + 1 < len(sorted_pts_4):
            width = sorted_pts_4[i + 1][0] - x
        else:
            width = ref_point[0] - x
        height = ref_point[1] - y
        hv_4 += width * height

    results["hv_2d_complex"] = {
        "ref_point": ref_point,
        "pareto_points": pareto_points_4,
        "hypervolume": hv_4
    }

    return results


def gen_successive_halving_detailed():
    """SuccessiveHalving 详细 rung 决策验证"""
    results = {}

    # 场景: reduction_factor=3, min_resource=1, max_resource=27
    # n_brackets = floor(log_3(27/1)) + 1 = 4
    rf = 3
    min_r = 1
    max_r = 27

    n_brackets = int(math.floor(math.log(max_r / min_r) / math.log(rf))) + 1

    # 对每个 bracket，计算 rung 资源
    for bracket_id in range(n_brackets):
        bracket_min_r = max_r / (rf ** (n_brackets - 1 - bracket_id))
        rungs = []
        r = bracket_min_r
        while r <= max_r + 1e-6:
            rungs.append(round(r))
            r *= rf
        results[f"bracket_{bracket_id}_rungs"] = rungs

    # 晋升决策: 在 rung r，top 1/rf 的试验晋级
    # 示例: 9 个试验, rung_0 values
    trial_values = [0.5, 0.3, 0.8, 0.1, 0.9, 0.2, 0.7, 0.4, 0.6]
    # minimize: 值越小越好, top 1/3 = 3 个
    n_promote = max(1, len(trial_values) // rf)
    sorted_idx = sorted(range(len(trial_values)), key=lambda i: trial_values[i])
    promoted = sorted_idx[:n_promote]

    results["promotion"] = {
        "trial_values": trial_values,
        "n_promote": n_promote,
        "promoted_indices": promoted,
        "promoted_values": [trial_values[i] for i in promoted]
    }
    results["n_brackets"] = n_brackets

    return results


def gen_percentile_pruner_decisions():
    """PercentilePruner 决策过程验证"""
    results = {}

    # 场景：5 个完成试验, 1 个进行中的试验
    # percentile = 50 (median)
    completed_values_at_step_5 = [0.3, 0.5, 0.7, 0.2, 0.9]
    current_value = 0.6

    median = float(np.nanpercentile(completed_values_at_step_5, 50))
    # minimize: 若 current > median → 剪枝
    should_prune = current_value > median

    results["basic_prune"] = {
        "completed_values": completed_values_at_step_5,
        "current_value": current_value,
        "median": median,
        "should_prune": should_prune
    }

    # NaN 处理场景
    values_with_nan = [0.3, float("nan"), 0.7, float("nan"), 0.9]
    median_nan = float(np.nanpercentile(values_with_nan, 50))
    results["nan_handling"] = {
        "values": [v if not math.isnan(v) else "NaN" for v in values_with_nan],
        "median": median_nan,
        "n_valid": sum(1 for v in values_with_nan if not math.isnan(v))
    }

    # 不同百分位数
    for percentile in [25, 50, 75, 90]:
        val = float(np.nanpercentile(completed_values_at_step_5, percentile))
        results[f"percentile_{percentile}"] = val

    return results


def gen_truncnorm_detailed():
    """截断正态分布的详细验证点"""
    results = {}

    # CDF 验证
    test_cases = [
        # (x, a, b, loc, scale)
        (0.5, 0.0, 1.0, 0.5, 0.2),
        (0.0, 0.0, 1.0, 0.5, 0.2),
        (1.0, 0.0, 1.0, 0.5, 0.2),
        (0.3, 0.0, 1.0, 0.0, 0.5),
        (0.7, 0.0, 1.0, 1.0, 0.5),
        # 极端情况
        (0.001, 0.0, 1.0, 0.5, 0.001),
        (0.999, 0.0, 1.0, 0.5, 0.001),
    ]

    for i, (x, low, high, loc, scale) in enumerate(test_cases):
        a = (low - loc) / scale
        b = (high - loc) / scale

        cdf_val = float(stats.truncnorm.cdf(x, a, b, loc=loc, scale=scale))
        pdf_val = float(stats.truncnorm.pdf(x, a, b, loc=loc, scale=scale))
        logpdf_val = float(stats.truncnorm.logpdf(x, a, b, loc=loc, scale=scale))
        ppf_05 = float(stats.truncnorm.ppf(0.5, a, b, loc=loc, scale=scale))

        results[f"truncnorm_{i}"] = {
            "x": x, "a": a, "b": b, "loc": loc, "scale": scale,
            "cdf": cdf_val,
            "pdf": pdf_val,
            "logpdf": logpdf_val,
            "ppf_0.5": ppf_05
        }

    return results


def main():
    all_results = {}
    all_results["parzen_logpdf"] = gen_parzen_log_pdf()
    all_results["tpe_ei"] = gen_tpe_ei()
    all_results["nsga_sorting"] = gen_nsga_sorting()
    all_results["hypervolume"] = gen_hypervolume()
    all_results["successive_halving_detail"] = gen_successive_halving_detailed()
    all_results["percentile_pruner"] = gen_percentile_pruner_decisions()
    all_results["truncnorm_detail"] = gen_truncnorm_detailed()

    def sanitize(obj):
        """将 Infinity/NaN 替换为 JSON 合法值"""
        if isinstance(obj, float):
            if math.isinf(obj):
                return 1e308 if obj > 0 else -1e308
            if math.isnan(obj):
                return None
            return obj
        if isinstance(obj, dict):
            return {k: sanitize(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [sanitize(v) for v in obj]
        return obj

    output_path = "tests/detailed_baseline.json"
    with open(output_path, "w") as f:
        json.dump(sanitize(all_results), f, indent=2)
    print(f"✅ 详细基线数据已生成: {output_path}")
    print(f"   共 {sum(len(v) for v in all_results.values())} 个验证组")


if __name__ == "__main__":
    main()
