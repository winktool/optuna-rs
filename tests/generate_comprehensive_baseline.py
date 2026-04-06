#!/usr/bin/env python3
"""综合交叉验证基线生成脚本

为 optuna-rs 生成 Python 参考值，覆盖以下模块：
1. Distributions (分布) — contains / single / 验证
2. Hyperband CRC32 — 括号分配
3. Percentile Pruner — 百分位数计算
4. Successive Halving — rung 分配
5. Search Space Transform — 连续化 + 逆变换
6. TPE Parzen Estimator — 权重、gamma、核密度
7. QMC Halton/Sobol — 低差异序列
"""

import binascii
import json
import math
import sys

import numpy as np

# ==============================================================
# 1. Distributions
# ==============================================================

def gen_distributions():
    """测试 contains / single 方法"""
    results = {}

    # FloatDistribution contains
    float_cases = [
        {"low": 0.0, "high": 1.0, "log": False, "step": None,
         "values": [0.0, 0.5, 1.0, -0.1, 1.1, float("nan"), float("inf")]},
        {"low": 1e-5, "high": 1.0, "log": True, "step": None,
         "values": [1e-5, 0.01, 1.0, 0.0, -1.0, 2.0]},
        {"low": 0.0, "high": 1.0, "log": False, "step": 0.25,
         "values": [0.0, 0.25, 0.5, 0.75, 1.0, 0.1, 0.3]},
    ]
    for i, fc in enumerate(float_cases):
        contains = []
        for v in fc["values"]:
            if math.isnan(v) or math.isinf(v):
                contains.append(False)
            elif v < fc["low"] or v > fc["high"]:
                contains.append(False)
            elif fc["step"] is not None:
                contains.append(abs((v - fc["low"]) % fc["step"]) < 1e-12 or
                               abs((v - fc["low"]) % fc["step"] - fc["step"]) < 1e-12)
            else:
                contains.append(True)
        results[f"float_contains_{i}"] = contains

    # FloatDistribution single
    results["float_single_0_1"] = False  # low=0, high=1 → not single
    results["float_single_5_5"] = True   # low=5, high=5 → single
    results["float_single_step"] = True  # low=0, high=0, step=0.5 → single

    # IntDistribution contains
    int_cases = [
        {"low": 0, "high": 10, "log": False, "step": 1,
         "values": [0, 5, 10, -1, 11, 3]},
        {"low": 1, "high": 100, "log": True, "step": 1,
         "values": [1, 10, 100, 0, 101]},
        {"low": 0, "high": 10, "log": False, "step": 2,
         "values": [0, 2, 4, 6, 8, 10, 1, 3, 5]},
    ]
    for i, ic in enumerate(int_cases):
        contains = []
        for v in ic["values"]:
            if v < ic["low"] or v > ic["high"]:
                contains.append(False)
            elif (v - ic["low"]) % ic["step"] != 0:
                contains.append(False)
            else:
                contains.append(True)
        results[f"int_contains_{i}"] = contains

    # CategoricalDistribution contains
    results["cat_contains"] = [True, True, True, False, False]
    # choices = ["a", "b", "c"], values = [0, 1, 2, 3, -1]

    return results


# ==============================================================
# 2. Hyperband CRC32
# ==============================================================

def gen_crc32():
    """CRC32 哈希对齐验证"""
    results = {}
    test_strings = [
        "my_study_0", "my_study_1", "my_study_99",
        "test_5", "study_abc_42", "123456789",
        "", "a", "hello_world_12345",
        # 特殊字符
        "study_with_underscores_100",
        "s_0", "s_1", "s_2", "s_3", "s_4",
        # 大量数据
    ]
    # 添加连续编号用于 bracket 分配验证
    for i in range(100):
        test_strings.append(f"hyperband_study_{i}")

    for s in test_strings:
        results[f"crc32_{s}"] = binascii.crc32(s.encode())

    # 括号分配测试 (n_brackets=4, budgets=[81, 27, 9, 3])
    budgets = [81, 27, 9, 3]
    total = sum(budgets)
    bracket_assignments = []
    for i in range(100):
        h = binascii.crc32(f"hyperband_study_{i}".encode())
        n = h % total
        bracket_id = 0
        for bid, b in enumerate(budgets):
            n -= b
            if n < 0:
                bracket_id = bid
                break
        bracket_assignments.append(bracket_id)
    results["bracket_assignments_100"] = bracket_assignments

    return results


# ==============================================================
# 3. Percentile Pruner
# ==============================================================

def gen_percentile():
    """numpy nanpercentile 对齐验证"""
    results = {}

    # 基本百分位数计算
    data_sets = [
        [1.0, 2.0, 3.0, 4.0, 5.0],
        [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0],
        [1.0, float("nan"), 3.0, float("nan"), 5.0],
        [100.0, 1.0, 50.0, 25.0, 75.0],
        [1.0, 1.0, 1.0, 1.0, 1.0],  # 全相同
        [float("nan"), float("nan"), float("nan"), 1.0],  # 大量 NaN
    ]
    percentiles = [25.0, 50.0, 75.0, 90.0]

    for i, data in enumerate(data_sets):
        arr = np.array(data)
        for p in percentiles:
            val = float(np.nanpercentile(arr, p))
            results[f"percentile_{i}_p{int(p)}"] = val

    # 线性插值验证
    results["percentile_interp_33"] = float(np.nanpercentile([1.0, 2.0, 3.0, 4.0], 33.33))
    results["percentile_interp_66"] = float(np.nanpercentile([1.0, 2.0, 3.0, 4.0], 66.67))

    return results


# ==============================================================
# 4. Successive Halving
# ==============================================================

def gen_successive_halving():
    """Successive Halving rung 计算验证"""
    results = {}

    # min_resource 自动估计
    # formula: max(1, floor(n_trials * reduction_factor^(1 - n_brackets)))
    test_cases = [
        {"reduction_factor": 3, "max_resource": 81},
        {"reduction_factor": 2, "max_resource": 64},
        {"reduction_factor": 4, "max_resource": 256},
    ]

    for i, tc in enumerate(test_cases):
        rf = tc["reduction_factor"]
        mr = tc["max_resource"]
        n_brackets = int(math.floor(math.log(mr) / math.log(rf))) + 1

        # 计算每个 rung 的资源量
        rungs = []
        for bracket_id in range(n_brackets):
            min_r = mr / (rf ** (n_brackets - 1 - bracket_id))
            rung_resources = []
            r = min_r
            while r <= mr:
                rung_resources.append(r)
                r *= rf
            rungs.append(rung_resources)
        results[f"sh_rungs_{i}"] = rungs
        results[f"sh_n_brackets_{i}"] = n_brackets

    return results


# ==============================================================
# 5. Search Space Transform
# ==============================================================

def gen_search_space_transform():
    """搜索空间变换和逆变换"""
    results = {}

    # Float 变换
    # 连续分布: 直接透传
    results["transform_float_linear"] = {
        "input": [0.0, 0.5, 1.0],
        "transformed": [0.0, 0.5, 1.0]
    }

    # Log 分布: x → log(x)
    log_inputs = [1e-5, 0.001, 0.01, 0.1, 1.0]
    results["transform_float_log"] = {
        "input": log_inputs,
        "transformed": [math.log(x) for x in log_inputs]
    }

    # Step 分布: 量化
    step_inputs = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
    step = 0.25
    results["transform_float_step"] = {
        "input": step_inputs,
        "quantized": [round(round((x - 0.0) / step) * step + 0.0, 10)
                      for x in step_inputs]
    }

    # Int 变换
    # 线性 int: x → x
    results["transform_int_linear"] = {
        "input": [0, 1, 2, 5, 10],
        "transformed": [0.0, 1.0, 2.0, 5.0, 10.0]
    }

    # Log int: x → log(x) — 对齐 Python optuna _transform._transform_numerical_param
    log_int_inputs = [1, 2, 5, 10, 100]
    results["transform_int_log"] = {
        "input": log_int_inputs,
        "transformed": [math.log(x) for x in log_int_inputs]
    }

    # 往返精度测试
    round_trip_cases = [
        {"low": 0.0, "high": 1.0, "log": False, "values": [0.0, 0.123456789, 0.5, 1.0]},
        {"low": 1e-5, "high": 1.0, "log": True, "values": [1e-5, 0.001, 0.5, 1.0]},
    ]
    for i, rt in enumerate(round_trip_cases):
        for j, v in enumerate(rt["values"]):
            if rt["log"]:
                encoded = math.log(v)
                decoded = math.exp(encoded)
            else:
                encoded = v
                decoded = encoded
            results[f"round_trip_{i}_{j}"] = {
                "original": v,
                "encoded": encoded,
                "decoded": decoded,
                "error": abs(v - decoded)
            }

    return results


# ==============================================================
# 6. TPE 补充验证
# ==============================================================

def gen_tpe_supplementary():
    """TPE 采样器的补充验证点"""
    results = {}

    # default_gamma 函数: min(ceil(0.1 * n), 25) — 对齐 Python optuna 实际实现
    for n in [1, 4, 9, 16, 25, 50, 100, 1000]:
        results[f"gamma_{n}"] = min(math.ceil(0.1 * n), 25)

    # 权重函数验证 — 对齐 Python optuna default_weights:
    # n < 25: np.ones(n)
    # n >= 25: np.concatenate([np.linspace(1/n, 1, n-25), np.ones(25)])
    for n in [1, 3, 5, 10, 25]:
        if n == 0:
            weights = []
        elif n < 25:
            weights = [1.0] * n
        else:
            import numpy as np
            ramp = list(np.linspace(1.0 / n, 1.0, num=n - 25))
            flat = [1.0] * 25
            weights = ramp + flat
        results[f"weights_{n}"] = weights

    return results


# ==============================================================
# 7. QMC 序列
# ==============================================================

def gen_qmc():
    """低差异序列验证"""
    results = {}

    # Van der Corput 序列 (base=2)
    def van_der_corput(n, base):
        result = 0.0
        denom = 1.0
        while n > 0:
            denom *= base
            digit = n % base
            n //= base
            result += digit / denom
        return result

    # 基本 Van der Corput
    for base in [2, 3, 5, 7]:
        seq = [van_der_corput(i, base) for i in range(1, 21)]
        results[f"vdc_base{base}"] = seq

    # Halton 序列 (2D, 3D)
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
    for dim in [2, 3, 5, 10]:
        points = []
        for i in range(1, 21):
            point = [van_der_corput(i, primes[d]) for d in range(dim)]
            points.append(point)
        results[f"halton_{dim}d"] = points

    return results


# ==============================================================
# 8. 综合端到端优化验证
# ==============================================================

def gen_e2e():
    """端到端优化流程验证数据"""
    results = {}

    # InMemoryStorage 操作序列验证
    results["storage_ops"] = {
        "create_study": {"study_name": "test_study", "direction": "minimize"},
        "n_trials": 10,
        "expected_states": ["complete"] * 8 + ["pruned"] * 2,
    }

    # is_first_in_interval_step 函数
    def is_first_in_interval(step, warmup, interval, steps):
        nearest = (step - warmup) // interval * interval + warmup
        second_last = max((s for s in steps if s < step), default=-1)
        return second_last < nearest

    interval_cases = [
        {"step": 5, "warmup": 0, "interval": 5, "steps": [0, 1, 2, 3, 4, 5]},
        {"step": 10, "warmup": 3, "interval": 5, "steps": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
        {"step": 7, "warmup": 0, "interval": 3, "steps": [0, 1, 2, 3, 4, 5, 6, 7]},
        {"step": 1, "warmup": 0, "interval": 1, "steps": [0, 1]},
    ]
    for i, ic in enumerate(interval_cases):
        val = is_first_in_interval(ic["step"], ic["warmup"], ic["interval"], ic["steps"])
        results[f"interval_check_{i}"] = val

    return results


# ==============================================================
# 主函数
# ==============================================================

def main():
    all_results = {}
    all_results["distributions"] = gen_distributions()
    all_results["crc32"] = gen_crc32()
    all_results["percentile"] = gen_percentile()
    all_results["successive_halving"] = gen_successive_halving()
    all_results["search_space_transform"] = gen_search_space_transform()
    all_results["tpe_supplementary"] = gen_tpe_supplementary()
    all_results["qmc"] = gen_qmc()
    all_results["e2e"] = gen_e2e()

    output_path = "tests/comprehensive_baseline.json"
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"✅ 基线数据已生成: {output_path}")
    print(f"   共 {sum(len(v) for v in all_results.values())} 个测试点")


if __name__ == "__main__":
    main()
