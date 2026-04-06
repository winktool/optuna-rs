#!/usr/bin/env python3
"""
Distribution 模块深度交叉验证 — Python 黄金值生成器。

生成 FloatDistribution / IntDistribution / CategoricalDistribution
在各种边界条件下的精确参考值，供 Rust 交叉验证使用。
"""

import json
import math
import sys
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

from optuna.distributions import (
    FloatDistribution,
    IntDistribution,
    CategoricalDistribution,
    check_distribution_compatibility,
    distribution_to_json,
    json_to_distribution,
    _adjust_discrete_uniform_high,
    _adjust_int_uniform_high,
    _get_single_value,
    _is_distribution_log,
)

results = {}


# ============================================================================
# 1. FloatDistribution: _adjust_discrete_uniform_high 精确对比
# ============================================================================
adjust_float_cases = [
    # (low, high, step, label)
    (0.0, 1.0, 0.3, "step03"),       # high → 0.9
    (0.0, 1.0, 0.25, "step025"),     # high → 1.0 (整除)
    (0.0, 1.0, 0.7, "step07"),       # high → 0.7
    (0.0, 1.0, 0.1, "step01"),       # high → 1.0 (整除)
    (0.0, 1.0, 0.15, "step015"),     # high = ?
    (0.0, 1.0, 0.4, "step04"),       # high → 0.8
    (0.1, 0.95, 0.3, "offset_step03"),  # 带偏移
    (0.05, 1.0, 0.3, "offset005_step03"),
    (0.0, 10.0, 3.0, "large_step3"), # 大范围
    (0.0, 10.0, 0.3, "large_step03"),
    (1.5, 5.5, 0.7, "mid_range_step07"),
    (0.0, 0.1, 0.3, "tiny_range"),   # high < step → adjusted to low
    (0.0, 0.3, 0.3, "exact_step"),   # 刚好一步
    (0.0, 0.0, 0.3, "zero_range"),   # low == high
]

adjust_results = []
for low, high, step, label in adjust_float_cases:
    adjusted = _adjust_discrete_uniform_high(low, high, step)
    fd = FloatDistribution(low, high, step=step)
    s = fd.single()
    
    # contains 测试值: low, adjusted_high, high, 中间值
    contains_tests = {}
    for val in [low, adjusted, high, (low + adjusted) / 2, low + step, low + 2*step]:
        contains_tests[str(val)] = fd._contains(val)
    
    adjust_results.append({
        "label": label,
        "low": low,
        "high_input": high,
        "step": step,
        "adjusted_high": adjusted,
        "single": s,
        "contains": contains_tests,
    })

results["float_adjust_high"] = adjust_results


# ============================================================================
# 2. FloatDistribution: single() 精度边界
# ============================================================================
single_float_cases = []
test_configs = [
    (0.0, 1.0, 0.3, False),
    (0.0, 0.1, 0.3, True),     # (0.1-0.0) < 0.3 → adjusted high=0.0 → single
    (0.0, 0.3, 0.3, False),    # (0.3-0.0) == 0.3 → NOT single
    (0.0, 0.29, 0.3, True),    # adjusted high=0.0 → single
    (5.0, 5.0, None, True),    # equal, no step
    (5.0, 5.0, 0.1, True),     # equal, with step
    (1e-10, 1e-10, None, True),  # very small equal
    (0.0, 1e-15, None, False),   # very close but not equal
    (0.0, 1.0, 2.0, True),     # step > range → single (adjusted high=0.0)
    (0.0, 1.0, 1.0, False),    # step == range → NOT single
    (0.0, 1.0, 0.999, False),  # step == adjusted range (0.999) → NOT single
    (0.0, 1.0, 1.001, True),   # step > range → single (adjusted high=0.0)
]

for low, high, step, expected in test_configs:
    fd = FloatDistribution(low, high, step=step)
    actual = fd.single()
    assert actual == expected, f"MISMATCH: FloatDist({low},{high},step={step}).single() = {actual}, expected {expected}"
    single_float_cases.append({
        "low": low,
        "high_input": high,
        "step": step,
        "adjusted_high": fd.high,
        "single": actual,
    })

results["float_single"] = single_float_cases


# ============================================================================
# 3. FloatDistribution: contains() 精度边界
# ============================================================================
contains_float_cases = []

# 基本连续分布
fd = FloatDistribution(0.0, 1.0)
for val, exp in [(0.0, True), (0.5, True), (1.0, True), (-1e-15, False),
                 (1.0 + 1e-15, False), (float('nan'), False),
                 (float('inf'), False), (float('-inf'), False)]:
    contains_float_cases.append({
        "dist": "float_0_1",
        "value": val if not (isinstance(val, float) and (math.isnan(val) or math.isinf(val))) else str(val),
        "expected": exp,
    })

# step 分布
fd_step = FloatDistribution(0.0, 1.0, step=0.25)
for val, exp in [(0.0, True), (0.25, True), (0.5, True), (0.75, True), (1.0, True),
                 (0.1, False), (0.3, False), (0.125, False),
                 (0.25 + 1e-9, True), (0.25 - 1e-9, True),   # 在1e-8容差内
                 (0.25 + 1e-6, False), (0.25 - 1e-6, False)]: # 超出容差
    contains_float_cases.append({
        "dist": "float_0_1_step025",
        "value": val,
        "expected": exp,
    })

# log 分布
fd_log = FloatDistribution(0.001, 100.0, log=True)
for val, exp in [(0.001, True), (1.0, True), (100.0, True), (0.0, False),
                 (-1.0, False), (100.001, False)]:
    contains_float_cases.append({
        "dist": "float_log_001_100",
        "value": val,
        "expected": exp,
    })

results["float_contains"] = contains_float_cases


# ============================================================================
# 4. FloatDistribution: to_internal_repr / to_external_repr
# ============================================================================
repr_float_cases = []

# 普通分布
fd = FloatDistribution(0.0, 10.0)
for v in [0.0, 1.5, 5.0, 10.0, 3.14159265, 1e-10, 9.999999]:
    internal = fd.to_internal_repr(v)
    external = fd.to_external_repr(internal)  # FloatDist: identity
    repr_float_cases.append({
        "dist": "float_0_10",
        "value": v,
        "internal": internal,
        "external": external,
    })

# log 分布
fd_log = FloatDistribution(0.001, 100.0, log=True)
for v in [0.001, 0.1, 1.0, 50.0, 100.0]:
    internal = fd_log.to_internal_repr(v)
    external = fd_log.to_external_repr(internal)
    repr_float_cases.append({
        "dist": "float_log_001_100",
        "value": v,
        "internal": internal,
        "external": external,
    })

results["float_repr"] = repr_float_cases


# ============================================================================
# 5. IntDistribution: _adjust_int_uniform_high 精确对比
# ============================================================================
adjust_int_cases = []
int_configs = [
    (0, 10, 3, "step3"),       # high → 9
    (0, 10, 2, "step2"),       # high → 10 (整除)
    (0, 10, 1, "step1"),       # high → 10 (整除)
    (0, 20, 7, "step7"),       # high → 14
    (0, 100, 30, "step30"),    # high → 90
    (5, 25, 4, "offset5_step4"),  # high → 25 (range=20, 20%4=0)
    (5, 26, 4, "offset5_step4_adj"),  # range=21, 21%4=1 → high=25
    (0, 2, 5, "tiny_range"),   # range < step → high=0
    (0, 0, 1, "zero_range"),   # low == high → high=0
    (1, 1000, 1, "large_range"),
    (1, 1000, 7, "large_step7"),  # 1000-1=999, 999//7=142, 142*7+1=995
]

for low, high, step, label in int_configs:
    adjusted = _adjust_int_uniform_high(low, high, step)
    id_dist = IntDistribution(low, high, step=step)
    s = id_dist.single()
    
    # contains 测试
    contains = {}
    for val in [low, adjusted, high, low + step, low + 2*step]:
        contains[str(val)] = id_dist._contains(float(val))
    
    adjust_int_cases.append({
        "label": label,
        "low": low,
        "high_input": high,
        "step": step,
        "adjusted_high": adjusted,
        "single": s,
        "contains": contains,
    })

results["int_adjust_high"] = adjust_int_cases


# ============================================================================
# 6. IntDistribution: single()
# ============================================================================
single_int_cases = []
int_single_configs = [
    (0, 10, 1, False, False),
    (5, 5, 1, False, True),
    (5, 5, 1, True, True),       # log, equal
    (1, 100, 1, True, False),    # log, range
    (0, 2, 5, False, True),      # step > range
    (0, 10, 5, False, False),    # step == range → NOT single (2 values: 0, 5, 10)
    (0, 5, 5, False, False),     # step == range → NOT single (2 values: 0, 5)
    (0, 4, 5, False, True),      # adjusted high=0 → single
]

for low, high, step, log, expected in int_single_configs:
    id_dist = IntDistribution(low, high, step=step, log=log)
    actual = id_dist.single()
    assert actual == expected, f"INT MISMATCH: ({low},{high},step={step},log={log}).single() = {actual}, expected {expected}"
    single_int_cases.append({
        "low": low,
        "high_input": high,
        "step": step,
        "log": log,
        "adjusted_high": id_dist.high,
        "single": actual,
    })

results["int_single"] = single_int_cases


# ============================================================================
# 7. IntDistribution: contains() / repr()
# ============================================================================
int_contains_cases = []

id_dist = IntDistribution(0, 10, step=3)
for val, exp in [(0.0, True), (3.0, True), (6.0, True), (9.0, True),
                 (10.0, False), (1.0, False), (2.0, False),
                 (-1.0, False), (12.0, False), (0.5, False)]:
    int_contains_cases.append({
        "dist": "int_0_10_step3",
        "value": val,
        "expected": exp,
    })

# NaN, Inf for IntDistribution
id_basic = IntDistribution(0, 10)
for val_s in ["nan", "inf", "-inf"]:
    v = float(val_s)
    try:
        c = id_basic._contains(v)
    except (ValueError, OverflowError):
        c = False
    int_contains_cases.append({
        "dist": "int_0_10",
        "value": val_s,
        "expected": c,
    })

results["int_contains"] = int_contains_cases


# repr 测试
int_repr_cases = []
for v in [0, 5, 10]:
    id_dist2 = IntDistribution(0, 10, step=1)
    internal = id_dist2.to_internal_repr(v)
    external = id_dist2.to_external_repr(internal)
    int_repr_cases.append({
        "dist": "int_0_10",
        "value": v,
        "internal": internal,
        "external": external,
    })

results["int_repr"] = int_repr_cases


# ============================================================================
# 8. CategoricalDistribution: 完整测试
# ============================================================================
cat_cases = []

# 字符串选项
cd = CategoricalDistribution(["a", "b", "c"])
for choice, idx in [("a", 0), ("b", 1), ("c", 2)]:
    cat_cases.append({
        "dist": "cat_abc",
        "choice": choice,
        "internal": cd.to_internal_repr(choice),
        "external": cd.to_external_repr(float(idx)),
        "single": cd.single(),
    })

# 混合类型
cd_mixed = CategoricalDistribution([None, True, 42, 3.14, "hello"])
for choice, idx in [(None, 0), (True, 1), (42, 2), (3.14, 3), ("hello", 4)]:
    internal = cd_mixed.to_internal_repr(choice)
    external = cd_mixed.to_external_repr(float(idx))
    cat_cases.append({
        "dist": "cat_mixed",
        "choice": str(choice),
        "choice_type": type(choice).__name__,
        "internal": internal,
        "external_type": type(external).__name__,
        "external": str(external),
        "single": cd_mixed.single(),
    })

# contains 测试
cat_contains = []
cd3 = CategoricalDistribution(["x", "y", "z"])
for val, exp in [(0.0, True), (1.0, True), (2.0, True), (3.0, False),
                 (-1.0, False), (0.5, True), (2.9, True)]:  # int(0.5)=0, int(2.9)=2
    cat_contains.append({
        "value": val,
        "expected": exp,
    })
cat_cases.append({"dist": "cat_xyz_contains", "contains": cat_contains})

# 单选项
cd_one = CategoricalDistribution([42])
cat_cases.append({
    "dist": "cat_single",
    "single": cd_one.single(),
    "internal_42": cd_one.to_internal_repr(42),
    "external_0": cd_one.to_external_repr(0.0),
})

results["categorical"] = cat_cases


# ============================================================================
# 9. check_distribution_compatibility
# ============================================================================
compat_cases = []

def test_compat(a, b, label, expect_ok):
    try:
        check_distribution_compatibility(a, b)
        ok = True
    except ValueError:
        ok = False
    assert ok == expect_ok, f"COMPAT MISMATCH: {label}: got {ok}, expected {expect_ok}"
    compat_cases.append({"label": label, "compatible": ok})

# Float vs Float: same log → OK
test_compat(
    FloatDistribution(0.0, 1.0),
    FloatDistribution(0.0, 10.0),
    "float_same_log_false", True
)

# Float: different log → NOT OK
test_compat(
    FloatDistribution(0.01, 1.0, log=True),
    FloatDistribution(0.0, 1.0),
    "float_diff_log", False
)

# Float: different step → OK (Python 不检查 step)
test_compat(
    FloatDistribution(0.0, 1.0, step=0.1),
    FloatDistribution(0.0, 1.0, step=0.2),
    "float_diff_step_ok", True
)

# Float: step vs None → OK
test_compat(
    FloatDistribution(0.0, 1.0, step=0.1),
    FloatDistribution(0.0, 1.0),
    "float_step_vs_none_ok", True
)

# Int vs Int: same log → OK
test_compat(
    IntDistribution(0, 10),
    IntDistribution(0, 100),
    "int_same_log_false", True
)

# Int: different log → NOT OK
test_compat(
    IntDistribution(1, 100, log=True),
    IntDistribution(0, 100),
    "int_diff_log", False
)

# Int: different step → OK
test_compat(
    IntDistribution(0, 10, step=1),
    IntDistribution(0, 10, step=2),
    "int_diff_step_ok", True
)

# Categorical: same choices → OK
test_compat(
    CategoricalDistribution(["a", "b"]),
    CategoricalDistribution(["a", "b"]),
    "cat_same", True
)

# Categorical: different choices → NOT OK
test_compat(
    CategoricalDistribution(["a", "b"]),
    CategoricalDistribution(["a", "c"]),
    "cat_diff", False
)

# Float vs Int → NOT OK
test_compat(
    FloatDistribution(0.0, 1.0),
    IntDistribution(0, 1),
    "float_vs_int", False
)

# Float vs Categorical → NOT OK
test_compat(
    FloatDistribution(0.0, 1.0),
    CategoricalDistribution(["a"]),
    "float_vs_cat", False
)

# Int vs Categorical → NOT OK
test_compat(
    IntDistribution(0, 10),
    CategoricalDistribution(["a"]),
    "int_vs_cat", False
)

results["compatibility"] = compat_cases


# ============================================================================
# 10. _get_single_value
# ============================================================================
single_value_cases = []

# Float single
fd_s = FloatDistribution(5.0, 5.0)
sv = _get_single_value(fd_s)
single_value_cases.append({"type": "float", "low": 5.0, "value": sv, "value_type": type(sv).__name__})

# Float with step (single after adjustment)
fd_adj = FloatDistribution(3.0, 3.1, step=0.5)
# adjusted high = 3.0 (since (3.1-3.0)=0.1 < 0.5)
sv2 = _get_single_value(fd_adj)
single_value_cases.append({"type": "float_step", "low": 3.0, "high_input": 3.1, "step": 0.5,
                            "adjusted_high": fd_adj.high, "value": sv2})

# Int single
id_s = IntDistribution(7, 7)
sv3 = _get_single_value(id_s)
single_value_cases.append({"type": "int", "low": 7, "value": sv3, "value_type": type(sv3).__name__})

# Int single (step > range)
id_adj = IntDistribution(0, 2, step=5)
sv4 = _get_single_value(id_adj)
single_value_cases.append({"type": "int_step", "low": 0, "high_input": 2, "step": 5,
                            "adjusted_high": id_adj.high, "value": sv4})

# Categorical single
cd_s = CategoricalDistribution(["only"])
sv5 = _get_single_value(cd_s)
single_value_cases.append({"type": "categorical", "choices": ["only"], "value": sv5})

results["single_value"] = single_value_cases


# ============================================================================
# 11. _is_distribution_log
# ============================================================================
log_cases = []

log_tests = [
    (FloatDistribution(0.0, 1.0), False, "float_no_log"),
    (FloatDistribution(0.01, 1.0, log=True), True, "float_log"),
    (IntDistribution(0, 10), False, "int_no_log"),
    (IntDistribution(1, 100, log=True), True, "int_log"),
    (CategoricalDistribution(["a"]), False, "categorical"),
]

for dist, expected, label in log_tests:
    actual = _is_distribution_log(dist)
    assert actual == expected
    log_cases.append({"label": label, "is_log": actual})

results["is_log"] = log_cases


# ============================================================================
# 12. JSON 序列化/反序列化 往返
# ============================================================================
json_cases = []

dists = [
    ("float_basic", FloatDistribution(0.0, 10.0)),
    ("float_log", FloatDistribution(1e-5, 1.0, log=True)),
    ("float_step", FloatDistribution(0.0, 1.0, step=0.1)),
    ("float_step03", FloatDistribution(0.0, 1.0, step=0.3)),  # high adjusted to 0.9
    ("int_basic", IntDistribution(0, 100)),
    ("int_log", IntDistribution(1, 1000, log=True)),
    ("int_step3", IntDistribution(0, 10, step=3)),  # high adjusted to 9
    ("cat_str", CategoricalDistribution(["x", "y", "z"])),
    ("cat_mixed", CategoricalDistribution([None, 1, 2.5, "hi"])),
]

for label, dist in dists:
    json_str = distribution_to_json(dist)
    parsed = json_to_distribution(json_str)
    roundtrip = (dist == parsed)
    json_cases.append({
        "label": label,
        "json": json_str,
        "roundtrip_ok": roundtrip,
    })

results["json_roundtrip"] = json_cases


# ============================================================================
# 13. 验证统计 
# ============================================================================
total_golden = 0
for group, items in results.items():
    if isinstance(items, list):
        total_golden += len(items)

print(f"✅ Generated {total_golden} golden values across {len(results)} groups", file=sys.stderr)


# ============================================================================
# 输出
# ============================================================================
# NaN/Inf → null/字符串
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj):
                return None
            if math.isinf(obj):
                return "inf" if obj > 0 else "-inf"
        return super().default(obj)

out_path = "tests/distributions_deep_golden_values.json"
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, cls=CustomEncoder)
print(f"Written to {out_path}", file=sys.stderr)
print(json.dumps({"total_golden_values": total_golden, "groups": list(results.keys())}, indent=2))
