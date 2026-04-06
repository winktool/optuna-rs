#!/usr/bin/env python3
"""
Pruners 模块深度交叉验证金标准值生成器

生成 Python optuna 剪枝器的参考值，用于与 Rust 实现的精确对比。
覆盖：PercentilePruner, PatientPruner, SuccessiveHalvingPruner,
       HyperbandPruner, WilcoxonPruner
"""

import json
import math
import binascii
import datetime
import numpy as np
from scipy import stats

from optuna.pruners._percentile import (
    _get_best_intermediate_result_over_steps,
    _get_percentile_intermediate_result_over_trials,
    _is_first_in_interval_step,
)
from optuna.pruners._successive_halving import (
    _estimate_min_resource,
    _is_trial_promotable_to_next_rung,
)
from optuna.study._study_direction import StudyDirection
from optuna.trial import FrozenTrial
from optuna.trial._state import TrialState

results = {}


# ── 辅助函数 ──────────────────────────────────────────────────
def make_frozen(number, state, iv, value=None):
    return FrozenTrial(
        number=number,
        state=state,
        value=value,
        datetime_start=datetime.datetime.now(),
        datetime_complete=datetime.datetime.now() if state == TrialState.COMPLETE else None,
        params={},
        distributions={},
        user_attrs={},
        system_attrs={},
        intermediate_values=iv,
        trial_id=number,
    )


# ── Group 1: nanpercentile 精确值 (13 cases) ──────────────────
group = []
cases_np = [
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 0.0),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 25.0),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 50.0),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 75.0),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], 100.0),
    ([0.1, 0.2, 0.3], 50.0),
    ([0.1, 0.2, 0.3], 10.0),
    ([0.1, 0.2, 0.3], 90.0),
    ([100.0], 50.0),
    ([1.0, 1000.0], 50.0),
    ([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11], 33.3),
    ([1.0, 3.0, 4.0, 5.0], 50.0),  # nanpercentile ignores NaN if present
    ([2.0, 4.0, 6.0, 8.0, 10.0], 37.5),
]
for vals, pct in cases_np:
    group.append({"values": [float(v) for v in vals], "percentile": pct,
                  "result": float(np.nanpercentile(vals, pct))})
results["nanpercentile_precision"] = group

# ── Group 2: Wilcoxon p-value 精确值 (8 cases) ───────────────
group = []

wilcoxon_cases = [
    ("all_positive_n5", [1.0, 2.0, 3.0, 4.0, 5.0]),
    ("all_negative_n5", [-1.0, -2.0, -3.0, -4.0, -5.0]),
    ("mixed_n8", [3.0, -1.0, 2.5, -0.5, 1.0, -0.2, 0.8, -0.3]),
    ("with_ties_n6", [1.0, 1.0, -1.0, 2.0, 2.0, -3.0]),
    ("with_zeros_n8", [1.0, 0.0, -2.0, 0.0, 3.0, -0.5, 0.0, 1.5]),
    ("mixed_n15", [0.5, -0.3, 1.2, -0.8, 0.1, -0.1, 0.9, -0.7,
                   1.5, -0.2, 0.3, -0.4, 0.6, -0.5, 1.0]),
    ("symmetric_n6", [1.0, -1.0, 1.0, -1.0, 1.0, -1.0]),
    ("large_spread", [100.0, -50.0, 200.0, -30.0, 150.0]),
]

for label, diffs in wilcoxon_cases:
    rg = stats.wilcoxon(diffs, alternative='greater', zero_method='zsplit')
    rl = stats.wilcoxon(diffs, alternative='less', zero_method='zsplit')
    group.append({
        "label": label, "diffs": diffs,
        "p_greater": float(rg.pvalue), "p_less": float(rl.pvalue),
    })
results["wilcoxon_p_values"] = group

# ── Group 3: CRC32 hash 交叉验证 (10 cases) ──────────────────
group = []
for s in ["study_0", "study_1", "study_42", "study_100",
          "my_study_0", "my_study_1", "test_5",
          "experiment_alpha_99", "optuna_study_name_123", ""]:
    group.append({"input": s, "crc32": binascii.crc32(s.encode())})
results["crc32_hashes"] = group

# ── Group 4: is_first_in_interval_step (12 cases) ────────────
group = []
interval_cases = [
    (0, [0], 0, 1, True),
    (1, [0, 1], 0, 1, True),
    (2, [0, 1, 2], 0, 3, False),
    (3, [0, 1, 2, 3], 0, 3, True),
    (5, list(range(10)), 3, 2, True),
    (4, list(range(10)), 3, 2, False),
    (6, list(range(10)), 3, 2, False),
    (7, list(range(10)), 3, 2, True),
    (0, [0], 0, 5, True),
    (4, list(range(5)), 0, 5, False),
    (5, list(range(6)), 0, 5, True),
    (10, list(range(11)), 0, 5, True),
]
for step, steps_list, warmup, interval, _ in interval_cases:
    iv = {s: 0.0 for s in steps_list}
    result = _is_first_in_interval_step(step, iv, warmup, interval)
    group.append({"step": step, "steps": steps_list,
                  "n_warmup": warmup, "interval": interval, "expected": result})
results["is_first_in_interval_step"] = group

# ── Group 5: best_intermediate_result (8 cases) ──────────────
group = []
best_cases = [
    ({0: 1.0, 1: 2.0, 2: 3.0}, "minimize"),
    ({0: 1.0, 1: 2.0, 2: 3.0}, "maximize"),
    ({0: 5.0, 1: 3.0, 2: 7.0, 3: 1.0}, "minimize"),
    ({0: 5.0, 1: 3.0, 2: 7.0, 3: 1.0}, "maximize"),
    ({0: -10.0, 1: -20.0, 2: -5.0}, "minimize"),
    ({0: -10.0, 1: -20.0, 2: -5.0}, "maximize"),
]
for iv, d in best_cases:
    trial = make_frozen(0, TrialState.RUNNING, iv)
    direction = StudyDirection.MINIMIZE if d == "minimize" else StudyDirection.MAXIMIZE
    r = _get_best_intermediate_result_over_steps(trial, direction)
    group.append({
        "intermediate_values": {str(k): v for k, v in iv.items()},
        "direction": d,
        "best": r,
    })
results["best_intermediate_result"] = group

# ── Group 6: percentile_over_trials (6 cases) ────────────────
group = []

# 6a: 3 trials at step 0, p50, minimize → percentile of [1,2,3]
c3 = [make_frozen(i, TrialState.COMPLETE, {0: float(i + 1)}, value=float(i + 1))
      for i in range(3)]
p_min = _get_percentile_intermediate_result_over_trials(
    c3, StudyDirection.MINIMIZE, 0, 50.0, 1)
p_max = _get_percentile_intermediate_result_over_trials(
    c3, StudyDirection.MAXIMIZE, 0, 50.0, 1)
group.append({"label": "3t_s0_p50_min", "result": float(p_min)})
group.append({"label": "3t_s0_p50_max", "result": float(p_max)})

# 6b: 5 trials, p25 and p75
c5 = [make_frozen(i, TrialState.COMPLETE, {0: float((i + 1) * 10)},
      value=float((i + 1) * 10)) for i in range(5)]
p25 = _get_percentile_intermediate_result_over_trials(
    c5, StudyDirection.MINIMIZE, 0, 25.0, 1)
p75 = _get_percentile_intermediate_result_over_trials(
    c5, StudyDirection.MINIMIZE, 0, 75.0, 1)
group.append({"label": "5t_s0_p25_min", "result": float(p25)})
group.append({"label": "5t_s0_p75_min", "result": float(p75)})

# 6c: n_min_trials not met → NaN
r_insuf = _get_percentile_intermediate_result_over_trials(
    [make_frozen(0, TrialState.COMPLETE, {0: 1.0}, value=1.0)],
    StudyDirection.MINIMIZE, 0, 50.0, 3)
group.append({"label": "n_min_not_met", "result": None if math.isnan(r_insuf) else float(r_insuf)})

# 6d: missing step → NaN
r_miss = _get_percentile_intermediate_result_over_trials(
    [make_frozen(0, TrialState.COMPLETE, {0: 1.0}, value=1.0)],
    StudyDirection.MINIMIZE, 5, 50.0, 1)
group.append({"label": "missing_step", "result": None if math.isnan(r_miss) else float(r_miss)})

results["percentile_over_trials"] = group

# ── Group 7: PatientPruner 窗口逻辑 (6 cases) ────────────────
group = []
patient_cases = [
    ([(0, 1.0), (1, 0.5), (2, 0.8), (3, 0.9), (4, 1.0)], 2, 0.0, "minimize", True),
    ([(0, 1.0), (1, 0.9), (2, 0.8), (3, 0.7)], 2, 0.0, "minimize", False),
    ([(0, 1.0), (1, 2.0), (2, 1.5), (3, 1.0), (4, 0.5)], 2, 0.0, "maximize", True),
    ([(0, 1.0), (1, 0.95), (2, 0.92), (3, 0.91)], 2, 0.1, "minimize", False),
    ([(0, 0.5), (1, 0.8)], 0, 0.0, "minimize", True),
    ([(0, 8.0), (1, 10.0), (2, 9.5), (3, 9.8), (4, 9.2)], 2, 0.5, "maximize", False),
]
for values, patience, delta, d, _ in patient_cases:
    steps = sorted([s for s, _ in values])
    vdict = dict(values)
    if len(steps) <= patience + 1:
        result = False
    else:
        split = len(steps) - (patience + 1)
        bv = [vdict[s] for s in steps[:split] if not math.isnan(vdict[s])]
        av = [vdict[s] for s in steps[split:] if not math.isnan(vdict[s])]
        if not bv or not av:
            result = False
        elif d == "minimize":
            result = min(bv) + delta < min(av)
        else:
            result = max(bv) - delta > max(av)
    group.append({"values": values, "patience": patience, "min_delta": delta,
                  "direction": d, "maybe_prune": result})
results["patient_pruner_decisions"] = group

# ── Group 8: SuccessiveHalving rung 晋升步骤 (11 cases) ──────
group = []
sh_cases = [
    (1, 4, 0, 0), (1, 4, 0, 1), (1, 4, 0, 2),
    (10, 4, 0, 0), (10, 4, 0, 1),
    (1, 3, 0, 0), (1, 3, 0, 1), (1, 3, 0, 2),
    (1, 3, 1, 0), (1, 3, 1, 1), (1, 3, 2, 0),
]
for mr, rf, rate, rung in sh_cases:
    group.append({"min_resource": mr, "reduction_factor": rf,
                  "min_early_stopping_rate": rate, "rung": rung,
                  "promotion_step": mr * rf ** (rate + rung)})
results["successive_halving_rung_steps"] = group

# ── Group 9: Hyperband bracket allocation (6 cases) ──────────
group = []
hb_cases = [
    (1, 27, 3), (1, 81, 3), (1, 100, 3),
    (1, 9, 3), (1, 16, 4), (1, 64, 4),
]
for mr, mxr, rf in hb_cases:
    nb = math.floor(math.log(mxr / mr, rf)) + 1
    budgets = [math.ceil(nb * rf ** (nb - 1 - bid) / (nb - bid))
               for bid in range(nb)]
    group.append({"min_resource": mr, "max_resource": mxr, "reduction_factor": rf,
                  "n_brackets": nb, "budgets": budgets, "total_budget": sum(budgets)})
results["hyperband_bracket_allocation"] = group

# ── Group 10: _is_trial_promotable_to_next_rung (9 cases) ────
group = []
prom_cases = [
    (1.0, [1.0, 2.0, 3.0, 4.0], 4, "minimize", True),
    (4.0, [1.0, 2.0, 3.0, 4.0], 4, "minimize", False),
    (2.0, [1.0, 2.0, 3.0, 4.0], 4, "minimize", False),
    (4.0, [1.0, 2.0, 3.0, 4.0], 4, "maximize", True),
    (3.0, [1.0, 2.0, 3.0, 4.0], 4, "maximize", False),
    (2.0, [1.0, 2.0, 3.0, 4.0], 2, "minimize", True),
    (3.0, [1.0, 2.0, 3.0, 4.0], 2, "minimize", False),
    (1.0, [1.0, 2.0], 4, "minimize", True),
    (2.0, [1.0, 2.0], 4, "minimize", False),
]
for val, comp, rf, d, _ in prom_cases:
    direction = StudyDirection.MINIMIZE if d == "minimize" else StudyDirection.MAXIMIZE
    r = _is_trial_promotable_to_next_rung(val, comp.copy(), rf, direction)
    group.append({"value": val, "competing": comp, "reduction_factor": rf,
                  "direction": d, "promotable": r})
results["promotable_to_next_rung"] = group

# ── Group 11: estimate_min_resource (5 cases) ────────────────
group = []
est_cases = [([100], 1), ([500], 5), ([99], 1), ([50, 200, 100], 2), ([1000], 10)]
for last_steps, _ in est_cases:
    trials = [make_frozen(i, TrialState.COMPLETE, {ls: 0.0}, value=0.0)
              for i, ls in enumerate(last_steps)]
    r = _estimate_min_resource(trials)
    group.append({"last_steps": last_steps, "result": r})
results["estimate_min_resource"] = group

# ── 输出 ──────────────────────────────────────────────────────
output = "tests/pruners_deep_golden_values.json"
with open(output, "w") as f:
    json.dump(results, f, indent=2)

print(f"Generated {output}")
for name, cases in results.items():
    print(f"  {name}: {len(cases)} cases")
print(f"Total: {sum(len(c) for c in results.values())} golden values")
