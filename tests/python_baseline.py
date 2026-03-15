#!/usr/bin/env python3
"""Generate Python reference values for cross-validation with Rust tests."""
import json, sys
sys.path.insert(0, "/Users/lichangqing/Rust/optuna/optuna")

from optuna.distributions import FloatDistribution, IntDistribution, CategoricalDistribution

r = {}

# === FloatDistribution ===
d = FloatDistribution(0.0, 1.0, step=0.3)
r["float_step03_high"] = d.high
r["float_step03_single"] = d.single()
r["float_step03_contains_09"] = d._contains(0.9)
r["float_step03_contains_10"] = d._contains(1.0)
r["float_step03_contains_06"] = d._contains(0.6)
r["float_step03_contains_03"] = d._contains(0.3)
r["float_step03_contains_00"] = d._contains(0.0)
r["float_step03_contains_04"] = d._contains(0.4)

d2 = FloatDistribution(0.0, 1.0, step=0.25)
r["float_step025_high"] = d2.high
r["float_step025_single"] = d2.single()

d3 = FloatDistribution(0.0, 0.1, step=0.2)
r["float_small_range_single"] = d3.single()
r["float_small_range_high"] = d3.high

d4 = FloatDistribution(0.001, 10.0, log=True)
r["float_log_single"] = d4.single()

d5 = FloatDistribution(5.0, 5.0)
r["float_equal_single"] = d5.single()

# === IntDistribution ===
di = IntDistribution(0, 10, step=3)
r["int_step3_high"] = di.high
r["int_step3_single"] = di.single()
r["int_step3_contains_0"] = di._contains(0)
r["int_step3_contains_3"] = di._contains(3)
r["int_step3_contains_6"] = di._contains(6)
r["int_step3_contains_9"] = di._contains(9)
r["int_step3_contains_10"] = di._contains(10)
r["int_step3_contains_1"] = di._contains(1)

di2 = IntDistribution(0, 10, step=2)
r["int_step2_high"] = di2.high

di3 = IntDistribution(0, 2, step=5)
r["int_small_range_single"] = di3.single()
r["int_small_range_high"] = di3.high

di4 = IntDistribution(1, 100, log=True)
r["int_log_single"] = di4.single()

# === CategoricalDistribution ===
dc = CategoricalDistribution(["a", "b", "c"])
r["cat_to_internal_a"] = dc.to_internal_repr("a")
r["cat_to_internal_c"] = dc.to_internal_repr("c")
r["cat_to_external_0"] = dc.to_external_repr(0.0)
r["cat_single"] = dc.single()
r["cat_contains_0"] = dc._contains(0.0)
r["cat_contains_3"] = dc._contains(3.0)

dc2 = CategoricalDistribution([42])
r["cat_one_single"] = dc2.single()

# === Wilcoxon cross-validation ===
import numpy as np
from scipy.stats import wilcoxon

# Test 1: clear difference
diff1 = [10.0] * 20
stat1 = wilcoxon(diff1, alternative="greater", zero_method="zsplit")
r["wilcoxon_clear_diff_pvalue"] = float(stat1.pvalue)
r["wilcoxon_clear_diff_stat"] = float(stat1.statistic)

# Test 2: identical values
diff2 = [0.0] * 10
stat2 = wilcoxon(diff2, alternative="greater", zero_method="zsplit")
r["wilcoxon_zero_pvalue"] = float(stat2.pvalue)

# Test 3: mixed
diff3 = [1.0, -0.5, 2.0, -0.3, 1.5, 0.8, -0.1, 1.2, 0.5, -0.2]
stat3 = wilcoxon(diff3, alternative="greater", zero_method="zsplit")
r["wilcoxon_mixed_pvalue"] = float(stat3.pvalue)
r["wilcoxon_mixed_stat"] = float(stat3.statistic)

# Test 4: Test 3 with alternative="less" (for maximize)
stat4 = wilcoxon(diff3, alternative="less", zero_method="zsplit")
r["wilcoxon_mixed_less_pvalue"] = float(stat4.pvalue)

# === Pruner cross-validation ===
# MedianPruner
from optuna.pruners import MedianPruner, PercentilePruner, ThresholdPruner

# Percentile pruner: compute the percentile manually
import numpy as np
vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
r["percentile_25"] = float(np.percentile(vals, 25))
r["percentile_50"] = float(np.percentile(vals, 50))
r["percentile_75"] = float(np.percentile(vals, 75))

# === Search space transform cross-validation ===
from optuna._transform import _SearchSpaceTransform
from optuna.distributions import FloatDistribution, IntDistribution

ss = {"x": FloatDistribution(0.0, 1.0), "y": IntDistribution(1, 10)}
t = _SearchSpaceTransform(ss)
# Transform a sample point
transformed = t.transform({"x": 0.5, "y": 5})
r["transform_x"] = float(transformed[0])
r["transform_y"] = float(transformed[1])

# Untransform
untransformed = t.untransform(transformed)
r["untransform_x"] = float(untransformed["x"])
r["untransform_y"] = int(untransformed["y"])

# Log transform
ss_log = {"lr": FloatDistribution(0.001, 1.0, log=True)}
t_log = _SearchSpaceTransform(ss_log)
transformed_log = t_log.transform({"lr": 0.01})
r["transform_log_lr"] = float(transformed_log[0])
untransformed_log = t_log.untransform(transformed_log)
r["untransform_log_lr"] = float(untransformed_log["lr"])

# Step transform
ss_step = {"x": FloatDistribution(0.0, 1.0, step=0.25)}
t_step = _SearchSpaceTransform(ss_step)
transformed_step = t_step.transform({"x": 0.5})
r["transform_step_x"] = float(transformed_step[0])
untransformed_step = t_step.untransform(transformed_step)
r["untransform_step_x"] = float(untransformed_step["x"])

# === Importance cross-validation: use high-level API ===
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)
study_imp = optuna.create_study(direction="minimize")
for i in range(30):
    trial = study_imp.ask()
    x = trial.suggest_float("x", -5.0, 5.0)
    y = trial.suggest_float("y", -5.0, 5.0)
    z = trial.suggest_float("z", -5.0, 5.0)
    study_imp.tell(trial, x**2 + 0.1 * y**2 + 0.01 * z**2)

imp_result = optuna.importance.get_param_importances(study_imp)
# x should be most important
r["importance_x_is_max"] = bool(max(imp_result, key=imp_result.get) == "x")
r["importance_all_positive"] = bool(all(v >= 0 for v in imp_result.values()))
r["importance_sum_leq_1"] = bool(sum(imp_result.values()) <= 1.0 + 1e-6)

# === Crowding distance cross-validation ===
from optuna._hypervolume.wfg import compute_hypervolume as wfg_hv
pts = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
ref = np.array([5.0, 5.0])
r["hypervolume_2d"] = float(wfg_hv(pts, ref))

print(json.dumps(r, indent=2))
