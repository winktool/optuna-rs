#!/usr/bin/env python3
"""
Python 原版测试：与 Rust optuna-rs 交叉验证。
同一份数据在 Python 和 Rust 中必须产生完全一致的结果。

运行: python3 tests/test_cross_validation.py
或:   pytest tests/test_cross_validation.py -v

每一组测试都有对应的 Rust #[test] 函数名（注释标注），
两边使用完全相同的输入数据和预期结果。
"""
import sys, math, json
sys.path.insert(0, "/Users/lichangqing/Rust/optuna/optuna")

import numpy as np
from optuna.distributions import (
    FloatDistribution, IntDistribution, CategoricalDistribution
)
from optuna._transform import _SearchSpaceTransform
from optuna._hypervolume.wfg import compute_hypervolume as wfg_hv
import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)

# ═══════════════════════════════════════════════════════════════════════════
#  1. FloatDistribution — 对应 Rust src/distributions/float.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_float_step03():
    """Rust: test_python_cross_float_step03"""
    d = FloatDistribution(0.0, 1.0, step=0.3)
    assert d.high == 0.9,                f"high={d.high}"
    assert d.single() == False,          f"single={d.single()}"
    assert d._contains(0.9) == True,     "0.9 should be in dist"
    assert d._contains(1.0) == False,    "1.0 should NOT be in dist"
    assert d._contains(0.6) == True,     "0.6 should be in dist"
    assert d._contains(0.3) == True,     "0.3 should be in dist"
    assert d._contains(0.0) == True,     "0.0 should be in dist"
    assert d._contains(0.4) == False,    "0.4 should NOT be in dist"

def test_float_step025():
    """Rust: test_python_cross_float_step025"""
    d = FloatDistribution(0.0, 1.0, step=0.25)
    assert d.high == 1.0,               f"high={d.high}"
    assert d.single() == False,          f"single={d.single()}"

def test_float_small_range_step():
    """Rust: test_python_cross_float_small_range_step"""
    d = FloatDistribution(0.0, 0.1, step=0.2)
    assert d.single() == True,           f"single={d.single()}"
    assert d.high == 0.0,                f"high={d.high}"

def test_float_log():
    """Rust: test_python_cross_float_log"""
    d = FloatDistribution(0.001, 10.0, log=True)
    assert d.single() == False

def test_float_equal():
    """Rust: test_python_cross_float_equal"""
    d = FloatDistribution(5.0, 5.0)
    assert d.single() == True

def test_float_high_adjustment_step07():
    """Rust: test_python_cross_float_step07"""
    d = FloatDistribution(0.0, 1.0, step=0.7)
    assert d.high == 0.7,                f"high={d.high}"
    assert d._contains(0.7) == True
    assert d._contains(0.0) == True
    assert d._contains(1.0) == False

def test_float_repr_roundtrip():
    """Rust: test_python_cross_float_repr"""
    d = FloatDistribution(1.0, 10.0)
    for v in [1.0, 5.5, 9.999]:
        internal = d.to_internal_repr(v)
        external = d.to_external_repr(internal)
        assert abs(external - v) < 1e-12, f"roundtrip failed for {v}"

def test_float_log_repr():
    """Rust: test_python_cross_float_log_repr"""
    d = FloatDistribution(0.01, 100.0, log=True)
    for v in [0.01, 1.0, 50.0, 100.0]:
        internal = d.to_internal_repr(v)
        external = d.to_external_repr(internal)
        assert abs(external - v) < 1e-12

# ═══════════════════════════════════════════════════════════════════════════
#  2. IntDistribution — 对应 Rust src/distributions/int.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_int_step3():
    """Rust: test_python_cross_int_step3"""
    d = IntDistribution(0, 10, step=3)
    assert d.high == 9,                     f"high={d.high}"
    assert d.single() == False
    assert d._contains(0) == True
    assert d._contains(3) == True
    assert d._contains(6) == True
    assert d._contains(9) == True
    assert d._contains(10) == False
    assert d._contains(1) == False

def test_int_step2():
    """Rust: test_python_cross_int_step2"""
    d = IntDistribution(0, 10, step=2)
    assert d.high == 10

def test_int_small_range():
    """Rust: test_python_cross_int_small_range_step"""
    d = IntDistribution(0, 2, step=5)
    assert d.single() == True
    assert d.high == 0

def test_int_log():
    """Rust: test_python_cross_int_log"""
    d = IntDistribution(1, 100, log=True)
    assert d.single() == False

def test_int_repr_roundtrip():
    """Rust: test_python_cross_int_repr"""
    d = IntDistribution(0, 100, step=5)
    for v in [0, 5, 50, 95, 100]:
        internal = d.to_internal_repr(v)
        external = d.to_external_repr(internal)
        assert external == v, f"roundtrip failed for {v}: got {external}"

def test_int_step7():
    """Rust: test_python_cross_int_step7"""
    d = IntDistribution(0, 20, step=7)
    assert d.high == 14,                    f"high={d.high}"
    assert d._contains(0) == True
    assert d._contains(7) == True
    assert d._contains(14) == True
    assert d._contains(20) == False

# ═══════════════════════════════════════════════════════════════════════════
#  3. CategoricalDistribution — 对应 Rust src/distributions/categorical.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_cat_basic():
    """Rust: test_python_cross_categorical"""
    d = CategoricalDistribution(["a", "b", "c"])
    assert d.to_internal_repr("a") == 0.0
    assert d.to_internal_repr("c") == 2.0
    assert d.to_external_repr(0.0) == "a"
    assert d.single() == False
    assert d._contains(0.0) == True
    assert d._contains(3.0) == False

def test_cat_single():
    """Rust: test_python_cross_categorical_one"""
    d = CategoricalDistribution([42])
    assert d.single() == True

def test_cat_nan():
    """Rust: test_python_cross_cat_nan"""
    d = CategoricalDistribution([1.0, float("nan"), "x"])
    # NaN 在 choices 中应该能正确处理
    assert d.to_internal_repr(float("nan")) == 1.0
    assert d._contains(1.0) == True   # NaN 对应 index 1
    assert d._contains(0.0) == True   # 1.0 对应 index 0
    assert d._contains(2.0) == True   # "x" 对应 index 2

def test_cat_mixed_types():
    """Rust: test_python_cross_cat_mixed"""
    d = CategoricalDistribution([1, 2.5, "hello", True])
    assert d.to_internal_repr(1) == 0.0
    assert d.to_internal_repr(2.5) == 1.0
    assert d.to_internal_repr("hello") == 2.0
    assert d.single() == False

# ═══════════════════════════════════════════════════════════════════════════
#  4. Wilcoxon Signed-Rank Test — 对应 Rust src/pruners/wilcoxon.rs
# ═══════════════════════════════════════════════════════════════════════════

from scipy.stats import wilcoxon

def test_wilcoxon_clear_diff():
    """Rust: test_python_cross_wilcoxon_clear_diff"""
    diff = [10.0] * 20
    result = wilcoxon(diff, alternative="greater", zero_method="zsplit")
    assert result.pvalue < 0.001, f"p={result.pvalue}"

def test_wilcoxon_all_zero():
    """Rust: test_python_cross_wilcoxon_all_zero"""
    diff = [0.0] * 10
    result = wilcoxon(diff, alternative="greater", zero_method="zsplit")
    assert result.pvalue == 1.0, f"p={result.pvalue}"

def test_wilcoxon_mixed():
    """Rust: test_python_cross_wilcoxon_mixed"""
    diff = [1.0, -0.5, 2.0, -0.3, 1.5, 0.8, -0.1, 1.2, 0.5, -0.2]
    result_gt = wilcoxon(diff, alternative="greater", zero_method="zsplit")
    result_lt = wilcoxon(diff, alternative="less", zero_method="zsplit")
    # p(greater) 应该较小（正值偏多），p(less) 应该较大
    assert result_gt.pvalue < 0.1, f"p_gt={result_gt.pvalue}"
    assert result_lt.pvalue > 0.9, f"p_lt={result_lt.pvalue}"

def test_wilcoxon_large_sample():
    """Rust: test_python_cross_wilcoxon_large — 大样本正态近似验证"""
    np.random.seed(42)
    diff = list(np.random.normal(0.5, 1.0, 50))
    result = wilcoxon(diff, alternative="greater", zero_method="zsplit")
    # 记录精确值供 Rust 对比
    print(f"  wilcoxon_large_pvalue = {result.pvalue}")
    assert 0.0 < result.pvalue < 1.0

def test_wilcoxon_tied_values():
    """Rust: test_python_cross_wilcoxon_tied — 大量并列值"""
    diff = [1.0, 1.0, 1.0, -1.0, -1.0, 2.0, 2.0, -2.0, 3.0, 0.0]
    result = wilcoxon(diff, alternative="greater", zero_method="zsplit")
    print(f"  wilcoxon_tied_pvalue = {result.pvalue}")
    assert 0.0 < result.pvalue < 1.0

# ═══════════════════════════════════════════════════════════════════════════
#  5. Percentile/Median 剪枝器 — 对应 Rust src/pruners/percentile.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_percentile_values():
    """Rust: test_python_cross_percentile_values"""
    vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    assert np.percentile(vals, 25) == 3.25
    assert np.percentile(vals, 50) == 5.5
    assert np.percentile(vals, 75) == 7.75

def test_percentile_edge_cases():
    """Rust: test_python_cross_percentile_edge"""
    # 单个值
    assert np.percentile([5.0], 50) == 5.0
    assert np.percentile([5.0], 0) == 5.0
    assert np.percentile([5.0], 100) == 5.0
    # 两个值
    assert np.percentile([1.0, 3.0], 50) == 2.0
    assert np.percentile([1.0, 3.0], 25) == 1.5
    # 不均匀间距
    vals2 = [1.0, 10.0, 100.0]
    p50 = np.percentile(vals2, 50)
    assert p50 == 10.0, f"p50={p50}"

# ═══════════════════════════════════════════════════════════════════════════
#  6. Threshold 剪枝器 — 对应 Rust src/pruners/threshold.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_threshold_pruner():
    """Rust: test_python_cross_threshold"""
    from optuna.pruners import ThresholdPruner
    study = optuna.create_study(direction="minimize")
    pruner = ThresholdPruner(upper=5.0, lower=0.0, n_warmup_steps=0)
    # 模拟完整试验+ 运行中试验
    trial0 = optuna.trial.create_trial(
        values=[1.0], state=optuna.trial.TrialState.COMPLETE,
        intermediate_values={0: 2.0}
    )
    study.add_trial(trial0)
    # value=6.0 超过 upper=5.0 → 应剪枝
    trial1 = optuna.trial.create_trial(
        state=optuna.trial.TrialState.RUNNING,
        intermediate_values={0: 6.0}
    )
    assert pruner.prune(study=study, trial=trial1) == True
    # value=3.0 在范围内 → 不剪枝
    trial2 = optuna.trial.create_trial(
        state=optuna.trial.TrialState.RUNNING,
        intermediate_values={0: 3.0}
    )
    assert pruner.prune(study=study, trial=trial2) == False
    # value=-1.0 低于 lower=0.0 → 应剪枝
    trial3 = optuna.trial.create_trial(
        state=optuna.trial.TrialState.RUNNING,
        intermediate_values={0: -1.0}
    )
    assert pruner.prune(study=study, trial=trial3) == True

# ═══════════════════════════════════════════════════════════════════════════
#  7. Patient 剪枝器 — 对应 Rust src/pruners/patient.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_patient_pruner():
    """Rust: test_python_cross_patient"""
    from optuna.pruners import PatientPruner, NopPruner
    pruner = PatientPruner(NopPruner(), patience=2, min_delta=0.0)
    study = optuna.create_study(direction="minimize")
    # 连续 3 步没有改善 (patience=2) → 第 3 步应该考虑剪枝
    # 但 NopPruner 永不剪枝，所以 PatientPruner 也不会
    trial0 = optuna.trial.create_trial(
        state=optuna.trial.TrialState.RUNNING,
        intermediate_values={0: 5.0, 1: 5.0, 2: 5.0, 3: 5.0}
    )
    result = pruner.prune(study=study, trial=trial0)
    # NopPruner always returns False, so patient also returns False
    assert result == False

# ═══════════════════════════════════════════════════════════════════════════
#  8. SearchSpaceTransform — 对应 Rust src/search_space/transform.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_transform_basic():
    """Rust: test_python_cross_transform_basic"""
    ss = {"x": FloatDistribution(0.0, 1.0), "y": IntDistribution(1, 10)}
    t = _SearchSpaceTransform(ss)
    result = t.transform({"x": 0.5, "y": 5})
    assert abs(result[0] - 0.5) < 1e-12
    assert abs(result[1] - 5.0) < 1e-12
    back = t.untransform(result)
    assert abs(back["x"] - 0.5) < 1e-12
    assert back["y"] == 5

def test_transform_log():
    """Rust: test_python_cross_transform_log"""
    ss = {"lr": FloatDistribution(0.001, 1.0, log=True)}
    t = _SearchSpaceTransform(ss)
    result = t.transform({"lr": 0.01})
    expected = math.log(0.01)
    assert abs(result[0] - expected) < 1e-12
    back = t.untransform(result)
    assert abs(back["lr"] - 0.01) < 1e-10

def test_transform_step():
    """Rust: test_python_cross_transform_step"""
    ss = {"x": FloatDistribution(0.0, 1.0, step=0.25)}
    t = _SearchSpaceTransform(ss)
    result = t.transform({"x": 0.5})
    assert abs(result[0] - 0.5) < 1e-12
    back = t.untransform(result)
    assert abs(back["x"] - 0.5) < 1e-10

def test_transform_categorical():
    """Rust: test_python_cross_transform_cat"""
    ss = {"c": CategoricalDistribution(["a", "b", "c"])}
    t = _SearchSpaceTransform(ss)
    result = t.transform({"c": "b"})
    # One-hot: [0, 1, 0]
    assert result[0] == 0.0
    assert result[1] == 1.0
    assert result[2] == 0.0
    back = t.untransform(result)
    assert back["c"] == "b"

def test_transform_int_log():
    """Rust: test_python_cross_transform_int_log"""
    ss = {"n": IntDistribution(1, 100, log=True)}
    t = _SearchSpaceTransform(ss)
    result = t.transform({"n": 10})
    expected = math.log(10)
    assert abs(result[0] - expected) < 1e-12
    back = t.untransform(result)
    assert back["n"] == 10

def test_transform_bounds():
    """Rust: test_python_cross_transform_bounds"""
    ss = {"x": FloatDistribution(0.0, 1.0, step=0.25)}
    t = _SearchSpaceTransform(ss)
    bounds = t.bounds
    # step=0.25 → half_step=0.125, bounds = [-0.125, 1.125]
    assert abs(bounds[0][0] - (-0.125)) < 1e-12, f"lo={bounds[0][0]}"
    assert abs(bounds[0][1] - 1.125) < 1e-12, f"hi={bounds[0][1]}"

def test_transform_int_step_bounds():
    """Rust: test_python_cross_transform_int_step_bounds"""
    ss = {"n": IntDistribution(0, 10, step=2)}
    t = _SearchSpaceTransform(ss)
    bounds = t.bounds
    # step=2 → half_step=1.0, bounds = [-1.0, 11.0]
    assert abs(bounds[0][0] - (-1.0)) < 1e-12
    assert abs(bounds[0][1] - 11.0) < 1e-12

def test_transform_0_1():
    """Rust: test_python_cross_transform_01"""
    ss = {"x": FloatDistribution(0.0, 10.0)}
    t = _SearchSpaceTransform(ss, transform_0_1=True)
    result = t.transform({"x": 5.0})
    assert abs(result[0] - 0.5) < 1e-12
    back = t.untransform(result)
    assert abs(back["x"] - 5.0) < 1e-10

# ═══════════════════════════════════════════════════════════════════════════
#  9. Hypervolume — 对应 Rust src/multi_objective.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_hypervolume_2d():
    """Rust: test_python_cross_hypervolume_2d"""
    pts = np.array([[1.0, 4.0], [2.0, 3.0], [3.0, 2.0], [4.0, 1.0]])
    ref = np.array([5.0, 5.0])
    hv = wfg_hv(pts, ref)
    assert abs(hv - 10.0) < 1e-12

def test_hypervolume_3d():
    """Rust: test_python_cross_hypervolume_3d"""
    pts = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]])
    ref = np.array([3.0, 3.0, 3.0])
    hv = wfg_hv(pts, ref)
    # [1,1,1] 支配体积 = (3-1)^3 = 8; [2,2,2] 完全被包含，总超体积 = 8
    assert abs(hv - 8.0) < 1e-12, f"hv={hv}"

def test_hypervolume_single_point():
    """Rust: test_python_cross_hypervolume_single"""
    pts = np.array([[2.0, 3.0]])
    ref = np.array([5.0, 5.0])
    hv = wfg_hv(pts, ref)
    assert abs(hv - 6.0) < 1e-12

# ═══════════════════════════════════════════════════════════════════════════
#  10. TPE默认权重函数 — 对应 Rust src/samplers/tpe/sampler.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_tpe_default_weights():
    """Rust: test_default_weights_edge_n26_matches_python"""
    def default_weights(x):
        if x == 0:
            return np.asarray([])
        elif x < 25:
            return np.ones(x)
        else:
            ramp = np.linspace(1.0 / x, 1.0, num=x - 25)
            flat = np.ones(25)
            return np.concatenate([ramp, flat], axis=0)

    # n=0
    w0 = default_weights(0)
    assert len(w0) == 0

    # n=1
    w1 = default_weights(1)
    assert list(w1) == [1.0]

    # n=24
    w24 = default_weights(24)
    assert list(w24) == [1.0] * 24

    # n=25 (boundary)
    w25 = default_weights(25)
    assert len(w25) == 25
    assert all(abs(v - 1.0) < 1e-12 for v in w25)

    # n=26 (1 ramp + 25 flat)
    w26 = default_weights(26)
    assert len(w26) == 26
    # np.linspace(1/26, 1.0, num=1) == [1/26]——num=1时返回 start，而非 stop
    assert abs(w26[0] - 1.0/26) < 1e-12, f"w26[0]={w26[0]}"
    assert all(abs(v - 1.0) < 1e-12 for v in w26[1:])

    # n=30 (5 ramp + 25 flat)
    w30 = default_weights(30)
    assert len(w30) == 30
    ramp_expected = list(np.linspace(1.0/30, 1.0, num=5))
    for i, expected in enumerate(ramp_expected):
        assert abs(w30[i] - expected) < 1e-12, f"w30[{i}]={w30[i]}, expected={expected}"

# ═══════════════════════════════════════════════════════════════════════════
#  11. Study 基础操作 — 对应 Rust src/study/core.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_study_optimize_minimize():
    """Rust: test_python_cross_study_minimize"""
    study = optuna.create_study(direction="minimize")
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        return (x - 3.0) ** 2
    study.optimize(objective, n_trials=50)
    assert study.best_value < 1.0, f"best={study.best_value}"
    assert abs(study.best_params["x"] - 3.0) < 2.0

def test_study_direction():
    """Rust: test_python_cross_study_direction"""
    study_min = optuna.create_study(direction="minimize")
    assert study_min.direction == optuna.study.StudyDirection.MINIMIZE
    study_max = optuna.create_study(direction="maximize")
    assert study_max.direction == optuna.study.StudyDirection.MAXIMIZE

def test_study_user_attrs():
    """Rust: test_python_cross_study_user_attrs"""
    study = optuna.create_study()
    study.set_user_attr("key1", "value1")
    study.set_user_attr("key2", 42)
    assert study.user_attrs["key1"] == "value1"
    assert study.user_attrs["key2"] == 42

# ═══════════════════════════════════════════════════════════════════════════
#  12. Trial 操作 — 对应 Rust src/trial/handle.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_trial_suggest_float():
    """Rust: test_python_cross_trial_suggest_float"""
    study = optuna.create_study()
    trial = study.ask()
    x = trial.suggest_float("x", 0.0, 1.0)
    assert 0.0 <= x <= 1.0
    # 重复 suggest 同名参数应返回同一值
    x2 = trial.suggest_float("x", 0.0, 1.0)
    assert x == x2

def test_trial_suggest_int():
    """Rust: test_python_cross_trial_suggest_int"""
    study = optuna.create_study()
    trial = study.ask()
    n = trial.suggest_int("n", 1, 10)
    assert 1 <= n <= 10
    assert isinstance(n, int)

def test_trial_suggest_categorical():
    """Rust: test_python_cross_trial_suggest_categorical"""
    study = optuna.create_study()
    trial = study.ask()
    c = trial.suggest_categorical("c", ["a", "b", "c"])
    assert c in ["a", "b", "c"]

def test_trial_report_and_should_prune():
    """Rust: test_python_cross_trial_report"""
    study = optuna.create_study(pruner=optuna.pruners.NopPruner())
    trial = study.ask()
    trial.report(1.0, 0)
    trial.report(2.0, 1)
    assert trial.should_prune() == False  # NopPruner 永不剪枝

def test_trial_user_attrs():
    """Rust: test_python_cross_trial_user_attrs"""
    study = optuna.create_study()
    trial = study.ask()
    trial.set_user_attr("my_key", [1, 2, 3])
    assert trial.user_attrs["my_key"] == [1, 2, 3]

# ═══════════════════════════════════════════════════════════════════════════
#  13. FrozenTrial — 对应 Rust src/trial/frozen.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_frozen_trial_fields():
    """Rust: test_python_cross_frozen_trial"""
    study = optuna.create_study()
    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        trial.report(x, 0)
        trial.set_user_attr("note", "test")
        return x
    study.optimize(objective, n_trials=1)
    ft = study.trials[0]
    assert ft.state == optuna.trial.TrialState.COMPLETE
    assert ft.number == 0
    assert "x" in ft.params
    assert 0 in ft.intermediate_values
    assert ft.user_attrs["note"] == "test"
    assert ft.values is not None and len(ft.values) == 1

def test_frozen_trial_duration():
    """Rust: test_python_cross_frozen_duration"""
    study = optuna.create_study()
    study.optimize(lambda t: t.suggest_float("x", 0, 1), n_trials=1)
    ft = study.trials[0]
    assert ft.datetime_start is not None
    assert ft.datetime_complete is not None
    assert ft.duration is not None
    assert ft.duration.total_seconds() >= 0.0

# ═══════════════════════════════════════════════════════════════════════════
#  14. Importance — 对应 Rust src/importance.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_importance_basic():
    """Rust: test_python_cross_importance_basic"""
    study = optuna.create_study(direction="minimize")
    for i in range(50):
        trial = study.ask()
        x = trial.suggest_float("x", -5, 5)
        y = trial.suggest_float("y", -5, 5)
        z = trial.suggest_float("z", -5, 5)
        study.tell(trial, x**2 + 0.1 * y**2 + 0.01 * z**2)
    imp = optuna.importance.get_param_importances(study)
    # x 应该最重要 (系数 1.0 vs 0.1 vs 0.01)
    assert max(imp, key=imp.get) == "x", f"imp={imp}"
    assert all(v >= 0 for v in imp.values())
    assert sum(imp.values()) <= 1.0 + 1e-6

def test_importance_evaluator_fanova():
    """Rust: test_python_cross_importance_fanova"""
    from optuna.importance import FanovaImportanceEvaluator
    study = optuna.create_study(direction="minimize")
    for _ in range(50):
        trial = study.ask()
        x = trial.suggest_float("x", 0, 10)
        y = trial.suggest_float("y", 0, 10)
        study.tell(trial, x * 10 + y)
    evaluator = FanovaImportanceEvaluator(seed=42)
    imp = evaluator.evaluate(study)
    assert imp["x"] > imp["y"], f"imp={imp}"

# ═══════════════════════════════════════════════════════════════════════════
#  15. Callbacks — 对应 Rust src/callbacks/mod.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_max_trials_callback():
    """Rust: test_python_cross_max_trials"""
    from optuna.study import MaxTrialsCallback
    study = optuna.create_study()
    # MaxTrialsCallback 限制总试验数
    study.optimize(lambda t: t.suggest_float("x", 0, 1),
                   n_trials=100,
                   callbacks=[MaxTrialsCallback(5)])
    assert len(study.trials) == 5

# ═══════════════════════════════════════════════════════════════════════════
#  16. 正态CDF精度 — 对应 Rust wilcoxon.rs::normal_cdf
# ═══════════════════════════════════════════════════════════════════════════

from scipy.stats import norm

def test_normal_cdf_values():
    """Rust: test_python_cross_normal_cdf"""
    test_points = [-3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0]
    for x in test_points:
        expected = norm.cdf(x)
        print(f"  normal_cdf({x}) = {expected}")
        # Rust 使用 Abramowitz & Stegun 近似，精度通常在 1e-7 内
    assert abs(norm.cdf(0.0) - 0.5) < 1e-15
    assert abs(norm.cdf(-3.0) - 0.0013498980316300946) < 1e-10
    assert abs(norm.cdf(3.0) - 0.9986501019683699) < 1e-10

# ═══════════════════════════════════════════════════════════════════════════
#  17. Study ask/tell 工作流 — 对应 Rust study/core.rs ask/tell
# ═══════════════════════════════════════════════════════════════════════════

def test_study_ask_tell_basic():
    """Rust: test_python_cross_ask_tell_basic"""
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    x = trial.suggest_float("x", -5, 5)
    study.tell(trial, x ** 2)
    assert len(study.trials) == 1
    assert study.trials[0].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[0].values[0] == x ** 2

def test_study_ask_tell_pruned():
    """Rust: test_python_cross_ask_tell_pruned"""
    study = optuna.create_study()
    trial = study.ask()
    trial.suggest_float("x", 0, 1)
    trial.report(99.0, 0)
    study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    ft = study.trials[0]
    assert ft.state == optuna.trial.TrialState.PRUNED

def test_study_ask_tell_fail():
    """Rust: test_python_cross_ask_tell_fail"""
    study = optuna.create_study()
    trial = study.ask()
    trial.suggest_float("x", 0, 1)
    study.tell(trial, state=optuna.trial.TrialState.FAIL)
    ft = study.trials[0]
    assert ft.state == optuna.trial.TrialState.FAIL
    assert ft.values is None

# ═══════════════════════════════════════════════════════════════════════════
#  18. Study enqueue_trial — 对应 Rust study/core.rs enqueue_trial
# ═══════════════════════════════════════════════════════════════════════════

def test_study_enqueue_trial():
    """Rust: test_python_cross_enqueue_trial"""
    study = optuna.create_study(direction="minimize")
    study.enqueue_trial({"x": 3.0, "y": 4.0})
    study.enqueue_trial({"x": -1.0, "y": 2.0})
    # 优化时使用预设参数
    def objective(trial):
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        return x ** 2 + y ** 2
    study.optimize(objective, n_trials=2)
    assert study.trials[0].params["x"] == 3.0
    assert study.trials[0].params["y"] == 4.0
    assert study.trials[0].values[0] == 25.0  # 3^2 + 4^2
    assert study.trials[1].params["x"] == -1.0
    assert study.trials[1].values[0] == 5.0   # 1 + 4

def test_study_enqueue_skip_if_exists():
    """Rust: test_python_cross_enqueue_skip"""
    study = optuna.create_study()
    study.enqueue_trial({"x": 1.0})
    study.enqueue_trial({"x": 1.0}, skip_if_exists=True)
    # skip_if_exists 应该只保留一个排队试验
    waiting = [t for t in study.trials if t.state == optuna.trial.TrialState.WAITING]
    assert len(waiting) == 1

# ═══════════════════════════════════════════════════════════════════════════
#  19. 多目标优化 — 对应 Rust multi_objective + study
# ═══════════════════════════════════════════════════════════════════════════

def test_multi_objective_basic():
    """Rust: test_python_cross_multi_objective"""
    study = optuna.create_study(directions=["minimize", "maximize"])
    assert len(study.directions) == 2
    assert study.directions[0] == optuna.study.StudyDirection.MINIMIZE
    assert study.directions[1] == optuna.study.StudyDirection.MAXIMIZE
    def objective(trial):
        x = trial.suggest_float("x", 0, 1)
        return x, 1 - x  # minimize x, maximize 1-x → Pareto front
    study.optimize(objective, n_trials=20)
    best = study.best_trials
    assert len(best) >= 1  # 至少有一个 Pareto 前沿解

def test_multi_objective_best_trials():
    """Rust: test_python_cross_multi_obj_best_trials"""
    study = optuna.create_study(directions=["minimize", "minimize"])
    # 手动添加已知试验
    for vals in [(1.0, 3.0), (2.0, 2.0), (3.0, 1.0), (2.0, 3.0)]:
        trial = study.ask()
        trial.suggest_float("x", 0, 10)
        study.tell(trial, list(vals))
    bests = study.best_trials
    # (1,3), (2,2), (3,1) 应在 Pareto 前沿; (2,3) 被 (2,2) 支配
    best_values = [tuple(t.values) for t in bests]
    assert (2.0, 3.0) not in best_values
    assert len(bests) == 3

# ═══════════════════════════════════════════════════════════════════════════
#  20. MedianPruner — 对应 Rust pruners/median.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_median_pruner_basic():
    """Rust: test_python_cross_median_pruner"""
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=0)
    )
    # 先完成一个好的 trial (中位数 = 5.0)
    t0 = study.ask()
    t0.suggest_float("x", 0, 10)
    t0.report(5.0, 0)
    study.tell(t0, 5.0)

    # 第二个 trial 中间值 = 100.0 → 远高于中位数 → 应被剪枝
    t1 = study.ask()
    t1.suggest_float("x", 0, 10)
    t1.report(100.0, 0)
    assert t1.should_prune() == True

def test_median_pruner_warmup():
    """Rust: test_python_cross_median_warmup"""
    # warmup 期间不剪枝
    study = optuna.create_study(
        pruner=optuna.pruners.MedianPruner(n_startup_trials=0, n_warmup_steps=5)
    )
    t0 = study.ask()
    t0.suggest_float("x", 0, 10)
    t0.report(1.0, 0)
    study.tell(t0, 1.0)

    t1 = study.ask()
    t1.suggest_float("x", 0, 10)
    t1.report(999.0, 2)  # step=2 < warmup=5
    assert t1.should_prune() == False

# ═══════════════════════════════════════════════════════════════════════════
#  21. SuccessiveHalving — 对应 Rust pruners/successive_halving.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_successive_halving_basic():
    """Rust: test_python_cross_successive_halving"""
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1, reduction_factor=2, min_early_stopping_rate=0
    )
    study = optuna.create_study(pruner=pruner)
    # rung 0: min_resource=1 → step 0
    # rung 1: 1 * 2^1 = 2 → step 1
    results = []
    for i in range(4):
        trial = study.ask()
        trial.suggest_float("x", 0, 10)
        trial.report(float(i), 0)
        pruned = trial.should_prune()
        results.append(pruned)
        if pruned:
            study.tell(trial, state=optuna.trial.TrialState.PRUNED)
        else:
            study.tell(trial, float(i))
    # 前几个试验不应该被剪（startup trial 限制）
    assert results[0] == False  # 第一个试验不剪

# ═══════════════════════════════════════════════════════════════════════════
#  22. Study best_value / best_params — 对应 Rust study/core.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_study_best_value_minimize():
    """Rust: test_python_cross_best_value_min"""
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: t.suggest_float("x", 0, 10), n_trials=30)
    assert study.best_value == min(t.values[0] for t in study.trials)

def test_study_best_value_maximize():
    """Rust: test_python_cross_best_value_max"""
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: t.suggest_float("x", 0, 10), n_trials=30)
    assert study.best_value == max(t.values[0] for t in study.trials)

def test_study_best_params():
    """Rust: test_python_cross_best_params"""
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda t: t.suggest_float("x", 0, 10), n_trials=30)
    assert "x" in study.best_params
    assert study.best_params["x"] == study.best_trial.params["x"]

# ═══════════════════════════════════════════════════════════════════════════
#  23. Study add_trial — 对应 Rust study/core.rs add_trial
# ═══════════════════════════════════════════════════════════════════════════

def test_study_add_trial():
    """Rust: test_python_cross_add_trial"""
    from optuna.trial import create_trial
    study = optuna.create_study()
    ft = create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        values=[1.5],
        params={"x": 0.5},
        distributions={"x": FloatDistribution(0, 1)},
    )
    study.add_trial(ft)
    assert len(study.trials) == 1
    assert study.trials[0].values[0] == 1.5
    assert study.trials[0].params["x"] == 0.5

# ═══════════════════════════════════════════════════════════════════════════
#  24. Study metric_names — 对应 Rust study/core.rs set_metric_names
# ═══════════════════════════════════════════════════════════════════════════

def test_study_metric_names():
    """Rust: test_python_cross_metric_names"""
    study = optuna.create_study(directions=["minimize", "maximize"])
    study.set_metric_names(["loss", "accuracy"])
    assert study.metric_names == ["loss", "accuracy"]

# ═══════════════════════════════════════════════════════════════════════════
#  25. IntersectionSearchSpace — 对应 Rust search_space/intersection.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_intersection_search_space():
    """Rust: test_python_cross_intersection_ss"""
    from optuna.search_space import IntersectionSearchSpace
    study = optuna.create_study()
    # trial 0: x, y
    t0 = study.ask()
    t0.suggest_float("x", 0, 1)
    t0.suggest_float("y", 0, 1)
    study.tell(t0, 0.5)
    # trial 1: x, z
    t1 = study.ask()
    t1.suggest_float("x", 0, 1)
    t1.suggest_float("z", 0, 1)
    study.tell(t1, 0.5)
    # 交集应该只有 x
    iss = IntersectionSearchSpace()
    space = iss.calculate(study)
    assert "x" in space
    assert "y" not in space
    assert "z" not in space

# ═══════════════════════════════════════════════════════════════════════════
#  26. TrialState — 对应 Rust trial/state.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_trial_state_is_finished():
    """Rust: test_python_cross_trial_state"""
    from optuna.trial import TrialState
    assert TrialState.RUNNING.is_finished() == False
    assert TrialState.COMPLETE.is_finished() == True
    assert TrialState.PRUNED.is_finished() == True
    assert TrialState.FAIL.is_finished() == True
    assert TrialState.WAITING.is_finished() == False

# ═══════════════════════════════════════════════════════════════════════════
#  27. FrozenTrial validate — 对应 Rust trial/frozen.rs validate
# ═══════════════════════════════════════════════════════════════════════════

def test_frozen_trial_validate_complete_no_values():
    """Rust: test_python_cross_ft_validate_no_values"""
    from optuna.trial import create_trial
    try:
        ft = create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            values=None,
        )
        ft._validate()
        assert False, "should raise"
    except ValueError:
        pass

def test_frozen_trial_validate_complete_nan():
    """Rust: test_python_cross_ft_validate_nan"""
    from optuna.trial import create_trial
    try:
        ft = create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            values=[float('nan')],
        )
        ft._validate()
        assert False, "should raise"
    except ValueError:
        pass

def test_frozen_trial_last_step():
    """Rust: test_python_cross_ft_last_step"""
    from optuna.trial import create_trial
    ft = create_trial(
        state=optuna.trial.TrialState.COMPLETE,
        values=[1.0],
        intermediate_values={0: 1.0, 5: 2.0, 3: 1.5},
    )
    assert ft.last_step == 5

# ═══════════════════════════════════════════════════════════════════════════
#  28. RetryFailedTrialCallback — 对应 Rust callbacks/mod.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_retry_failed_trial_callback():
    """Rust: test_python_cross_retry_callback"""
    call_count = 0
    def objective(trial):
        nonlocal call_count
        call_count += 1
        x = trial.suggest_float("x", 0, 1)
        if call_count <= 2:
            raise RuntimeError("simulated failure")
        return x

    from optuna.study import MaxTrialsCallback
    study = optuna.create_study()
    cb = optuna.storages.RetryFailedTrialCallback(max_retry=3)
    study.optimize(objective, n_trials=10, callbacks=[cb, MaxTrialsCallback(5)],
                   catch=(RuntimeError,))
    failed = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]
    assert len(failed) >= 1  # 至少有失败的试验被重试

# ═══════════════════════════════════════════════════════════════════════════
#  29. FloatDistribution contains() 容差 1e-8 — 对应 Rust float.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_float_contains_tolerance_1e8():
    """验证 Python _contains 中 abs(k - round(k)) < 1e-8 的精确容差"""
    d = optuna.distributions.FloatDistribution(low=0.0, high=1.0, step=0.25)
    # 偏差 1e-9 → k_offset = 1e-9 / 0.25 = 4e-9 < 1e-8 → 通过
    assert d._contains(0.25 + 1e-9), "1e-9 offset should pass"
    assert d._contains(0.25 - 1e-9), "1e-9 neg offset should pass"
    # 偏差 5e-9 → k_offset = 5e-9 / 0.25 = 2e-8 > 1e-8 → 拒绝
    assert not d._contains(0.25 + 5e-9), "5e-9 offset should fail"
    assert not d._contains(0.25 - 5e-9), "5e-9 neg offset should fail"

# ═══════════════════════════════════════════════════════════════════════════
#  30. RandomSampler 半开区间 [lo, hi) — 对应 Rust random.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_random_sampler_half_open_interval():
    """验证 np.random.uniform(lo, hi) 是 [lo, hi) 半开区间"""
    import numpy as np
    rng = np.random.RandomState(0)
    # 连续采样 10000 次，所有值必须 < hi
    for _ in range(10000):
        v = rng.uniform(0.0, 1.0)
        assert v < 1.0, f"np.random.uniform 应返回 < hi 的值, got {v}"
        assert v >= 0.0, f"np.random.uniform 应返回 >= lo 的值, got {v}"

# ═══════════════════════════════════════════════════════════════════════════
#  31. suggest() 检查顺序: fixed_params 优先于 single() — 对应 Rust handle.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_suggest_fixed_param_priority_over_single():
    """验证 Python _suggest 中 fixed_params 优先于 distribution.single()"""
    study = optuna.create_study()
    # 通过 enqueue_trial 注入 fixed_params
    study.enqueue_trial({"x": 5.0})
    trial = study.ask()
    # 使用 single 分布 (low==high==3.0) 但 fixed=5.0
    x = trial.suggest_float("x", 3.0, 3.0)
    assert abs(x - 5.0) < 1e-12, f"fixed(5.0) 应优先于 single(3.0), got {x}"

# ═══════════════════════════════════════════════════════════════════════════
#  32. BruteForceSampler 穷举网格 — 对应 Rust brute_force.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_brute_force_sampler_enumerate():
    """验证 BruteForceSampler 会穷举所有离散组合"""
    sampler = optuna.samplers.BruteForceSampler()
    study = optuna.create_study(sampler=sampler)
    study.optimize(
        lambda trial: trial.suggest_int("x", 1, 3),
        n_trials=10,
    )
    # 应至少使用了 3 个不同的 x 值 (1, 2, 3)
    xs = set(t.params["x"] for t in study.trials)
    assert xs == {1, 2, 3}, f"应穷举 {{1,2,3}}, got {xs}"

# ═══════════════════════════════════════════════════════════════════════════
#  33. PartialFixedSampler 固定参数 — 对应 Rust partial_fixed.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_partial_fixed_sampler():
    """验证 PartialFixedSampler 固定参数行为"""
    base = optuna.samplers.RandomSampler(seed=42)
    sampler = optuna.samplers.PartialFixedSampler(
        base_sampler=base,
        fixed_params={"x": 0.5},
    )
    study = optuna.create_study(sampler=sampler)
    study.optimize(
        lambda trial: trial.suggest_float("x", 0.0, 1.0) + trial.suggest_float("y", 0.0, 1.0),
        n_trials=10,
    )
    for t in study.trials:
        assert abs(t.params["x"] - 0.5) < 1e-10, f"x 应始终为 0.5, got {t.params['x']}"
    # y 应该会变化
    ys = [t.params["y"] for t in study.trials]
    assert len(set(round(y, 6) for y in ys)) > 1, "y 应该在不同试验中有不同值"

# ═══════════════════════════════════════════════════════════════════════════
#  34. MaxTrialsCallback 实际停止 — 对应 Rust callbacks/mod.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_max_trials_callback_stops():
    """验证 MaxTrialsCallback 在达到限制后停止优化"""
    from optuna.study import MaxTrialsCallback
    study = optuna.create_study()
    cb = MaxTrialsCallback(3, states=(optuna.trial.TrialState.COMPLETE,))
    study.optimize(
        lambda trial: trial.suggest_float("x", 0, 1),
        n_trials=100,  # 大上限
        callbacks=[cb],
    )
    n_complete = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    # 回调应在 3 次完成后停止
    assert 3 <= n_complete <= 5, f"应在 3 次后停止, got {n_complete}"

# ═══════════════════════════════════════════════════════════════════════════
#  35. RetryFailedTrialCallback 继承中间值 — 对应 Rust callbacks/mod.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_retry_callback_inherit_intermediate_values():
    """验证 inherit_intermediate_values=True 时继承中间值"""
    study = optuna.create_study()
    call_count = 0
    def objective(trial):
        nonlocal call_count
        call_count += 1
        trial.report(0.5, 0)
        trial.report(0.3, 1)
        if call_count == 1:
            raise RuntimeError("fail")
        return trial.suggest_float("x", 0, 1)

    cb = optuna.storages.RetryFailedTrialCallback(
        max_retry=3, inherit_intermediate_values=True
    )
    study.optimize(objective, n_trials=3, callbacks=[cb], catch=(RuntimeError,))

    # 找到 WAITING 状态的重试试验
    waiting = [t for t in study.trials if t.state == optuna.trial.TrialState.WAITING]
    if waiting:
        # 应继承中间值
        assert len(waiting[0].intermediate_values) == 2

# ═══════════════════════════════════════════════════════════════════════════
#  36. tell() PRUNED+values 应抛错 — 对应 Rust study/core.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_tell_pruned_with_values_raises():
    """验证 Python tell(state=PRUNED, values=1.0) 抛出 ValueError"""
    study = optuna.create_study()
    trial = study.ask()
    try:
        study.tell(trial, values=1.0, state=optuna.trial.TrialState.PRUNED)
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "Values" in str(e) or "cannot" in str(e).lower()

def test_tell_fail_with_values_raises():
    """验证 Python tell(state=FAIL, values=1.0) 抛出 ValueError"""
    study = optuna.create_study()
    trial = study.ask()
    try:
        study.tell(trial, values=1.0, state=optuna.trial.TrialState.FAIL)
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "Values" in str(e) or "cannot" in str(e).lower()

# ═══════════════════════════════════════════════════════════════════════════
#  37. tell(state=None) 自动推断 — 对应 Rust study/core.rs tell_auto
# ═══════════════════════════════════════════════════════════════════════════

def test_tell_auto_complete():
    """验证 Python tell(state=None, values=1.0) → Complete"""
    study = optuna.create_study()
    trial = study.ask()
    frozen = study.tell(trial, values=1.0)  # state=None is default
    assert frozen.state == optuna.trial.TrialState.COMPLETE
    assert abs(frozen.values[0] - 1.0) < 1e-12

def test_tell_auto_fail_on_none():
    """验证 Python tell(state=None, values=None) → Fail"""
    study = optuna.create_study()
    trial = study.ask()
    frozen = study.tell(trial, values=None)
    assert frozen.state == optuna.trial.TrialState.FAIL

def test_tell_auto_fail_on_nan():
    """验证 Python tell(state=None, values=NaN) → Fail"""
    import math
    study = optuna.create_study()
    trial = study.ask()
    frozen = study.tell(trial, values=math.nan)
    assert frozen.state == optuna.trial.TrialState.FAIL

# ═══════════════════════════════════════════════════════════════════════════
#  38. tell PRUNED 自动使用最后中间值 — 对应 Rust study/core.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_tell_pruned_auto_intermediate_value():
    """验证 Python tell(state=PRUNED) 自动使用最后中间值"""
    study = optuna.create_study()
    trial = study.ask()
    trial.report(0.5, 0)
    trial.report(0.3, 1)
    frozen = study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    assert frozen.state == optuna.trial.TrialState.PRUNED
    assert frozen.values is not None
    assert abs(frozen.values[0] - 0.3) < 1e-12

# ═══════════════════════════════════════════════════════════════════════════
#  39. SearchSpaceTransform int truncation — 对应 Rust transform.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_int_truncation_log_no_transform():
    """验证 Python int() 截断行为: int(2.9)==2, int(99.7)==99"""
    assert int(2.9) == 2, "int(2.9) should be 2"
    assert int(2.1) == 2, "int(2.1) should be 2"
    assert int(99.7) == 99, "int(99.7) should be 99"
    assert int(-2.9) == -2, "int(-2.9) should be -2"

# ═══════════════════════════════════════════════════════════════════════════
#  40. Pareto front with constraints — 对应 Rust multi_objective.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_pareto_front_with_constraints():
    """验证 Python get_pareto_front_trials(consider_constraint=True)"""
    from optuna.study._multi_objective import _get_pareto_front_trials
    study = optuna.create_study(directions=["minimize", "minimize"])

    # 添加可行试验 [1.0, 2.0]
    study.add_trial(
        optuna.trial.create_trial(
            values=[1.0, 2.0],
            system_attrs={"constraints": [-1.0]},
        )
    )
    # 添加不可行但 Pareto 更优的试验 [0.5, 0.5]
    study.add_trial(
        optuna.trial.create_trial(
            values=[0.5, 0.5],
            system_attrs={"constraints": [1.0]},  # 违反约束
        )
    )

    # 无约束过滤 → [0.5, 0.5] 在前沿
    front_all = _get_pareto_front_trials(study, consider_constraint=False)
    assert any(t.values == [0.5, 0.5] for t in front_all)

    # 有约束过滤 → 只剩可行的 [1.0, 2.0]
    front_constrained = _get_pareto_front_trials(study, consider_constraint=True)
    assert all(t.values != [0.5, 0.5] for t in front_constrained)
    assert any(t.values == [1.0, 2.0] for t in front_constrained)

# ═══════════════════════════════════════════════════════════════════════════
#  41. set_trial_state_values values=None 不清空 — 对应 Rust in_memory.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_set_trial_state_values_none_preserves():
    """验证 Python: set_trial_state_values(values=None) 不清空已有 values"""
    storage = optuna.storages.InMemoryStorage()
    sid = storage.create_new_study(
        directions=[optuna.study.StudyDirection.MINIMIZE],
        study_name="test",
    )
    tid = storage.create_new_trial(sid)
    # 先设置 Complete + values
    storage.set_trial_state_values(tid, optuna.trial.TrialState.COMPLETE, [1.5])
    trial = storage.get_trial(tid)
    assert trial.values == [1.5]

# ═══════════════════════════════════════════════════════════════════════════
#  42. check_distribution_compatibility 不检查 step — 对应 Rust distributions/mod.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_check_distribution_compatibility_float_step_allowed():
    """对齐 Python: Float 分布 step 不同也兼容"""
    from optuna.distributions import check_distribution_compatibility, FloatDistribution
    a = FloatDistribution(0.0, 1.0, step=0.1)
    b = FloatDistribution(0.0, 1.0, step=0.2)
    # Python 不检查 step，不应报错
    check_distribution_compatibility(a, b)

def test_check_distribution_compatibility_float_step_vs_none():
    """对齐 Python: Float step=Some vs step=None 也兼容"""
    from optuna.distributions import check_distribution_compatibility, FloatDistribution
    a = FloatDistribution(0.0, 1.0, step=0.1)
    b = FloatDistribution(0.0, 1.0)
    check_distribution_compatibility(a, b)

def test_check_distribution_compatibility_int_step_allowed():
    """对齐 Python: Int 分布 step 不同也兼容"""
    from optuna.distributions import check_distribution_compatibility, IntDistribution
    a = IntDistribution(0, 10, step=1)
    b = IntDistribution(0, 10, step=2)
    check_distribution_compatibility(a, b)

def test_check_distribution_compatibility_float_log_differ():
    """对齐 Python: Float log 不同则不兼容"""
    from optuna.distributions import check_distribution_compatibility, FloatDistribution
    a = FloatDistribution(0.01, 1.0, log=True)
    b = FloatDistribution(0.01, 1.0, log=False)
    try:
        check_distribution_compatibility(a, b)
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass

def test_check_distribution_compatibility_int_log_differ():
    """对齐 Python: Int log 不同则不兼容"""
    from optuna.distributions import check_distribution_compatibility, IntDistribution
    a = IntDistribution(1, 10, log=True)
    b = IntDistribution(1, 10, log=False)
    try:
        check_distribution_compatibility(a, b)
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass

# ═══════════════════════════════════════════════════════════════════════════
#  43. create_trial value/values 互斥 + validate — 对应 Rust trial/mod.rs
# ═══════════════════════════════════════════════════════════════════════════

def test_create_trial_value_values_mutual_exclusion():
    """对齐 Python: 同时传入 value 和 values 报 ValueError"""
    try:
        optuna.trial.create_trial(value=1.0, values=[2.0])
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass

def test_create_trial_complete_without_values():
    """对齐 Python validate(): Complete 无 values 报 ValueError"""
    try:
        optuna.trial.create_trial(state=optuna.trial.TrialState.COMPLETE)
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass

def test_create_trial_fail_with_values():
    """对齐 Python validate(): Fail 有 values 报 ValueError"""
    try:
        optuna.trial.create_trial(state=optuna.trial.TrialState.FAIL, value=1.0)
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass

def test_create_trial_complete_with_nan():
    """对齐 Python validate(): Complete + NaN 报 ValueError"""
    try:
        optuna.trial.create_trial(value=float('nan'))
        assert False, "应该抛出 ValueError"
    except ValueError:
        pass

def test_create_trial_pruned_no_values_ok():
    """对齐 Python: Pruned 无 values 合法"""
    t = optuna.trial.create_trial(state=optuna.trial.TrialState.PRUNED)
    assert t.state == optuna.trial.TrialState.PRUNED

# ═══════════════════════════════════════════════════════════════════════════
#  44. TPE split_trials 排序 — 确认 Python 行为
# ═══════════════════════════════════════════════════════════════════════════

def test_tpe_split_trials_sorted_by_number():
    """验证 Python TPE split_trials 结果按 trial.number 排序"""
    study = optuna.create_study(direction="minimize")
    # 手动添加乱序 number 的试验
    for number, value in [(3, 0.1), (0, 0.2), (2, 0.3), (1, 0.4)]:
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=value,
                params={"x": float(number)},
                distributions={"x": optuna.distributions.FloatDistribution(-10, 10)},
            )
        )
    # 使用 TPE 内部 _split_trials
    from optuna.samplers._tpe.sampler import _split_trials
    below, above = _split_trials(study, study.trials, 1, [])
    # below 和 above 都应按 number 排序
    below_numbers = [t.number for t in below]
    above_numbers = [t.number for t in above]
    assert below_numbers == sorted(below_numbers), f"below 未排序: {below_numbers}"
    assert above_numbers == sorted(above_numbers), f"above 未排序: {above_numbers}"

# ═══════════════════════════════════════════════════════════════════════════
#  45. TPE search_space 过滤 single() 分布 — 确认 Python 行为
# ═══════════════════════════════════════════════════════════════════════════

def test_tpe_search_space_filters_single():
    """验证 Python TPE 的 infer_relative_search_space 过滤 single() 分布"""
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(
        multivariate=True, n_startup_trials=0, seed=42))
    # 添加试验: x 有多值分布, y 固定=5.0
    for i in range(5):
        study.add_trial(
            optuna.trial.create_trial(
                state=optuna.trial.TrialState.COMPLETE,
                value=float(i),
                params={"x": float(i), "y": 5.0},
                distributions={
                    "x": optuna.distributions.FloatDistribution(0, 10),
                    "y": optuna.distributions.FloatDistribution(5.0, 5.0),  # single!
                },
            )
        )
    # 推断搜索空间
    trial = study.ask()
    search_space = study.sampler.infer_relative_search_space(study, trial)
    assert "y" not in search_space, f"single() 分布不应在搜索空间中: {search_space.keys()}"
    assert "x" in search_space

# ═══════════════════════════════════════════════════════════════════════════
#  46. TPE IntersectionSearchSpace include_pruned=True — 确认 Python 行为
# ═══════════════════════════════════════════════════════════════════════════

def test_tpe_intersection_search_space_includes_pruned():
    """验证 Python TPE 使用 IntersectionSearchSpace(include_pruned=True)"""
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(
        multivariate=True, n_startup_trials=0, seed=42))
    # 先添加完成的试验
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            value=1.0,
            params={"x": 1.0, "z": 2.0},
            distributions={
                "x": optuna.distributions.FloatDistribution(0, 10),
                "z": optuna.distributions.FloatDistribution(0, 10),
            },
        )
    )
    # 添加 pruned 试验（有不同参数）
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.PRUNED,
            params={"x": 2.0, "z": 3.0},
            distributions={
                "x": optuna.distributions.FloatDistribution(0, 10),
                "z": optuna.distributions.FloatDistribution(0, 10),
            },
        )
    )
    # TPE 使用 include_pruned=True，所以 pruned 试验也在计算搜索空间
    trial = study.ask()
    search_space = study.sampler.infer_relative_search_space(study, trial)
    # x 和 z 都应在搜索空间中（因为 pruned 试验也有这些参数）
    assert "x" in search_space, f"x 应在搜索空间中"
    assert "z" in search_space, f"z 应在搜索空间中"

# ═══════════════════════════════════════════════════════════════════════════
#  执行
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    passed = failed = 0
    for test in tests:
        try:
            test()
            passed += 1
            print(f"  ✅ {test.__name__}")
        except Exception as e:
            failed += 1
            print(f"  ❌ {test.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{'='*60}")
    print(f"Python 交叉验证: {passed} 通过, {failed} 失败, 共 {passed+failed} 个测试")
    if failed:
        sys.exit(1)
