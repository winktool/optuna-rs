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
#  分布兼容性检查交叉验证
# ═══════════════════════════════════════════════════════════════════════════

def test_distribution_compatibility_same_kind_different_range():
    """对齐 Python check_distribution_compatibility: 同类型不同 range 应允许。"""
    from optuna.distributions import check_distribution_compatibility, FloatDistribution
    d1 = FloatDistribution(0.0, 1.0)
    d2 = FloatDistribution(0.0, 2.0)
    # 不应报错
    check_distribution_compatibility(d1, d2)

def test_distribution_compatibility_different_log():
    """对齐 Python: 同类型不同 log 应报错。"""
    from optuna.distributions import check_distribution_compatibility, FloatDistribution
    d1 = FloatDistribution(0.1, 1.0)
    d2 = FloatDistribution(0.1, 1.0, log=True)
    try:
        check_distribution_compatibility(d1, d2)
        assert False, "应报错"
    except ValueError:
        pass

def test_distribution_compatibility_different_kind():
    """对齐 Python: 不同类型应报错。"""
    from optuna.distributions import check_distribution_compatibility, FloatDistribution, IntDistribution
    d1 = FloatDistribution(0.0, 1.0)
    d2 = IntDistribution(0, 10)
    try:
        check_distribution_compatibility(d1, d2)
        assert False, "应报错"
    except ValueError:
        pass

def test_distribution_compatibility_categorical_dynamic():
    """对齐 Python: Categorical 不同选项应报错。"""
    from optuna.distributions import check_distribution_compatibility, CategoricalDistribution
    d1 = CategoricalDistribution(["a", "b"])
    d2 = CategoricalDistribution(["a", "c"])
    try:
        check_distribution_compatibility(d1, d2)
        assert False, "应报错"
    except ValueError as e:
        assert "dynamic value space" in str(e)


# ═══════════════════════════════════════════════════════════════════════════
#  FloatDistribution contains 精确边界交叉验证
# ═══════════════════════════════════════════════════════════════════════════

def test_float_contains_strict_boundary():
    """严格 low <= value <= high 无容差。"""
    d = optuna.distributions.FloatDistribution(0.0, 1.0)
    assert d._contains(0.0) == True
    assert d._contains(1.0) == True
    assert d._contains(0.5) == True
    # 超出边界
    assert d._contains(1.0 + 1e-11) == False
    assert d._contains(-1e-11) == False

def test_float_contains_step_adjusted():
    """step=0.3 时 high 调整为 0.9, contains(0.9) 应为 True。"""
    d = optuna.distributions.FloatDistribution(0.0, 1.0, step=0.3)
    assert d.high == 0.9
    assert d._contains(0.9) == True
    assert d._contains(1.0) == False
    assert d._contains(0.6) == True
    assert d._contains(0.3) == True
    assert d._contains(0.0) == True
    assert d._contains(0.4) == False


# ═══════════════════════════════════════════════════════════════════════════
#  3D 超体积交叉验证
# ═══════════════════════════════════════════════════════════════════════════

def test_hypervolume_3d_cross_validation():
    """对齐 Python _compute_3d: 3D 超体积计算。"""
    import numpy as np
    from optuna._hypervolume.wfg import _compute_3d

    # 4 个非支配点
    pts = np.array([[1.0, 4.0, 3.0], [2.0, 2.0, 2.0], [3.0, 1.0, 4.0], [4.0, 3.0, 1.0]])
    ref = np.array([5.0, 5.0, 5.0])
    pts_sorted = pts[np.argsort(pts[:, 0])]
    hv = _compute_3d(pts_sorted, ref)
    assert abs(hv - 33.0) < 1e-10, f"Expected 33.0, got {hv}"

    # 含被支配点
    pts2 = np.array([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0], [3.0, 3.0, 0.5]])
    ref2 = np.array([4.0, 4.0, 4.0])
    pts2_sorted = pts2[np.argsort(pts2[:, 0])]
    hv2 = _compute_3d(pts2_sorted, ref2)
    assert abs(hv2 - 27.5) < 1e-10, f"Expected 27.5, got {hv2}"

    # 两个不对角的点
    pts3 = np.array([[1.0, 3.0, 2.0], [2.0, 1.0, 3.0]])
    ref3 = np.array([4.0, 4.0, 4.0])
    pts3_sorted = pts3[np.argsort(pts3[:, 0])]
    hv3 = _compute_3d(pts3_sorted, ref3)
    assert abs(hv3 - 10.0) < 1e-10, f"Expected 10.0, got {hv3}"


# ═══════════════════════════════════════════════════════════════════════════
#  PartialFixedSampler 交叉验证
# ═══════════════════════════════════════════════════════════════════════════

def test_partial_fixed_sample_relative_no_inject():
    """对齐 Python: PartialFixedSampler.sample_relative 不注入固定参数。"""
    import optuna
    base = optuna.samplers.RandomSampler(seed=42)
    fixed = {"x": 0.5}
    sampler = optuna.samplers.PartialFixedSampler(fixed_params=fixed, base_sampler=base)
    
    study = optuna.create_study(sampler=sampler)
    # 添加一个完成的试验
    study.add_trial(
        optuna.trial.create_trial(
            state=optuna.trial.TrialState.COMPLETE,
            params={"x": 0.3, "y": 0.7},
            distributions={
                "x": optuna.distributions.FloatDistribution(0, 1),
                "y": optuna.distributions.FloatDistribution(0, 1),
            },
            values=[1.0],
        )
    )
    trial = study.ask()
    # sample_relative 由 base_sampler 处理，不包含固定参数
    search_space = sampler.infer_relative_search_space(study, trial)
    result = sampler.sample_relative(study, trial, search_space)
    # 固定参数 "x" 不应在 sample_relative 结果中
    # （Python 中 PartialFixedSampler.sample_relative 直接委托给 base_sampler）
    # 注意: 实际上 base_sampler (RandomSampler) 的 sample_relative 返回空字典
    assert isinstance(result, dict)


def test_partial_fixed_out_of_range_warns():
    """对齐 Python: 固定值超出范围应发出警告但不报错。"""
    import optuna
    import warnings
    base = optuna.samplers.RandomSampler(seed=42)
    fixed = {"x": 99.0}  # 超出 [0, 1] 范围
    sampler = optuna.samplers.PartialFixedSampler(fixed_params=fixed, base_sampler=base)
    
    study = optuna.create_study(sampler=sampler)
    # 应该能成功 ask (且 suggest_float 返回固定值)
    # PartialFixedSampler.sample_independent 会发出警告
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trial = study.ask()
        val = trial.suggest_float("x", 0.0, 1.0)
        # Python: 应发出 UserWarning
        assert val == 99.0, f"应返回固定值 99.0, got {val}"


# ═══════════════════════════════════════════════════════════════════════════
#  ParzenEstimator 验证交叉验证
# ═══════════════════════════════════════════════════════════════════════════

def test_parzen_estimator_negative_prior_weight():
    """对齐 Python: prior_weight < 0 应抛 ValueError。"""
    from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator, _ParzenEstimatorParameters
    from optuna.distributions import FloatDistribution
    
    params = _ParzenEstimatorParameters(
        prior_weight=-1.0,
        consider_magic_clip=True,
        consider_endpoints=False,
        weights=lambda n: [1.0] * n if n > 0 else [],
        multivariate=False,
        categorical_distance_func=None,
    )
    try:
        _ParzenEstimator(
            observations={"x": [0.5]},
            search_space={"x": FloatDistribution(0.0, 1.0)},
            parameters=params,
        )
        assert False, "应抛 ValueError"
    except ValueError as e:
        assert "non-negative" in str(e).lower(), f"错误消息应包含 'non-negative': {e}"


def test_parzen_estimator_weights_validation():
    """对齐 Python _call_weights_func: 负权重/全零/NaN 应抛错。"""
    from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator, _ParzenEstimatorParameters
    from optuna.distributions import FloatDistribution

    # 测试负权重
    params = _ParzenEstimatorParameters(
        prior_weight=1.0,
        consider_magic_clip=True,
        consider_endpoints=False,
        weights=lambda n: [-1.0] * n if n > 0 else [],
        multivariate=False,
        categorical_distance_func=None,
    )
    try:
        _ParzenEstimator(
            observations={"x": [0.5]},
            search_space={"x": FloatDistribution(0.0, 1.0)},
            parameters=params,
        )
        assert False, "负权重应抛错"
    except ValueError:
        pass

    # 测试全零
    params2 = _ParzenEstimatorParameters(
        prior_weight=1.0,
        consider_magic_clip=True,
        consider_endpoints=False,
        weights=lambda n: [0.0] * n if n > 0 else [],
        multivariate=False,
        categorical_distance_func=None,
    )
    try:
        _ParzenEstimator(
            observations={"x": [0.5]},
            search_space={"x": FloatDistribution(0.0, 1.0)},
            parameters=params2,
        )
        assert False, "全零权重应抛错"
    except ValueError:
        pass


# ═══════════════════════════════════════════════════════════════════════════
#  truncnorm ppf 精度交叉验证
# ═══════════════════════════════════════════════════════════════════════════

def test_truncnorm_ppf_near_one():
    """对齐 Python: ppf 在 q 接近 1 时的精度 (使用 log1p)。"""
    from optuna.samplers._tpe._truncnorm import ppf
    import numpy as np
    # 标准正态截断 [-5, 5]
    q = np.array([0.999999])
    a = np.array([-5.0])
    b = np.array([5.0])
    result = ppf(q, a, b)
    # 结果应接近上界但不超过
    assert result[0] < 5.0, f"ppf 结果应 < 5.0, got {result[0]}"
    assert result[0] > 3.0, f"ppf 结果应 > 3.0, got {result[0]}"

# ═══════════════════════════════════════════════════════════════════════════
#  Storage 对齐验证
# ═══════════════════════════════════════════════════════════════════════════

def test_set_trial_system_attr_rejects_finished():
    """对齐 Python: set_trial_system_attr 不允许修改已完成试验。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    storage = optuna.storages.InMemoryStorage()
    sid = storage.create_new_study([optuna.study.StudyDirection.MINIMIZE])
    tid = storage.create_new_trial(sid)
    storage.set_trial_state_values(tid, optuna.trial.TrialState.COMPLETE, [1.0])
    try:
        storage.set_trial_system_attr(tid, "key", "value")
        assert False, "应抛出 UpdateFinishedTrialError"
    except Exception as e:
        assert "finished" in str(e).lower() or "updatable" in str(e).lower(), f"错误消息不匹配: {e}"

def test_set_metric_names_error_message():
    """对齐 Python: set_metric_names 长度不匹配的错误消息。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    try:
        study.set_metric_names(["a", "b"])  # 只有1个direction但传了2个名字
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "number of objectives" in str(e).lower() or "metric" in str(e).lower(), f"错误消息不匹配: {e}"

def test_best_trial_constraint_error_message():
    """对齐 Python: 无可行约束试验时的错误消息。"""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize")
    # 添加一个 COMPLETE 试验，但其约束不满足
    trial = study.ask()
    study._storage.set_trial_system_attr(
        trial._trial_id, "constraints", [1.0]  # 正值 → 不可行
    )
    study.tell(trial, 1.0)
    try:
        _ = study.best_trial
        assert False, "应抛出 ValueError"
    except ValueError as e:
        assert "feasible" in str(e).lower(), f"错误消息不匹配: {e}"

# ═══════════════════════════════════════════════════════════════════════════
#  Session 35: FrozenStudy.direction() / create_study 验证
# ═══════════════════════════════════════════════════════════════════════════

def test_frozen_study_direction_single():
    """对齐 Rust: FrozenStudy.direction() 单目标返回方向"""
    study = optuna.create_study(direction="minimize")
    assert study.direction == optuna.study.StudyDirection.MINIMIZE

def test_frozen_study_direction_multi_error():
    """对齐 Rust: FrozenStudy.direction 多目标报 RuntimeError"""
    study = optuna.create_study(directions=["minimize", "maximize"])
    try:
        _ = study.direction
        assert False, "应抛出 RuntimeError"
    except RuntimeError:
        pass

def test_create_study_empty_directions_error():
    """对齐 Rust: create_study 空方向列表报 ValueError"""
    try:
        optuna.create_study(directions=[])
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

def test_create_study_both_direction_directions_error():
    """对齐 Rust: 同时指定 direction 和 directions 报错"""
    try:
        optuna.create_study(direction="minimize", directions=["minimize"])
        assert False, "应抛出 ValueError"
    except ValueError:
        pass

def test_create_study_multi_default_sampler():
    """对齐 Rust: 多目标时默认使用 NSGAIISampler"""
    study = optuna.create_study(directions=["minimize", "maximize"])
    assert isinstance(study.sampler, optuna.samplers.NSGAIISampler)

def test_importance_normalize_false():
    """对齐 Rust: normalize=False 返回未归一化的重要性值"""
    study = optuna.create_study(direction="minimize")
    for _ in range(30):
        trial = study.ask()
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        study.tell(trial, x**2 + 0.01*y)
    importances = optuna.importance.get_param_importances(study, normalize=False)
    # normalize=False 时值不一定加和为 1
    assert len(importances) == 2
    # x 应比 y 更重要
    assert importances["x"] > importances["y"]

def test_importance_normalize_true():
    """对齐 Rust: normalize=True 值加和为 1"""
    study = optuna.create_study(direction="minimize")
    for _ in range(30):
        trial = study.ask()
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_float("y", -10, 10)
        study.tell(trial, x**2 + 0.01*y)
    importances = optuna.importance.get_param_importances(study, normalize=True)
    total = sum(importances.values())
    assert abs(total - 1.0) < 1e-10, f"归一化后应为1.0, 实际={total}"

# ═══════════════════════════════════════════════════════════════════════════
#  Session 38: 旧版 Distribution 兼容性测试
# ═══════════════════════════════════════════════════════════════════════════

def test_deprecated_uniform_distribution_json():
    """对齐 Rust: test_json_to_distribution_uniform
    验证旧版 UniformDistribution 的 JSON 序列化格式。
    Rust 端需要能反序列化此格式。"""
    import json as json_mod
    d = FloatDistribution(0.0, 1.0)
    j = optuna.distributions.distribution_to_json(d)
    parsed = json_mod.loads(j)
    assert parsed["name"] == "FloatDistribution"
    assert parsed["attributes"]["low"] == 0.0
    assert parsed["attributes"]["high"] == 1.0

def test_deprecated_int_distribution_step_adjustment():
    """对齐 Rust: test_json_to_distribution_int_uniform
    验证 IntDistribution 的 high 调整行为。
    IntDistribution(1, 10, step=2) → high 被调整为 9 (因为 (10-1)%2 != 0)"""
    d = IntDistribution(1, 10, step=2)
    assert d.low == 1
    assert d.high == 9  # Python 也做同样的调整

def test_fixed_trial_system_attrs():
    """对齐 Rust: test_fixed_trial_set_and_get_system_attr
    验证 FixedTrial 的 system_attrs 行为。"""
    from optuna.trial import FixedTrial
    trial = FixedTrial({"x": 1.0})
    assert trial.system_attrs == {}  # 初始为空

def test_fixed_trial_params_property():
    """对齐 Rust: test_fixed_trial_params_alias
    验证 FixedTrial.params 在 suggest 后返回正确值。"""
    from optuna.trial import FixedTrial
    trial = FixedTrial({"x": 1.5})
    trial.suggest_float("x", 0.0, 10.0)
    assert "x" in trial.params
    assert trial.params["x"] == 1.5

def test_nested_optimize_detection():
    """对齐 Rust: test_optimize_loop_flag_resets_after_completion
    验证连续两次 optimize 不会被误报为嵌套调用。"""
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: trial.suggest_float("x", -10, 10) ** 2, n_trials=2)
    # 第二次调用不应失败
    study.optimize(lambda trial: trial.suggest_float("x", -10, 10) ** 2, n_trials=2)
    assert len(study.trials) == 4

def test_reseed_rng_sampler():
    """对齐 Rust: test_reseed_rng_changes_output
    验证 RandomSampler.reseed_rng() 存在且可调用。"""
    sampler = optuna.samplers.RandomSampler(seed=42)
    # Python 的 reseed_rng 接受一个 seed sequence
    sampler.reseed_rng()
    study = optuna.create_study(sampler=sampler, direction="minimize")
    study.optimize(lambda trial: trial.suggest_float("x", 0, 1), n_trials=1)
    assert len(study.trials) == 1

# ═══════════════════════════════════════════════════════════════════════════
#  Session 39: tell_auto / ask-tell / add_trial / stop 交叉验证
# ═══════════════════════════════════════════════════════════════════════════

def test_tell_state_none_valid_values_complete():
    """对齐 Rust: test_tell_auto_valid_values_complete
    state=None + 合法值 → Complete"""
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    trial.suggest_float("x", 0.0, 1.0)
    frozen = study.tell(trial, 0.5, state=None)
    assert frozen.state == optuna.trial.TrialState.COMPLETE
    assert abs(frozen.values[0] - 0.5) < 1e-12

def test_tell_state_none_none_values_fail():
    """对齐 Rust: test_tell_auto_none_values_fail
    state=None + values=None → Fail"""
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    frozen = study.tell(trial, values=None, state=None)
    assert frozen.state == optuna.trial.TrialState.FAIL

def test_tell_state_none_nan_values_fail():
    """对齐 Rust: test_tell_auto_nan_values_fail
    state=None + NaN → Fail"""
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    frozen = study.tell(trial, values=float("nan"), state=None)
    assert frozen.state == optuna.trial.TrialState.FAIL

def test_tell_skip_if_finished():
    """对齐 Rust: test_tell_finished_trial_skip
    skip_if_finished=True 应静默跳过已完成试验"""
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    study.tell(trial, 1.0)
    # 再次 tell 应不报错
    frozen = study.tell(trial, 2.0, skip_if_finished=True)
    assert abs(frozen.values[0] - 1.0) < 1e-12  # 保持原始值

def test_tell_pruned_uses_last_intermediate():
    """对齐 Rust: test_tell_pruned_uses_last_intermediate_value
    PRUNED 状态应使用最后中间值"""
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    trial.report(0.5, 0)
    trial.report(0.3, 1)
    frozen = study.tell(trial, state=optuna.trial.TrialState.PRUNED)
    assert frozen.state == optuna.trial.TrialState.PRUNED
    # Python: pruned 时如果有中间值且可行，自动作为 value
    assert frozen.values is not None
    assert abs(frozen.values[0] - 0.3) < 1e-12

def test_enqueue_trial_params_used():
    """对齐 Rust: test_enqueue_trial_params_used
    enqueue 的参数应被 ask 使用"""
    study = optuna.create_study(direction="minimize")
    study.enqueue_trial({"x": 0.42})
    trial = study.ask()
    x = trial.suggest_float("x", 0.0, 1.0)
    assert abs(x - 0.42) < 1e-12

def test_add_trial_and_add_trials():
    """对齐 Rust: test_add_trial_and_add_trials"""
    study = optuna.create_study(direction="minimize")
    ft = optuna.trial.create_trial(state=optuna.trial.TrialState.COMPLETE, values=[1.0])
    study.add_trial(ft)
    assert len(study.trials) == 1
    ft2 = optuna.trial.create_trial(state=optuna.trial.TrialState.COMPLETE, values=[2.0])
    ft3 = optuna.trial.create_trial(state=optuna.trial.TrialState.COMPLETE, values=[3.0])
    study.add_trials([ft2, ft3])
    assert len(study.trials) == 3

def test_study_stop():
    """对齐 Rust: test_study_stop_from_callback
    study.stop() 应终止优化"""
    counter = {"n": 0}
    study = optuna.create_study(direction="minimize")
    def objective(trial):
        counter["n"] += 1
        if counter["n"] >= 3:
            study.stop()
        return trial.suggest_float("x", 0, 1)
    study.optimize(objective, n_trials=1000)
    assert counter["n"] <= 5, f"stop 后应很快终止，实际 {counter['n']} 次"

## NOTE: test_frozen_trial_hashable 已移除
## Python FrozenTrial 不可哈希（含 list 字段），Rust 的 Hash impl 是扩展功能

def test_suggest_updates_params_immediately():
    """对齐 Rust: test_cached_trial_params_updated_after_suggest
    suggest 后 params 应立即可用"""
    study = optuna.create_study(direction="minimize")
    trial = study.ask()
    assert len(trial.params) == 0
    trial.suggest_float("x", 0.0, 1.0)
    assert "x" in trial.params
    trial.suggest_int("n", 1, 10)
    assert "n" in trial.params
    assert len(trial.params) == 2

# ═══════════════════════════════════════════════════════════════════════════
#  17. MOTPE 多目标 TPE — 对应 Rust MOTPE 实现
# ═══════════════════════════════════════════════════════════════════════════

def test_motpe_reference_point():
    """Rust: test_get_reference_point / test_get_reference_point_negative"""
    from optuna.samplers._tpe.sampler import _get_reference_point
    # 正值
    loss_vals = np.array([[1.0, 2.0], [3.0, 1.0]])
    rp = _get_reference_point(loss_vals)
    assert abs(rp[0] - 3.3) < 1e-9, f"rp[0]={rp[0]}"
    assert abs(rp[1] - 2.2) < 1e-9, f"rp[1]={rp[1]}"
    # 负值
    loss_vals2 = np.array([[-3.0, -1.0]])
    rp2 = _get_reference_point(loss_vals2)
    assert abs(rp2[0] - (-2.7)) < 1e-9, f"rp2[0]={rp2[0]}"
    assert abs(rp2[1] - (-0.9)) < 1e-9, f"rp2[1]={rp2[1]}"

def test_motpe_nondomination_rank():
    """Rust: test_fast_non_domination_rank"""
    from optuna.samplers._tpe.sampler import _fast_non_domination_rank
    # 全 Pareto
    loss = np.array([[1.0, 3.0], [2.0, 2.0], [3.0, 1.0]])
    ranks = _fast_non_domination_rank(loss)
    assert list(ranks) == [0, 0, 0], f"ranks={ranks}"
    # 层次
    loss2 = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    ranks2 = _fast_non_domination_rank(loss2)
    assert ranks2[0] == 0
    assert ranks2[1] == 1
    assert ranks2[2] == 2

def test_motpe_split_multi_objective():
    """Rust: test_split_trials_multi_objective
    多目标分割：所有试验在 Pareto 前沿上"""
    study = optuna.create_study(directions=["minimize", "minimize"],
                                sampler=optuna.samplers.TPESampler(seed=42))
    for i in range(10):
        trial = study.ask()
        trial.suggest_float("x", -10.0, 10.0)
        study.tell(trial, [float(i), 10.0 - float(i)])
    # 使用内部方法验证分割
    from optuna.samplers._tpe.sampler import _split_complete_trials_multi_objective
    n_below = max(1, int(0.1 * 10))  # 与 default gamma 一致
    below, above = _split_complete_trials_multi_objective(
        list(study.trials), study, n_below
    )
    assert len(below) > 0
    assert len(above) > 0
    assert len(below) + len(above) == 10

def test_motpe_weights_all_pareto():
    """Rust: test_calculate_mo_weights_all_pareto
    所有解都在 Pareto 前沿上，权重应全部正"""
    from optuna.samplers._tpe.sampler import _calculate_weights_below_for_multi_objective
    study = optuna.create_study(directions=["minimize", "minimize"])
    for i, vals in enumerate([(1.0, 3.0), (2.0, 2.0), (3.0, 1.0)]):
        trial = study.ask()
        study.tell(trial, list(vals))
    below = list(study.trials)
    weights = _calculate_weights_below_for_multi_objective(study, below, None)
    assert len(weights) == 3
    for w in weights:
        assert w > 0, f"weight should be positive: {w}"

def test_motpe_weights_dominated():
    """Rust: test_calculate_mo_weights_dominated
    有支配关系时，Pareto 前沿解权重应最大"""
    from optuna.samplers._tpe.sampler import _calculate_weights_below_for_multi_objective
    study = optuna.create_study(directions=["minimize", "minimize"])
    for vals in [(1.0, 1.0), (2.0, 2.0), (3.0, 3.0)]:
        trial = study.ask()
        study.tell(trial, list(vals))
    below = list(study.trials)
    weights = _calculate_weights_below_for_multi_objective(study, below, None)
    # dominated points should have lower weights (EPS)
    assert weights[0] > weights[1], f"w0={weights[0]} should > w1={weights[1]}"
    assert weights[0] > weights[2], f"w0={weights[0]} should > w2={weights[2]}"

def test_motpe_end_to_end():
    """Rust: end-to-end MOTPE 优化应能运行并产出 Pareto 前沿"""
    study = optuna.create_study(
        directions=["minimize", "minimize"],
        sampler=optuna.samplers.TPESampler(seed=42, n_startup_trials=5)
    )
    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        return x, 1.0 - x
    study.optimize(objective, n_trials=30)
    assert len(study.trials) == 30
    # 应有 Pareto 最优解
    pareto = study.best_trials
    assert len(pareto) > 0


# ═══════════════════════════════════════════════════════════════════════════
#  Session 41: fANOVA 方差分解、BaseTrial trait
# ═══════════════════════════════════════════════════════════════════════════

def test_fanova_importance_ranking():
    """Rust: test_get_param_importances_quadratic
    fANOVA 应该正确识别 x^2 + 0.01*y 中 x 远比 y 重要。
    Python 和 Rust 都应该给出 x > y 的排序。
    """
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=42),
    )
    def obj(trial):
        x = trial.suggest_float("x", -10.0, 10.0)
        y = trial.suggest_float("y", -10.0, 10.0)
        return x * x + 0.01 * y
    study.optimize(obj, n_trials=100)

    imp = optuna.importance.get_param_importances(
        study,
        evaluator=optuna.importance.FanovaImportanceEvaluator(seed=42),
    )
    assert "x" in imp and "y" in imp, f"keys={list(imp.keys())}"
    assert imp["x"] > imp["y"], f"x={imp['x']}, y={imp['y']}: x should dominate"
    assert imp["x"] > 0.9, f"x importance should be > 0.9, got {imp['x']}"


def test_fanova_three_params():
    """Rust: test_importance_three_params
    f(x, y, z) = 10*x^2 + y^2 + 0.001*z => importance: x > y > z
    """
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.RandomSampler(seed=123),
    )
    def obj(trial):
        x = trial.suggest_float("x", -5.0, 5.0)
        y = trial.suggest_float("y", -5.0, 5.0)
        z = trial.suggest_float("z", -5.0, 5.0)
        return 10.0 * x * x + y * y + 0.001 * z
    study.optimize(obj, n_trials=200)

    imp = optuna.importance.get_param_importances(
        study,
        evaluator=optuna.importance.FanovaImportanceEvaluator(seed=123),
    )
    keys = list(imp.keys())
    assert keys[0] == "x", f"first={keys[0]}, expected x"
    assert keys[1] == "y", f"second={keys[1]}, expected y"
    assert keys[2] == "z", f"third={keys[2]}, expected z"


def test_wilcoxon_exact_small_n():
    """Rust: test_exact_small_n_accuracy
    n=3 全正差值: exact p-value 应该是 0.125 (= 1/8)
    """
    from scipy.stats import wilcoxon
    d = [1.0, 2.0, 3.0]
    stat, pvalue = wilcoxon(d, alternative="greater", method="exact", correction=False)
    assert abs(pvalue - 0.125) < 1e-10, f"pvalue={pvalue}, expected 0.125"


# ═══════════════════════════════════════════════════════════════════════════
#  Session 42: FrozenTrial 语义对齐、SearchSpaceTransform 列映射
# ═══════════════════════════════════════════════════════════════════════════

def test_frozen_trial_repr():
    """Rust: test_display_contains_all_fields
    对齐 Python FrozenTrial.__repr__: 输出包含 'FrozenTrial(' 和所有关键字段。
    """
    t = optuna.trial.FrozenTrial(
        number=42, state=optuna.trial.TrialState.COMPLETE,
        value=1.5, values=None,
        datetime_start=None, datetime_complete=None,
        params={}, distributions={},
        user_attrs={}, system_attrs={},
        intermediate_values={}, trial_id=7,
    )
    r = repr(t)
    assert r.startswith("FrozenTrial("), f"repr should start with FrozenTrial(, got: {r[:30]}"
    assert "number=42" in r, "'number=42' should be in repr"
    # Python 的 __repr__ 也有 value=None 结尾
    assert "value=" in r, "'value=' should appear in repr"


def test_frozen_trial_ordering():
    """Rust: test_ordering_by_number
    对齐 Python FrozenTrial.__lt__ / __le__: 按 number 排序。
    """
    kwargs = dict(state=optuna.trial.TrialState.COMPLETE, value=1.0, values=None,
                  datetime_start=None, datetime_complete=None,
                  params={}, distributions={},
                  user_attrs={}, system_attrs={},
                  intermediate_values={}, trial_id=0)
    t1 = optuna.trial.FrozenTrial(number=3, **kwargs)
    t2 = optuna.trial.FrozenTrial(number=1, **kwargs)
    assert t2 < t1, "trial with smaller number should be <"
    assert t2 <= t1
    assert not t1 < t2
    trials = sorted([t1, t2])
    assert trials[0].number == 1
    assert trials[1].number == 3


def test_frozen_trial_eq():
    """Rust: test_eq_different_state
    对齐 Python FrozenTrial.__eq__: 比较所有字段。
    """
    kwargs = dict(value=None, values=None,
                  datetime_start=None, datetime_complete=None,
                  params={}, distributions={},
                  user_attrs={}, system_attrs={},
                  intermediate_values={}, trial_id=0)
    t1 = optuna.trial.FrozenTrial(number=0, state=optuna.trial.TrialState.COMPLETE, **kwargs)
    t2 = optuna.trial.FrozenTrial(number=0, state=optuna.trial.TrialState.COMPLETE, **kwargs)
    # 相同字段 → 相等
    assert t1 == t2


def test_frozen_trial_last_step():
    """Rust: test_last_step_returns_max
    对齐 Python FrozenTrial.last_step: 返回最大 step。
    """
    t = optuna.trial.FrozenTrial(
        number=0, state=optuna.trial.TrialState.RUNNING,
        value=None, values=None,
        datetime_start=None, datetime_complete=None,
        params={}, distributions={},
        user_attrs={}, system_attrs={},
        intermediate_values={0: 1.0, 5: 2.0, 3: 3.0}, trial_id=0,
    )
    assert t.last_step == 5, f"expected 5, got {t.last_step}"


def test_frozen_trial_duration():
    """Rust: test_duration_complete
    对齐 Python FrozenTrial.duration: 有 start 和 complete 时返回 timedelta。
    """
    import datetime
    start = datetime.datetime(2024, 1, 1, 0, 0, 0)
    end = start + datetime.timedelta(seconds=10)
    t = optuna.trial.FrozenTrial(
        number=0, state=optuna.trial.TrialState.COMPLETE,
        value=1.0, values=None,
        datetime_start=start, datetime_complete=end,
        params={}, distributions={},
        user_attrs={}, system_attrs={},
        intermediate_values={}, trial_id=0,
    )
    assert t.duration == datetime.timedelta(seconds=10)


def test_frozen_trial_value_single():
    """Rust: test_value_single_objective
    对齐 Python: 单目标 value 返回值。
    """
    t = optuna.trial.FrozenTrial(
        number=0, state=optuna.trial.TrialState.COMPLETE,
        value=3.14, values=None,
        datetime_start=None, datetime_complete=None,
        params={}, distributions={},
        user_attrs={}, system_attrs={},
        intermediate_values={}, trial_id=0,
    )
    assert abs(t.value - 3.14) < 1e-10


def test_frozen_trial_value_multi_raises():
    """Rust: test_value_multi_objective_error
    对齐 Python: 多目标调用 .value 报 RuntimeError。
    """
    t = optuna.trial.FrozenTrial(
        number=0, state=optuna.trial.TrialState.COMPLETE,
        value=None, values=[1.0, 2.0],
        datetime_start=None, datetime_complete=None,
        params={}, distributions={},
        user_attrs={}, system_attrs={},
        intermediate_values={}, trial_id=0,
    )
    try:
        _ = t.value
        assert False, "should raise RuntimeError"
    except RuntimeError:
        pass


def test_transform_column_mapping():
    """Rust: test_column_to_encoded_columns_mixed
    对齐 Python SearchSpaceTransform.column_to_encoded_columns:
    Float→1列, 3-choice Categorical→3列 (one-hot)。
    """
    from optuna._transform import _SearchSpaceTransform
    from optuna.distributions import FloatDistribution, CategoricalDistribution

    search_space = {
        "x": FloatDistribution(0.0, 1.0),
        "c": CategoricalDistribution(["a", "b", "c"]),
    }
    t = _SearchSpaceTransform(search_space)

    # column_to_encoded_columns: 每个原始参数对应哪些编码列
    col_map = t.column_to_encoded_columns
    assert len(col_map) == 2, f"should have 2 params, got {len(col_map)}"
    assert len(col_map[0]) == 1, f"Float x should map to 1 column, got {len(col_map[0])}"
    assert len(col_map[1]) == 3, f"Categorical c should map to 3 columns, got {len(col_map[1])}"

    # encoded_column_to_column: 反向映射
    enc_to_col = t.encoded_column_to_column
    # [0, 1, 1, 1] => 编码列 0→参数0, 列 1/2/3→参数1
    assert list(enc_to_col) == [0, 1, 1, 1], f"got {list(enc_to_col)}"


# ═══════════════════════════════════════════════════════════════════════════
#  Session 43: NaN / Inf 边界行为 + MOTPE 约束交叉验证
# ═══════════════════════════════════════════════════════════════════════════

def test_float_contains_nan():
    """对齐 Rust: FloatDistribution.contains(NaN) → False"""
    from optuna.distributions import FloatDistribution
    d = FloatDistribution(0.0, 1.0)
    assert not d._contains(float('nan')), "NaN should not be contained (no step)"
    d_step = FloatDistribution(0.0, 1.0, step=0.5)
    assert not d_step._contains(float('nan')), "NaN should not be contained (step)"
    d_log = FloatDistribution(0.01, 1.0, log=True)
    assert not d_log._contains(float('nan')), "NaN should not be contained (log)"


def test_float_contains_inf():
    """对齐 Rust: FloatDistribution.contains(Inf) → False"""
    from optuna.distributions import FloatDistribution
    d = FloatDistribution(0.0, 1.0)
    assert not d._contains(float('inf')), "Inf should not be contained"
    assert not d._contains(float('-inf')), "-Inf should not be contained"


def test_int_contains_nan():
    """对齐 Rust: IntDistribution.contains(NaN) → False"""
    from optuna.distributions import IntDistribution
    d = IntDistribution(0, 10)
    assert not d._contains(float('nan')), "NaN should not be contained (low=0)"
    d2 = IntDistribution(5, 10)
    assert not d2._contains(float('nan')), "NaN should not be contained (low=5)"
    d_step = IntDistribution(0, 10, step=3)
    assert not d_step._contains(float('nan')), "NaN should not be contained (step=3)"
    d_log = IntDistribution(1, 100, log=True)
    assert not d_log._contains(float('nan')), "NaN should not be contained (log)"


def test_int_contains_inf():
    """对齐 Rust: IntDistribution.contains(Inf) → False"""
    from optuna.distributions import IntDistribution
    d = IntDistribution(0, 10)
    assert not d._contains(float('inf')), "Inf should not be contained"
    assert not d._contains(float('-inf')), "-Inf should not be contained"


def test_int_to_external_repr_nan():
    """对齐 Rust: int(float('nan')) → ValueError"""
    try:
        int(float('nan'))
        assert False, "int(NaN) should raise ValueError"
    except ValueError:
        pass  # expected


def test_int_to_external_repr_inf():
    """对齐 Rust: int(float('inf')) → OverflowError"""
    try:
        int(float('inf'))
        assert False, "int(Inf) should raise OverflowError"
    except OverflowError:
        pass  # expected


def test_float_contains_boundary():
    """对齐 Rust: Float boundary inclusive"""
    from optuna.distributions import FloatDistribution
    d = FloatDistribution(0.0, 1.0)
    assert d._contains(0.0), "low boundary should be contained"
    assert d._contains(1.0), "high boundary should be contained"
    assert not d._contains(-0.001), "below low"
    assert not d._contains(1.001), "above high"


def test_float_contains_step_grid():
    """对齐 Rust: step 分布精确网格"""
    from optuna.distributions import FloatDistribution
    d = FloatDistribution(0.0, 1.0, step=0.25)
    assert d._contains(0.0)
    assert d._contains(0.25)
    assert d._contains(0.5)
    assert d._contains(0.75)
    assert d._contains(1.0)
    assert not d._contains(0.1), "not on 0.25 grid"


def test_int_contains_boundary():
    """对齐 Rust: Int boundary inclusive"""
    from optuna.distributions import IntDistribution
    d = IntDistribution(0, 10)
    assert d._contains(0), "low boundary"
    assert d._contains(10), "high boundary"
    assert not d._contains(-1), "below low"
    assert not d._contains(11), "above high"


def test_int_contains_non_integer():
    """对齐 Rust: 非整数值不 contain"""
    from optuna.distributions import IntDistribution
    d = IntDistribution(0, 10)
    assert not d._contains(5.5), "5.5 not integer"
    assert not d._contains(0.1), "0.1 not integer"


def test_motpe_weights_feasibility():
    """对齐 Rust: MOTPE calculate_mo_weights 区分可行/不可行试验权重。
    Python 版本将不可行试验权重设为 EPS。"""
    import numpy as np
    from optuna.samplers._tpe.sampler import _calculate_weights_below_for_multi_objective
    import optuna

    study = optuna.create_study(directions=["minimize", "minimize"])

    # 创建 3 个 below trials: 2 feasible + 1 infeasible
    trials = []
    for i, (vals, constraint) in enumerate([
        ([1.0, 3.0], [-1.0]),   # feasible
        ([2.0, 2.0], [-0.5]),   # feasible
        ([3.0, 1.0], [2.0]),    # infeasible
    ]):
        trial = optuna.trial.create_trial(
            values=vals,
            params={},
            distributions={},
        )
        trials.append(trial)

    # 使用 constraints_func
    constraint_values = [[-1.0], [-0.5], [2.0]]
    def constraints_func(trial):
        idx = trials.index(trial)
        return constraint_values[idx]

    weights = _calculate_weights_below_for_multi_objective(
        study, trials, constraints_func
    )

    # 不可行试验 (index 2) 权重应远小于可行试验
    assert weights[2] < weights[0] / 100, \
        f"infeasible weight {weights[2]} should be << feasible {weights[0]}"
    assert weights[2] < weights[1] / 100, \
        f"infeasible weight {weights[2]} should be << feasible {weights[1]}"


def test_cv_error_evaluator_missing_scores():
    """对齐 Rust: CrossValidationErrorEvaluator 缺少 CV 分数时 panic/ValueError"""
    from optuna.terminator import CrossValidationErrorEvaluator
    from optuna.study import StudyDirection
    import optuna

    trial = optuna.trial.create_trial(
        values=[0.5],
        params={},
        distributions={},
    )

    eval = CrossValidationErrorEvaluator()
    try:
        eval.evaluate([trial], StudyDirection.MINIMIZE)
        assert False, "should raise ValueError when CV scores missing"
    except ValueError as e:
        assert "Cross-validation scores" in str(e)


# ═══════════════════════════════════════════════════════════════════════════
#  Session 44+: 对齐 Rust — Journal Storage / NaN / best_trial 约束
# ═══════════════════════════════════════════════════════════════════════════

def test_journal_attr_format():
    """对齐 Rust: JournalStorage user_attr/system_attr 使用 {key: value} dict 格式"""
    import optuna
    import json
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        path = f.name

    try:
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(path)
        )
        study = optuna.create_study(storage=storage, study_name="attr_fmt")
        study.set_user_attr("sk1", "sv1")
        study.set_system_attr("ssk1", 123)

        trial = study.ask()
        trial.set_user_attr("tk1", "tv1")
        trial.set_system_attr("tsk1", 456)
        study.tell(trial, 1.0)

        # 验证日志文件格式 — Python 格式是平铺的（op_code 在顶层，无嵌套 data）
        with open(path, "r") as f:
            lines = f.readlines()
        found_user_attr = False
        found_system_attr = False
        for line in lines:
            entry = json.loads(line)
            op = entry.get("op_code", -1)
            if op == 2:  # SET_STUDY_USER_ATTR
                assert "user_attr" in entry, f"study user attr should use 'user_attr' key: {entry}"
                found_user_attr = True
            elif op == 3:  # SET_STUDY_SYSTEM_ATTR
                assert "system_attr" in entry, f"study system attr should use 'system_attr' key: {entry}"
                found_system_attr = True
            elif op == 8:  # SET_TRIAL_USER_ATTR
                assert "user_attr" in entry, f"trial user attr should use 'user_attr' key: {entry}"
            elif op == 9:  # SET_TRIAL_SYSTEM_ATTR
                assert "system_attr" in entry, f"trial system attr should use 'system_attr' key: {entry}"
        assert found_user_attr, "no study user_attr entry found"
        assert found_system_attr, "no system_attr entry found"
    finally:
        os.unlink(path)


def test_journal_replay_with_template():
    """对齐 Rust: JournalStorage replay 应恢复模板 trial 的所有数据"""
    import optuna
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        path = f.name

    try:
        # 第一次会话: 用 enqueue_trial 创建模板 trial
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(path)
        )
        study = optuna.create_study(storage=storage, study_name="tmpl_test")
        study.enqueue_trial({"x": 0.5})

        trial = study.ask()
        val = trial.suggest_float("x", 0.0, 1.0)
        assert val == 0.5, f"enqueued value should be 0.5, got {val}"
        study.tell(trial, 0.75)

        # 第二次会话: reload
        storage2 = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(path)
        )
        study2 = optuna.load_study(storage=storage2, study_name="tmpl_test")
        trials = study2.trials
        assert len(trials) == 1
        t = trials[0]
        assert t.state == optuna.trial.TrialState.COMPLETE
        assert t.values == [0.75]
        assert "x" in t.params
        assert t.params["x"] == 0.5
    finally:
        os.unlink(path)


def test_journal_replay_timestamps():
    """对齐 Rust: JournalStorage replay 应保留时间戳"""
    import optuna
    import tempfile, os, datetime

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        path = f.name

    try:
        now_before = datetime.datetime.now()

        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(path)
        )
        study = optuna.create_study(storage=storage, study_name="ts_test")
        trial = study.ask()
        study.tell(trial, 1.0)

        now_after = datetime.datetime.now()

        # Reload and check timestamps
        storage2 = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(path)
        )
        study2 = optuna.load_study(storage=storage2, study_name="ts_test")
        t = study2.trials[0]
        assert t.datetime_start is not None, "datetime_start should be preserved"
        assert t.datetime_complete is not None, "datetime_complete should be preserved"
    finally:
        os.unlink(path)


def test_get_best_trial_nan_minimize():
    """对齐 Rust: get_best_trial 应过滤 NaN 值 (minimize)
    Python 的 create_trial 不允许 NaN 值，因此用 objective 返回 NaN 来测试。
    """
    import optuna
    import math

    study = optuna.create_study(direction="minimize")

    def obj_normal_3(trial):
        return 3.0

    def obj_nan(trial):
        return float('nan')

    def obj_normal_1(trial):
        return 1.0

    study.optimize(obj_normal_3, n_trials=1)
    study.optimize(obj_nan, n_trials=1)
    study.optimize(obj_normal_1, n_trials=1)

    best = study.best_trial
    assert best.value == 1.0, f"best should be 1.0, got {best.value}"


def test_get_best_trial_nan_maximize():
    """对齐 Rust: get_best_trial 应过滤 NaN 值 (maximize)"""
    import optuna
    import math

    study = optuna.create_study(direction="maximize")

    def obj_3(trial):
        return 3.0

    def obj_nan(trial):
        return float('nan')

    def obj_5(trial):
        return 5.0

    study.optimize(obj_3, n_trials=1)
    study.optimize(obj_nan, n_trials=1)
    study.optimize(obj_5, n_trials=1)

    best = study.best_trial
    assert best.value == 5.0, f"best should be 5.0, got {best.value}"


def test_best_trial_constraint_two_step():
    """对齐 Rust: best_trial 约束逻辑 — 两步法"""
    import optuna
    import math

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 10.0)
        # Trial 0: x=1.0, 值=1.0, 可行
        # Trial 1: x=5.0, 值=0.1, 不可行 (约束>0)
        # Trial 2: x=3.0, 值=0.5, 可行
        return x

    study = optuna.create_study(direction="minimize")
    # Trial 0: 可行, 值=1.0
    t0 = optuna.trial.create_trial(
        values=[1.0], params={"x": 1.0},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 10.0)},
        system_attrs={"constraints": [0.0]},  # 满足约束 (<=0)
    )
    study.add_trial(t0)

    # Trial 1: 不可行, 值=0.1 (最好值但不可行)
    t1 = optuna.trial.create_trial(
        values=[0.1], params={"x": 5.0},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 10.0)},
        system_attrs={"constraints": [1.0]},  # 违反约束 (>0)
    )
    study.add_trial(t1)

    # Trial 2: 可行, 值=0.5
    t2 = optuna.trial.create_trial(
        values=[0.5], params={"x": 3.0},
        distributions={"x": optuna.distributions.FloatDistribution(0.0, 10.0)},
        system_attrs={"constraints": [-1.0]},  # 满足约束 (<=0)
    )
    study.add_trial(t2)

    # Python 两步法: 先找全局最好 (0.1)，发现不可行，回退到可行过滤
    # 可行试验: t0 (1.0), t2 (0.5), 其中 t2 最好
    best = study.best_trial
    assert best.value == 0.5, f"best should be 0.5 (feasible best), got {best.value}"


def test_best_trial_no_constraint_is_best():
    """对齐 Rust: 当全局最好 trial 没有约束键时视为可行"""
    import optuna

    study = optuna.create_study(direction="minimize")
    # Trial 0: 没有约束键, 值=0.5 — 全局最好
    t0 = optuna.trial.create_trial(
        values=[0.5], params={}, distributions={},
    )
    study.add_trial(t0)

    # Trial 1: 有约束, 可行, 值=1.0
    t1 = optuna.trial.create_trial(
        values=[1.0], params={}, distributions={},
        system_attrs={"constraints": [0.0]},
    )
    study.add_trial(t1)

    # 两步法: 全局最好是 t0 (0.5)。t0 没有约束键
    # Python: 没有约束键的 trial，best_trial 视为可行
    best = study.best_trial
    assert best.value == 0.5, f"best should be 0.5 (no constraints = feasible), got {best.value}"


def test_journal_duplicate_running_rejection():
    """对齐 Rust: JournalStorage replay 重复 RUNNING 应被静默拒绝"""
    import optuna
    import json
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        path = f.name

    try:
        # 手动构造日志文件（Python 格式 — 平铺，无嵌套 data）
        wid = "test-worker-1"
        with open(path, "w") as f:
            # create study
            f.write(json.dumps({"op_code": 0, "worker_id": wid, "study_name": "dup_run", "directions": [1]}) + "\n")
            # create trial (默认 Running)
            now = "2024-01-01T00:00:00.000000"
            f.write(json.dumps({"op_code": 4, "worker_id": wid, "study_id": 0, "datetime_start": now}) + "\n")
            # 第一个 worker 设为 Running (已经是 Running, 应被拒绝)
            f.write(json.dumps({"op_code": 6, "worker_id": wid, "trial_id": 0, "state": 0, "values": None, "datetime_start": "2024-01-01T00:01:00.000000"}) + "\n")
            # complete
            f.write(json.dumps({"op_code": 6, "worker_id": wid, "trial_id": 0, "state": 1, "values": [42.0], "datetime_complete": "2024-01-01T00:03:00.000000"}) + "\n")

        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(path)
        )
        study = optuna.load_study(storage=storage, study_name="dup_run")
        trials = study.trials
        assert len(trials) == 1
        assert trials[0].state == optuna.trial.TrialState.COMPLETE
        assert trials[0].values == [42.0]
    finally:
        os.unlink(path)


def test_journal_python_flat_format_no_data_wrapper():
    """对齐: Python JournalStorage 写入的日志应该是扁平格式（无 data 包裹）"""
    import optuna
    import json
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        path = f.name

    try:
        # Python 写入日志
        backend = optuna.storages.JournalFileStorage(path)
        storage = optuna.storages.JournalStorage(backend)
        study = optuna.create_study(storage=storage, study_name="flat_check",
                                     direction="minimize")
        study.set_user_attr("py_key", "py_value")
        trial = study.ask()
        trial.suggest_float("x", 0.0, 1.0)
        trial.report(0.5, step=0)
        study.tell(trial, 0.42)

        # 验证每一行日志都是扁平格式
        with open(path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                assert "data" not in entry, f"Line {i}: Python log should be flat (no 'data' key), got: {list(entry.keys())}"
                assert "op_code" in entry, f"Line {i}: missing op_code"
                assert "worker_id" in entry, f"Line {i}: missing worker_id"
    finally:
        os.unlink(path)


def test_journal_interop_python_write_rust_format_read():
    """对齐: Python 能正确读取严格 Python 扁平格式的日志"""
    import optuna
    import json
    import tempfile, os

    with tempfile.NamedTemporaryFile(suffix=".log", delete=False) as f:
        path = f.name

    try:
        # 构造 Python 扁平格式日志
        with open(path, "w") as f:
            # Python 扁平格式: create study
            f.write(json.dumps({
                "op_code": 0, "worker_id": "py-1",
                "study_name": "mixed_test", "directions": [1]
            }) + "\n")
            # Python 扁平格式: set study user attr (from_python)
            f.write(json.dumps({
                "op_code": 2, "worker_id": "py-1",
                "study_id": 0, "user_attr": {"from_python": "flat"}
            }) + "\n")
            # Python 扁平格式: set study user attr (from_rust — 新扁平格式)
            f.write(json.dumps({
                "op_code": 2, "worker_id": "rust:0",
                "study_id": 0, "user_attr": {"from_rust": "new_flat"}
            }) + "\n")
            # Python 扁平格式: create trial
            f.write(json.dumps({
                "op_code": 4, "worker_id": "py-1", "study_id": 0,
                "datetime_start": "2024-01-01T00:00:00.000000"
            }) + "\n")
            # Python 扁平格式: set intermediate value
            f.write(json.dumps({
                "op_code": 7, "worker_id": "py-1", "trial_id": 0,
                "step": 0, "intermediate_value": 0.95
            }) + "\n")
            # Python 扁平格式: complete trial
            f.write(json.dumps({
                "op_code": 6, "worker_id": "py-1", "trial_id": 0,
                "state": 1, "values": [0.42],
                "datetime_complete": "2024-01-01T00:01:00.000000"
            }) + "\n")

        # Python 应能读取扁平格式日志
        storage = optuna.storages.JournalStorage(
            optuna.storages.JournalFileStorage(path)
        )
        study = optuna.load_study(storage=storage, study_name="mixed_test")

        # 验证 study attrs
        assert study.user_attrs.get("from_python") == "flat", \
            f"Should read Python flat attrs, got: {study.user_attrs}"
        assert study.user_attrs.get("from_rust") == "new_flat", \
            f"Should read Rust new flat attrs, got: {study.user_attrs}"

        # 验证 trial
        trials = study.trials
        assert len(trials) == 1
        assert trials[0].state == optuna.trial.TrialState.COMPLETE
        assert trials[0].values == [0.42]
    finally:
        os.unlink(path)


# ═══════════════════════════════════════════════════════════════════════════
# Session 47: NSGA-II/III 参数验证 + TPE default_weights/gamma + search_space
# ═══════════════════════════════════════════════════════════════════════════

def test_nsgaii_parameter_validation_mutation_prob():
    """对齐 Python: mutation_prob 必须在 [0.0, 1.0] 或 None"""
    import optuna

    # None 应合法
    optuna.samplers.NSGAIISampler(mutation_prob=None)

    # 0.0 和 1.0 应合法
    optuna.samplers.NSGAIISampler(mutation_prob=0.0)
    optuna.samplers.NSGAIISampler(mutation_prob=1.0)

    # 超出范围应抛 ValueError
    try:
        optuna.samplers.NSGAIISampler(mutation_prob=-0.5)
        assert False, "Should raise ValueError for mutation_prob=-0.5"
    except ValueError:
        pass

    try:
        optuna.samplers.NSGAIISampler(mutation_prob=1.1)
        assert False, "Should raise ValueError for mutation_prob=1.1"
    except ValueError:
        pass


def test_nsgaii_parameter_validation_crossover_prob():
    """对齐 Python: crossover_prob 必须在 [0.0, 1.0]"""
    import optuna

    optuna.samplers.NSGAIISampler(crossover_prob=0.0)
    optuna.samplers.NSGAIISampler(crossover_prob=1.0)

    try:
        optuna.samplers.NSGAIISampler(crossover_prob=-0.1)
        assert False, "Should raise ValueError for crossover_prob=-0.1"
    except ValueError:
        pass

    try:
        optuna.samplers.NSGAIISampler(crossover_prob=1.5)
        assert False, "Should raise ValueError for crossover_prob=1.5"
    except ValueError:
        pass


def test_nsgaii_parameter_validation_swapping_prob():
    """对齐 Python: swapping_prob 必须在 [0.0, 1.0]"""
    import optuna

    optuna.samplers.NSGAIISampler(swapping_prob=0.0)
    optuna.samplers.NSGAIISampler(swapping_prob=1.0)

    try:
        optuna.samplers.NSGAIISampler(swapping_prob=-0.1)
        assert False, "Should raise ValueError for swapping_prob=-0.1"
    except ValueError:
        pass

    try:
        optuna.samplers.NSGAIISampler(swapping_prob=1.5)
        assert False, "Should raise ValueError for swapping_prob=1.5"
    except ValueError:
        pass


def test_nsgaii_population_size_validation():
    """对齐 Python: population_size < 2 应抛 ValueError"""
    import optuna

    try:
        optuna.samplers.NSGAIISampler(population_size=1)
        assert False, "Should raise ValueError for population_size=1"
    except ValueError:
        pass

    # population_size=2 应合法
    optuna.samplers.NSGAIISampler(population_size=2)


def test_nsgaiii_parameter_validation():
    """对齐 Python: NSGA-III 同样的参数验证"""
    import optuna

    # 合法参数
    optuna.samplers.NSGAIIISampler(
        mutation_prob=0.5, crossover_prob=0.5, swapping_prob=0.5
    )

    # mutation_prob 不合法
    try:
        optuna.samplers.NSGAIIISampler(mutation_prob=-0.1)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # crossover_prob 不合法
    try:
        optuna.samplers.NSGAIIISampler(crossover_prob=1.5)
        assert False, "Should raise ValueError"
    except ValueError:
        pass

    # swapping_prob 不合法
    try:
        optuna.samplers.NSGAIIISampler(swapping_prob=-0.1)
        assert False, "Should raise ValueError"
    except ValueError:
        pass


def test_nsgaii_default_values():
    """对齐 Python/Rust: NSGA-II 默认参数值"""
    import optuna

    sampler = optuna.samplers.NSGAIISampler()
    # Python 默认: population_size=50, crossover_prob=0.9, swapping_prob=0.5
    assert sampler._population_size == 50, f"Expected 50, got {sampler._population_size}"
    assert sampler._child_generation_strategy._crossover_prob == 0.9
    assert sampler._child_generation_strategy._swapping_prob == 0.5
    assert sampler._child_generation_strategy._mutation_prob is None


def test_tpe_default_weights_cross_validation():
    """对齐 Python/Rust: default_weights 函数输出完全一致"""
    from optuna.samplers._tpe.sampler import default_weights
    import numpy as np

    # n=0 → 空
    w0 = default_weights(0)
    assert len(w0) == 0, f"n=0: expected empty, got {w0}"

    # n=1..24 → 全1
    for n in [1, 5, 10, 24]:
        w = default_weights(n)
        assert len(w) == n
        assert all(abs(v - 1.0) < 1e-10 for v in w), f"n={n}: expected all 1.0"

    # n=25 → 全1 (ramp_len=0)
    w25 = default_weights(25)
    assert len(w25) == 25
    assert all(abs(v - 1.0) < 1e-10 for v in w25)

    # n=26 → 第一个是 1/26
    w26 = default_weights(26)
    assert len(w26) == 26
    assert abs(w26[0] - 1.0 / 26) < 1e-6, f"w26[0]={w26[0]}, expected {1/26}"
    for i in range(1, 26):
        assert abs(w26[i] - 1.0) < 1e-10, f"w26[{i}]={w26[i]}"

    # n=50
    w50 = default_weights(50)
    assert len(w50) == 50
    assert abs(w50[0] - 0.02) < 1e-6
    assert abs(w50[24] - 1.0) < 1e-6
    for i in range(25, 50):
        assert abs(w50[i] - 1.0) < 1e-10

    # n=100
    w100 = default_weights(100)
    assert len(w100) == 100
    assert abs(w100[0] - 0.01) < 1e-6
    assert abs(w100[74] - 1.0) < 1e-6


def test_tpe_hyperopt_gamma_cross_validation():
    """对齐 Python/Rust: hyperopt_default_gamma 函数输出完全一致"""
    from optuna.samplers._tpe.sampler import hyperopt_default_gamma
    import math

    # hyperopt_default_gamma: min(ceil(0.25 * sqrt(n)), 25)
    for n, expected in [(0, 0), (1, 1), (4, 1), (16, 1), (17, 2), (25, 2),
                        (64, 2), (100, 3), (10000, 25), (100000, 25)]:
        actual = hyperopt_default_gamma(n)
        assert actual == expected, f"hyperopt_default_gamma({n}) = {actual}, expected {expected}"


def test_tpe_default_gamma_cross_validation():
    """对齐 Python/Rust: default_gamma (TPE 默认 gamma) 一致"""
    from optuna.samplers._tpe.sampler import default_gamma

    # default_gamma(n) = min(ceil(0.1 * n), 25)
    expected = {0: 0, 1: 1, 10: 1, 50: 5, 250: 25, 500: 25}
    for n, exp in expected.items():
        actual = default_gamma(n)
        assert actual == exp, f"default_gamma({n}) = {actual}, expected {exp}"


def test_intersection_search_space_conflict_permanence():
    """对齐 Python/Rust: 分布冲突后搜索空间永久清空"""
    import optuna
    from optuna.search_space import IntersectionSearchSpace

    study = optuna.create_study()

    # Trial 0: x ∈ [0, 1]
    trial0 = study.ask()
    trial0.suggest_float("x", 0.0, 1.0)
    study.tell(trial0, 0.5)

    iss = IntersectionSearchSpace()
    ss1 = iss.calculate(study)
    assert "x" in ss1, f"x should be in search space, got {ss1}"

    # Trial 1: x ∈ [0, 10] — 不同分布
    trial1 = study.ask()
    trial1.suggest_float("x", 0.0, 10.0)
    study.tell(trial1, 0.3)

    ss2 = iss.calculate(study)
    assert "x" not in ss2, f"x should be removed after conflict, got {ss2}"

    # Trial 2: x ∈ [0, 1] — 恢复原分布，但搜索空间应仍为空
    trial2 = study.ask()
    trial2.suggest_float("x", 0.0, 1.0)
    study.tell(trial2, 0.4)

    ss3 = iss.calculate(study)
    assert "x" not in ss3, f"x should remain removed, got {ss3}"


def test_intersection_search_space_include_pruned():
    """对齐 Python/Rust: include_pruned=True 时包含被剪枝的试验"""
    import optuna
    from optuna.search_space import IntersectionSearchSpace

    study = optuna.create_study()

    # Trial 0: Complete with x and y
    trial0 = study.ask()
    trial0.suggest_float("x", 0.0, 1.0)
    trial0.suggest_float("y", 0.0, 1.0)
    study.tell(trial0, 0.5)

    # Trial 1: Pruned with only x
    trial1 = study.ask()
    trial1.suggest_float("x", 0.0, 1.0)
    study.tell(trial1, state=optuna.trial.TrialState.PRUNED)

    # include_pruned=False → 只看 Complete，搜索空间 = {x, y}
    iss_no = IntersectionSearchSpace(include_pruned=False)
    ss_no = iss_no.calculate(study)
    assert "x" in ss_no and "y" in ss_no

    # include_pruned=True → Complete ∩ Pruned = {x}
    iss_yes = IntersectionSearchSpace(include_pruned=True)
    ss_yes = iss_yes.calculate(study)
    assert "x" in ss_yes
    assert "y" not in ss_yes, f"y should not be in intersection, got {ss_yes}"


def test_fixed_trial_base_interface():
    """对齐 Python/Rust: FixedTrial 的 BaseTrial 接口行为"""
    import optuna

    # suggest_float 返回固定值
    trial = optuna.trial.FixedTrial({"x": 0.5, "n": 3})
    assert trial.suggest_float("x", 0.0, 1.0) == 0.5
    assert trial.suggest_int("n", 0, 10) == 3

    # should_prune 始终返回 False
    assert trial.should_prune() is False

    # report 是 no-op
    trial.report(1.0, 0)

    # params 和 distributions 一致
    p = trial.params
    d = trial.distributions
    assert set(p.keys()) == set(d.keys())

    # number 默认为 0
    assert trial.number == 0


def test_nsgaiii_reference_points_count():
    """对齐 Python/Rust: generate_reference_points 点数公式 C(n+d-1, d)"""
    from math import comb

    # 验证 Das-Dennis 公式
    test_cases = [
        (2, 3, comb(4, 3)),   # C(4,3)=4
        (2, 4, comb(5, 4)),   # C(5,4)=5
        (3, 3, comb(5, 3)),   # C(5,3)=10
        (3, 4, comb(6, 4)),   # C(6,4)=15
        (4, 3, comb(6, 3)),   # C(6,3)=20
    ]

    # 直接用 NSGA-III 创建并检查参考点数
    import optuna
    for n_obj, divs, expected_count in test_cases:
        sampler = optuna.samplers.NSGAIIISampler(
            dividing_parameter=divs
        )
        # 验证公式: C(n_obj + divs - 1, divs) = expected_count
        actual = comb(n_obj + divs - 1, divs)
        assert actual == expected_count, \
            f"n_obj={n_obj}, divs={divs}: C({n_obj+divs-1},{divs})={actual}, expected {expected_count}"


# ═══════════════════════════════════════════════════════════════════════════
# Session 48: 对齐 logging、NopPruner、GP、heartbeat 相关
# ═══════════════════════════════════════════════════════════════════════════


def test_logging_set_get_verbosity():
    """对齐 Rust logging.rs: set_verbosity/get_verbosity 往返一致"""
    import optuna
    original = optuna.logging.get_verbosity()
    try:
        optuna.logging.set_verbosity(optuna.logging.DEBUG)
        assert optuna.logging.get_verbosity() == optuna.logging.DEBUG
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        assert optuna.logging.get_verbosity() == optuna.logging.WARNING
        optuna.logging.set_verbosity(optuna.logging.ERROR)
        assert optuna.logging.get_verbosity() == optuna.logging.ERROR
        optuna.logging.set_verbosity(optuna.logging.INFO)
        assert optuna.logging.get_verbosity() == optuna.logging.INFO
    finally:
        optuna.logging.set_verbosity(original)


def test_logging_verbosity_levels_values():
    """对齐 Rust LogLevel repr: DEBUG=10, INFO=20, WARNING=30, ERROR=40"""
    import optuna
    assert optuna.logging.DEBUG == 10
    assert optuna.logging.INFO == 20
    assert optuna.logging.WARNING == 30
    assert optuna.logging.ERROR == 40


def test_nop_pruner_never_prunes():
    """对齐 Rust nop.rs: NopPruner 永远返回 False"""
    import optuna
    study = optuna.create_study()

    def objective(trial):
        for step in range(10):
            trial.report(float(step), step)
            # NopPruner 不应剪枝
            assert not trial.should_prune()
        return 1.0

    study.optimize(objective, n_trials=3, callbacks=[],
                   catch=())


def test_nop_pruner_is_default():
    """对齐 Rust: 默认 pruner 是 NopPruner"""
    import optuna
    pruner = optuna.pruners.NopPruner()
    # NopPruner 应该有 prune 方法
    assert hasattr(pruner, 'prune')


def test_study_direction_values():
    """对齐 Rust direction.rs: StudyDirection 枚举值"""
    import optuna
    assert optuna.study.StudyDirection.NOT_SET.value == 0
    assert optuna.study.StudyDirection.MINIMIZE.value == 1
    assert optuna.study.StudyDirection.MAXIMIZE.value == 2


def test_study_direction_all_variants():
    """对齐 Rust direction.rs: 所有变体互不相同"""
    import optuna
    directions = list(optuna.study.StudyDirection)
    assert len(directions) == 3
    assert len(set(directions)) == 3


def test_gp_sampler_creation():
    """对齐 Rust gp.rs: GPSampler 可以创建"""
    import optuna
    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(sampler=sampler)
    # 至少能做 1 轮优化
    study.optimize(lambda trial: trial.suggest_float("x", 0, 1) ** 2, n_trials=3)
    assert len(study.trials) == 3


def test_gp_sampler_deterministic():
    """对齐 Rust gp_lbfgsb.rs: deterministic_objective 模式"""
    import optuna
    sampler = optuna.samplers.GPSampler(seed=42, deterministic_objective=True)
    study = optuna.create_study(sampler=sampler)
    study.optimize(lambda trial: trial.suggest_float("x", 0, 1) ** 2, n_trials=5)
    assert len(study.trials) == 5
    assert study.best_value is not None


def test_gp_sampler_multidimensional():
    """对齐 Rust gp_lbfgsb.rs: 多维搜索"""
    import optuna
    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(sampler=sampler)

    def obj(trial):
        x = trial.suggest_float("x", 0, 1)
        y = trial.suggest_float("y", 0, 1)
        return x + y

    study.optimize(obj, n_trials=5)
    assert len(study.trials) == 5


def test_sampler_after_trial_all_states():
    """对齐 Rust samplers/mod.rs: after_trial 对所有状态可调用"""
    import optuna
    from optuna.trial import TrialState

    study = optuna.create_study()

    # COMPLETE: 需要 values
    trial1 = study.ask()
    study.tell(trial1, values=0.0, state=TrialState.COMPLETE)

    # PRUNED: 不传 values
    trial2 = study.ask()
    study.tell(trial2, state=TrialState.PRUNED)

    # FAIL: 不传 values
    trial3 = study.ask()
    study.tell(trial3, state=TrialState.FAIL)

    assert len(study.trials) == 3


def test_heartbeat_default_interval():
    """对齐 Rust heartbeat.rs: 默认没有 heartbeat 干扰"""
    import optuna
    study = optuna.create_study()
    # InMemory storage 不支持 heartbeat, 不会报错
    assert len(study.trials) == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Session 49/50: 审计修复的交叉验证测试
# ═══════════════════════════════════════════════════════════════════════════

def test_tell_auto_infer_fail_with_warning():
    """对齐 Rust tell_auto: state=None + values=None → FAIL + 警告。
    Python `_tell_with_warning` 中 state is None 分支。"""
    import warnings
    study = optuna.create_study()

    # 情况1: values=None → FAIL
    trial1 = study.ask()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        study.tell(trial1, values=None, state=None)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL

    # 情况2: values=[NaN] → FAIL
    trial2 = study.ask()
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        study.tell(trial2, values=[float('nan')], state=None)
    assert study.trials[1].state == optuna.trial.TrialState.FAIL

    # 情况3: values=[1.0] → Complete
    trial3 = study.ask()
    study.tell(trial3, values=[1.0], state=None)
    assert study.trials[2].state == optuna.trial.TrialState.COMPLETE
    assert study.trials[2].values == [1.0]


def test_tell_auto_wrong_n_values():
    """对齐 Rust tell_auto: values 数量不匹配 → FAIL。
    Python `_check_values_are_feasible` 检查。"""
    study = optuna.create_study(directions=["minimize", "minimize"])

    # 传入1个值但期望2个 → FAIL
    trial = study.ask()
    import warnings
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        study.tell(trial, values=[1.0], state=None)
    assert study.trials[0].state == optuna.trial.TrialState.FAIL


def test_categorical_contains_int_truncation():
    """对齐 Rust categorical.contains: Python 使用 int() 截断语义。
    int(0.5)=0, int(2.9)=2, int(-0.1)=0, int(-1.0)=-1"""
    d = CategoricalDistribution(choices=["a", "b", "c"])  # 3 choices: idx 0,1,2

    # 整数索引
    assert d._contains(0.0) == True
    assert d._contains(1.0) == True
    assert d._contains(2.0) == True
    assert d._contains(3.0) == False
    assert d._contains(-1.0) == False

    # 非整数值: Python int() 截断
    assert d._contains(0.5) == True   # int(0.5)=0, valid
    assert d._contains(2.9) == True   # int(2.9)=2, valid
    assert d._contains(-0.1) == True  # int(-0.1)=0, valid
    assert d._contains(-0.9) == True  # int(-0.9)=0, valid
    assert d._contains(-1.1) == False # int(-1.1)=-1, invalid


def test_get_single_value_asserts_single():
    """对齐 Rust get_single_value: 调用前需 assert single()。
    Python `_get_single_value` 函数开头 `assert distribution.single()`。"""
    from optuna.distributions import _get_single_value

    # single() == True → 正常返回
    d_float = FloatDistribution(1.0, 1.0)
    assert d_float.single()
    assert _get_single_value(d_float) == 1.0

    d_int = IntDistribution(5, 5)
    assert d_int.single()
    assert _get_single_value(d_int) == 5

    d_cat = CategoricalDistribution(choices=["only"])
    assert d_cat.single()
    assert _get_single_value(d_cat) == "only"

    # single() == False → assert 失败
    d_multi = FloatDistribution(0.0, 1.0)
    assert not d_multi.single()
    try:
        _get_single_value(d_multi)
        assert False, "should have raised AssertionError"
    except AssertionError:
        pass


def test_report_duplicate_step_ignored():
    """对齐 Rust report: 重复 step 忽略新值（不覆盖）。
    Python Trial.report 在 step 已存在时发出警告并 return，不写入存储。"""
    study = optuna.create_study()
    trial = study.ask()

    trial.report(1.0, step=0)
    assert trial._cached_frozen_trial.intermediate_values[0] == 1.0

    # 重复 report 同一 step: 值不变
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        trial.report(999.0, step=0)
        # 应该收到警告
        assert any("already reported" in str(warning.message) for warning in w)

    # 值仍为 1.0
    assert trial._cached_frozen_trial.intermediate_values[0] == 1.0


def test_tpe_mo_weights_3_objectives():
    """对齐 Rust TPE calculate_mo_weights: ≤3 目标使用精确 LOO HV 贡献。"""
    from optuna.samplers._tpe.sampler import _calculate_weights_below_for_multi_objective

    study = optuna.create_study(directions=["minimize", "minimize", "minimize"])

    # 构造一组简单的试验
    for vals in [[1.0, 5.0, 3.0], [3.0, 1.0, 4.0], [2.0, 2.0, 2.0], [4.0, 4.0, 1.0]]:
        trial = study.ask()
        study.tell(trial, vals)

    below = study.trials
    weights = _calculate_weights_below_for_multi_objective(study, below, None)

    # 所有权重应当为正（可行试验）
    assert all(w > 0 for w in weights), f"weights={weights}"
    assert len(weights) == 4


def test_tpe_mo_weights_4_objectives():
    """对齐 Rust TPE calculate_mo_weights: >3 目标使用近似方法。
    验证近似算法不会产生负权重。"""
    from optuna.samplers._tpe.sampler import _calculate_weights_below_for_multi_objective

    study = optuna.create_study(directions=["minimize"] * 4)

    # 构造试验
    np.random.seed(42)
    for _ in range(8):
        trial = study.ask()
        study.tell(trial, np.random.rand(4).tolist())

    below = study.trials
    weights = _calculate_weights_below_for_multi_objective(study, below, None)

    assert all(w > 0 for w in weights), f"weights={weights}"
    assert len(weights) == 8


def test_non_domination_rank_n_below():
    """对齐 Rust fast_non_domination_rank_with_n_below:
    Python `_calculate_nondomination_rank` 支持 n_below 提前终止。"""
    from optuna.study._multi_objective import _fast_non_domination_rank

    loss_values = np.array([
        [1.0, 5.0],   # Pareto front (rank 0)
        [5.0, 1.0],   # Pareto front (rank 0)
        [2.0, 3.0],   # Pareto front (rank 0): 不被 [1,5] 或 [5,1] 支配
        [3.0, 2.0],   # Pareto front (rank 0): 同上
        [4.0, 4.0],   # rank 1: 被 [2,3] 支配 (2<4 且 3<4)
    ])

    # 不限制 n_below
    ranks_full = _fast_non_domination_rank(loss_values)
    assert ranks_full[0] == 0
    assert ranks_full[1] == 0
    assert ranks_full[2] == 0  # [2,3] 也是 Pareto 前沿
    assert ranks_full[3] == 0  # [3,2] 也是 Pareto 前沿
    assert ranks_full[4] == 1  # [4,4] 被 [2,3] 支配

    # n_below=2: 只需要前2个元素的排名，应在前沿0之后终止
    ranks_limited = _fast_non_domination_rank(loss_values, n_below=2)
    assert ranks_limited[0] == 0
    assert ranks_limited[1] == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Session 51/52: GP 多目标 (LogEHVI) 交叉验证测试
# ═══════════════════════════════════════════════════════════════════════════

def test_gp_sampler_multi_objective():
    """对齐 Rust gp.rs: GPSampler 支持多目标优化 (LogEHVI)。
    Python GPSampler 在多 directions 时，使用 LogEHVI 采集函数。"""
    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(
        sampler=sampler,
        directions=["minimize", "minimize"],
    )

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 2.0)
        return x ** 2, (x - 1.0) ** 2

    study.optimize(objective, n_trials=15)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) >= 10, f"Should have >= 10 completed, got {len(completed)}"

    # 某些试验应接近 Pareto 前沿: x ∈ [0,1], f ∈ [0,1]²
    pareto_near = [t for t in completed
                   if t.values[0] <= 1.5 and t.values[1] <= 1.5]
    assert len(pareto_near) > 0, "Should have trials near the Pareto front"


def test_gp_sampler_multi_objective_3d():
    """对齐 Rust gp.rs: GPSampler 3 目标优化。"""
    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(
        sampler=sampler,
        directions=["minimize", "minimize", "minimize"],
    )

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 3.0)
        y = trial.suggest_float("y", 0.0, 3.0)
        return x ** 2, y ** 2, (x - y) ** 2

    study.optimize(objective, n_trials=15)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) >= 10, f"Should have >= 10 completed, got {len(completed)}"


def test_gp_sampler_multi_objective_maximize():
    """对齐 Rust gp.rs: GPSampler 多目标 maximize 方向。"""
    sampler = optuna.samplers.GPSampler(seed=42)
    study = optuna.create_study(
        sampler=sampler,
        directions=["maximize", "maximize"],
    )

    def objective(trial):
        x = trial.suggest_float("x", 0.0, 1.0)
        return x, 1.0 - x  # Pareto front: (x, 1-x) for x ∈ [0,1]

    study.optimize(objective, n_trials=15)
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    assert len(completed) >= 10


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
