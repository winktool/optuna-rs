#!/usr/bin/env python3
"""
生成 constrained_dominates 和 _evaluate_penalty 的金标准值。
"""
import numpy as np
from optuna.samplers.nsgaii._constraints_evaluation import _constrained_dominates, _evaluate_penalty
from optuna.study import StudyDirection
from optuna.trial import FrozenTrial, TrialState
from optuna.samplers._base import _CONSTRAINTS_KEY
from datetime import datetime

def make_trial(number, values, constraints=None, state=TrialState.COMPLETE):
    t = FrozenTrial(
        number=number,
        trial_id=number,
        state=state,
        value=None,
        values=values,
        datetime_start=datetime.now(),
        datetime_complete=datetime.now(),
        params={},
        distributions={},
        user_attrs={},
        system_attrs={_CONSTRAINTS_KEY: constraints} if constraints is not None else {},
        intermediate_values={},
    )
    return t

dirs = [StudyDirection.MINIMIZE, StudyDirection.MINIMIZE]

print("=== constrained_dominates ===")

# Case 1: Both feasible, a dominates b
a = make_trial(0, [1.0, 2.0], [-1.0, -1.0])
b = make_trial(1, [3.0, 4.0], [-0.5, -0.5])
print(f"Both feasible, a dom b: {_constrained_dominates(a, b, dirs)}")  # True
print(f"Both feasible, b dom a: {_constrained_dominates(b, a, dirs)}")  # False

# Case 2: a feasible, b infeasible
a = make_trial(0, [5.0, 5.0], [-1.0, -1.0])
b = make_trial(1, [1.0, 1.0], [1.0, 0.5])
print(f"a feasible, b infeasible: {_constrained_dominates(a, b, dirs)}")  # True
print(f"b infeasible, a feasible: {_constrained_dominates(b, a, dirs)}")  # False

# Case 3: Both infeasible, a has smaller violation
a = make_trial(0, [5.0, 5.0], [0.5, 0.3])  # violation = 0.8
b = make_trial(1, [1.0, 1.0], [1.0, 2.0])  # violation = 3.0
print(f"Both infeasible, a < b violation: {_constrained_dominates(a, b, dirs)}")  # True
print(f"Both infeasible, b < a violation: {_constrained_dominates(b, a, dirs)}")  # False

# Case 4: Both feasible, tradeoff (no domination)
a = make_trial(0, [1.0, 4.0], [-1.0, -1.0])
b = make_trial(1, [3.0, 2.0], [-0.5, -0.5])
print(f"Tradeoff a→b: {_constrained_dominates(a, b, dirs)}")  # False
print(f"Tradeoff b→a: {_constrained_dominates(b, a, dirs)}")  # False

# Case 5: a has constraints, b doesn't
a = make_trial(0, [3.0, 3.0], [-1.0, -1.0])
b = make_trial(1, [1.0, 1.0])
import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    print(f"a has constraints, b doesn't: {_constrained_dominates(a, b, dirs)}")  # True
    print(f"b doesn't, a has: {_constrained_dominates(b, a, dirs)}")  # False

# Case 6: State gating
a_running = make_trial(0, [1.0, 1.0], [-1.0, -1.0], state=TrialState.RUNNING)
b_complete = make_trial(1, [5.0, 5.0], [-1.0, -1.0])
print(f"a running, b complete: {_constrained_dominates(a_running, b_complete, dirs)}")  # False
print(f"b complete, a running: {_constrained_dominates(b_complete, a_running, dirs)}")  # True

print("\n=== _evaluate_penalty ===")
trials = [
    make_trial(0, [1.0, 1.0], [-1.0, -1.0]),   # feasible → 0
    make_trial(1, [2.0, 2.0], [0.5, 1.5]),      # infeasible → 2.0
    make_trial(2, [3.0, 3.0]),                    # no constraints → nan
    make_trial(3, [4.0, 4.0], [-0.3, 0.7, -0.1, 0.2]),  # infeasible → 0.7+0.2=0.9
]
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    pen = _evaluate_penalty(trials)
print(f"penalties: {pen.tolist()}")  # [0.0, 2.0, nan, 0.9]
