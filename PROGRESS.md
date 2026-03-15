# optuna-rs 对齐进度

## 总览

- **Rust 测试基线**: 629 default / 705 all-features
- **Python 交叉验证**: 90 tests (全部通过)
- **最新提交**: Session 28 on gitlab/main

## 功能对齐状态 (对比 Python optuna)

| 模块 | 状态 | 备注 |
|------|------|------|
| Study (study/core.rs) | ✅ 100% | optimize, ask, tell, tell_auto, enqueue_trial, best_trial, best_trials, add_trial, metric_names |
| Trial (trial/handle.rs) | ✅ 100% | suggest_float/int/categorical, report, should_prune, set_user_attr |
| FrozenTrial (trial/frozen.rs) | ✅ 100% | validate, value, last_step, duration, suggest_* |
| Distributions | ✅ 100% | Float, Int, Categorical (含 step/log) |
| Samplers | ✅ 100% | TPE, Random, Grid, BruteForce, CmaEs, GP, QMC, NSGAII, NSGAIII, PartialFixed |
| Pruners | ✅ 100% | Median, Percentile, Hyperband, SuccessiveHalving, Threshold, Patient, Wilcoxon, Nop |
| Storage | ✅ 100% | InMemory, Cached, Journal, RDB, Redis, gRPC (6 backends) |
| Callbacks | ✅ 100% | MaxTrialsCallback, RetryFailedTrialCallback, TerminatorCallback |
| SearchSpace | ✅ 100% | IntersectionSearchSpace, GroupDecomposedSearchSpace, SearchSpaceTransform |
| Terminator | ✅ 100% | ErrorEvaluator, ImprovementEvaluator |
| Importance | ✅ 100% | Fanova, MeanDecreaseImpurity, PedAnova |
| Multi-Objective | ✅ 100% | Pareto front, NSGA-II/III |
| Hypervolume | ✅ 100% | WFG 算法 |
| CLI | ✅ 100% | create/delete/ask/tell/best-trial/best-trials/studies |

## Bug 修复记录

### Session 24 (commit 0a3852f)
- MaxTrialsCallback states=None → 传 None 给 get_n_trials
- SuccessiveHalvingPruner bootstrap 验证
- HyperbandPruner 双重初始化 + bootstrap 验证
- CMA-ES categorical 过滤 + pruned trials 处理
- Study.tell() 允许 Pruned+values
- Study.enqueue_trial() 检查所有状态

### Session 25 (commit 001c5b3)
- [HIGH] CLI tell 绕过 Study.tell() 验证 → 使用 load_study + tell_with_options
- [LOW] Trial.suggest() 缺少 distribution.single() 短路返回
- [BUG] InMemoryStorage.set_trial_system_attr 错误检查 updatable

### Session 26 (commit bb70d72)
- 深度审计确认 98% 功能对齐
- 新增 21 个 Python 交叉验证测试 (53→74)
- 新增 22 个 Rust 内联测试 (571→593)

### Session 27 (commit 4f99c53)
- [MEDIUM] FloatDistribution.contains() 容差 1e-6 → 1e-8 (对齐 Python _contains 精确容差)
- [CRITICAL] RandomSampler gen_range(*lo..=*hi) → gen_range(*lo..*hi) (对齐 Python np.random.uniform [lo, hi) 半开区间)
- [HIGH] Trial.suggest() 检查顺序: single() 在 fixed_params 之前 → fixed_params 在 single() 之前 (对齐 Python _suggest 精确顺序)
- 新增 17 个 Rust 内联测试 (593→610)
- 新增 7 个 Python 交叉验证测试 (74→81)

### Session 28 (当前)
- [CRITICAL] tell() PRUNED/FAIL+values: Python 抛 ValueError, Rust 静默丢弃 → 修复为抛错
- [CRITICAL] TPE after_trial 不存储约束值 → 新增 compute_constraints() trait 方法 + tell() 自动存储
- [HIGH] tell() 不支持 state=None 自动推断 → 新增 tell_auto() 方法
- [MEDIUM] set_trial_state_values values=None 清空已有值 → 修复为仅 Some 时覆盖
- [MEDIUM] SearchSpaceTransform IntDist log+transform_log=False: round() → 截断 (对齐 Python int())
- [MEDIUM] get_pareto_front_trials 缺 consider_constraint → 新增 with_constraint 变体
- 深度审计: study/core.rs, storage/in_memory.rs, trial/frozen.rs, samplers/tpe, pruners, search_space, multi_objective
- 新增 19 个 Rust 内联测试 (610→629)
- 新增 9 个 Python 交叉验证测试 (81→90)

## 测试覆盖

### 高覆盖文件 (>10 tests)
- multi_objective.rs: 43 tests
- study/core.rs: 44 tests
- storage/in_memory.rs: 25 tests
- trial/frozen.rs: 24 tests
- search_space/transform.rs: 21 tests
- distributions/float.rs: 16 tests
- trial/handle.rs: 16 tests
- storage/journal.rs: 12 tests
- random.rs: 14 tests
- callbacks/mod.rs: 11 tests
- distributions/int.rs, categorical.rs: 合计 ~25 tests

### 中覆盖文件 (4-10 tests)
- samplers/brute_force.rs: 9 tests
- storage/mod.rs: 8 tests
- search_space/intersection.rs: 8 tests
- search_space/group_decomposed.rs: 8 tests
- samplers/partial_fixed.rs: 8 tests
- pruners/hyperband.rs: 9 tests
- pruners/successive_halving.rs: 11 tests
- error.rs: 6 tests
- trial/state.rs: 6 tests
- study/frozen.rs: 5 tests
- study/mod.rs: 9 tests
- pruners/mod.rs: 5 tests
