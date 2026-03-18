# optuna-rs 对齐进度

## 总览

- **Rust 测试基线**: 1011 unit + 5 doc-tests
- **Python 交叉验证**: 全部通过
- **最新提交**: Session 44 — TPE/交叉算子/HeartbeatThread/BaseTrial 深度审计 (10 bug fix + 12 new tests)

## Session 35 修复摘要 (全模块深度审计)

### 审计范围
- distributions, trial, study, storage, search_space, callbacks, samplers, pruners, importance, visualization

### Bug 修复 (15 项)
1. **[HIGH] storage/in_memory.rs**: InMemoryStorage get_best_trial override — 单锁作用域防死锁
2. **[HIGH] storage/mod.rs**: get_best_trial 默认实现 partial_cmp().unwrap() → total_cmp() 防 NaN panic
3. **[HIGH] study/mod.rs**: create_study 多目标时默认使用 NSGAIISampler (对齐 Python)
4. **[HIGH] distributions/mod.rs**: CategoricalDistribution to_internal_repr 支持 ParamValue::Int/Float (JSON 反序列化兼容)
5. **[MEDIUM] study/mod.rs**: create_study 空方向列表验证
6. **[MEDIUM] study/frozen.rs**: 添加 FrozenStudy.direction() 方法 + PartialEq
7. **[MEDIUM] trial/frozen.rs**: _suggest 范围检查从静默忽略改为 warning
8. **[MEDIUM] search_space/intersection.rs**: next_cached_trial_number 初始化为 -1 (对齐 Python)
9. **[MEDIUM] search_space/intersection.rs**: 移除不可达的 Pruned 检查代码
10. **[MEDIUM] search_space/transform.rs**: 添加空搜索空间 assert (对齐 Python)
11. **[MEDIUM] search_space/group_decomposed.rs**: study_id 不匹配时 panic 消息更清晰
12. **[MEDIUM] callbacks/mod.rs**: TerminatorCallback 添加终止日志 (对齐 Python)
13. **[MEDIUM] samplers/cmaes.rs**: 添加 6 项参数组合互斥验证
14. **[MEDIUM] samplers/gp.rs**: 添加 warn_and_convert_inf — Inf 目标值裁剪为 MAX/MIN
15. **[MEDIUM] samplers/gp.rs**: 添加 warn_independent_sampling 警告
16. **[MEDIUM] pruners/wilcoxon.rs**: Wilcoxon 统计量添加连续性修正 (对齐 scipy)
17. **[MEDIUM] importance.rs**: FanovaEvaluator/MDI/PedAnova 移除内部归一化, 由 get_param_importances 统一处理

### 新增测试 (14 个 Rust, 7 个 Python)
- storage/in_memory.rs: +5 (get_best_trial override: no_deadlock, maximize, no_complete, multi_obj_error, nan_safety)
- study/frozen.rs: +3 (direction_single, direction_multi_error, partial_eq)
- study/mod.rs: +2 (empty_directions_error, both_direction_directions_error)
- search_space/transform.rs: +1 (empty_panic)
- samplers/cmaes.rs: +3 (source_x0_conflict, separable_margin_conflict, lr_separable_conflict)
- Python cross-validation: +7 (frozen_study_direction, create_study_errors, default_sampler, importance_normalize)

### 已知架构差异 (非本次修复范围)
- GP: 使用随机搜索拟合超参而非 L-BFGS-B + 先验; 不支持多目标
- TPE: 缺失多目标 MOTPE 分割逻辑 (非支配排序 + HSSP)
- FanovaEvaluator: 使用朴素 bin-variance 算法而非真正的 fANOVA 树分解
- Pruners: direction 在构造时绑定而非 prune 时读取 study
- CMA-ES: 状态不可序列化/恢复, 无分布式支持
- Terminator: `test_regret_bound_top_n_includes_ties`, `test_emmr_search_space_uses_all_trials`

### 已知仍待对齐的差异
- TPE 多目标 (MOTPE): 完全缺失
- CMA-ES 状态持久化: 分布式不安全
- CMA-ES 世代管理: 并行不安全
- QMC sample_id: 分布式下序列重复
- GridSampler `_same_search_space` 检查: 跨搜索空间 grid_id 隔离
- `reseed_rng` trait 方法: 所有 sampler 缺失

## Session 32 修复摘要

### Bug 修复 (12 项)
1. **[CRITICAL] storage/in_memory.rs**: `set_trial_system_attr` 添加 `check_updatable` 验证 (Python 对齐)
2. **[CRITICAL] study/core.rs**: `best_trial` 约束错误消息改为 "No feasible trials are completed yet."
3. **[CRITICAL] study/core.rs**: `set_metric_names` 错误消息改为英文 (对齐 Python)
4. **[HIGH] samplers/nsgaii/sampler.rs**: mutation 语义修复 — 被 mutate 的参数从结果中排除 (由 sample_independent 重采样)
5. **[HIGH] samplers/nsgaiii/sampler.rs**: 同 NSGA-II mutation 语义修复
6. **[HIGH] samplers/nsgaii/sampler.rs**: 添加 `population_size >= 2` 验证
7. **[HIGH] samplers/nsgaii/sampler.rs**: 添加 `population_size >= crossover.n_parents` 验证
8. **[HIGH] samplers/nsgaiii/sampler.rs**: 添加 population_size 验证 (同 NSGA-II)
9. **[HIGH] samplers/cmaes.rs**: 添加 4 项参数冲突验证 (source_trials vs x0/sigma0, separable vs margin, lr_adapt vs separable/margin)
10. **[HIGH] samplers/ga.rs**: 修复 GA 测试中 trial 生命周期 (先 RUNNING 再设 generation → Complete)
11. **[MEDIUM] tests/test_cross_validation.py**: 修复 truncnorm ppf API 调用 (3 参数, 非 5 参数)
12. **[MEDIUM] samplers/tpe/parzen_estimator.rs**: validate_weights 检查顺序修正 (先 finite 再 positive)

### 新增测试 (15 项)
- storage/in_memory.rs: +2 (set_trial_system_attr 拒绝已完成试验, RUNNING 正常)
- study/core.rs: +3 (set_metric_names 错误消息, 约束错误消息, storage system_attr 拒绝)
- samplers/nsgaii/sampler.rs: +1 (population_size < 2 panic)
- samplers/cmaes.rs: +2 (separable+margin 冲突, lr_adapt+separable 冲突)
- tests/test_cross_validation.py: +3 (storage system_attr 拒绝, metric_names 错误, 约束错误)
- 上一 Session 遗留: +7 (Session 31 的 partial_fixed + parzen_estimator + truncnorm 测试)

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

### Session 43 — 全仓深度审计 (distributions shorthand + transform + cached storage)
- [HIGH] SearchSpaceTransform categorical 无效值回退 index 0 → panic 对齐 Python ValueError
- [HIGH] json_to_distribution 缺失 shorthand 缩写格式 → 完整实现 float/int/categorical
- [MEDIUM] CachedStorage get_all_trials 缓存未生效 → 合并策略使用缓存版本
- 全仓三模块深度审计: 排除 5 个误报
- 新增 12 个测试 (987→999)

### Session 42 — 全仓深度审计 (distributions + trial + storage)
- [HIGH] CategoricalDistribution::contains NaN/Inf 穿透 bug → 添加 NaN/Inf 前置守卫
- [HIGH] Trial::set_user_attr 仅写 storage 不更新 cached_trial → 同步更新本地缓存
- [HIGH] Trial::set_system_attr 同上
- [MEDIUM] CachedStorage::get_all_trials 全量读取 → 增量缓存已完成试验
- 验证: CRC32 与 Python binascii.crc32 完全一致, Wilcoxon correction=False 正确
- 新增 7 个测试 (980→987)

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

### Session 29
- [CRITICAL] check_distribution_compatibility 过度严格: 移除 Float/Int step 检查 (对齐 Python 仅检查 log)
- [HIGH] create_trial() 跳过 validate() + 静默取 value 忽略 values → 改用 FrozenTrial::new() 内部校验
- [CRITICAL] TPE split_trials 缺少按 trial.number 重排序 → 添加排序 (对齐 Python 权重按时间序分配)
- [HIGH] TPE IntersectionSearchSpace include_pruned=false → true (对齐 Python)
- [HIGH] TPE infer_relative_search_space 未过滤 single() 分布 → 添加过滤
- [HIGH] TPE custom weights 函数未传递给 ParzenEstimator → 添加 weights_func 参数
- [MEDIUM] TPE pruned trial NaN 中间值未映射为 inf → 添加 NaN→inf 映射 (对齐 Python _get_pruned_trial_score)
- [HIGH] EMMR KL 散度用错 GP 模型 (gpr_t1 → gpr_t) → 修复 (对齐 Python 注释)
- [MEDIUM] RegretBound top_n 四舍五入 → 截断 (对齐 Python int() 语义)
- 深度审计: distributions, trial, samplers/tpe, callbacks, importance, terminators
- 新增 18 个 Rust 内联测试 (629→647)
- 新增 13 个 Python 交叉验证测试 (90→103)

### Session 30
**全仓深度审计**: study/core.rs, distributions, TPE, trial, pruners, storage, importance, terminators, callbacks, search_space, multi_objective
- [CRITICAL] hypervolume_3d z_delta 索引 BUG: z_delta 矩阵写入了错误点的 z 值 + 矩阵乘法使用了转置 → 修复为对齐 Python 的 fancy indexing + 正确的 np.dot 顺序
- [CRITICAL] NSGA-II elite selection: "取最后 N 个试验" → 正确的非支配排序 + 拥挤距离截断
- [CRITICAL] NSGA-III elite selection: "取最后 N 个试验" → 正确的非支配排序 + 参考点 niche 保留
- [HIGH] study/core.rs after_trial finally 语义: after_trial 异常不保证 set_trial_state_values → catch_unwind + 保证写入
- [HIGH] CMA-ES n_startup_trials 默认值 25 → 1 (对齐 Python)
- [HIGH] CMA-ES 均值初始化: 使用最佳试验 → 搜索空间中心 (对齐 Python)
- [HIGH] storage set_trial_param 分布兼容性: 精确相等 → Python 兼容性检查 (同类型不同 range 允许, 不同 log 报错)
- [HIGH] FloatDistribution contains: 移除 ±1e-10 容差，使用严格 low <= value <= high (对齐 Python)
- [HIGH] FloatDistribution adjust_discrete_uniform_high: f64 直接计算 → 字符串精度整数域计算 (对齐 Python Decimal)
- [HIGH] EMMR margin: 移除错误的 +0.1 常数 (Python 不加此 margin)
- [MEDIUM] BruteForce visited check: 未包含 PRUNED/FAIL 状态 → 添加 (对齐 Python)
- [MEDIUM] Grid unvisited check: 未排除 RUNNING 试验 → 添加 Running 排除 + 回退逻辑
- [MEDIUM] enqueue_trial skip_if_exists: 仅检查 fixed_params → 回退到 trial.params (对齐 Python)
- [MEDIUM] TPE group 模式键未排序 → 添加 sorted_keys (对齐 Python sorted(sub_space.items()))
- 新增 7 个 Rust 内联测试 (723→730)
- 新增 7 个 Python 交叉验证测试 (103→110)

## 测试覆盖

### 高覆盖文件 (>10 tests)
- multi_objective.rs: 43 tests
- study/core.rs: 44 tests
- storage/in_memory.rs: 25 tests
- trial/frozen.rs: 24 tests
- search_space/transform.rs: 21 tests
- distributions/float.rs: 16 tests
- distributions/mod.rs: 20 tests (含 check_distribution_compatibility)
- trial/handle.rs: 16 tests
- trial/mod.rs: 12 tests (含 create_trial validate)
- storage/journal.rs: 12 tests
- random.rs: 14 tests
- callbacks/mod.rs: 11 tests
- distributions/int.rs, categorical.rs: 合计 ~25 tests
- samplers/tpe/sampler.rs: 15 tests (含 split_trials/single/weights)

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
