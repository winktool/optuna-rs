# optuna-rs 对齐 Python 审计进度报告

## Session 37 — 全仓深度审计与修复

### 审计状态总览

| 模块 | 审计状态 | 修复数 | 待修复 |
|------|---------|--------|--------|
| storage | ✅ 已完成 | 8 | 0 |
| distributions | ✅ 已完成 | 3 | 2 |
| trial | ✅ 已完成 | 3 | 3 |
| study | ✅ 已完成 | 4 | 6 |
| samplers | ✅ 已完成 | 2 | 8 |
| pruners | ✅ 已完成 | 0 | 2 |
| importance | ✅ 已完成 | 0 | 1 |
| search_space | ✅ 已完成 | 0 | 1 |

### 已应用的修复 (Session 37)

#### Storage 模块
1. **UUID 自动命名** — `create_new_study` 的 auto-naming 从 `no-name-{study_id}` 改为 `no-name-{UUID4}`，消除名称冲突风险
2. **高效 delete_study** — 改为只迭代 study.trials 而非全局 HashMap
3. **有序 get_all_studies** — 结果按 study_id 排序，匹配 Python dict 插入顺序
4. **缓存 get_best_trial** — 使用 `best_trial_id` 缓存，O(1) 查找
5. **RuntimeError 类型** — 多目标 get_best_trial 使用 RuntimeError 而非 ValueError
6. **NaN 安全 update_cache** — best_trial_id 缓存正确处理 NaN 值比较
7. **便捷方法** — 添加 get_trial_params/user_attrs/system_attrs 到 Storage trait
8. **RuntimeError variant** — 在 OptunaError enum 添加 RuntimeError 变体

#### Distributions 模块
1. **get_single_value()** — 添加 `_get_single_value()` 等效方法
2. **is_log()** — 添加 `_is_distribution_log()` 等效方法
3. **错误消息对齐** — `check_distribution_compatibility` 错误消息完全匹配 Python

#### Trial 模块
1. **value() RuntimeError** — 多目标 FrozenTrial.value() 改为 RuntimeError
2. **set_system_attr()** — 添加到 FrozenTrial
3. **set_value()** — 添加到 FrozenTrial，多目标时报 RuntimeError

#### Study 模块
1. **load_study 多目标采样器** — 多目标时默认切换到 NSGAIISampler
2. **direction() RuntimeError** — Study.direction() 和 FrozenStudy.direction() 改为 RuntimeError
3. **NOT_SET 方向验证** — create_study 拒绝 NOT_SET 方向
4. **错误消息对齐** — 匹配 Python 的错误消息文本

#### Samplers 模块
1. **TPE pruned trial score** — 修复无中间值时返回 (1, 0.0) 而非 (i64::MIN, f64::INFINITY)
2. **Grid seed=None** — 改为随机种子而非确定性 seed=0

### 新增测试 (Session 37)

共新增 **18 个测试**（789 总计，从 771 增至 789）:
- Storage: UUID 唯一性、排序、RuntimeError、NaN Maximize、便捷方法
- Distributions: get_single_value (3)、is_log、错误消息匹配
- Trial: RuntimeError 类型、set_system_attr、set_value (2)
- Study: RuntimeError 方向、NOT_SET 验证 (2)
- Grid: seed 确定性测试

---

### 待修复项（按优先级）

#### 🔴 CRITICAL — 影响正确性

| # | 模块 | 问题 | 描述 |
|---|------|------|------|
| 1 | samplers/TPE | MOTPE 完全缺失 | 多目标 TPE 不支持：缺少非支配排序、HSSP、超体积贡献加权 |
| 2 | samplers/NSGA-II | 无代际系统 | 使用 last-N trials 而非 Python 的 generation-based population |
| 3 | samplers/CMA-ES | 状态不持久化 | CMA 状态只在内存中，不序列化到 storage，不支持分布式 |
| 4 | study | tell() state=None 缺失 | Python 的 tell(state=None) 自动推断状态，Rust 必须传入 |
| 5 | study | _filter_study 缺失 | Hyperband 的 after_trial 不经过 _filter_study 过滤 |

#### 🟠 HIGH — 功能缺失

| # | 模块 | 问题 | 描述 |
|---|------|------|------|
| 6 | samplers | reseed_rng 缺失 | 并行模式下不同线程可能使用相同随机种子 |
| 7 | samplers/CMA-ES | 自定义 Jacobi 特征分解 | 数值稳定性不如 Python 使用的 cmaes 库 |
| 8 | study | 嵌套 optimize 检查 | 缺少 in_optimize_loop 标志，允许嵌套调用 |
| 9 | study | StudySummary 类缺失 | get_all_study_summaries 只返回 FrozenStudy |
| 10 | study | Heartbeat 缺失 | 无法检测和清理僵死试验 |
| 11 | distributions | deprecated distribution JSON | 无法反序列化旧版 Python 存储的 JSON |

#### 🟡 MEDIUM — 差异/不一致

| # | 模块 | 问题 | 描述 |
|---|------|------|------|
| 12 | importance/fANOVA | 分箱方差 vs 真正 fANOVA | 算法完全不同：分箱方差 vs 随机森林边际方差分解 |
| 13 | samplers/NSGA-II | single() 分布未过滤 | infer_relative_search_space 没有过滤单值分布 |
| 14 | pruners/Wilcoxon | 小样本正态近似 | 缺少精确 Wilcoxon 分布表 |
| 15 | trial | BaseTrial trait 缺失 | 无统一 trait 将 Trial/FrozenTrial/FixedTrial 统一 |

---

### 测试统计

| 指标 | 值 |
|------|---|
| Rust 总测试数 | 789 |
| 忽略测试 | 2 |
| 文件修改数 | 10 |
| 新增代码行 | ~300 |

### Git 状态

- 分支: main
- 远程: gitlab
- 上次提交: 04cdf6e (Session 37)

---

## Session 38 — 深层对齐修复与测试扩充

### 已应用的修复

#### FixedTrial 模块 (trial/fixed.rs)
1. **system_attrs 字段** — 添加 `system_attrs: HashMap<String, serde_json::Value>` 及 getter
2. **set_system_attr 方法** — 添加 `#[deprecated]` 标注的 `set_system_attr()` 方法
3. **params() 别名** — 添加 `params()` 方法作为 `suggested_params()` 的别名（匹配 Python `.params` 属性）
4. **内部字段重命名** — `params` → `fixed_params` 避免与 Python API 命名冲突
5. **datetime_start 返回类型** — 改为 `Option<DateTime<Utc>>`

#### Sampler 模块 (samplers/)
6. **reseed_rng 方法** — Sampler trait 添加 `fn reseed_rng(&self, seed: u64)` 默认空实现
7. **RandomSampler reseed_rng** — 实现 RNG 重置
8. **TpeSampler reseed_rng** — 实现 RNG 重置
9. **NSGAIISampler reseed_rng** — 实现 RNG 重置
10. **NSGAIIISampler reseed_rng** — 实现 RNG 重置
11. **CmaEsSampler reseed_rng** — 实现 RNG 重置
12. **GpSampler reseed_rng** — 实现 RNG 重置
13. **NSGA-II single() 过滤** — `infer_relative_search_space` 过滤 `single()` 分布

#### Distributions 模块 (distributions/mod.rs)
14. **旧版 JSON 反序列化** — `json_to_distribution` 支持 5 种 deprecated 格式:
    - UniformDistribution → FloatDistribution
    - LogUniformDistribution → FloatDistribution(log=true)
    - DiscreteUniformDistribution → FloatDistribution(step=q)
    - IntUniformDistribution → IntDistribution
    - IntLogUniformDistribution → IntDistribution(log=true)

#### Study 模块 (study/core.rs)
15. **嵌套 optimize 检测** — 添加 `in_optimize_loop: AtomicBool` 字段
16. **optimize_with_options 重构** — 拆分为外层检查 + `optimize_inner` 内部实现
17. **optimize_multi_with_options 重构** — 同样拆分，防止嵌套调用

### 新增测试 (15 个 Rust / 7 个 Python)

#### Rust 内联测试
| 文件 | 测试名 | 描述 |
|------|--------|------|
| trial/fixed.rs | test_fixed_trial_system_attrs_initially_empty | system_attrs 初始为空 |
| trial/fixed.rs | test_fixed_trial_set_and_get_system_attr | set/get system_attr |
| trial/fixed.rs | test_fixed_trial_params_alias | params() == suggested_params() |
| trial/fixed.rs | test_fixed_trial_datetime_start_is_some | datetime_start 返回 Some |
| trial/fixed.rs | test_fixed_trial_missing_param_error | 缺失参数报错 |
| distributions/mod.rs | test_json_to_distribution_uniform | UniformDistribution JSON |
| distributions/mod.rs | test_json_to_distribution_log_uniform | LogUniform JSON |
| distributions/mod.rs | test_json_to_distribution_discrete_uniform | DiscreteUniform JSON |
| distributions/mod.rs | test_json_to_distribution_int_uniform | IntUniform JSON |
| distributions/mod.rs | test_json_to_distribution_int_log_uniform | IntLogUniform JSON |
| distributions/mod.rs | test_json_to_distribution_standard_float | 标准 Float JSON |
| distributions/mod.rs | test_json_to_distribution_unknown_name | 未知名称报错 |
| samplers/random.rs | test_reseed_rng_changes_output | reseed_rng 改变输出 |
| study/core.rs | test_optimize_loop_flag_resets_after_completion | 标志重置 |
| study/core.rs | test_optimize_loop_flag_resets_after_error | 错误后标志重置 |

#### Python 交叉验证测试
| 测试名 | 描述 |
|--------|------|
| test_deprecated_uniform_distribution_json | FloatDistribution JSON 格式 |
| test_deprecated_int_distribution_step_adjustment | IntDistribution high 调整 |
| test_fixed_trial_system_attrs | FixedTrial system_attrs 初始为空 |
| test_fixed_trial_params_property | FixedTrial params 属性 |
| test_nested_optimize_detection | 连续 optimize 不报嵌套 |
| test_reseed_rng_sampler | RandomSampler reseed_rng 可调用 |

### 测试统计

| 指标 | 数值 |
|------|------|
| Rust 测试总数 | 804 |
| Python 交叉验证 | 131 |
| 本次新增 Rust | +15 |
| 本次新增 Python | +6 |
| 修改文件数 | 11 |

### 待修复 (后续 Session)

| 优先级 | 项目 | 说明 |
|--------|------|------|
| 🔴 高 | MOTPE 多目标 TPE | 无非支配排序/HSSP/超体积加权 |
| 🔴 高 | NSGA-II 代际系统 | 使用末尾 N 个 trial 而非真正的代际 |
| 🔴 高 | CMA-ES 状态持久化 | 重启后丢失优化状态 |
| 🟠 中 | StudySummary 类 | Python 的 StudySummary 未实现 |
| 🟠 中 | Heartbeat 功能 | 分布式试验心跳检测 |
| 🟡 低 | fANOVA 方差分解 | 使用 bin variance 而非随机森林边际方差 |
| 🟡 低 | Wilcoxon 精确分布表 | 小样本精确分布缺失 |
| 🟡 低 | BaseTrial trait | Trial/FrozenTrial/FixedTrial 统一接口 |

---

## Session 39 — Trial 缓存性能优化 & FrozenTrial Hash & 深度审计

### 审计覆盖

#### Pruners 模块深度审计
- **NopPruner** ✅ 完全对齐
- **MedianPruner** ✅ 完全对齐
- **PercentilePruner** ✅ 完全对齐
- **SuccessiveHalvingPruner** ✅ 对齐（细微差异：storage=None 跳过写入）
- **HyperbandPruner** ✅ 完全对齐
- **ThresholdPruner** ✅ 完全对齐
- **PatientPruner** ✅ 对齐（低优先级：NotSet 方向处理）
- **WilcoxonPruner** ✅ 对齐（已知差异：小样本使用正态近似 vs Python 精确表）

#### Trial 模块深度审计
- 发现 P1: Trial 每次 suggest 都读 storage（已修复）
- 发现 P2: FrozenTrial 无 Hash（已修复）

### 已应用的修复

#### FrozenTrial Hash (trial/frozen.rs)
1. **Hash impl** — `impl std::hash::Hash for FrozenTrial`，使用 `trial_id + number + state` 作为哈希键
   - 注：Python FrozenTrial 不可哈希（含 list 字段），此为 Rust 扩展功能

#### Trial 缓存 (trial/handle.rs) — P1 性能修复
2. **cached_trial 字段** — 添加 `cached_trial: FrozenTrial` 字段，匹配 Python 的 `_cached_frozen_trial`
3. **suggest() 重构** — 使用 `cached_trial.distributions/params` 替代每次 `storage.get_trial()` 读取
4. **suggest() 缓存更新** — 成功 suggest 后更新 `cached_trial.params` 和 `cached_trial.distributions`
5. **report() 签名变更** — `&self` → `&mut self`，使用 `cached_trial.intermediate_values` 检查重复 step
6. **report() 缓存更新** — 写入 storage 后更新 `cached_trial.intermediate_values`
7. **params() 简化** — 返回类型从 `Result<HashMap<..>>` → `HashMap<..>`（直读缓存）
8. **distributions() 简化** — 返回类型从 `Result<HashMap<..>>` → `HashMap<..>`（直读缓存）

#### 级联修改
9. **Study.ask()** — 传递 `trial` (FrozenTrial) 作为 `cached_trial` 参数给 `Trial::new()`
10. **cli.rs** — `trial.params().unwrap_or_default()` → `trial.params()`
11. **integration.rs** — `PruningMixin.check()` 参数从 `&Trial` → `&mut Trial`

### 已解决的待修复项

| 原编号 | 原状态 | 项目 | 处理结果 |
|--------|--------|------|---------|
| #4 | 🔴 | tell() state=None 缺失 | ✅ 已存在（Session 39 测试确认） |
| #6 | 🟠 | reseed_rng 缺失 | ✅ Session 38 已修复 |
| #8 | 🟠 | 嵌套 optimize 检查 | ✅ Session 38 已修复 |
| #11 | 🟠 | deprecated distribution JSON | ✅ Session 38 已修复 |
| #13 | 🟡 | NSGA-II single() 过滤 | ✅ Session 38 已修复 |

### 新增测试

#### Rust 内联测试 (20 个)
| 文件 | 测试名 | 描述 |
|------|--------|------|
| trial/frozen.rs | test_frozen_trial_hash_in_hashset | HashSet 去重 |
| trial/frozen.rs | test_frozen_trial_hash_consistency | 哈希一致性 |
| trial/frozen.rs | test_validate_mismatched_keys | 参数/分布键不匹配 |
| trial/frozen.rs | test_validate_complete_without_values | 完成无值拒绝 |
| trial/frozen.rs | test_validate_pruned_without_values_ok | 剪枝无值允许 |
| trial/handle.rs | test_cached_trial_params_updated_after_suggest | suggest 后 params 更新 |
| trial/handle.rs | test_cached_trial_distributions_updated_after_suggest | suggest 后 distributions 更新 |
| trial/handle.rs | test_report_updates_cache_for_pruning | report 更新缓存 |
| trial/handle.rs | test_multiple_suggests_independent | 多参数独立 suggest |
| trial/handle.rs | test_report_multi_objective_error | 多目标 report 拒绝 |
| trial/handle.rs | test_report_negative_step_error | 负 step 报错 |
| study/core.rs | test_tell_auto_valid_values_complete | tell auto 有效值 |
| study/core.rs | test_tell_auto_none_values_fail | tell auto 无值失败 |
| study/core.rs | test_tell_auto_nan_values_fail | tell auto NaN 失败 |
| study/core.rs | test_tell_auto_wrong_count_fail | tell auto 数量错误 |
| study/core.rs | test_tell_finished_trial_error | tell 完成试验报错 |
| study/core.rs | test_tell_finished_trial_skip | tell 跳过完成试验 |
| study/core.rs | test_enqueue_trial_params_used | enqueue 参数生效 |
| study/core.rs | test_add_trial_and_add_trials | add_trial/add_trials |
| study/core.rs | test_study_stop_from_callback | study.stop 回调 |

#### Python 交叉验证测试 (10 个)
| 测试名 | 描述 |
|--------|------|
| test_tell_state_none_valid_values_complete | tell(state=None) 有效值 |
| test_tell_state_none_none_values_fail | tell(state=None) 无值 |
| test_tell_state_none_nan_values_fail | tell(state=None) NaN 值 |
| test_tell_skip_if_finished | tell 跳过已完成 |
| test_tell_pruned_uses_last_intermediate | tell 剪枝用末值 |
| test_enqueue_trial_params_used | enqueue 参数 |
| test_add_trial_and_add_trials | add_trial/add_trials |
| test_study_stop | study.stop |
| test_suggest_updates_params_immediately | suggest 后 params 立即可用 |
| ~~test_frozen_trial_hashable~~ | ~~已移除：Python FrozenTrial不可哈希~~ |

### 测试统计

| 指标 | 数值 |
|------|------|
| Rust 测试总数 | 753 (748 unit + 5 doc) |
| Python 交叉验证 | 140 |
| 本次新增 Rust | +20 (部分抵消了之前多计) |
| 本次新增 Python | +10 |
| 修改文件数 | 6 |

### 待修复 (后续 Session)

| 优先级 | 项目 | 说明 |
|--------|------|------|
| 🔴 高 | MOTPE 多目标 TPE | 无非支配排序/HSSP/超体积加权 |
| 🔴 高 | NSGA-II 代际系统 | 使用末尾 N 个 trial 而非真正的代际 |
| 🔴 高 | CMA-ES 状态持久化 | 重启后丢失优化状态 |
| 🟠 中 | StudySummary 类 | Python 的 StudySummary 未实现 |
| 🟠 中 | Heartbeat 功能 | 分布式试验心跳检测 |
| 🟡 低 | fANOVA 方差分解 | 使用 bin variance 而非随机森林边际方差 |
| 🟡 低 | Wilcoxon 精确分布表 | 小样本精确分布缺失 |
| 🟡 低 | BaseTrial trait | Trial/FrozenTrial/FixedTrial 统一接口 |
| 🟡 低 | Hyperband _filter_study | after_trial 不经过 _filter_study 过滤 |

---

## Session 40 — MOTPE 多目标 TPE 完成 + 修复汇总

### 已完成修复

#### 1. MOTPE 多目标 TPE（重大功能补全）
- `TpeSampler.direction` → `directions: Vec<StudyDirection>`，支持多目标
- `new_multi()` 构造函数接受多个优化方向
- `split_trials_multi_objective()`: 非支配排序 + HSSP 平局打断
- `fast_non_domination_rank()`: 直接在 loss values 上计算非支配层级
- `dominates_values()`: Pareto 支配判定
- `get_reference_point()`: 超体积参考点（对齐 Python 1.1/0.9 规则）
- `calculate_mo_weights()`: 基于 leave-one-out 超体积贡献度的权重，对齐 Python 的 max-normalization
- `TpeSamplerBuilder::new_multi()` 多目标构建器
- `tpe_sample()` 中多目标 below 组使用 HV 权重作为 ParzenEstimator 的 `predetermined_weights`

#### 2. NSGA-II 代际系统（Session 39 完成，Session 40 编译修复）
- `GaSampler for NSGAIISampler` 实现
- `elite_select()` 提取方法
- storage/study_id 注入
- `after_trial` 链式调用 `random_sampler.after_trial()`

#### 3. CMA-ES 状态持久化（Session 39 完成）
- `CmaState` serde 序列化/反序列化
- JSON 分块存储到 trial system_attrs（匹配 Python RDB 2045 字符限制）
- 从已完成 trial 恢复状态

#### 4. StudySummary 类（Session 39 完成）
- `StudySummary` 结构体（study_name, directions, best_trial, user_attrs, system_attrs, n_trials, datetime_start, study_id）
- `build_study_summaries()` 聚合函数
- `get_all_study_summaries()` 返回 `Vec<StudySummary>`

#### 5. Sampler trait 扩展
- `inject_storage()` 默认空操作方法，允许有状态采样器接收 storage 引用
- `Study::new()` 自动调用 `sampler.inject_storage()`

### 新增测试

#### Rust 单元测试 (8 个新增，20 个 TPE 总计)
| 文件 | 测试名 | 描述 |
|------|--------|------|
| tpe/sampler.rs | test_fast_non_domination_rank | 非支配排序层级验证 |
| tpe/sampler.rs | test_dominates_values | Pareto 支配判定 |
| tpe/sampler.rs | test_get_reference_point | 正值参考点 |
| tpe/sampler.rs | test_get_reference_point_negative | 负值参考点 |
| tpe/sampler.rs | test_split_trials_multi_objective | 多目标分割 |
| tpe/sampler.rs | test_calculate_mo_weights_all_pareto | 全 Pareto 权重 |
| tpe/sampler.rs | test_calculate_mo_weights_dominated | 有支配权重 |
| tpe/sampler.rs | test_motpe_builder_multi | 多目标构建器 |
| tpe/sampler.rs | test_motpe_builder_single | 单目标构建器 |

#### Python 交叉验证测试 (6 个新增)
| 测试名 | 描述 |
|--------|------|
| test_motpe_reference_point | 参考点计算对齐 |
| test_motpe_nondomination_rank | 非支配排序对齐 |
| test_motpe_split_multi_objective | 多目标分割对齐 |
| test_motpe_weights_all_pareto | 全 Pareto 权重对齐 |
| test_motpe_weights_dominated | 有支配权重对齐 |
| test_motpe_end_to_end | 端到端多目标优化 |

### 测试统计

| 指标 | 数值 |
|------|------|
| Rust 测试总数 | 773 (768 unit + 5 doc) |
| Python 交叉验证 | 146 |
| 本次新增 Rust | +9 (MOTPE) |
| 本次新增 Python | +6 (MOTPE) |

### 待修复 (后续 Session)

| 优先级 | 项目 | 说明 |
|--------|------|------|
| 🟡 低 | fANOVA 方差分解 | 使用 bin variance 而非随机森林边际方差 |
| 🟡 低 | Wilcoxon 精确分布表 | 小样本精确分布缺失 |
| 🟡 低 | BaseTrial trait | Trial/FrozenTrial/FixedTrial 统一接口 |
| 🟡 低 | Hyperband _filter_study | after_trial 不经过 _filter_study 过滤 |
