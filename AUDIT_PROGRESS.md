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
