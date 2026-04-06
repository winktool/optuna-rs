# Importance 模块深度对比审计报告

**日期**: 2026-07-28  
**模块**: `optuna.importance` (Python) ↔ `optuna_rs::importance` (Rust)  
**Python 代码**: ~810 行 (6 文件)  
**Rust 代码**: 1850 行 (1 文件)

---

## 1. 模块架构对比

| 组件 | Python | Rust | 状态 |
|------|--------|------|------|
| BaseImportanceEvaluator | `_base.py` (185行) | `ImportanceEvaluator` trait | ✅ 对齐 |
| FanovaImportanceEvaluator | `_fanova/_evaluator.py` (130行) + `_fanova.py` (108行) + `_tree.py` (319行) | `FanovaEvaluator` (557行内含) | ✅ 算法对齐 |
| MeanDecreaseImpurityEvaluator | `_mean_decrease_impurity.py` (95行) | `MeanDecreaseImpurityEvaluator` | ✅ 算法对齐 |
| PedAnovaImportanceEvaluator | `_ped_anova/evaluator.py` (266行) + `scott_parzen_estimator.py` (156行) | `PedAnovaEvaluator` | ⚠️ 差异见下 |
| get_param_importances | `__init__.py` (120行) | `get_param_importances` 函数 | ✅ 基本对齐 |

---

## 2. fANOVA 评估器 — 函数级对比

### 2.1 随机森林实现

| 方面 | Python | Rust | 影响 |
|------|--------|------|------|
| 随机森林引擎 | sklearn `RandomForestRegressor` (C++) | 自定义实现 (xorshift64 RNG) | 结果不完全一致 |
| 分裂策略 | sklearn BestSplitter (排序优化) | 遍历中点分裂 | 分裂点可能不同 |
| 特征选择 | sklearn max_features=sqrt | Fisher-Yates shuffle 取 sqrt | ✅ 对齐策略 |
| Bootstrap | sklearn _parallel_build_trees | 逐树 bootstrap 有放回 | ✅ 对齐 |
| min_samples_split/leaf | 2 / 1 | 2 / 1 | ✅ 对齐 |
| 不纯度度量 | MSE (variance) | variance_impurity() | ✅ 对齐 |

### 2.2 FanovaTree 方差分解

| 函数 | Python (_tree.py) | Rust (FanovaTree) | 状态 |
|------|-------------------|-------------------|------|
| `variance` (总方差) | 叶子 weighted_variance | `weighted_variance()` | ✅ 精确对齐 |
| `get_marginal_variance` | 笛卡尔积遍历 midpoints × sizes | 笛卡尔积 + 索引推进 | ✅ 对齐 |
| `_get_marginalized_statistics` | active_nodes 栈 + search_spaces 传播 | active_nodes + search_spaces | ✅ 对齐 |
| `_precompute_statistics` | 前向传播 ss + 后向汇总 | 前向 + 后向 | ✅ 对齐 |
| `_precompute_split_midpoints_and_sizes` | 收集阈值 + 加边界 + 中点/大小 | 相同逻辑 | ✅ 对齐 |
| `_precompute_subtree_active_features` | 后向或运算 | 后向或运算 | ✅ 对齐 |
| `_get_cardinality` | `np.prod(ss[:,1] - ss[:,0])` | `ss.iter().map(r[1]-r[0]).product()` | ✅ 对齐 |

**Golden 验证 (Group 2)**:
- Tree A (单特征2叶子): total_var=1.0, mv_feat0=1.0 — 边际方差=总方差 ✅
- Tree B (两特征3叶子): total_var=1.86, mv_feat0=1.44, mv_feat1=0.21 ✅
- Tree C (三特征4叶子): total_var=3.81, mv_feat0=2.25, mv_feat1=0.54, mv_feat2=0.24 ✅

### 2.3 差异 D1: 分类变量编码

| 编码方式 | Python | Rust |
|----------|--------|------|
| FanovaEvaluator | **One-hot** (通过 `_SearchSpaceTransform`) | **Ordinal** (choice index) |
| MeanDecreaseImpurityEvaluator | One-hot (通过 `_SearchSpaceTransform`) | **One-hot** ✅ |
| `column_to_encoded_columns` | ✅ 有映射回原参数 | FanovaEvaluator 无 / MDI 有 |

**影响**: Rust FanovaEvaluator 对分类变量使用 ordinal 编码而非 one-hot, 多选项分类变量的重要性评估可能不准确。MDI 已正确使用 one-hot。

---

## 3. MDI 评估器 — 函数级对比

| 函数 | Python | Rust | 状态 |
|------|--------|------|------|
| 特征重要性算法 | sklearn `feature_importances_` (MDI/Gini) | `extract_feature_importances` | ✅ 算法对齐 |
| 归一化 | per-tree normalize → average | per-tree normalize → average | ✅ 对齐 |
| One-hot 反变换 | `np.add.at(importances, encoded_column_to_column, ...)` | `importances[column_to_param[col]] += imp` | ✅ 对齐 |
| 不纯度加权 | `weighted_n_node_samples * impurity_decrease` | `best_impurity_decrease * n as f64` | ✅ 对齐 |

---

## 4. PED-ANOVA 评估器 — 函数级对比

### 4.1 _QuantileFilter

| 方面 | Python | Rust | 状态 |
|------|--------|------|------|
| 算法 | `np.partition` + cutoff | 完全排序 + cutoff | ✅ 结果等价 |
| quantile 索引 | `ceil(q * n) - 1` | `ceil(q * n) - 1` | ✅ 对齐 |
| min_n_top 保护 | `max(partition[min-1], quantile)` | `.max(min_n_top)` on index | ✅ 等价 |

**Golden 验证 (Group 4)**:
- minimize q=0.3: top=[1.0, 2.0, 3.0] ✅
- maximize q=0.3: top=[8.0, 9.0, 10.0] ✅
- minimize q=0.1 min=3: top=[1.0, 2.0, 3.0] ✅

### 4.2 ScottParzenEstimator

| 方面 | Python | Rust | 状态 |
|------|--------|------|------|
| 基类 | 继承 `_ParzenEstimator` (TPE框架) | 独立实现 | ⚠️ D4 |
| 带宽计算 | Scott's rule: `1.059 * min(IQR/1.34, σ) * n^{-0.2}` | 相同公式 | ✅ 对齐 |
| 最小带宽 | `σ_min = 0.5/1.64` | `0.5/1.64` | ✅ 对齐 |
| 先验核 | `μ=mean(domain), σ=domain_size` | `μ=mid, σ=domain_size` | ✅ 对齐 |
| PDF 计算 | **离散截断正态**: `Φ((x+0.5-μ)/σ) - Φ((x-0.5-μ)/σ)` | **连续截断正态**: `φ((x-μ)/σ) / σ / denom` | ⚠️ D4 |
| IQR Q75 索引 | `searchsorted(side="right")` | `position(c >= target)` = side="left" | ⚠️ D5 |

### 4.3 差异 D2: 离散化 Grid 范围

| 方面 | Python | Rust |
|------|--------|------|
| Grid 范围 | **Distribution bounds** (`dist.low`, `dist.high`) | **观测数据范围** (`min(values)`, `max(values)`) |
| Grid 映射 | `searchsorted(grids, params - step/2)` | 最近邻 |
| n_steps 自适应 | log: `min(log2_domain, n_steps)`, step: `min((h-l)/step+1, n_steps)` | 固定 50 |

**影响**: Python 使用完整搜索空间边界建 grid, Rust 使用观测到的数据范围。当参数空间大但采样集中时, 差异显著。

### 4.4 差异 D4: PDF 计算

Python 使用**离散截断正态** (概率质量 = 对连续密度在 [x-0.5, x+0.5] 上积分):
```python
log_pdf = log(Φ((x+0.5-μ)/σ) - Φ((x-0.5-μ)/σ)) - log(Φ((high+0.5-μ)/σ) - Φ((low-0.5-μ)/σ))
```

Rust 使用**连续截断正态** (在点 x 处的密度):
```rust
p = φ((x-μ)/σ) / σ / (Φ((high-μ)/σ) - Φ((low-μ)/σ))
```

**影响**: 对于小 σ 值, 离散版本会在整数点处给出更集中的概率; 连续版本则是逐点密度。差异在 σ >> 1 时可忽略, σ < 1 时显著。

### 4.5 Pearson 散度

| 方面 | Python | Rust | 状态 |
|------|--------|------|------|
| 公式 | `Σ q * ((p/q - 1)²)` | 相同 | ✅ 对齐 |
| eps 处理 | `+1e-12` 在 PDF 上 | `+1e-12` 在函数内 | ✅ 等价 |

**Golden 验证 (Group 5)**:
- identical: ~0.0 ✅
- peaked vs uniform: ~2.25 ✅
- bimodal vs uniform: ~0.675 ✅
- two peaked: ~4.13 ✅

### 4.6 差异 D7: 分位数过滤结构

| 方面 | Python | Rust |
|------|--------|------|
| target 过滤 | 从**全部试验**中过滤 | 从**区域试验**中过滤 (调整 quantile) |
| region 过滤 | 从**全部试验**中过滤 | 从**全部试验**中过滤 |
| quantile 调整 | 无 | `target_quantile / region_quantile` |

**影响**: 当 `region_quantile = 1.0` (默认) 时完全等价。`region_quantile < 1.0` 时可能有微小差异。

---

## 5. get_param_importances 顶层函数

| 方面 | Python | Rust | 状态 |
|------|--------|------|------|
| 默认评估器 | FanovaImportanceEvaluator | FanovaEvaluator | ✅ |
| 归一化 (sum=1) | `value / s` for each value | `*val /= total` | ✅ |
| 全零重要性 | `1.0 / n_params` (均分) | 保持全零 | ⚠️ D9 |
| 参数交集 | `intersection_search_space` | 手动 HashSet 交集 | ✅ 对齐 |
| 目标值过滤 | 过滤 NaN/Inf | `val.is_finite()` | ✅ 对齐 |
| 最少试验数 | ≥ 2 | ≥ 2 | ✅ 对齐 |

### 差异 D9: 全零归一化

Python: `if s == 0.0: return {param: 1.0/n_params for param in res}`  
Rust: `if total > 0.0 { normalize }` (全零时不做归一化)

**影响**: 仅在所有参数重要性都为 0 时触发 (极端边界情况)。

---

## 6. 已知差异汇总

| ID | 差异描述 | 严重度 | 影响 |
|----|----------|--------|------|
| D1 | FanovaEvaluator 分类变量: ordinal vs one-hot | 中 | 多选项分类参数重要性不准确 |
| D2 | PED-ANOVA Grid 范围: 观测范围 vs 分布边界 | 中 | 参数空间大但采样集中时差异大 |
| D4 | Scott-Parzen PDF: 连续 vs 离散截断正态 | 中 | σ < 1 时差异明显 |
| D5 | IQR Q75: searchsorted side="left" vs "right" | 低 | 微小带宽差异 |
| D7 | 分位数过滤结构: 嵌套 vs 独立 | 低 | 仅 region_quantile < 1.0 时 |
| D9 | 全零归一化: 均分 vs 保持零 | 极低 | 极端边界情况 |

---

## 7. 交叉验证测试结果

| 测试 | 描述 | 结果 |
|------|------|------|
| test_fanova_importance_ordering_x_dominant | f(x,y)=x²+0.001y → x > y | ✅ PASS |
| test_fanova_three_param_ordering | f(x,y,z)=10x²+y²+0.001z → x首位 | ✅ PASS |
| test_mdi_importance_ordering_x_dominant | MDI: f(x,y)=x²+0.001y | ✅ PASS |
| test_mdi_three_param_ordering | MDI: f(x,y,z)=10x²+y²+0.001z | ✅ PASS |
| test_ped_anova_importance_ordering_x_dominant | PED-ANOVA: f(x,y)=x²+0.001y | ✅ PASS |
| test_ped_anova_three_param_ordering | PED-ANOVA: f(x,y,z)=10x²+y²+0.001z | ✅ PASS |
| test_importance_normalization_sum_to_one | 三种评估器归一化 sum=1.0 | ✅ PASS |
| test_importance_no_normalization | 未归一化非负性 | ✅ PASS |
| test_all_evaluators_agree_on_dominant_param | 三种评估器一致性 | ✅ PASS |
| test_importance_empty_study_error | 空研究错误 | ✅ PASS |
| test_importance_single_trial_error | 单试验错误 | ✅ PASS |
| test_importance_param_subset | 参数子集 | ✅ PASS |
| test_fanova_reproducible_with_same_seed | fANOVA 种子可复现 | ✅ PASS |
| test_mdi_reproducible_with_same_seed | MDI 种子可复现 | ✅ PASS |
| test_fanova_single_important_param_near_one | f=x²: x重要性>0.85 | ✅ PASS |
| test_mdi_single_important_param_near_one | MDI f=x²: x>0.85 | ✅ PASS |
| test_ped_anova_uniform_noise_equal_importance | 常数目标: 参数差<0.6 | ✅ PASS |
| test_fanova_maximize_direction | 最大化方向 | ✅ PASS |
| test_importance_custom_target | 自定义 target 函数 | ✅ PASS |
| test_linear_function_ordering_all_evaluators | 线性函数三评估器 | ✅ PASS |
| test_quadratic_function_ordering | 二次函数排序 | ✅ PASS |
| test_importance_with_int_params | 整数参数支持 | ✅ PASS |

**结果: 22/22 测试全部通过** ✅

---

## 8. 结论

Rust importance 模块整体上**算法层面与 Python 对齐**。三种评估器 (fANOVA, MDI, PED-ANOVA) 在重要性排序判断上与 Python 一致。

主要差异集中在:
1. **底层随机森林实现不同** (sklearn C++ vs 自定义 Rust) — 导致精确数值不匹配, 但排序一致
2. **FanovaEvaluator 分类变量编码** (D1) — 建议添加 one-hot 编码
3. **PED-ANOVA 的 PDF 计算** (D4) — 连续 vs 离散截断正态, 对小 σ 有影响
4. **PED-ANOVA Grid 范围** (D2) — 观测范围 vs 分布边界

这些差异在实际使用中影响有限, 因为重要性评估主要用于**排序**而非精确数值比较。
