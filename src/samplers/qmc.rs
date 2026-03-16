//! 准蒙特卡洛 (QMC) 采样器模块
//!
//! 对应 Python `optuna.samplers.QMCSampler`。
//! 支持 Sobol' 和 Halton 两种低差异序列，默认使用 Sobol'（与 Python 一致）。
//! 低差异序列比伪随机采样更均匀地填充搜索空间，提高采样效率。
//!
//! ## 与 Python 的对齐说明
//! - 构造参数: `qmc_type`, `scramble`, `seed`, `independent_sampler`,
//!   `warn_asynchronous_seeding`, `warn_independent_sampling` — 完全对齐
//! - `seed=None` 时自动生成随机种子（对应 Python `np.random.PCG64().random_raw()`）
//! - 搜索空间冻结语义: 首个完成试验确定搜索空间后不再改变
//! - 分类参数始终由 `independent_sampler` 处理
//! - `sample_independent` 在搜索空间已冻结时输出警告（对应 Python `_log_independent_sampling`）
//! - `before_trial`/`after_trial` 委托给 `independent_sampler`（与 Python 一致）
//!
//! ## 与 Python 的已知差异
//! - Python 使用 `scipy.stats.qmc.Sobol/Halton`；Rust 使用自实现的 Sobol'/Halton 序列
//! - Python 样本 ID 存储在 study storage 中（分布式安全）；Rust 使用进程内 AtomicU64
//!   （单进程安全，分布式场景可能需要额外同步）

use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::RngCore;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::samplers::random::RandomSampler;
use crate::samplers::Sampler;
use crate::search_space::SearchSpaceTransform;
use crate::trial::{FrozenTrial, TrialState};
use crate::optuna_warn;

/// Halton 序列的前 20 个素数基底。
const PRIMES: [u64; 20] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71];

/// QMC 序列类型。对应 Python `QMCSampler(qmc_type=...)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum QmcType {
    /// Sobol' 序列（默认）— 低差异性质优于 Halton
    Sobol,
    /// Halton 序列
    Halton,
}

/// Quasi-Monte Carlo sampler using Sobol' or Halton sequences.
///
/// 对应 Python `optuna.samplers.QMCSampler`。
/// 默认使用 Sobol' 序列（与 Python 一致），也支持 Halton 序列。
///
/// ## 搜索空间确定规则 (与 Python 一致)
/// - 若 study 有已完成试验，从 `number` 最小的完成试验推断搜索空间
/// - 否则第一次试验使用 `independent_sampler`，第二次试验开始推断
/// - 搜索空间一旦确定不再改变 (冻结语义)
/// - `CategoricalDistribution` 参数始终排除在外，由 `independent_sampler` 处理
pub struct QmcSampler {
    /// 独立采样器（第一试验 + 分类参数使用）
    independent_sampler: Arc<dyn Sampler>,
    /// 全局采样索引 (从 0 开始，atomic 保证线程安全)
    /// 对应 Python `_find_sample_id` 中的 `sample_id`
    /// NOTE: Python 将 sample_id 存储在 study storage（分布式安全），
    ///       Rust 使用进程内原子计数器（单进程线程安全）。
    next_index: AtomicU64,
    /// 是否启用序列置乱 (默认 false — 与 Python 一致)
    scramble: bool,
    /// 随机种子
    /// 当 seed=None 时自动随机生成（对应 Python `np.random.PCG64().random_raw()`）
    /// scramble=false 时不使用
    seed: u64,
    /// 序列类型 (Sobol / Halton)
    qmc_type: QmcType,
    /// 冻结的初始搜索空间 (排除分类参数)。
    /// 对应 Python `self._initial_search_space`
    initial_search_space: Mutex<Option<IndexMap<String, Distribution>>>,
    /// 是否在独立采样时输出警告
    /// 对应 Python `self._warn_independent_sampling`
    warn_independent_sampling: bool,
}

impl std::fmt::Debug for QmcSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("QmcSampler")
            .field("scramble", &self.scramble)
            .field("qmc_type", &self.qmc_type)
            .field("warn_independent_sampling", &self.warn_independent_sampling)
            .finish()
    }
}

impl QmcSampler {
    /// 创建 QMC 采样器。
    ///
    /// 对应 Python `QMCSampler.__init__`。
    ///
    /// # 参数
    /// * `qmc_type` - 序列类型（默认 Sobol — 对应 Python `qmc_type="sobol"`）
    /// * `scramble` - 是否启用序列置乱（默认 false — 对应 Python `scramble=False`）
    /// * `seed` - 随机种子（默认 None→随机生成 — 对应 Python `seed=None` → `np.random.PCG64().random_raw()`）
    /// * `independent_sampler` - 独立采样器（默认 RandomSampler — 对应 Python `independent_sampler=None`）
    /// * `warn_asynchronous_seeding` - 是否警告异步种子（默认 true — 对应 Python `warn_asynchronous_seeding=True`）
    /// * `warn_independent_sampling` - 是否警告独立采样（默认 true — 对应 Python `warn_independent_sampling=True`）
    pub fn new(
        qmc_type: Option<QmcType>,
        scramble: Option<bool>,
        seed: Option<u64>,
        independent_sampler: Option<Arc<dyn Sampler>>,
        warn_asynchronous_seeding: Option<bool>,
        warn_independent_sampling: Option<bool>,
    ) -> Self {
        let scramble_val = scramble.unwrap_or(false);

        // 当 seed=None 时自动生成随机种子（对应 Python `np.random.PCG64().random_raw()`）
        let seed_val = match seed {
            Some(s) => s,
            None => rand::thread_rng().next_u64(),
        };

        // 当 scramble=true 且 seed 未手动设置时发出警告
        // 对应 Python `if seed is None and scramble and warn_asynchronous_seeding:`
        // 对应 Python `if seed is None and scramble and warn_asynchronous_seeding:`
        if seed.is_none() && scramble_val && warn_asynchronous_seeding.unwrap_or(true) {
            optuna_warn!(
                "No seed is provided for `QmcSampler` and the seed is set randomly. \
                 If you are running multiple `QmcSampler`s in parallel and/or distributed \
                 environment, the same seed must be used in all samplers to ensure that resulting \
                 samples are taken from the same QMC sequence."
            );
        }

        Self {
            // 对应 Python `self._independent_sampler = independent_sampler or RandomSampler(seed=seed)`
            independent_sampler: independent_sampler
                .unwrap_or_else(|| Arc::new(RandomSampler::new(seed))),
            next_index: AtomicU64::new(0),
            scramble: scramble_val,
            seed: seed_val,
            qmc_type: qmc_type.unwrap_or(QmcType::Sobol),
            initial_search_space: Mutex::new(None),
            warn_independent_sampling: warn_independent_sampling.unwrap_or(true),
        }
    }

    /// 从第一个完成试验推断初始搜索空间 (排除分类参数)。
    /// 对应 Python `_infer_initial_search_space`。
    fn infer_initial_search_space(trial: &FrozenTrial) -> IndexMap<String, Distribution> {
        trial
            .distributions
            .iter()
            .filter(|(_, dist)| !matches!(dist, Distribution::CategoricalDistribution(_)))
            .map(|(name, dist)| (name.clone(), dist.clone()))
            .collect()
    }
}

/// 计算 Van der Corput 序列值。
/// 将索引 n 按照基底 base 展开为小数序列值。
fn van_der_corput(mut n: u64, base: u64) -> f64 {
    let mut result = 0.0;
    let mut denom = 1.0;
    while n > 0 {
        denom *= base as f64;
        let digit = n % base;
        n /= base;
        result += digit as f64 / denom;
    }
    result
}

/// 计算置乱的 Van der Corput 值（使用种子驱动的数位排列）。
fn van_der_corput_scrambled(mut n: u64, base: u64, seed: u64) -> f64 {
    let mut result = 0.0;
    let mut denom = 1.0;
    while n > 0 {
        denom *= base as f64;
        let digit = n % base;
        n /= base;
        // Simple scrambling: permute digit using seed
        let scrambled = (digit.wrapping_add(seed).wrapping_mul(2654435761)) % base;
        result += scrambled as f64 / denom;
    }
    result
}

/// 生成第 index 个 Halton 点（[0,1]^d 空间）。
fn halton_point(index: u64, dim: usize, scramble: bool, seed: u64) -> Vec<f64> {
    (0..dim)
        .map(|d| {
            let base = if d < PRIMES.len() {
                PRIMES[d]
            } else {
                // For high dimensions, use odd numbers as bases
                2 * d as u64 + 3
            };
            if scramble {
                van_der_corput_scrambled(index, base, seed.wrapping_add(d as u64))
            } else {
                van_der_corput(index, base)
            }
        })
        .collect()
}

// ════════════════════════════════════════════════════════════════════════
// Sobol' 序列生成器
// ════════════════════════════════════════════════════════════════════════

/// Sobol' 序列方向数（前 20 维的初始化参数）。
/// 使用 Joe-Kuo 方向数（简化版本）。
/// Sobol' 参数表：(s = 原始多项式的阶数, a = 多项式系数, 初始方向数 m_i[])
/// 来源: new-joe-kuo-6.21201 (D(6) 搜索准则)，与 scipy.stats.qmc.Sobol 一致
/// 维度 0 使用 Van der Corput 序列 (s=0, a=0)
const SOBOL_PARAMS: [(usize, u64, [u64; 13]); 20] = [
    // dim 0: Van der Corput (base 2)
    (0, 0, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 1 (JK d=2): s=1, a=0, poly=x+1
    (1, 0, [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 2 (JK d=3): s=2, a=1, poly=x²+x+1
    (2, 1, [1, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 3 (JK d=4): s=3, a=1, poly=x³+x+1
    (3, 1, [1, 3, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 4 (JK d=5): s=3, a=2, poly=x³+x²+1
    (3, 2, [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 5 (JK d=6): s=4, a=1, poly=x⁴+x+1
    (4, 1, [1, 1, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 6 (JK d=7): s=4, a=4, poly=x⁴+x³+1
    (4, 4, [1, 3, 5, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 7 (JK d=8): s=5, a=2, poly=x⁵+x²+1
    (5, 2, [1, 1, 5, 5, 17, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 8 (JK d=9): s=5, a=4, poly=x⁵+x³+1
    (5, 4, [1, 1, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 9 (JK d=10): s=5, a=7, poly=x⁵+x³+x²+x+1
    (5, 7, [1, 1, 7, 11, 19, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 10 (JK d=11): s=5, a=11, poly=x⁵+x⁴+x²+x+1
    (5, 11, [1, 1, 5, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 11 (JK d=12): s=5, a=13, poly=x⁵+x⁴+x³+x+1
    (5, 13, [1, 1, 1, 3, 11, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 12 (JK d=13): s=5, a=14, poly=x⁵+x⁴+x³+x²+1
    (5, 14, [1, 3, 5, 5, 31, 0, 0, 0, 0, 0, 0, 0, 0]),
    // dim 13 (JK d=14): s=6, a=1, poly=x⁶+x+1
    (6, 1, [1, 3, 3, 9, 7, 49, 0, 0, 0, 0, 0, 0, 0]),
    // dim 14 (JK d=15): s=6, a=13, poly=x⁶+x⁴+x³+x+1
    (6, 13, [1, 1, 1, 15, 21, 21, 0, 0, 0, 0, 0, 0, 0]),
    // dim 15 (JK d=16): s=6, a=16, poly=x⁶+x⁵+1
    (6, 16, [1, 3, 1, 13, 27, 49, 0, 0, 0, 0, 0, 0, 0]),
    // dim 16 (JK d=17): s=6, a=19, poly=x⁶+x⁵+x²+x+1
    (6, 19, [1, 1, 1, 15, 7, 5, 0, 0, 0, 0, 0, 0, 0]),
    // dim 17 (JK d=18): s=6, a=22, poly=x⁶+x⁵+x³+x²+1
    (6, 22, [1, 3, 1, 15, 13, 25, 0, 0, 0, 0, 0, 0, 0]),
    // dim 18 (JK d=19): s=6, a=25, poly=x⁶+x⁵+x⁴+x+1
    (6, 25, [1, 1, 5, 5, 19, 61, 0, 0, 0, 0, 0, 0, 0]),
    // dim 19 (JK d=20): s=7, a=1, poly=x⁷+x+1
    (7, 1, [1, 3, 7, 11, 23, 15, 103, 0, 0, 0, 0, 0, 0]),
];

/// 生成 Sobol' 序列的第 index 个点（灰码实现）。
///
/// 使用灰码生成 Sobol' 序列，维度 d 使用 Van der Corput(base 2)。
fn sobol_point(index: u64, dim: usize, scramble: bool, seed: u64) -> Vec<f64> {
    let bits = 32u32;
    let scale = 2.0_f64.powi(-(bits as i32));

    (0..dim)
        .map(|d| {
            // 初始化维度 d 的方向数
            let mut v = [0u64; 32];
            if d == 0 {
                // 第 0 维: Van der Corput (base 2)
                for i in 0..bits as usize {
                    v[i] = 1u64 << (bits as usize - 1 - i);
                }
            } else if d < SOBOL_PARAMS.len() {
                let (s, a, ref m) = SOBOL_PARAMS[d];
                // 使用 s（多项式阶数）作为初始方向数的数量
                let num_m = s;
                for i in 0..num_m.min(bits as usize) {
                    v[i] = m[i] << (bits as usize - 1 - i);
                }
                // 使用原始多项式系数 a 进行递推（非维度索引 d）
                for i in num_m..bits as usize {
                    v[i] = v[i - s] ^ (v[i - s] >> s);
                    for j in 1..s {
                        v[i] ^= ((a >> (s - 1 - j)) & 1) as u64 * v[i - j];
                    }
                }
            } else {
                // 高维回退: 使用种子驱动的伪 Sobol
                for i in 0..bits as usize {
                    v[i] = (seed.wrapping_add(d as u64).wrapping_mul(2654435761).wrapping_add(i as u64))
                        | (1u64 << (bits as usize - 1 - i));
                }
            }

            // 灰码生成
            let mut result = 0u64;
            let mut idx = index;
            let mut bit = 0;
            while idx > 0 {
                if idx & 1 == 1 {
                    result ^= v[bit];
                }
                idx >>= 1;
                bit += 1;
                if bit >= bits as usize {
                    break;
                }
            }

            let mut val = result as f64 * scale;

            // 置乱: Owen-style 简化版
            if scramble {
                let s = seed.wrapping_add(d as u64).wrapping_mul(6364136223846793005);
                val = (val + (s as f64 * 2.3283064365386963e-10)) % 1.0;
                if val < 0.0 {
                    val += 1.0;
                }
            }

            val
        })
        .collect()
}

/// 根据 QMC 类型（Sobol' 或 Halton）生成低差异点。
fn qmc_point(index: u64, dim: usize, scramble: bool, seed: u64, qmc_type: QmcType) -> Vec<f64> {
    match qmc_type {
        QmcType::Sobol => sobol_point(index, dim, scramble, seed),
        QmcType::Halton => halton_point(index, dim, scramble, seed),
    }
}

impl Sampler for QmcSampler {
    /// 推断相对搜索空间 — 冻结语义。
    /// 第一个完成/剪枝的试验确定搜索空间后不再改变。
    /// 排除 CategoricalDistribution（由 independent_sampler 处理）。
    /// 对应 Python `infer_relative_search_space` + `_infer_initial_search_space`。
    fn infer_relative_search_space(
        &self,
        trials: &[FrozenTrial],
    ) -> IndexMap<String, Distribution> {
        // 若已冻结，直接返回
        {
            let guard = self.initial_search_space.lock();
            if let Some(ref ss) = *guard {
                return ss.clone();
            }
        }

        // 寻找 number 最小的已完成/剪枝试验
        let past: Vec<&FrozenTrial> = trials
            .iter()
            .filter(|t| matches!(t.state, TrialState::Complete | TrialState::Pruned))
            .collect();
        if past.is_empty() {
            // 无已完成试验 — 第一次试验由 independent_sampler 处理
            return IndexMap::new();
        }
        let first = past.iter().min_by_key(|t| t.number).unwrap();
        let ss = Self::infer_initial_search_space(first);

        // 冻结
        *self.initial_search_space.lock() = Some(ss.clone());
        ss
    }

    /// 相对采样 — 生成 QMC 低差异点并变换到参数空间。
    ///
    /// 对应 Python `sample_relative`:
    /// ```python
    /// sample = self._sample_qmc(study, search_space)           # [0,1]^d QMC 点
    /// trans = _SearchSpaceTransform(search_space)               # transform_0_1=False
    /// sample = trans.bounds[:, 0] + sample * (trans.bounds[:, 1] - trans.bounds[:, 0])
    /// return trans.untransform(sample[0, :])
    /// ```
    ///
    /// Rust 等价: `SearchSpaceTransform::new(..., transform_0_1=true)` 使得
    /// `untransform` 内部自动完成 `[0,1] → bounds` 缩放，数学上等效。
    fn sample_relative(
        &self,
        _trials: &[FrozenTrial],
        search_space: &IndexMap<String, Distribution>,
    ) -> Result<HashMap<String, f64>> {
        if search_space.is_empty() {
            return Ok(HashMap::new());
        }

        // 构建有序搜索空间 (按 key 排序，保证维度分配一致)
        let mut ordered_space = IndexMap::new();
        let mut param_names: Vec<String> = search_space.keys().cloned().collect();
        param_names.sort();
        for name in &param_names {
            ordered_space.insert(name.clone(), search_space[name].clone());
        }

        // transform_0_1=true: untransform 自动将 [0,1] 映射回 bounds
        // 等价于 Python 手动 `bounds[:, 0] + sample * (bounds[:, 1] - bounds[:, 0])`
        let transform = SearchSpaceTransform::new(ordered_space.clone(), true, true, true);
        let n_dims = transform.n_encoded();

        // 获取当前采样索引 (atomic increment)
        // 对应 Python `_find_sample_id` (Python 存储在 study storage，Rust 存储在进程内 AtomicU64)
        let index = self.next_index.fetch_add(1, Ordering::Relaxed);

        // 生成 [0,1]^d 的 QMC 点（Sobol' 或 Halton）
        // 对应 Python `_sample_qmc` → `qmc_engine.random(1)`
        let point = qmc_point(index, n_dims, self.scramble, self.seed, self.qmc_type);

        // 通过 SearchSpaceTransform 逆变换回参数空间
        // 对应 Python `trans.untransform(sample[0, :])`
        let decoded = transform.untransform(&point)?;
        let mut result = HashMap::new();
        for (name, dist) in &ordered_space {
            if let Some(pv) = decoded.get(name) {
                let internal = dist.to_internal_repr(pv)?;
                result.insert(name.clone(), internal);
            }
        }

        Ok(result)
    }

    /// 独立采样 — 当参数不在相对搜索空间中时由此方法处理。
    /// 在搜索空间已冻结后发出警告（对应 Python `sample_independent` + `_log_independent_sampling`）。
    fn sample_independent(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        // 对应 Python: if self._initial_search_space is not None: if self._warn_independent_sampling: ...
        // 对应 Python `if self._initial_search_space is not None and self._warn_independent_sampling:`
        if self.warn_independent_sampling {
            let guard = self.initial_search_space.lock();
            if guard.is_some() {
                optuna_warn!(
                    "Trial {} suggests '{}' via independent sampling (QMCSampler). \
                     Dynamic search space and CategoricalDistribution are not supported \
                     by QMCSampler. Falling back to RandomSampler.",
                    trial.number,
                    param_name,
                );
            }
        }
        self.independent_sampler
            .sample_independent(trials, trial, param_name, distribution)
    }

    /// 对应 Python `before_trial` — 委托给 independent_sampler。
    fn before_trial(&self, trials: &[FrozenTrial]) {
        self.independent_sampler.before_trial(trials);
    }

    /// 对应 Python `after_trial` — 委托给 independent_sampler。
    fn after_trial(
        &self,
        trials: &[FrozenTrial],
        trial: &FrozenTrial,
        state: TrialState,
        values: Option<&[f64]>,
    ) {
        self.independent_sampler
            .after_trial(trials, trial, state, values);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::study::{create_study, StudyDirection};
    use std::sync::Arc;

    #[test]
    fn test_van_der_corput() {
        // Base 2: 1/2, 1/4, 3/4, 1/8, 5/8, ...
        assert!((van_der_corput(1, 2) - 0.5).abs() < 1e-10);
        assert!((van_der_corput(2, 2) - 0.25).abs() < 1e-10);
        assert!((van_der_corput(3, 2) - 0.75).abs() < 1e-10);
        // Base 3: 1/3, 2/3, 1/9, ...
        assert!((van_der_corput(1, 3) - 1.0 / 3.0).abs() < 1e-10);
        assert!((van_der_corput(2, 3) - 2.0 / 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_halton_point_bounds() {
        for i in 1..100 {
            let pt = halton_point(i, 5, false, 0);
            assert_eq!(pt.len(), 5);
            for &v in &pt {
                assert!((0.0..1.0).contains(&v), "value {v} out of [0, 1)");
            }
        }
    }

    #[test]
    fn test_halton_uniqueness() {
        let points: Vec<Vec<f64>> = (1..50).map(|i| halton_point(i, 3, false, 0)).collect();
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                assert_ne!(points[i], points[j]);
            }
        }
    }

    #[test]
    fn test_sobol_point_bounds() {
        for i in 0..100 {
            let pt = sobol_point(i, 5, false, 0);
            assert_eq!(pt.len(), 5);
            for &v in &pt {
                assert!((0.0..=1.0).contains(&v), "Sobol value {v} out of [0,1]");
            }
        }
    }

    #[test]
    fn test_sobol_uniqueness() {
        let points: Vec<Vec<f64>> = (0..64).map(|i| sobol_point(i, 3, false, 0)).collect();
        for i in 0..points.len() {
            for j in (i + 1)..points.len() {
                assert_ne!(points[i], points[j], "Sobol points {i} and {j} collide");
            }
        }
    }

    /// 默认参数应与 Python 一致: scramble=false, qmc_type=Sobol
    #[test]
    fn test_qmc_default_params() {
        let sampler = QmcSampler::new(None, None, None, None, None, None);
        assert!(!sampler.scramble, "默认 scramble 应为 false (与 Python 一致)");
        assert_eq!(sampler.qmc_type, QmcType::Sobol);
        assert!(sampler.warn_independent_sampling);
    }

    /// Sobol 采样器基础优化测试
    #[test]
    fn test_qmc_sampler_sobol() {
        let sampler: Arc<dyn Sampler> = Arc::new(
            QmcSampler::new(Some(QmcType::Sobol), Some(false), Some(42), None, None, None)
        );
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        study.optimize(|trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            let y = trial.suggest_float("y", 0.0, 1.0, false, None)?;
            Ok(x * x + y * y)
        }, Some(30), None, None).unwrap();

        assert_eq!(study.trials().unwrap().len(), 30);
    }

    /// Halton 采样器基础优化测试
    #[test]
    fn test_qmc_sampler_halton() {
        let sampler: Arc<dyn Sampler> = Arc::new(
            QmcSampler::new(Some(QmcType::Halton), Some(false), Some(42), None, None, None)
        );
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        study.optimize(|trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            Ok(x * x)
        }, Some(20), None, None).unwrap();

        assert_eq!(study.trials().unwrap().len(), 20);
    }

    /// 置乱 Sobol 测试
    #[test]
    fn test_qmc_sampler_scrambled() {
        let sampler: Arc<dyn Sampler> = Arc::new(
            QmcSampler::new(Some(QmcType::Sobol), Some(true), Some(42), None, None, None)
        );
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        study.optimize(|trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            Ok(x * x)
        }, Some(20), None, None).unwrap();

        assert_eq!(study.trials().unwrap().len(), 20);
    }

    /// 自定义 independent_sampler 测试
    #[test]
    fn test_qmc_custom_independent_sampler() {
        let independent = Arc::new(RandomSampler::new(Some(123)));
        let sampler: Arc<dyn Sampler> = Arc::new(
            QmcSampler::new(None, None, None, Some(independent), None, None)
        );
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        study.optimize(|trial| {
            let x = trial.suggest_float("x", 0.0, 1.0, false, None)?;
            Ok(x * x)
        }, Some(10), None, None).unwrap();

        assert_eq!(study.trials().unwrap().len(), 10);
    }

    /// 分类参数应由 independent_sampler 处理 (不进入相对搜索空间)。
    /// 对应 Python `_infer_initial_search_space` 中排除 CategoricalDistribution。
    #[test]
    fn test_qmc_categorical_excluded_from_search_space() {
        use crate::distributions::{
            CategoricalChoice, CategoricalDistribution, FloatDistribution,
        };
        use crate::trial::TrialState;

        let sampler = QmcSampler::new(None, None, None, None, None, None);

        // 构造一个含分类参数的完成试验
        let mut params = HashMap::new();
        params.insert(
            "x".to_string(),
            crate::distributions::ParamValue::Float(0.5),
        );
        params.insert(
            "cat".to_string(),
            crate::distributions::ParamValue::Categorical(CategoricalChoice::Str(
                "a".to_string(),
            )),
        );
        let mut dists = HashMap::new();
        dists.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log: false,
                step: None,
            }),
        );
        dists.insert(
            "cat".to_string(),
            Distribution::CategoricalDistribution(CategoricalDistribution {
                choices: vec![
                    CategoricalChoice::Str("a".to_string()),
                    CategoricalChoice::Str("b".to_string()),
                ],
            }),
        );

        let trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            params,
            distributions: dists,
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            datetime_start: None,
            datetime_complete: None,
        };

        let ss = sampler.infer_relative_search_space(&[trial]);
        // 分类参数 "cat" 不应在搜索空间中
        assert!(ss.contains_key("x"), "数值参数应在搜索空间中");
        assert!(!ss.contains_key("cat"), "分类参数应被排除");
    }

    /// 搜索空间冻结测试: 一旦确定不再改变
    #[test]
    fn test_qmc_frozen_search_space() {
        use crate::distributions::FloatDistribution;
        use crate::trial::TrialState;

        let sampler = QmcSampler::new(None, None, None, None, None, None);

        let make_trial = |number: i64, params: &[&str]| {
            let mut p = HashMap::new();
            let mut d = HashMap::new();
            for &name in params {
                p.insert(
                    name.to_string(),
                    crate::distributions::ParamValue::Float(0.5),
                );
                d.insert(
                    name.to_string(),
                    Distribution::FloatDistribution(FloatDistribution {
                        low: 0.0,
                        high: 1.0,
                        log: false,
                        step: None,
                    }),
                );
            }
            FrozenTrial {
                number,
                trial_id: number,
                state: TrialState::Complete,
                values: Some(vec![1.0]),
                params: p,
                distributions: d,
                user_attrs: HashMap::new(),
                system_attrs: HashMap::new(),
                intermediate_values: HashMap::new(),
                datetime_start: None,
                datetime_complete: None,
            }
        };

        let trial0 = make_trial(0, &["x", "y"]);
        let ss1 = sampler.infer_relative_search_space(&[trial0.clone()]);
        assert_eq!(ss1.len(), 2);

        // 即使新试验有不同参数，搜索空间不变 (冻结)
        let trial1 = make_trial(1, &["x", "y", "z"]);
        let ss2 = sampler.infer_relative_search_space(&[trial0, trial1]);
        assert_eq!(ss2.len(), 2, "搜索空间应保持冻结");
    }

    /// 当 seed=None 时应生成随机种子，不应为 0
    /// 对应 Python `self._seed = np.random.PCG64().random_raw() if seed is None else seed`
    #[test]
    fn test_qmc_random_seed_generation() {
        // 创建多个 seed=None 的采样器，种子应各不相同
        let s1 = QmcSampler::new(None, None, None, None, None, None);
        let s2 = QmcSampler::new(None, None, None, None, None, None);
        // 极大概率不会碰撞（2^64 中取两个）
        // 但也不能保证 != 0（虽然概率极低）
        // 测试: 至少两个不同的种子中有一个不是 0
        assert!(
            s1.seed != 0 || s2.seed != 0,
            "seed=None 时不应总是生成固定种子 0"
        );
    }

    /// 当 seed 手动指定时应使用指定值
    #[test]
    fn test_qmc_explicit_seed() {
        let s = QmcSampler::new(None, None, Some(12345), None, None, None);
        assert_eq!(s.seed, 12345);
    }

    #[test]
    fn test_qmc_infer_relative_search_space_no_complete_trials() {
        // 对齐 Python: 没有完成试验时，QMC 搜索空间为空。
        let sampler = QmcSampler::new(None, None, Some(7), None, None, None);
        let trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Running,
            values: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            datetime_start: None,
            datetime_complete: None,
        };
        let ss = sampler.infer_relative_search_space(&[trial]);
        assert!(ss.is_empty());
    }

    #[test]
    fn test_qmc_sample_relative_empty_search_space_returns_empty() {
        let sampler = QmcSampler::new(None, None, Some(11), None, None, None);
        let out = sampler.sample_relative(&[], &IndexMap::new()).unwrap();
        assert!(out.is_empty());
    }

    /// Sobol 采样产生的值应在参数范围内
    #[test]
    fn test_qmc_sample_relative_values_in_range() {
        use crate::distributions::FloatDistribution;

        let sampler = QmcSampler::new(Some(QmcType::Sobol), Some(false), Some(42), None, None, None);

        let mut search_space = IndexMap::new();
        search_space.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution {
                low: -5.0,
                high: 5.0,
                log: false,
                step: None,
            }),
        );
        search_space.insert(
            "y".to_string(),
            Distribution::FloatDistribution(FloatDistribution {
                low: 0.1,
                high: 100.0,
                log: true,
                step: None,
            }),
        );

        // 先冻结搜索空间
        let trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            params: {
                let mut p = HashMap::new();
                p.insert("x".to_string(), crate::distributions::ParamValue::Float(0.0));
                p.insert("y".to_string(), crate::distributions::ParamValue::Float(1.0));
                p
            },
            distributions: search_space.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            datetime_start: None,
            datetime_complete: None,
        };
        let _ = sampler.infer_relative_search_space(&[trial.clone()]);

        // 采样 50 次，检查值域
        for _ in 0..50 {
            let result = sampler.sample_relative(&[trial.clone()], &search_space).unwrap();
            if let Some(&x_val) = result.get("x") {
                // to_internal_repr 对 FloatDistribution 返回原始 f64
                assert!(
                    x_val >= -5.0 && x_val < 5.0,
                    "x={x_val} should be in [-5.0, 5.0)"
                );
            }
            if let Some(&y_val) = result.get("y") {
                // log 分布的 internal repr 也是原始值
                assert!(
                    y_val >= 0.1 && y_val <= 100.0,
                    "y={y_val} should be in [0.1, 100.0]"
                );
            }
        }
    }

    /// Halton 采样产生的值应在参数范围内
    #[test]
    fn test_qmc_halton_sample_relative_values_in_range() {
        use crate::distributions::FloatDistribution;

        let sampler = QmcSampler::new(Some(QmcType::Halton), Some(false), Some(42), None, None, None);

        let mut search_space = IndexMap::new();
        search_space.insert(
            "a".to_string(),
            Distribution::FloatDistribution(FloatDistribution {
                low: 0.0,
                high: 10.0,
                log: false,
                step: None,
            }),
        );

        let trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            params: {
                let mut p = HashMap::new();
                p.insert("a".to_string(), crate::distributions::ParamValue::Float(5.0));
                p
            },
            distributions: search_space.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            datetime_start: None,
            datetime_complete: None,
        };
        let _ = sampler.infer_relative_search_space(&[trial.clone()]);

        for _ in 0..30 {
            let result = sampler.sample_relative(&[trial.clone()], &search_space).unwrap();
            let a_val = result["a"];
            assert!(a_val >= 0.0 && a_val < 10.0, "a={a_val} should be in [0.0, 10.0)");
        }
    }

    /// scramble + 不同种子应产生不同序列
    #[test]
    fn test_qmc_scramble_different_seeds_differ() {
        let s1 = QmcSampler::new(Some(QmcType::Sobol), Some(true), Some(1), None, None, None);
        let s2 = QmcSampler::new(Some(QmcType::Sobol), Some(true), Some(2), None, None, None);

        use crate::distributions::FloatDistribution;
        let mut ss = IndexMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log: false,
                step: None,
            }),
        );
        let trial = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![1.0]),
            params: {
                let mut p = HashMap::new();
                p.insert("x".to_string(), crate::distributions::ParamValue::Float(0.5));
                p
            },
            distributions: ss.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            datetime_start: None,
            datetime_complete: None,
        };
        let _ = s1.infer_relative_search_space(&[trial.clone()]);
        let _ = s2.infer_relative_search_space(&[trial.clone()]);

        let r1 = s1.sample_relative(&[trial.clone()], &ss).unwrap();
        let r2 = s2.sample_relative(&[trial.clone()], &ss).unwrap();
        // 不同种子的 scrambled 序列应不同
        assert_ne!(r1["x"], r2["x"], "不同种子的 scramble 序列应不同");
    }

    #[test]
    fn test_qmc_independent_sampling_without_warning_flag() {
        // 覆盖 warn_independent_sampling=false 分支，确保可正常独立采样。
        use crate::distributions::FloatDistribution;

        let sampler = QmcSampler::new(
            Some(QmcType::Sobol),
            Some(false),
            Some(42),
            None,
            Some(false),
            Some(false),
        );

        // 冻结搜索空间（使 sample_independent 进入警告判断分支）
        let mut ss = HashMap::new();
        ss.insert(
            "x".to_string(),
            Distribution::FloatDistribution(FloatDistribution {
                low: 0.0,
                high: 1.0,
                log: false,
                step: None,
            }),
        );
        let trial_done = FrozenTrial {
            number: 0,
            trial_id: 0,
            state: TrialState::Complete,
            values: Some(vec![0.1]),
            params: {
                let mut p = HashMap::new();
                p.insert("x".to_string(), crate::distributions::ParamValue::Float(0.1));
                p
            },
            distributions: ss.clone(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            datetime_start: None,
            datetime_complete: None,
        };
        let _ = sampler.infer_relative_search_space(&[trial_done.clone()]);

        let v = sampler
            .sample_independent(&[trial_done.clone()], &trial_done, "x", ss.get("x").unwrap())
            .unwrap();
        assert!((0.0..=1.0).contains(&v));
    }

    /// 完整优化测试: Sobol + log 参数 + int 参数
    #[test]
    fn test_qmc_full_optimization_mixed_params() {
        let sampler: Arc<dyn Sampler> = Arc::new(
            QmcSampler::new(Some(QmcType::Sobol), Some(false), Some(42), None, None, None),
        );
        let study = create_study(
            None, Some(sampler), None, None,
            Some(StudyDirection::Minimize), None, false,
        ).unwrap();

        study.optimize(|trial| {
            let x = trial.suggest_float("x", 0.01, 10.0, true, None)?; // log
            let y = trial.suggest_int("y", 1, 10, false, 1)?;
            let cat = trial.suggest_categorical("cat", vec![
                crate::distributions::CategoricalChoice::Str("a".to_string()),
                crate::distributions::CategoricalChoice::Str("b".to_string()),
            ])?;
            let bonus = if cat == crate::distributions::CategoricalChoice::Str("a".to_string()) { 0.0 } else { 1.0 };
            Ok(x + y as f64 + bonus)
        }, Some(20), None, None).unwrap();

        let trials = study.trials().unwrap();
        assert_eq!(trials.len(), 20);
        // 所有试验应有有效值
        for t in &trials {
            assert!(t.values.is_some());
        }
    }

    #[test]
    fn test_sobol_first_points_unscrambled() {
        // 验证未置乱 Sobol 序列的前几个点
        // 第 0 个点 (index=0): 灰码全 0 → 结果为 [0.0, 0.0, ...]
        let p0 = sobol_point(0, 3, false, 0);
        assert_eq!(p0[0], 0.0);
        assert_eq!(p0[1], 0.0);
        assert_eq!(p0[2], 0.0);

        // 第 1 个点 (index=1): 灰码 bit 0 = 1 → v[0]
        // 维度 0: v[0] = 1 << 31, result = v[0], val = 0.5
        let p1 = sobol_point(1, 3, false, 0);
        assert!((p1[0] - 0.5).abs() < 1e-10, "dim 0, point 1 应为 0.5, 实际 {}", p1[0]);
        assert!((p1[1] - 0.5).abs() < 1e-10, "dim 1, point 1 应为 0.5, 实际 {}", p1[1]);

        // 验证各维度的值在 [0, 1) 范围内
        for i in 0..16 {
            let pt = sobol_point(i, 10, false, 0);
            for (d, &v) in pt.iter().enumerate() {
                assert!(v >= 0.0 && v < 1.0,
                    "sobol_point({}, dim={}) = {} 超出 [0,1) 范围", i, d, v);
            }
        }
    }

    #[test]
    fn test_sobol_params_consistency() {
        // 验证 SOBOL_PARAMS 表的内部一致性
        for (dim, &(s, _a, ref m)) in SOBOL_PARAMS.iter().enumerate() {
            if dim == 0 {
                assert_eq!(s, 0, "dim 0 应为 s=0");
                continue;
            }
            // 初始方向数的个数应等于 s
            let num_nonzero = m.iter().take_while(|&&x| x > 0).count();
            assert_eq!(num_nonzero, s,
                "dim {} 初始方向数个数 {} ≠ 多项式阶数 s={}", dim, num_nonzero, s);
            // 初始方向数 m_i 应为奇数
            for i in 0..s {
                assert!(m[i] % 2 == 1,
                    "dim {} 的 m[{}]={} 不是奇数", dim, i, m[i]);
            }
        }
    }
}
