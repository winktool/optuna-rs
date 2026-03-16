use indexmap::IndexMap;
use parking_lot::Mutex;
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::distributions::Distribution;
use crate::error::Result;
use crate::samplers::Sampler;
use crate::search_space::SearchSpaceTransform;
use crate::trial::FrozenTrial;

/// A sampler that draws parameters uniformly at random.
///
/// Corresponds to Python `optuna.samplers.RandomSampler`.
///
/// Uses `SearchSpaceTransform` to convert distributions into a continuous
/// uniform space, samples uniformly, then untransforms back.
pub struct RandomSampler {
    rng: Mutex<ChaCha8Rng>,
}

impl std::fmt::Debug for RandomSampler {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RandomSampler").finish()
    }
}

impl RandomSampler {
    /// Create a new `RandomSampler` with an optional seed.
    pub fn new(seed: Option<u64>) -> Self {
        let rng = match seed {
            Some(s) => ChaCha8Rng::seed_from_u64(s),
            None => ChaCha8Rng::from_entropy(),
        };
        Self {
            rng: Mutex::new(rng),
        }
    }

    /// Sample a single parameter using SearchSpaceTransform.
    fn sample_with_transform(&self, distribution: &Distribution) -> Result<f64> {
        let mut space = IndexMap::new();
        space.insert("_param".to_string(), distribution.clone());
        let transform = SearchSpaceTransform::with_defaults(space);
        let bounds = transform.bounds();

        let mut rng = self.rng.lock();
        let mut encoded = Vec::with_capacity(bounds.len());
        for [lo, hi] in &bounds {
            // 对齐 Python np.random.uniform(lo, hi) → [lo, hi) 半开区间
            // lo == hi 时（single value）直接返回 lo
            let v: f64 = if (hi - lo).abs() < f64::EPSILON {
                *lo
            } else {
                rng.gen_range(*lo..*hi)
            };
            encoded.push(v);
        }
        drop(rng);

        let params = transform.untransform(&encoded)?;
        let value = &params["_param"];
        distribution.to_internal_repr(value)
    }
}

impl Sampler for RandomSampler {
    fn sample_independent(
        &self,
        _trials: &[FrozenTrial],
        _trial: &FrozenTrial,
        _param_name: &str,
        distribution: &Distribution,
    ) -> Result<f64> {
        self.sample_with_transform(distribution)
    }

    /// 对齐 Python `RandomSampler.reseed_rng(seed)`: 重新设置随机种子。
    fn reseed_rng(&self, seed: u64) {
        *self.rng.lock() = ChaCha8Rng::seed_from_u64(seed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::distributions::*;

    fn dummy_trial() -> FrozenTrial {
        use chrono::Utc;
        use std::collections::HashMap;
        FrozenTrial {
            number: 0,
            state: crate::trial::TrialState::Running,
            values: None,
            datetime_start: Some(Utc::now()),
            datetime_complete: None,
            params: HashMap::new(),
            distributions: HashMap::new(),
            user_attrs: HashMap::new(),
            system_attrs: HashMap::new(),
            intermediate_values: HashMap::new(),
            trial_id: 0,
        }
    }

    #[test]
    fn test_sample_float_bounds() {
        let sampler = RandomSampler::new(Some(42));
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
            assert!(
                (0.0..=1.0).contains(&v),
                "value {v} out of [0, 1]"
            );
        }
    }

    #[test]
    fn test_sample_float_log() {
        let sampler = RandomSampler::new(Some(42));
        let dist = Distribution::FloatDistribution(
            FloatDistribution::new(0.001, 1.0, true, None).unwrap(),
        );
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&[], &trial, "lr", &dist).unwrap();
            assert!(
                (0.001..=1.0).contains(&v),
                "value {v} out of [0.001, 1.0]"
            );
        }
    }

    #[test]
    fn test_sample_float_step() {
        let sampler = RandomSampler::new(Some(42));
        let fd = FloatDistribution::new(0.0, 1.0, false, Some(0.25)).unwrap();
        let dist = Distribution::FloatDistribution(fd.clone());
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
            assert!(fd.contains(v), "value {v} not on step grid");
        }
    }

    #[test]
    fn test_sample_int_bounds() {
        let sampler = RandomSampler::new(Some(42));
        let dist =
            Distribution::IntDistribution(IntDistribution::new(1, 10, false, 1).unwrap());
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&[], &trial, "n", &dist).unwrap() as i64;
            assert!((1..=10).contains(&v), "value {v} out of [1, 10]");
        }
    }

    #[test]
    fn test_sample_int_step() {
        let sampler = RandomSampler::new(Some(42));
        let id = IntDistribution::new(0, 10, false, 2).unwrap();
        let dist = Distribution::IntDistribution(id.clone());
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&[], &trial, "n", &dist).unwrap();
            assert!(id.contains(v), "value {v} not on step grid");
        }
    }

    #[test]
    fn test_sample_int_log() {
        let sampler = RandomSampler::new(Some(42));
        let dist =
            Distribution::IntDistribution(IntDistribution::new(1, 100, true, 1).unwrap());
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&[], &trial, "n", &dist).unwrap() as i64;
            assert!((1..=100).contains(&v), "value {v} out of [1, 100]");
        }
    }

    #[test]
    fn test_sample_categorical() {
        let sampler = RandomSampler::new(Some(42));
        let dist = Distribution::CategoricalDistribution(
            CategoricalDistribution::new(vec![
                CategoricalChoice::Str("a".into()),
                CategoricalChoice::Str("b".into()),
                CategoricalChoice::Str("c".into()),
            ])
            .unwrap(),
        );
        let trial = dummy_trial();
        for _ in 0..100 {
            let v = sampler.sample_independent(&[], &trial, "opt", &dist).unwrap() as usize;
            assert!(v < 3, "index {v} out of range");
        }
    }

    #[test]
    fn test_single_distributions() {
        let sampler = RandomSampler::new(Some(42));
        let trial = dummy_trial();

        let fd = Distribution::FloatDistribution(
            FloatDistribution::new(5.0, 5.0, false, None).unwrap(),
        );
        let v = sampler.sample_independent(&[], &trial, "x", &fd).unwrap();
        assert!((v - 5.0).abs() < 1e-10);

        let id =
            Distribution::IntDistribution(IntDistribution::new(3, 3, false, 1).unwrap());
        let v = sampler.sample_independent(&[], &trial, "n", &id).unwrap();
        assert!((v - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_deterministic_with_seed() {
        let s1 = RandomSampler::new(Some(123));
        let s2 = RandomSampler::new(Some(123));
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        let trial = dummy_trial();
        for _ in 0..20 {
            let v1 = s1.sample_independent(&[], &trial, "x", &dist).unwrap();
            let v2 = s2.sample_independent(&[], &trial, "x", &dist).unwrap();
            assert_eq!(v1, v2);
        }
    }

    /// 验证 RandomSampler 使用半开区间 [lo, hi)（对齐 Python np.random.uniform）。
    /// 连续 float 分布采样值必须严格 < high。
    #[test]
    fn test_sample_float_strictly_less_than_high() {
        let sampler = RandomSampler::new(Some(0));
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        let trial = dummy_trial();
        for _ in 0..10000 {
            let v = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
            assert!(v < 1.0, "value {v} must be strictly < high (Python np.random.uniform is [lo, hi))");
            assert!(v >= 0.0, "value {v} must be >= low");
        }
    }

    /// 验证 lo == hi 边界情况（single-value 分布不会 panic）。
    #[test]
    fn test_sample_float_lo_eq_hi_no_panic() {
        let sampler = RandomSampler::new(Some(42));
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(3.0, 3.0, false, None).unwrap());
        let trial = dummy_trial();
        let v = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();
        assert!((v - 3.0).abs() < 1e-15, "lo==hi should return lo");
    }

    // ========== 对齐 Python: reseed_rng 测试 ==========

    /// 测试 reseed_rng 改变 RandomSampler 的 RNG 状态。
    /// 对应 Python: `sampler.reseed_rng()` 在并行模式下为每个 worker 重置 RNG。
    #[test]
    fn test_reseed_rng_changes_output() {
        use crate::samplers::Sampler;

        let sampler = RandomSampler::new(Some(42));
        let dist =
            Distribution::FloatDistribution(FloatDistribution::new(0.0, 1.0, false, None).unwrap());
        let trial = dummy_trial();

        // 第一次采样
        let v1 = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();

        // reseed 到不同的种子
        sampler.reseed_rng(99999);

        // reseed 后采样序列应重新开始（使用新种子的第一个值）
        let v2 = sampler.sample_independent(&[], &trial, "x", &dist).unwrap();

        // 使用相同种子 99999 的新采样器应产生相同的值
        let sampler2 = RandomSampler::new(Some(99999));
        let v3 = sampler2.sample_independent(&[], &trial, "x", &dist).unwrap();

        assert!((v2 - v3).abs() < 1e-15, "reseed 后应与相同种子的新采样器等价");
        // v1 与 v2 大概率不同（极小概率相同，忽略）
        let _ = v1; // 仅确认不 panic
    }
}
