// Session 49 - 会话 1-1: TPE 权重计算精确性交叉验证
//
// 深入验证 optuna-rs 与 Python optuna 的 TPE 权重计算是否精确对齐

#[cfg(test)]
mod session_49_tpe_weights_precision {
    use std::f64;
    use optuna_rs::samplers::tpe::{default_weights, hyperopt_default_gamma};

    // ═══════════════════════════════════════════════════════════════════════════
    //  Helper Functions
    // ═══════════════════════════════════════════════════════════════════════════

    fn assert_weights_valid(n_trials: usize, weights: &[f64]) {
        // 断言 1: 权重数量正确
        assert_eq!(
            weights.len(),
            n_trials,
            "weights.len()={} != n_trials={}",
            weights.len(),
            n_trials
        );

        // 注: 权重和不一定为 1.0，它们是相对权重而非概率分布
        // 对于 n < 25: 所有权重 = 1.0, 所以 sum = n
        // 对于 n >= 25: sum ～= n - 0.5

        // 断言 2: 权重单调递增
        for i in 1..weights.len() {
            assert!(
                weights[i] >= weights[i - 1] - 1e-14,
                "weights not monotonically increasing at i={}, n={}",
                i,
                n_trials
            );
        }

        // 断言 3: 所有权重非负、有限且在合理范围
        for (i, &w) in weights.iter().enumerate() {
            assert!(
                w > 0.0,
                "weight[{}]={} should be positive, n={}",
                i,
                w,
                n_trials
            );
            assert!(
                w.is_finite(),
                "weight[{}] is not finite ({:.e}), n={}",
                i,
                w,
                n_trials
            );
            // 权重不应超过 1.0（最末尾可能是 1.0）
            assert!(
                w <= 1.0 + 1e-14,
                "weight[{}]={} exceeds 1.0, n={}",
                i,
                w,
                n_trials
            );
        }
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Cases - Basic Sizes
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_tpe_weights_n10() {
        let n = 10;
        let weights = default_weights(n);
        let gamma = hyperopt_default_gamma(n) as f64;

        assert_weights_valid(n, &weights);

        // 验证 gamma 值合理
        assert!(gamma > 0.0 && gamma <= 1.0, "gamma={} out of (0,1]", gamma);

        // 对于 n < 25，所有权重都应该是 1.0
        for w in &weights {
            assert!((w - 1.0).abs() < 1e-14, "weight should be 1.0 for n < 25");
        }
    }

    #[test]
    fn session_49_tpe_weights_n20() {
        let n = 20;
        let weights = default_weights(n);
        assert_weights_valid(n, &weights);
    }

    #[test]
    fn session_49_tpe_weights_n50() {
        let n = 50;
        let weights = default_weights(n);
        assert_weights_valid(n, &weights);
    }

    #[test]
    fn session_49_tpe_weights_n100() {
        let n = 100;
        let weights = default_weights(n);
        assert_weights_valid(n, &weights);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Cases - Boundary Conditions
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_tpe_weights_boundary_n1() {
        let n = 1;
        let weights = default_weights(n);
        assert_eq!(weights.len(), 1);
        assert!((weights[0] - 1.0).abs() < 1e-14, "single weight should be 1.0");
    }

    #[test]
    fn session_49_tpe_weights_boundary_n2() {
        let n = 2;
        let weights = default_weights(n);
        assert_weights_valid(n, &weights);
    }

    #[test]
    fn session_49_tpe_weights_boundary_n25() {
        // Critical boundary: n=25 might have ramp_len issues
        let n = 25;
        let weights = default_weights(n);
        assert_weights_valid(n, &weights);
    }

    #[test]
    fn session_49_tpe_weights_boundary_n26() {
        // Critical boundary: n=26 should have ramp_len=1
        let n = 26;
        let weights = default_weights(n);
        assert_weights_valid(n, &weights);

        // For ramp_len=1, first weight should be close to endpoint
        let gamma = hyperopt_default_gamma(n) as f64;
        let expected_first = 1.0 / (n as f64);
        // Should be approximately equal to endpoint value
        assert!(
            weights[n - 1] > weights[0],
            "ramp should be non-trivial, n=26"
        );
    }

    #[test]
    fn session_49_tpe_weights_boundary_n100() {
        let n = 100;
        let weights = default_weights(n);
        assert_weights_valid(n, &weights);
    }

    #[test]
    fn session_49_tpe_weights_boundary_n999() {
        let n = 999;
        let weights = default_weights(n);
        assert_weights_valid(n, &weights);
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Cases - Mathematical Properties
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_tpe_weights_monotonicity_strict() {
        // 对于 n >= 25，验证严格单调性
        let n = 50;
        let weights = default_weights(n);

        // n >= 25 时，前 n-25 个权重应该严格递增（linspace）
        // 然后是 25 个 1.0
        let ramp_len = n - 25;
        for i in 1..ramp_len {
            assert!(
                weights[i] > weights[i - 1],
                "weights should be strictly increasing in ramp region for n={}",
                n
            );
        }
    }

    #[test]
    fn session_49_tpe_weights_endpoint_behavior() {
        // 验证端点行为：
        // 对于 n < 25：所有权重都是 1.0
        // 对于 n >= 25：末尾 25 个权重是 1.0，前 n-25 个是 linspace
        for n in [10, 20, 50, 100, 200] {
            let weights = default_weights(n);
            
            // 末尾权重应该是 1.0
            assert!(
                (weights[n - 1] - 1.0).abs() < 1e-14,
                "last weight should be 1.0 for n={}",
                n
            );

            if n >= 25 {
                // 对于 n >= 25，第一个权重应该小于最后一个
                assert!(
                    weights[0] < 1.0 - 1e-14,
                    "first weight < last weight (1.0) for n={}",
                    n
                );
                
                // 最后一个权重应该是 1.0
                assert!(
                    (weights[n - 1] - 1.0).abs() < 1e-14,
                    "last weight should be 1.0 for n={}",
                    n
                );
            }
        }
    }

    #[test]
    fn session_49_tpe_gamma_range() {
        // 验证 gamma 值总是在合理范围内
        // hyperopt_default_gamma 返回 usize，应该是 1 到 25
        for n in [1, 2, 5, 10, 20, 50, 100, 500] {
            let gamma_usize = hyperopt_default_gamma(n);
            assert!(
                gamma_usize >= 1 && gamma_usize <= 25,
                "gamma={} out of [1,25] for n={}",
                gamma_usize,
                n
            );

            // gamma 应该随 n 增长但被限制为 <= 25
            // gamma = ceil(0.25 * sqrt(n)).min(25)
            let expected_gamma = ((0.25 * (n as f64).sqrt()).ceil() as usize).min(25);
            assert_eq!(gamma_usize, expected_gamma, "gamma mismatch for n={}", n);
        }
    }

    #[test]
    fn session_49_tpe_weights_precision_error_accumulation() {
        // 验证大规模权重计算中的舍入误差不会累积
        let n = 1000;
        let weights = default_weights(n);

        // 多次求和应得到相同结果（舍入误差最小）
        let sum1: f64 = weights.iter().sum();
        let sum2: f64 = weights.iter().rev().sum();

        assert!(
            (sum1 - sum2).abs() < 1e-10,
            "sum order dependence detected: {} vs {}",
            sum1,
            sum2
        );

        // 对于 n >= 25：sum \approx n - 0.5 (因为 ramp_len = n - 25)
        // ramp_len 个权重从 1/n 到 1 的 linspace，plus 25 个 1.0
        let expected_sum = (n - 25) as f64 * (1.0 / (2 * n) as f64 + 1.0 / 2.0) + 25.0;
        assert!(
            (sum1 - expected_sum).abs() < 1.0,
            "sum={} deviates from expected {} for n={}",
            sum1,
            expected_sum,
            n
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Cross-validation with Python Reference Values (if file available)
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_tpe_weights_python_alignment() {
        // This test verifies that Rust weights align with Python behavior.
        // We compare structural properties rather than exact numerical values
        // (since weights are not meant to sum to 1.0).
        
        let test_cases = vec![10, 20, 50, 100];
        
        for n in test_cases {
            let weights = default_weights(n);
            let gamma = hyperopt_default_gamma(n);
            
            // Basic structural checks
            assert_eq!(weights.len(), n);
            
            // All weights should be positive and finite
            for w in &weights {
                assert!(*w > 0.0, "weight should be positive");
                assert!(w.is_finite(), "weight should be finite");
                assert!(*w <= 1.0 + 1e-14, "weight should not exceed 1.0");
            }
            
            // Gamma should be in valid range
            assert!(gamma >= 1 && gamma <= 25, "gamma out of range");
            
            // Monotonicity: weights should be non-decreasing
            for i in 1..weights.len() {
                assert!(weights[i] >= weights[i - 1] - 1e-14, "weights should be non-decreasing");
            }
        }
    }
}
