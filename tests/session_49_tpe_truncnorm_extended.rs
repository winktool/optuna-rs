// Session 49 - 会话 1-2 扩展: TPE Truncnorm 采样 - 高级特性测试
// 注: ndtr 是私有函数，改为测试公开的 log_gauss_mass/ppf 的高级特性

#[cfg(test)]
mod session_49_tpe_truncnorm_extended {
    use optuna_rs::samplers::tpe::truncnorm;

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Category 2: log_gauss_mass 极端情况扩展
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_log_gauss_mass_very_tight() {
        let a = -0.1;
        let b = 0.1;
        // Python scipy: log(ndtr(0.1) - ndtr(-0.1)) ≈ -2.530042
        let result = truncnorm::log_gauss_mass(a, b);
        let expected = -2.530042;
        assert!(
            (result - expected).abs() < 0.01,
            "log_gauss_mass({},{})={} (expected ~{})",
            a, b, result, expected
        );
    }

    #[test]
    fn session_49_log_gauss_mass_tight() {
        let a = -0.5;
        let b = 0.5;
        // Python: log(ndtr(0.5) - ndtr(-0.5)) ≈ -0.959916
        let result = truncnorm::log_gauss_mass(a, b);
        let expected = -0.959916;
        assert!(
            (result - expected).abs() < 0.01,
            "log_gauss_mass({},{})={} (expected ~{})",
            a, b, result, expected
        );
    }

    #[test]
    fn session_49_log_gauss_mass_asymmetric_left() {
        let a = -20.0;
        let b = 0.5;
        // Python: log(ndtr(0.5) - ndtr(-20.0)) ≈ -0.368946
        let result = truncnorm::log_gauss_mass(a, b);
        let expected = -0.368946;
        assert!(
            (result - expected).abs() < 0.01,
            "log_gauss_mass({},{})={} (expected ~{})",
            a, b, result, expected
        );
    }

    #[test]
    fn session_49_log_gauss_mass_asymmetric_right() {
        let a = -0.5;
        let b = 20.0;
        // Python: log(ndtr(20.0) - ndtr(-0.5)) ≈ -0.368946
        let result = truncnorm::log_gauss_mass(a, b);
        let expected = -0.368946;
        assert!(
            (result - expected).abs() < 0.01,
            "log_gauss_mass({},{})={} (expected ~{})",
            a, b, result, expected
        );
    }

    #[test]
    fn session_49_log_gauss_mass_tail_far_left() {
        let a = -100.0;
        let b = -99.0;
        // Both in far left tail: should be -inf in log scale
        let result = truncnorm::log_gauss_mass(a, b);
        assert!(
            result.is_infinite() || result < -30.0,
            "log_gauss_mass({},{})={} should be very negative/inf",
            a, b, result
        );
    }

    #[test]
    fn session_49_log_gauss_mass_tail_far_right() {
        let a = 99.0;
        let b = 100.0;
        // Both in far right tail: should be -inf in log scale
        let result = truncnorm::log_gauss_mass(a, b);
        assert!(
            result.is_infinite() || result < -30.0,
            "log_gauss_mass({},{})={} should be very negative/inf",
            a, b, result
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Category 3: PPF 分位数单调性和稳定性
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_ppf_monotonicity_comprehensive() {
        let a = -2.0;
        let b = 2.0;
        
        // 从极端分位数到另一个极端：检查 PPF 单调性
        let quantiles = vec![0.0001, 0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999, 0.9999];
        
        let mut prev_ppf = f64::NEG_INFINITY;
        for q in quantiles {
            let ppf = truncnorm::ppf(q, a, b);
            
            // PPF 应该严格单调递增
            assert!(
                ppf >= prev_ppf - 1e-10,
                "PPF not monotonic: ppf({})={} < prev_ppf({}={})",
                q, ppf, prev_ppf, prev_ppf
            );
            
            // 检查范围
            assert!(
                ppf >= a - 1e-10 && ppf <= b + 1e-10,
                "PPF({})={} outside range [{}, {}]",
                q, ppf, a, b
            );
            
            prev_ppf = ppf;
        }
    }

    #[test]
    fn session_49_tail_quantile_stability() {
        let a = -10.0;
        let b = 10.0;
        
        // 验证极端分位数的稳定性
        let q_pairs = vec![(0.0001, 0.001), (0.001, 0.01), (0.99, 0.999), (0.999, 0.9999)];
        
        for (q1, q2) in q_pairs {
            let ppf1 = truncnorm::ppf(q1, a, b);
            let ppf2 = truncnorm::ppf(q2, a, b);
            
            // 更高的分位数应该产生更高的值
            assert!(
                ppf2 > ppf1 - 1e-6,
                "Tail stability failed: ppf({})={} >= ppf({})={}",
                q1, ppf1, q2, ppf2
            );
        }
    }

    #[test]
    fn session_49_logpdf_integration_test() {
        let a = -2.0;
        let b = 2.0;
        let loc = 0.0;
        let scale = 1.0;
        
        // 在范围内的点应该有有限的 logpdf
        let x_valid = truncnorm::ppf(0.5, a, b); // 中位数
        let logpdf = truncnorm::logpdf(x_valid, a, b, loc, scale);
        assert!(logpdf.is_finite(), "logpdf at median should be finite");
        
        // 在范围外的点应该返回 -inf
        let x_outside = b + 1.0;
        let logpdf_outside = truncnorm::logpdf(x_outside, a, b, loc, scale);
        assert!(
            logpdf_outside.is_infinite() && logpdf_outside < 0.0,
            "logpdf outside range should be -inf"
        );
    }
}
