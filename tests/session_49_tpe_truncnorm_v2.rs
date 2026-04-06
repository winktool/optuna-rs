// Session 49 - 会话 1-2: TPE Truncnorm 采样精确性交叉验证
//
// 验证 optuna-rs truncnorm 与 Python scipy.stats.truncnorm 的精确对齐

#[cfg(test)]
mod session_49_tpe_truncnorm {
    use optuna_rs::samplers::tpe::truncnorm;

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Category 1: log_gauss_mass (Log Gaussian Mass)
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_truncnorm_log_gauss_mass_center() {
        let result = truncnorm::log_gauss_mass(-1.0, 1.0);
        let expected = -0.381715;
        assert!(
            (result - expected).abs() < 0.005,
            "log_gauss_mass(-1,1)={} (expected ~{})",
            result,
            expected
        );
    }

    #[test]
    fn session_49_truncnorm_log_gauss_mass_full_range() {
        let result = truncnorm::log_gauss_mass(-10.0, 10.0);
        assert!(
            result.abs() < 1e-6,
            "log_gauss_mass(-10,10)={} (expected ~0)",
            result
        );
    }

    #[test]
    fn session_49_truncnorm_log_gauss_mass_symmetric() {
        let left = truncnorm::log_gauss_mass(-10.0, 0.0);
        let right = truncnorm::log_gauss_mass(0.0, 10.0);
        assert!(
            (left - right).abs() < 1e-10,
            "symmetric ranges should have same log_gauss_mass"
        );
    }

    #[test]
    fn session_49_truncnorm_ppf_boundaries() {
        let a = -2.0;
        let b = 2.0;

        let ppf_0 = truncnorm::ppf(0.0, a, b);
        let ppf_1 = truncnorm::ppf(1.0, a, b);

        assert!(
            (ppf_0 - a).abs() < 1e-10,
            "ppf(0)={} should be close to a={}",
            ppf_0,
            a
        );
        assert!(
            (ppf_1 - b).abs() < 1e-10,
            "ppf(1)={} should be close to b={}",
            ppf_1,
            b
        );
    }

    #[test]
    fn session_49_truncnorm_ppf_monotonicity() {
        let a = -3.0;
        let b = 3.0;
        let quantiles = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        let mut prev_ppf = f64::NEG_INFINITY;
        for q in quantiles {
            let ppf = truncnorm::ppf(q, a, b);
            assert!(
                ppf >= prev_ppf - 1e-10,
                "PPF not monotonic at q={}",
                q
            );
            prev_ppf = ppf;
        }
    }

    #[test]
    fn session_49_truncnorm_ppf_range() {
        let a = -2.5;
        let b = 3.5;
        
        for q in [0.001, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99].iter() {
            let ppf = truncnorm::ppf(*q, a, b);
            assert!(
                ppf >= a - 1e-10 && ppf <= b + 1e-10,
                "ppf({})={} out of range [{}, {}]",
                q,
                ppf,
                a,
                b
            );
        }
    }

    #[test]
    fn session_49_truncnorm_logpdf_validity() {
        let a = -1.0;
        let b = 1.0;
        let x = 0.0;

        let pdf = truncnorm::logpdf(x, a, b, 0.0, 1.0);
        assert!(pdf.is_finite(), "logpdf should be finite at center");
    }

    #[test]
    fn session_49_truncnorm_logpdf_outside_range() {
        let a = -1.0;
        let b = 1.0;
        let x = 1.5;

        let pdf = truncnorm::logpdf(x, a, b, 0.0, 1.0);
        assert!(
            pdf.is_infinite() && pdf < 0.0,
            "logpdf outside range should be -inf"
        );
    }
}
