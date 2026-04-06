// Session 49 - 会话 1-2: TPE Truncnorm 采样精确性交叉验证
//
// 验证 optuna-rs truncnorm 与 Python scipy.stats.truncnorm 的精确对齐
// 重点: log_gauss_mass, ndtr, log_ndtr, logpdf, ppf

#[cfg(test)]
mod session_49_truncnorm_precision {
    use optuna_rs::samplers::tpe::truncnorm::{log_gauss_mass, logpdf, ppf};

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Category 1: Standard Normal CDF (ndtr-equivalent)
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_log_gauss_mass_center() {
        // 中心值测试: Phi(b) - Phi(a)
        // log_gauss_mass(-1, 1) ≈ log(2 * Phi(1) - 1) ≈ log(0.6826) ≈ -0.3817
        let result = log_gauss_mass(-1.0, 1.0);
        let expected = -0.381715;
        assert!(
            (result - expected).abs() < 0.001,
            "log_gauss_mass(-1,1)={} (expected ~{})",
            result,
            expected
        );
    }

    #[test]
    fn session_49_log_gauss_mass_full_range() {
        // 全范围测试: log_gauss_mass(-10, 10) ≈ log(1) = 0
        let result = log_gauss_mass(-10.0, 10.0);
        assert!(
            result.abs() < 1e-6,
            "log_gauss_mass(-10,10)={} (expected ~0)",
            result
        );
    }

    #[test]
    fn session_49_log_gauss_mass_left_half() {
        // 左半范围测试: log_gauss_mass(-10, 0)
        let result = log_gauss_mass(-10.0, 0.0);
        let expected = -0.693147; // log(0.5)
        assert!(
            (result - expected).abs() < 0.001,
            "log_gauss_mass(-10,0)={} (expected ~{})",
            result,
            expected
        );
    }

    #[test]
    fn session_49_log_gauss_mass_right_half() {
        // 右半范围测试: log_gauss_mass(0, 10)
        let result = log_gauss_mass(0.0, 10.0);
        let expected = -0.693147; // log(0.5)
        assert!(
            (result - expected).abs() < 0.001,
            "log_gauss_mass(0,10)={} (expected ~{})",
            result,
            expected
        );
    }

    #[test]
    fn session_49_log_gauss_mass_left_tail() {
        // 左尾范围测试: log_gauss_mass(-3, -2)
        let result = log_gauss_mass(-3.0, -2.0);
        // Expected: log(Phi(-2) - Phi(-3)) ≈ log(0.0228 - 0.00135) ≈ log(0.0214) ≈ -3.84
        let expected = -3.84435;
        assert!(
            (result - expected).abs() < 0.01,
            "log_gauss_mass(-3,-2)={} (expected ~{})",
            result,
            expected
        );
    }

    #[test]
    fn session_49_log_gauss_mass_right_tail() {
        // 右尾范围测试: log_gauss_mass(2, 3)
        let result = log_gauss_mass(2.0, 3.0);
        // Expected: log(Phi(3) - Phi(2)) ≈ log(0.99865 - 0.97725) ≈ log(0.0214) ≈ -3.84
        let expected = -3.84435;
        assert!(
            (result - expected).abs() < 0.01,
            "log_gauss_mass(2,3)={} (expected ~{})",
            result,
            expected
        );
    }

    #[test]
    fn session_49_log_gauss_mass_narrow() {
        // 狭窄范围测试: log_gauss_mass(-0.5, 0.5)
        let result = log_gauss_mass(-0.5, 0.5);
        // Expected: log(Phi(0.5) - Phi(-0.5)) ≈ log(0.69146 - 0.30854) ≈ log(0.38292) ≈ -0.958
        let expected = -0.958;
        assert!(
            (result - expected).abs() < 0.01,
            "log_gauss_mass(-0.5,0.5)={} (expected ~{})",
            result,
            expected
        );
    }

    #[test]
    fn session_49_log_gauss_mass_extreme_left() {
        // 极端左尾: log_gauss_mass(-100, -50)
        // 应该产生非常小的数，log 应该是很负的值
        let result = log_gauss_mass(-100.0, -50.0);
        assert!(
            result < -10.0,
            "log_gauss_mass(-100,-50)={} (expected < -10)",
            result
        );
        assert!(
            result.is_finite(),
            "log_gauss_mass(-100,-50) should be finite"
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Category 2: Truncnorm PDF Range
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_truncnorm_pdf_basic() {
        // 基础 truncnorm PDF 测试
        // 标准截断正态分布 (a=-1, b=1, mean=0, std=1)
        // PDF(0) = phi(0) / (Phi(1) - Phi(-1))
        // phi(0) = 1/sqrt(2π) ≈ 0.3989
        // Phi(1) - Phi(-1) ≈ 0.6826
        // PDF(0) ≈ 0.584

        let a = -1.0;
        let b = 1.0;
        let x = 0.0;
        let loc = 0.0;
        let scale = 1.0;

        let pdf = logpdf(x, a, b, loc, scale);
        // log(0.584) ≈ -0.538
        let expected_log = -0.538;
        assert!(
            (pdf - expected_log).abs() < 0.05,
            "logpdf(0, -1, 1, 0, 1)={} (expected ~{})",
            pdf,
            expected_log
        );
    }

    #[test]
    fn session_49_truncnorm_pdf_boundary() {
        // 边界处的 PDF 应该接近 0 (log 应该非常负)
        let a = -2.0;
        let b = 2.0;
        let x_near_boundary = 1.99; // 接近上边界

        let pdf = logpdf(x_near_boundary, a, b, 0.0, 1.0);
        // logpdf(1.99) ≈ -2.85, 比中心值低很多
        assert!(
            pdf < -2.5,
            "logpdf(1.99, -2, 2, 0, 1)={} (expected < -2.5)",
            pdf
        );
    }

    #[test]
    fn session_49_truncnorm_pdf_outside_range() {
        // 范围外的 PDF 应该是 -inf
        let a = -1.0;
        let b = 1.0;
        let x = 1.5; // 超出范围

        let pdf = logpdf(x, a, b, 0.0, 1.0);
        assert!(
            pdf.is_infinite() && pdf < 0.0,
            "logpdf(1.5, -1, 1, 0, 1)={} (expected -inf)",
            pdf
        );
    }

    // ═══════════════════════════════════════════════════════════════════════════
    //  Test Category 3: Truncnorm PPF (quantile function)
    // ═══════════════════════════════════════════════════════════════════════════

    #[test]
    fn session_49_truncnorm_ppf_boundaries() {
        // PPF(0) 应该返回 a，PPF(1) 应该返回 b
        let a = -2.0;
        let b = 2.0;

        let ppf_0 = ppf(0.0, a, b);
        let ppf_1 = ppf(1.0, a, b);

        assert!(
            (ppf_0 - a).abs() < 1e-10,
            "ppf(0, -2, 2)={} (expected {})",
            ppf_0,
            a
        );
        assert!(
            (ppf_1 - b).abs() < 1e-10,
            "ppf(1, -2, 2)={} (expected {})",
            ppf_1,
            b
        );
    }

    #[test]
    fn session_49_truncnorm_ppf_center() {
        // PPF(0.5) 应该返回分布的中位数
        let a = -2.0;
        let b = 2.0;

        let ppf_05 = ppf(0.5, a, b);

        // 对于对称分布，中位数应该接近 0
        assert!(
            ppf_05.abs() < 0.1,
            "ppf(0.5, -2, 2)={} (expected ~0)",
            ppf_05
        );
    }

    #[test]
    fn session_49_truncnorm_ppf_monotonicity() {
        // PPF 应该是单调递增的
        let a = -3.0;
        let b = 3.0;
        let quantiles = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

        let mut prev_ppf = a;
        for q in quantiles {
            let p = ppf(q, a, b);
            assert!(
                p >= prev_ppf - 1e-10,
                "PPF not monotonic: ppf({})={} < prev={}",
                q,
                p,
                prev_ppf
            );
            prev_ppf = p;
        }
    }

    #[test]
    fn session_49_truncnorm_cdf_ppf_inverse() {
        // CDF(PPF(q)) ≈ q (逆函数关系)
        // Use log_gauss_mass to compute CDF: CDF(x) = exp(log_gauss_mass(a, x)) / exp(log_gauss_mass(a, b))
        let a = -2.0;
        let b = 2.0;
        let quantiles = vec![0.05, 0.25, 0.5, 0.75, 0.95];
        let log_total = log_gauss_mass(a, b);

        for q in quantiles {
            let x = ppf(q, a, b);
            let log_partial = log_gauss_mass(a, x);
            let cdf = (log_partial - log_total).exp();
            assert!(
                (cdf - q).abs() < 0.01,
                "CDF(PPF({q}))={} (expected {q})",
                cdf
            );
        }
    }
}
