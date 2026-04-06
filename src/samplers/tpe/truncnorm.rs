//! Truncated normal distribution math.
//!
//! Provides CDF, log-CDF, PPF (inverse CDF), log-PDF, and sampling for
//! the standard truncated normal distribution on [a, b].
//!
//! Port of Python optuna's `_truncnorm.py`.

use std::f64::consts::{FRAC_1_SQRT_2, PI};

const LOG_SQRT_2PI: f64 = 0.9189385332046727; // ln(sqrt(2*pi))

const EPS: f64 = 1e-12;

/// Standard normal log-PDF: log phi(x) = -x^2/2 - ln(sqrt(2*pi))
fn norm_logpdf(x: f64) -> f64 {
    -0.5 * x * x - LOG_SQRT_2PI
}

/// Standard normal CDF: Phi(x), aligned with Python `_ndtr_single`.
/// Uses erfc for tails to avoid catastrophic cancellation.
fn ndtr(x: f64) -> f64 {
    let t = x * FRAC_1_SQRT_2;
    if t < -FRAC_1_SQRT_2 {
        // Left tail: use erfc for precision
        0.5 * libm::erfc(-t)
    } else if t < FRAC_1_SQRT_2 {
        // Central region
        0.5 + 0.5 * libm::erf(t)
    } else {
        // Right tail: use erfc for precision
        1.0 - 0.5 * libm::erfc(t)
    }
}

/// Log of standard normal CDF, computed stably.
fn log_ndtr(a: f64) -> f64 {
    if a > 6.0 {
        // Phi(a) ~ 1, use: log(1 - Phi(-a))
        let v = ndtr(-a);
        if v <= 0.0 {
            0.0
        } else {
            (-v).ln_1p()
        }
    } else if a > -20.0 {
        let v = ndtr(a);
        if v <= 0.0 {
            f64::NEG_INFINITY
        } else {
            v.ln()
        }
    } else {
        // Asymptotic series for very negative a (aligned with Python _log_ndtr_single):
        // log Phi(a) = -a^2/2 - ln(-a) - 0.5*ln(2*pi) + ln(series)
        // where series = 1 - 1/a^2 + 1*3/a^4 - 1*3*5/a^6 + ...
        let log_lhs = -0.5 * a * a - (-a).ln() - LOG_SQRT_2PI;
        let denom_cons = 1.0 / (a * a);
        let mut rhs = 1.0;
        let mut last_total = 0.0_f64;
        let mut sign = 1.0;
        let mut denom_factor = 1.0;
        let mut numerator = 1.0;
        let mut i = 0;
        while (last_total - rhs).abs() > f64::EPSILON {
            i += 1;
            last_total = rhs;
            sign = -sign;
            denom_factor *= denom_cons;
            numerator *= (2 * i - 1) as f64;
            rhs += sign * numerator * denom_factor;
        }
        log_lhs + rhs.ln()
    }
}

/// log(exp(log_p) - exp(log_q)) computed stably, assuming log_p >= log_q.
fn log_diff(log_p: f64, log_q: f64) -> f64 {
    if log_p == f64::NEG_INFINITY {
        return f64::NEG_INFINITY;
    }
    log_p + (-(log_q - log_p).exp()).ln_1p()
}

/// log(Phi(b) - Phi(a)), the log of Gaussian mass in [a, b].
pub fn log_gauss_mass(a: f64, b: f64) -> f64 {
    if a >= b {
        return f64::NEG_INFINITY;
    }
    if b <= 0.0 {
        // Both in left tail
        log_diff(log_ndtr(b), log_ndtr(a))
    } else if a >= 0.0 {
        // Both in right tail — use symmetry: Phi(b)-Phi(a) = Phi(-a)-Phi(-b)
        log_diff(log_ndtr(-a), log_ndtr(-b))
    } else {
        // Spans zero: Phi(b) - Phi(a) = 1 - Phi(-b) - (1 - Phi(b_negative_part))
        // = 1 - ndtr(a) - ndtr(-b)
        let tail_mass = ndtr(a) + ndtr(-b);
        if tail_mass >= 1.0 {
            f64::NEG_INFINITY
        } else {
            (-tail_mass).ln_1p()
        }
    }
}

/// Inverse of log_ndtr: find x such that log(Phi(x)) = log_p.
/// Uses Newton's method. Aligned with Python `_ndtri_exp`.
fn ndtri_exp(log_p: f64) -> f64 {
    if log_p < -1e15 {
        return -100.0;
    }

    // For y close to zero, flip sign for numerical stability (Python: flipped = y > -1e-2).
    // If exp(y) > 0.5 (y > -ln(2) ≈ -0.693), x is positive.
    // We compute z = log(1 - exp(y)) = log(-expm1(y)), solve for positive x, then negate.
    let flipped = log_p > -1e-2;
    let z = if flipped {
        (-log_p.exp_m1()).ln() // log(-expm1(log_p))
    } else {
        log_p
    };

    // Initial guess.
    // Python: _ndtri_exp_approx_C = sqrt(3) / pi
    let c = 3.0_f64.sqrt() / PI;
    let mut x = if z < -5.0 {
        // Deep left tail: x ≈ -sqrt(-2 * (z + LOG_SQRT_2PI))
        // Aligned with Python: x = -np.sqrt(-2.0 * (z + _norm_pdf_logC))
        -(-2.0 * (z + LOG_SQRT_2PI)).sqrt()
    } else {
        // Moderate region: logistic approximation
        // x ≈ -sqrt(3)/pi * log(expm1(-z))
        -c * (-z).exp_m1().ln()
    };

    // Newton iterations: f(x) = log_ndtr(x) - z
    // dx = f(x) * exp(log_ndtr(x) - norm_logpdf(x))
    // Convergence: relative |dx| < 1e-8 * |x| (aligned with Python rtol=1e-8)
    for _ in 0..100 {
        let lnx = log_ndtr(x);
        let lnpdf = norm_logpdf(x);
        let dx = (lnx - z) * (lnx - lnpdf).exp();
        x -= dx;
        if dx.abs() < 1e-8 * x.abs() {
            break;
        }
    }

    if flipped { -x } else { x }
}

/// PPF (quantile function) of the standard truncated normal on [a, b].
///
/// Returns x such that P(a <= Z <= x) / P(a <= Z <= b) = q.
pub fn ppf(q: f64, a: f64, b: f64) -> f64 {
    if q <= 0.0 {
        return a;
    }
    if q >= 1.0 {
        return b;
    }
    if (a - b).abs() < EPS {
        return f64::NAN;
    }

    let lm = log_gauss_mass(a, b);

    if a < 0.0 {
        // Left case:
        // We want x such that Phi(x) = Phi(a) + q * (Phi(b) - Phi(a))
        // In log: log(Phi(x)) = logaddexp(log(Phi(a)), log(q) + lm)
        let log_phi_a = log_ndtr(a);
        let log_q_mass = q.ln() + lm;
        let log_phi_x = logaddexp(log_phi_a, log_q_mass);
        ndtri_exp(log_phi_x)
    } else {
        // Right case: work with upper tail for numerical stability.
        // Phi(-x) = Phi(-b) + (1-q) * (Phi(b) - Phi(a))
        let log_phi_neg_b = log_ndtr(-b);
        // 对齐 Python: 使用 log1p(-q) 替代 (1-q).ln() 提高 q 接近 1 时的精度
        let log_1mq_mass = (-q).ln_1p() + lm;
        let log_phi_neg_x = logaddexp(log_phi_neg_b, log_1mq_mass);
        -ndtri_exp(log_phi_neg_x)
    }
}

/// Sample from truncated normal: TN(a, b, loc, scale).
///
/// Uses inverse-CDF method: draw uniform q, then ppf(q, a_std, b_std) * scale + loc.
pub fn rvs(
    a: &[f64],
    b: &[f64],
    loc: &[f64],
    scale: &[f64],
    rng: &mut impl rand::RngExt,
) -> Vec<f64> {
    let n = a.len();
    let mut result = Vec::with_capacity(n);
    for i in 0..n {
        let a_std = (a[i] - loc[i]) / scale[i];
        let b_std = (b[i] - loc[i]) / scale[i];
        let q: f64 = rng.random();
        let x = ppf(q, a_std, b_std) * scale[i] + loc[i];
        result.push(x);
    }
    result
}

/// Log-PDF of truncated normal TN(a, b, loc, scale).
pub fn logpdf(x: f64, a: f64, b: f64, loc: f64, scale: f64) -> f64 {
    if (a - b).abs() < EPS {
        return f64::NAN;
    }
    let z = (x - loc) / scale;
    let a_std = (a - loc) / scale;
    let b_std = (b - loc) / scale;
    if z < a_std - EPS || z > b_std + EPS {
        return f64::NEG_INFINITY;
    }
    norm_logpdf(z) - log_gauss_mass(a_std, b_std) - scale.ln()
}

fn logaddexp(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ndtr_basic() {
        assert!((ndtr(0.0) - 0.5).abs() < 1e-10);
        assert!(ndtr(-40.0) < 1e-100);
        assert!((ndtr(40.0) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_log_gauss_mass_full_range() {
        // log(Phi(10) - Phi(-10)) ≈ 0
        let m = log_gauss_mass(-10.0, 10.0);
        assert!(m.abs() < 1e-6, "got {m}");
    }

    #[test]
    fn test_log_gauss_mass_half() {
        // log(Phi(0) - Phi(-10)) ≈ log(0.5)
        let m = log_gauss_mass(-10.0, 0.0);
        assert!((m - (-std::f64::consts::LN_2)).abs() < 1e-4, "got {m}");
    }

    #[test]
    fn test_ppf_midpoint() {
        // Median of standard normal truncated to [-10, 10] ≈ 0
        let x = ppf(0.5, -10.0, 10.0);
        assert!(x.abs() < 0.05, "got {x}");
    }

    #[test]
    fn test_ppf_boundaries() {
        let a = -2.0;
        let b = 2.0;
        assert!((ppf(0.0, a, b) - a).abs() < 1e-10);
        assert!((ppf(1.0, a, b) - b).abs() < 1e-10);
    }

    #[test]
    fn test_ppf_monotone() {
        let a = -1.0;
        let b = 1.0;
        let mut prev = a;
        for i in 1..10 {
            let q = i as f64 / 10.0;
            let x = ppf(q, a, b);
            assert!(x >= prev - 1e-10, "ppf not monotone at q={q}: {x} < {prev}");
            prev = x;
        }
    }

    #[test]
    fn test_rvs_within_bounds() {
        let mut rng = rand::rng();
        let n = 100;
        let a = vec![-1.0; n];
        let b = vec![1.0; n];
        let loc = vec![0.0; n];
        let scale = vec![1.0; n];
        let samples = rvs(&a, &b, &loc, &scale, &mut rng);
        for &s in &samples {
            assert!(
                s >= -1.0 - 1e-10 && s <= 1.0 + 1e-10,
                "sample {s} out of bounds"
            );
        }
    }

    #[test]
    fn test_logpdf_in_range() {
        let lp = logpdf(0.0, -1.0, 1.0, 0.0, 1.0);
        assert!(lp.is_finite());
        assert!(lp > -10.0);
    }

    #[test]
    fn test_logpdf_out_of_range() {
        let lp = logpdf(2.0, -1.0, 1.0, 0.0, 1.0);
        assert!(lp == f64::NEG_INFINITY);
    }

    #[test]
    fn test_logpdf_consistency_with_ppf() {
        // Numerical integration of PDF over [a, b] should ≈ 1.
        let a = -2.0;
        let b = 2.0;
        let n = 1000;
        let dx = (b - a) / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let x = a + (i as f64 + 0.5) * dx;
            integral += logpdf(x, a, b, 0.0, 1.0).exp() * dx;
        }
        assert!(
            (integral - 1.0).abs() < 0.01,
            "integral = {integral}, expected ~1.0"
        );
    }

    #[test]
    fn test_ndtri_exp_roundtrip() {
        for &y in &[-0.1, -1.0, -5.0, -10.0, -20.0] {
            let x = ndtri_exp(y);
            let back = log_ndtr(x);
            assert!(
                (back - y).abs() < 1e-6,
                "roundtrip failed: y={y}, x={x}, back={back}"
            );
        }
    }

    #[test]
    fn test_ppf_various_quantiles() {
        // Check PPF for a narrower range.
        let a = -1.0;
        let b = 1.0;
        for &q in &[0.1, 0.25, 0.5, 0.75, 0.9] {
            let x = ppf(q, a, b);
            assert!(
                x >= a - 1e-10 && x <= b + 1e-10,
                "ppf({q}, {a}, {b}) = {x} out of range"
            );
        }
    }
}
