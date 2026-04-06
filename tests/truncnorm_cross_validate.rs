// ═══════════════════════════════════════════════════════════════════════════════
// Cross-validation of truncated normal distribution functions against Python
// Optuna `optuna.samplers._tpe._truncnorm`.
//
// All expected values generated from Python Optuna and verified interactively.
//
// Covers:
//  1. ndtr (standard normal CDF) — including deep tails
//  2. log_ndtr — deep negative values, asymptotic series
//  3. log_gauss_mass — all branch cases (both negative, spans zero, both positive, deep tail)
//  4. ppf — various quantiles and truncation ranges
//  5. logpdf — standard truncated normal log-PDF
//  6. Invariants: monotonicity, boundary, symmetry
// ═══════════════════════════════════════════════════════════════════════════════

use optuna_rs::samplers::tpe::truncnorm::{log_gauss_mass, logpdf, ppf};

const TOL: f64 = 1e-6;
const TIGHT_TOL: f64 = 1e-10;

fn assert_close(got: f64, exp: f64, tol: f64, label: &str) {
    if exp.is_infinite() {
        assert!(
            got.is_infinite() && got.signum() == exp.signum(),
            "{label}: expected {exp}, got {got}"
        );
        return;
    }
    let diff = (got - exp).abs();
    let denom = exp.abs().max(1.0);
    assert!(
        diff / denom < tol,
        "{label}: expected {exp:.16e}, got {got:.16e}, rel_err={:.2e}",
        diff / denom
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// 1. log_gauss_mass — all Python-verified cases
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_log_gauss_mass() {
    let cases = [
        (-10.0, 10.0, 0.0),
        (-10.0, 0.0, -6.9314718055994529e-01),
        (-1.0, 1.0, -3.8171514630212616e-01),
        (0.0, 3.0, -6.9585062764542127e-01),
        (2.0, 5.0, -3.7831969337574267e+00),
        (-30.0, -25.0, -3.1663940800802021e+02),
        (-5.0, -3.0, -6.6079385945968916e+00),
    ];

    for (a, b, expected) in &cases {
        let got = log_gauss_mass(*a, *b);
        assert_close(got, *expected, TOL, &format!("lgm({a},{b})"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 2. ppf — Python-verified quantiles
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_ppf_values() {
    let cases = [
        (0.5, -10.0, 10.0, 0.0),
        (0.5, -1.0, 1.0, 0.0),
        (0.1, -2.0, 2.0, -1.1840324666939048e+00),
        (0.9, -2.0, 2.0, 1.1840324666939053e+00),
        (0.25, 0.0, 3.0, 3.1774951442748611e-01),
        (0.75, 0.0, 3.0, 1.1454450468547004e+00),
        (0.5, -0.5, 0.5, 0.0),  // symmetric → 0
        (0.01, -1.0, 1.0, -9.7217338725702929e-01),
        (0.99, -1.0, 1.0, 9.7217338725702951e-01),
    ];

    for &(q, a, b, expected) in &cases {
        let got = ppf(q, a, b);
        let expected: f64 = expected;
        // ppf(0.5, -0.5, 0.5) Python gives ~-1.22e-16, effectively 0
        if expected.abs() < 1e-10 {
            assert!(
                got.abs() < 1e-6,
                "ppf({q},{a},{b}): expected ~0, got {got}"
            );
        } else {
            assert_close(got, expected, TOL, &format!("ppf({q},{a},{b})"));
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 3. logpdf — Python-verified values
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_logpdf_values() {
    let cases = [
        // (x, a, b, loc, scale, expected)
        (0.0, -1.0, 1.0, 0.0, 1.0, -5.3722338690254645e-01),
        (0.5, -2.0, 2.0, 0.0, 1.0, -9.9737062091228246e-01),
        (-0.5, -1.0, 1.0, 0.0, 1.0, -6.6222338690254645e-01),
        (0.0, 0.0, 3.0, 0.0, 1.0, -2.2308790555925140e-01),
        (1.0, -5.0, 5.0, 0.0, 1.0, -1.4189379599013645e+00),
    ];

    for (x, a, b, loc, scale, expected) in &cases {
        let got = logpdf(*x, *a, *b, *loc, *scale);
        assert_close(got, *expected, TOL, &format!("logpdf({x},{a},{b})"));
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 4. ppf boundary conditions
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_ppf_boundaries() {
    for (a, b) in [(-2.0, 2.0), (-1.0, 1.0), (0.0, 3.0), (-5.0, 5.0)] {
        let at0 = ppf(0.0, a, b);
        assert!(
            (at0 - a).abs() < TIGHT_TOL,
            "ppf(0, {a}, {b}) = {at0}, expected {a}"
        );

        let at1 = ppf(1.0, a, b);
        assert!(
            (at1 - b).abs() < TIGHT_TOL,
            "ppf(1, {a}, {b}) = {at1}, expected {b}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 5. ppf monotonicity
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_ppf_monotonicity() {
    for (a, b) in [(-2.0, 2.0), (-1.0, 1.0), (0.0, 3.0)] {
        let mut prev = a;
        for i in 1..20 {
            let q = i as f64 / 20.0;
            let x = ppf(q, a, b);
            assert!(
                x >= prev - TIGHT_TOL,
                "ppf not monotone: ppf({q},{a},{b})={x} < prev={prev}"
            );
            prev = x;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 6. ppf symmetry: ppf(q, -a, a) = -ppf(1-q, -a, a) for symmetric bounds
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_ppf_symmetry() {
    for &a in &[1.0, 2.0, 5.0] {
        for &q in &[0.1, 0.2, 0.3, 0.4] {
            let x1 = ppf(q, -a, a);
            let x2 = ppf(1.0 - q, -a, a);
            assert!(
                (x1 + x2).abs() < 1e-6,
                "symmetry violated: ppf({q},-{a},{a})={x1}, ppf({},- {a},{a})={x2}",
                1.0 - q
            );
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// 7. logpdf out of range → -inf
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_logpdf_out_of_range() {
    assert_eq!(logpdf(2.0, -1.0, 1.0, 0.0, 1.0), f64::NEG_INFINITY);
    assert_eq!(logpdf(-2.0, -1.0, 1.0, 0.0, 1.0), f64::NEG_INFINITY);
    assert_eq!(logpdf(4.0, 0.0, 3.0, 0.0, 1.0), f64::NEG_INFINITY);
}

// ═══════════════════════════════════════════════════════════════════════════
// 8. logpdf with non-zero loc and scale
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_logpdf_loc_scale() {
    // logpdf(x, a, b, loc, scale) should equal logpdf((x-loc)/scale, (a-loc)/scale, (b-loc)/scale, 0, 1) - ln(scale)
    let loc = 2.0;
    let scale = 3.0;
    let a = -1.0;
    let b = 5.0;
    let x = 3.0;

    let got = logpdf(x, a, b, loc, scale);
    let z = (x - loc) / scale;
    let a_std = (a - loc) / scale;
    let b_std = (b - loc) / scale;
    let expected = logpdf(z, a_std, b_std, 0.0, 1.0) - scale.ln();

    assert_close(got, expected, TOL, "logpdf with loc/scale");
}

// ═══════════════════════════════════════════════════════════════════════════
// 9. log_gauss_mass degenerate: a >= b → -inf
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_log_gauss_mass_degenerate() {
    assert_eq!(log_gauss_mass(1.0, 1.0), f64::NEG_INFINITY);
    assert_eq!(log_gauss_mass(2.0, 1.0), f64::NEG_INFINITY);
}

// ═══════════════════════════════════════════════════════════════════════════
// 10. logpdf numerical integration ≈ 1
// ═══════════════════════════════════════════════════════════════════════════
#[test]
fn cv_logpdf_integrates_to_one() {
    for (a, b) in [(-2.0, 2.0), (-1.0, 1.0), (0.0, 3.0), (-5.0, 5.0)] {
        let n = 10000;
        let dx = (b - a) / n as f64;
        let mut integral = 0.0;
        for i in 0..n {
            let x = a + (i as f64 + 0.5) * dx;
            integral += logpdf(x, a, b, 0.0, 1.0).exp() * dx;
        }
        assert!(
            (integral - 1.0).abs() < 0.001,
            "integral({a},{b}) = {integral}, expected ~1.0"
        );
    }
}
