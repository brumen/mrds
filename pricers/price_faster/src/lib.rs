use pyo3::prelude::*;


/// computes the cdf od the floating point number
fn cdf(x: f64) -> f64 {

    let L = if x<0. {-x} else {x};
    let K = 1. / (1. + 0.2316419 * L);
    let w = 1. - 1. / (2. * std::f64::consts::PI).sqrt() * (-L * L / 2.).exp() * (
        0.31938153 * K - 0.356563782 * K*K + 1.781477937 * K.powi(3) -1.821255978 * K.powi(4) + 1.330274429 * K.powi(5)
    );

    if x < 0. {
        return 1. - w;
    }

    w
}

type JWSS7_STRUCT = (f64, f64, f64, f64, f64, f64, f64, );

fn _transform_from_jwss7(jwss7_p: JWSS7_STRUCT) -> JWSS7_STRUCT {

    let (sigma_0, skew, smile, put_slope, put_bend, call_slope, call_bend) = jwss7_p;

    let B = (2. * skew + put_slope) / (put_slope + call_slope);
    let A = 0.5 * B * (1. - B) * (call_slope + put_slope).powi(2) / (smile + skew*skew);
    let C = call_slope/A;
    let P = put_slope/A;
    let alpha_C = call_bend;
    let alpha_P = put_bend;

    (sigma_0, A, B, C, P, alpha_C, alpha_P)
}

fn _compute_from_jw7(z: f64, jw7_p: JWSS7_STRUCT) -> f64 {
    let (sigma_0, A, B, C, P, alpha_C, alpha_P) = jw7_p;

    // skew = d(vol)/dz  | z = 0
    // smile = d^2(vol)/dz^2 | z=0
    let first_term = B * (C * (z / (1.0 + z*z).powf(alpha_C/2.))).exp();
    let second_term = (1. - B) * (- P * (z / (1.0 + z*z).powf(alpha_P/2.))).exp();

    sigma_0 * (
        1. + A * (first_term + second_term).ln()
    ).sqrt()
}

fn _vol_from_jw7(S0: f64, K: f64, ttm: f64, jw7_p: JWSS7_STRUCT) -> f64 {
    let sigma_0 = jw7_p.0;
    let z = (K/S0).ln() / sigma_0 / ttm.sqrt();

    _compute_from_jw7(z, jw7_p)
}

enum CallPut {
    Call,
    Put,
}

fn black_fast(S_0: f64,  K: f64, r: f64, sigma: f64, T: f64, call_put: CallPut) -> f64 {

    let d1 = ((S_0/K).ln() + 0.5 * sigma * sigma * T) / (sigma * T.sqrt());
    let d2 = d1 - sigma * T.sqrt();

    let internal = match call_put {
        CallPut::Call => S_0 * cdf(d1) - K * cdf(d2),
        CallPut::Put => K * cdf(-d2) - S_0 * cdf (-d1),
    };

    (-r * T).exp() * internal
}

fn _calibrate_jwss7(
    jwss7_p: JWSS7_STRUCT,
    S0: f64,
    r: f64,
    ttm: f64,
    prices_strikes_cp: Vec<(f64, f64, CallPut)>,
) -> f64 {
    let mut diffs = 0.;
    for (price, strike, call_put) in prices_strikes_cp {
        let jw7_p = _transform_from_jwss7(jwss7_p);
        let vol = _vol_from_jw7(S0, strike, ttm, jw7_p);
        let black_price = black_fast(S0, strike, r, vol, ttm, call_put);
        diffs += (black_price - price).powi(2);
    }
    diffs
}

/// Formats the sum of two numbers as string.
#[pyfunction]
fn calibrate_jwss7(
    jwss7_p: JWSS7_STRUCT,
    S0: f64,
    r: f64,
    ttm: f64,
    prices_strikes_cp: Vec<(f64, f64, CallPut)>,
) -> PyResult<f64> {
    let res = _calibrate_jwss7(jwss7_p, S0, r, ttm, prices_strikes_cp);
    Ok(res)
}

/// A Python module implemented in Rust.
#[pymodule]
fn price_faster(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(calibrate_jwss7, m)?)?;
    Ok(())
}
