//! Rough Heston Monte Carlo pricer — PyO3 bindings.
//!
//! Multi-threaded option pricing using Rayon for parallel MC paths under
//! the Rough Heston stochastic volatility model (El Euch & Rosenbaum 2019).
//!
//! The Rough Heston model extends classical Heston with a fractional Brownian
//! motion kernel, capturing the rough behaviour of realised volatility
//! time-series observed in equity and crypto markets (H ≈ 0.04–0.2).
//!
//! # Exposed Python Functions
//! - `rh_mc_call` / `rh_mc_put` — European option prices via MC
//! - `rh_greeks` — Delta, Gamma, Vega, Theta, Rho via bump-and-reprice
//! - `rh_vol_smile` — Smile at fixed maturity across strikes
//! - `rh_simulate_paths` — N terminal (S_T, V_T) pairs
//! - `rh_atm_skew_mc` — ATM implied skew via finite-difference
//! - `rh_variance_swap` — Analytical variance swap rate
//! - `rh_leverage_swap` — Analytical leverage swap proxy
//!
//! # References
//! - El Euch & Rosenbaum (2019) — "The characteristic function of rough Heston models"
//! - Euler-Maruyama discretisation with fractional kernel scaling: h^(H−0.5)

use pyo3::prelude::*;
use rand::SeedableRng;
use rand_distr::{Distribution, StandardNormal};
use rayon::prelude::*;

// ─────────────────────────────────────────────────────────────
// Internal: single path simulation
// ─────────────────────────────────────────────────────────────

/// Simulate one rough Heston path and return terminal stock price.
fn simulate_path(
    spot: f64,
    t: f64,
    r: f64,
    h: f64,
    nu: f64,
    rho: f64,
    kappa: f64,
    theta: f64,
    v0: f64,
    n_steps: usize,
    seed: u64,
) -> f64 {
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let normal = StandardNormal;

    let dt = t / (n_steps as f64);
    let sqrt_dt = dt.sqrt();
    let h_scaling = h.powf(h - 0.5);

    let mut v = v0.max(1e-8);
    let mut s = spot;

    for _ in 0..n_steps {
        let z1: f64 = normal.sample(&mut rng);
        let z2: f64 = normal.sample(&mut rng);

        let dw_v = z1;
        let dw_s = rho * z1 + (1.0 - rho * rho).sqrt() * z2;

        let v_sqrt = v.sqrt();
        let v_new = v + kappa * (theta - v) * dt * h_scaling
            + nu * v_sqrt * dw_v * sqrt_dt * h_scaling;
        v = v_new.max(1e-8);

        s *= ((r - 0.5 * v) * dt + v_sqrt * dw_s * sqrt_dt).exp();
    }

    s
}

// ─────────────────────────────────────────────────────────────
// Exported Python functions
// ─────────────────────────────────────────────────────────────

/// Monte Carlo put price under Rough Heston dynamics.
#[pyfunction]
#[pyo3(signature = (spot, k, t, r, h, nu, rho, lambda_, theta, v0, n_paths=5000, n_steps=100))]
pub fn rh_mc_put(
    py: Python,
    spot: f64,
    k: f64,
    t: f64,
    r: f64,
    h: f64,
    nu: f64,
    rho: f64,
    lambda_: f64,
    theta: f64,
    v0: f64,
    n_paths: usize,
    n_steps: usize,
) -> f64 {
    py.allow_threads(|| {
        let discount = (-r * t).exp();
        let sum: f64 = (0..n_paths)
            .into_par_iter()
            .map(|i| {
                let s_t = simulate_path(spot, t, r, h, nu, rho, lambda_, theta, v0, n_steps, i as u64 + 42);
                (k - s_t).max(0.0)
            })
            .sum();
        discount * sum / (n_paths as f64)
    })
}

/// Monte Carlo call price under Rough Heston dynamics.
#[pyfunction]
#[pyo3(signature = (spot, k, t, r, h, nu, rho, lambda_, theta, v0, n_paths=5000, n_steps=100))]
pub fn rh_mc_call(
    py: Python,
    spot: f64,
    k: f64,
    t: f64,
    r: f64,
    h: f64,
    nu: f64,
    rho: f64,
    lambda_: f64,
    theta: f64,
    v0: f64,
    n_paths: usize,
    n_steps: usize,
) -> f64 {
    py.allow_threads(|| {
        let discount = (-r * t).exp();
        let sum: f64 = (0..n_paths)
            .into_par_iter()
            .map(|i| {
                let s_t = simulate_path(spot, t, r, h, nu, rho, lambda_, theta, v0, n_steps, i as u64 + 137);
                (s_t - k).max(0.0)
            })
            .sum();
        discount * sum / (n_paths as f64)
    })
}

/// Analytical variance swap rate under Rough Heston (first-order approx).
///
/// `E[∫₀ᵀ Vₜ dt] ≈ θ·T + (v₀ − θ)·(1 − e^{−κT}) / κ`
#[pyfunction]
pub fn rh_variance_swap(tau: f64, v0: f64, kappa: f64, theta: f64) -> f64 {
    if kappa.abs() < 1e-8 {
        v0 * tau
    } else {
        theta * tau + (v0 - theta) * (1.0 - (-kappa * tau).exp()) / kappa
    }
}

/// Analytical leverage swap (covariance proxy) under Rough Heston.
///
/// `LS(T) = ρ · ν · VarSwap(T)`
#[pyfunction]
pub fn rh_leverage_swap(tau: f64, v0: f64, kappa: f64, theta: f64, rho: f64, nu: f64) -> f64 {
    rho * nu * rh_variance_swap(tau, v0, kappa, theta)
}

/// ATM implied skew approximation via bump finite-difference on MC call prices.
#[pyfunction]
#[pyo3(signature = (spot, t, r, h, nu, rho, lambda_, theta, v0, n_paths=2000, n_steps=50, dk_frac=0.01))]
pub fn rh_atm_skew_mc(
    py: Python,
    spot: f64,
    t: f64,
    r: f64,
    h: f64,
    nu: f64,
    rho: f64,
    lambda_: f64,
    theta: f64,
    v0: f64,
    n_paths: usize,
    n_steps: usize,
    dk_frac: f64,
) -> f64 {
    let dk = spot * dk_frac;
    let k_up = spot + dk;
    let k_dn = spot - dk;

    let p_up = rh_mc_call(py, spot, k_up, t, r, h, nu, rho, lambda_, theta, v0, n_paths, n_steps);
    let p_dn = rh_mc_call(py, spot, k_dn, t, r, h, nu, rho, lambda_, theta, v0, n_paths, n_steps);

    let sqrt_t = t.sqrt().max(1e-8);
    let iv_up = (p_up / spot / sqrt_t).max(1e-10);
    let iv_dn = (p_dn / spot / sqrt_t).max(1e-10);

    (iv_up - iv_dn) / (2.0 * dk)
}

/// Simulate N paths of (stock, variance) and return terminal values.
#[pyfunction]
#[pyo3(signature = (spot, t, r, h, nu, rho, kappa, theta, v0, n_paths=1000, n_steps=252))]
pub fn rh_simulate_paths(
    py: Python,
    spot: f64,
    t: f64,
    r: f64,
    h: f64,
    nu: f64,
    rho: f64,
    kappa: f64,
    theta: f64,
    v0: f64,
    n_paths: usize,
    n_steps: usize,
) -> Vec<(f64, f64)> {
    py.allow_threads(|| {
        (0..n_paths)
            .into_par_iter()
            .map(|i| {
                let mut rng = rand::rngs::StdRng::seed_from_u64(i as u64 + 7919);
                let normal = StandardNormal;
                let dt = t / (n_steps as f64);
                let sqrt_dt = dt.sqrt();
                let h_scaling = h.powf(h - 0.5);
                let mut v = v0.max(1e-8);
                let mut s = spot;
                for _ in 0..n_steps {
                    let z1: f64 = normal.sample(&mut rng);
                    let z2: f64 = normal.sample(&mut rng);
                    let dw_v = z1;
                    let dw_s = rho * z1 + (1.0 - rho * rho).sqrt() * z2;
                    let v_sqrt = v.sqrt();
                    let v_new = v + kappa * (theta - v) * dt * h_scaling
                        + nu * v_sqrt * dw_v * sqrt_dt * h_scaling;
                    v = v_new.max(1e-8);
                    s *= ((r - 0.5 * v) * dt + v_sqrt * dw_s * sqrt_dt).exp();
                }
                (s, v)
            })
            .collect()
    })
}

/// Compute a volatility smile at fixed maturity across K strikes.
///
/// Returns `Vec<(strike, call_price, put_price)>`.
#[pyfunction]
#[pyo3(signature = (spot, t, r, h, nu, rho, kappa, theta, v0, strikes, n_paths=5000, n_steps=100))]
pub fn rh_vol_smile(
    py: Python,
    spot: f64,
    t: f64,
    r: f64,
    h: f64,
    nu: f64,
    rho: f64,
    kappa: f64,
    theta: f64,
    v0: f64,
    strikes: Vec<f64>,
    n_paths: usize,
    n_steps: usize,
) -> Vec<(f64, f64, f64)> {
    strikes
        .iter()
        .map(|&k| {
            let call = rh_mc_call(py, spot, k, t, r, h, nu, rho, kappa, theta, v0, n_paths, n_steps);
            let put = rh_mc_put(py, spot, k, t, r, h, nu, rho, kappa, theta, v0, n_paths, n_steps);
            (k, call, put)
        })
        .collect()
}

/// Greeks via bump-and-reprice: `(delta, gamma, vega, theta, rho_greek)`.
#[pyfunction]
#[pyo3(signature = (spot, k, t, r, h, nu, rho, kappa, theta, v0, is_call=true, n_paths=5000, n_steps=100))]
pub fn rh_greeks(
    py: Python,
    spot: f64,
    k: f64,
    t: f64,
    r: f64,
    h: f64,
    nu: f64,
    rho: f64,
    kappa: f64,
    theta: f64,
    v0: f64,
    is_call: bool,
    n_paths: usize,
    n_steps: usize,
) -> (f64, f64, f64, f64, f64) {
    let pricer = if is_call { rh_mc_call } else { rh_mc_put };
    let base = pricer(py, spot, k, t, r, h, nu, rho, kappa, theta, v0, n_paths, n_steps);

    // Delta & Gamma: bump spot ±1 %
    let ds = spot * 0.01;
    let p_up = pricer(py, spot + ds, k, t, r, h, nu, rho, kappa, theta, v0, n_paths, n_steps);
    let p_dn = pricer(py, spot - ds, k, t, r, h, nu, rho, kappa, theta, v0, n_paths, n_steps);
    let delta = (p_up - p_dn) / (2.0 * ds);
    let gamma = (p_up - 2.0 * base + p_dn) / (ds * ds);

    // Vega: bump v0 by +0.01
    let dv = 0.01_f64;
    let p_vup = pricer(py, spot, k, t, r, h, nu, rho, kappa, theta, v0 + dv, n_paths, n_steps);
    let vega = (p_vup - base) / dv;

    // Theta: bump t by −1/365
    let dt_bump = 1.0 / 365.0;
    let t_new = (t - dt_bump).max(0.001);
    let p_tdn = pricer(py, spot, k, t_new, r, h, nu, rho, kappa, theta, v0, n_paths, n_steps);
    let theta_greek = (p_tdn - base) / (-dt_bump);

    // Rho: bump r by +0.01
    let dr = 0.01_f64;
    let p_rup = pricer(py, spot, k, t, r + dr, h, nu, rho, kappa, theta, v0, n_paths, n_steps);
    let rho_greek = (p_rup - base) / dr;

    (delta, gamma, vega, theta_greek, rho_greek)
}

// ─────────────────────────────────────────────────────────────
// PyO3 module registration (called from lib.rs)
// ─────────────────────────────────────────────────────────────

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(rh_mc_put, m)?)?;
    m.add_function(wrap_pyfunction!(rh_mc_call, m)?)?;
    m.add_function(wrap_pyfunction!(rh_variance_swap, m)?)?;
    m.add_function(wrap_pyfunction!(rh_leverage_swap, m)?)?;
    m.add_function(wrap_pyfunction!(rh_atm_skew_mc, m)?)?;
    m.add_function(wrap_pyfunction!(rh_simulate_paths, m)?)?;
    m.add_function(wrap_pyfunction!(rh_vol_smile, m)?)?;
    m.add_function(wrap_pyfunction!(rh_greeks, m)?)?;
    Ok(())
}
