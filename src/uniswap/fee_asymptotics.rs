//! # Uniswap v3 fee asymptotics (Echenim, Gobet, Maurice 2025)
//!
//! Implements §4 of:
//! Echenim, Gobet, Maurice — "Uniswap v3: impermanent loss modeling and
//! swap fees asymptotic analysis" (HAL-04214315v4, Sep 2025).
//!
//! Main result (Echenim-Gobet Thm 4.1, informal): under the arbitrage
//! model of Angeris et al. 2021 with rebalancing band `γ → 0`, the
//! collected swap-fee value over `[0, T]` converges to
//!
//! ```text
//!   F = (φ / (1-φ)) · L · ∫_{s_a}^{s_b} ν_T(P) · g(P) dP
//! ```
//!
//! where `ν_T(P)` is the *expected occupation density* of the price
//! process at level `P` over `[0, T]`, and the kernel `g(P)` weights
//! the contribution of a unit liquidity at price `P` to fee revenue:
//!
//! ```text
//!   g(P) = 1 / (2 · sqrt(P))
//! ```
//!
//! The paper further shows (Cor 4.3) that for a GBM `dP = σ·P·dW`, the
//! occupation density admits a closed form via the local time, and the
//! expected fee value can be written as an integral of European call /
//! put prices struck at all `K ∈ [P_a, P_b]`. We expose:
//!
//!   - [`fee_kernel`] the per-price weighting `g(P)`.
//!   - [`expected_fee_gbm`] closed-form expected fee under GBM with
//!     no drift, computed by trapezoidal integration of Black-Scholes
//!     calls over the range `[P_a, P_b]`.
//!
//! Pure functions, no I/O.

use super::{UniswapError, UniswapResult};

/// Fee kernel `g(P) = 1 / (2 · sqrt(P))` (Echenim-Gobet eq. 4.5).
pub fn fee_kernel(price: f64) -> UniswapResult<f64> {
    if price <= 0.0 {
        return Err(UniswapError::InvalidInput);
    }
    Ok(1.0 / (2.0 * price.sqrt()))
}

/// Standard normal CDF via `erf` approximation (Abramowitz-Stegun 7.1.26).
fn norm_cdf(x: f64) -> f64 {
    // erf approximation with max error ~1.5e-7
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let xa = (x / std::f64::consts::SQRT_2).abs();
    let t = 1.0 / (1.0 + p * xa);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-xa * xa).exp();
    0.5 * (1.0 + sign * y)
}

/// Black-Scholes call price for a non-dividend asset (rate r=0):
/// ```text
///   C(S, K, σ, T) = S · N(d1) - K · N(d2)
///   d1 = (ln(S/K) + 0.5 σ² T) / (σ √T)
///   d2 = d1 - σ √T
/// ```
pub fn black_scholes_call(spot: f64, strike: f64, sigma: f64, t: f64) -> UniswapResult<f64> {
    if spot <= 0.0 || strike <= 0.0 || sigma <= 0.0 || t <= 0.0 {
        return Err(UniswapError::InvalidInput);
    }
    let sqrt_t = t.sqrt();
    let d1 = ((spot / strike).ln() + 0.5 * sigma * sigma * t) / (sigma * sqrt_t);
    let d2 = d1 - sigma * sqrt_t;
    Ok(spot * norm_cdf(d1) - strike * norm_cdf(d2))
}

/// Expected swap-fee value over `[0, T]` for a v3 LP position with
/// liquidity `L` over price range `[p_a, p_b]`, fee rate `phi`, and
/// price following GBM `dP = σ·P·dW` starting at `P0`.
///
/// Echenim-Gobet Cor 4.3 (specialised form for r=0):
///
/// ```text
///   E[F] ≈ (φ / (1-φ)) · L · ∫_{p_a}^{p_b} g(K) · ψ(K, T) dK
/// ```
///
/// where `ψ(K, T)` is the second derivative of the call price
/// w.r.t. strike (the Breeden-Litzenberger density). We compute the
/// integral by trapezoidal rule with `n_grid` points and approximate
/// `ψ ≈ ∂²C/∂K²` by central differences with step `h`.
pub fn expected_fee_gbm(
    liquidity: f64,
    p0: f64,
    p_a: f64,
    p_b: f64,
    fee_phi: f64,
    sigma: f64,
    t: f64,
    n_grid: usize,
) -> UniswapResult<f64> {
    if liquidity <= 0.0 || p0 <= 0.0 || p_a <= 0.0 || p_b <= p_a
        || fee_phi <= 0.0 || fee_phi >= 1.0 || sigma <= 0.0 || t <= 0.0 || n_grid < 8
    {
        return Err(UniswapError::InvalidInput);
    }
    let dk = (p_b - p_a) / (n_grid as f64);
    let h = 0.5 * dk; // central-difference step
    let mut acc = 0.0;
    for i in 0..=n_grid {
        let k = p_a + (i as f64) * dk;
        let k_lo = (k - h).max(1e-12);
        let k_hi = k + h;
        let c_lo = black_scholes_call(p0, k_lo, sigma, t)?;
        let c_mid = black_scholes_call(p0, k, sigma, t)?;
        let c_hi = black_scholes_call(p0, k_hi, sigma, t)?;
        let psi = (c_hi - 2.0 * c_mid + c_lo) / (h * h);
        let weight = if i == 0 || i == n_grid { 0.5 } else { 1.0 };
        acc += weight * fee_kernel(k)? * psi.max(0.0) * dk;
    }
    Ok((fee_phi / (1.0 - fee_phi)) * liquidity * acc)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "expected {b}, got {a}");
    }

    #[test]
    fn fee_kernel_at_unity_is_half() {
        approx(fee_kernel(1.0).unwrap(), 0.5, 1e-12);
    }

    #[test]
    fn norm_cdf_at_zero_is_half() {
        approx(norm_cdf(0.0), 0.5, 1e-6);
    }

    #[test]
    fn norm_cdf_tails() {
        assert!(norm_cdf(-5.0) < 1e-5);
        assert!(norm_cdf(5.0) > 1.0 - 1e-5);
    }

    #[test]
    fn black_scholes_atm_increases_with_sigma() {
        let c1 = black_scholes_call(100.0, 100.0, 0.20, 1.0).unwrap();
        let c2 = black_scholes_call(100.0, 100.0, 0.40, 1.0).unwrap();
        assert!(c2 > c1, "vol-monotonicity failed: {c1} vs {c2}");
    }

    #[test]
    fn expected_fee_positive_and_grows_with_sigma() {
        // Use a WIDE price range so neither σ truncates the GBM distribution.
        let f1 = expected_fee_gbm(1_000.0, 1.0, 0.05, 20.0, 0.003, 0.20, 1.0, 256).unwrap();
        let f2 = expected_fee_gbm(1_000.0, 1.0, 0.05, 20.0, 0.003, 0.40, 1.0, 256).unwrap();
        assert!(f1 > 0.0, "expected positive fee, got {f1}");
        assert!(f2 > f1, "fee should grow with vol over a wide range: {f1} vs {f2}");
    }

    #[test]
    fn expected_fee_grows_with_liquidity() {
        let f1 = expected_fee_gbm(1_000.0, 1.0, 0.5, 2.0, 0.003, 0.30, 1.0, 64).unwrap();
        let f2 = expected_fee_gbm(2_000.0, 1.0, 0.5, 2.0, 0.003, 0.30, 1.0, 64).unwrap();
        approx(f2 / f1, 2.0, 1e-9);
    }
}
