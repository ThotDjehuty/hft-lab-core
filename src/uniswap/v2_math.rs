//! # Uniswap v2 / Constant Product Market Maker (CPMM) exact math
//!
//! Pure functions, no I/O. Models published in:
//!   - Adams et al. 2020 (Uniswap v2 whitepaper)
//!   - Wang, Chen, Wu, Zhou, Deng, Wattenhofer 2022 — "Cyclic Arbitrage
//!     in Decentralized Exchanges" (arXiv:2105.02784v3)
//!   - Cartea, Drissi, Monga 2025 — "DeFi & AMM: Execution and
//!     Speculation" (arXiv:2307.03499v3) — eq. (2.4) convexity costs
//!
//! All quantities are `f64`. The crate consumer is responsible for
//! converting on-chain `U256` reserves to floating point with the
//! correct decimals scaling.

use super::{UniswapError, UniswapResult};

/// Output amount for an exact-input swap on a v2-style CPMM with fee in
/// basis points (e.g. `30` for 0.30%).
///
/// Formula (Adams et al. 2020, eq. 3):
/// ```text
///   dy = (R_y * dx_eff) / (R_x + dx_eff),    dx_eff = dx * (1 - φ)
/// ```
/// where φ = `fee_bps / 10_000`.
///
/// Returns `InvalidInput` if reserves are non-positive or `dx <= 0`.
pub fn amount_out_given_in(
    reserve_in: f64,
    reserve_out: f64,
    amount_in: f64,
    fee_bps: u32,
) -> UniswapResult<f64> {
    if reserve_in <= 0.0 || reserve_out <= 0.0 || amount_in <= 0.0 {
        return Err(UniswapError::InvalidInput);
    }
    let phi = fee_bps as f64 / 10_000.0;
    let dx_eff = amount_in * (1.0 - phi);
    Ok(reserve_out * dx_eff / (reserve_in + dx_eff))
}

/// Spot (marginal) price of asset Y in units of X for a v2 CPMM:
/// `p = R_x / R_y`. Excludes fee.
pub fn spot_price(reserve_in: f64, reserve_out: f64) -> UniswapResult<f64> {
    if reserve_in <= 0.0 || reserve_out <= 0.0 {
        return Err(UniswapError::InvalidInput);
    }
    Ok(reserve_in / reserve_out)
}

/// Convexity cost coefficient κ from Cartea-Drissi-Monga §2 eq. (2.4):
///
/// ```text
///   κ(R_x, R_y, S) = S^2 / R_y     (= S / R_x because S = R_x/R_y)
/// ```
///
/// Execution cost of an LT order of size `ν`-per-unit-time over `dt` is
/// `κ · ν^2 · dt`. Returned in *price-units squared per token*.
pub fn convexity_cost_coefficient(reserve_in: f64, reserve_out: f64) -> UniswapResult<f64> {
    let s = spot_price(reserve_in, reserve_out)?;
    Ok((s * s) / reserve_out)
}

/// Optimal arbitrage size between an external reference price `p_ext`
/// (CEX mid in *X-per-Y* units, i.e. same units as the CPMM spot
/// `R_x / R_y`) and a v2 CPMM with reserves `(R_x, R_y)` and fee
/// `fee_bps`.
///
/// Derivation (from the FOC of `profit(dx) = γ·R_y·p_ext·dx /
/// (R_x + γ·dx) - dx`):
///
/// ```text
///   Case A:  p_ext > p_dex = R_x/R_y    (Y cheap on DEX, sell on CEX)
///       dx* = ( sqrt(γ · R_x · R_y · p_ext) - R_x ) / γ
///
///   Case B:  p_ext < p_dex                (Y expensive on DEX, buy on CEX)
///       dy* = ( sqrt(γ · R_x · R_y / p_ext) - R_y ) / γ
/// ```
///
/// Returns the *signed* size: positive `dx` (BUY Y on DEX with X input)
/// or negative `-dy` (SELL Y on DEX with Y input). Returns `0.0` when
/// no profitable trade exists (e.g. fees exceed the discrepancy).
pub fn optimal_arb_size_two_venues(
    reserve_in: f64,
    reserve_out: f64,
    fee_bps: u32,
    p_ext: f64,
) -> UniswapResult<f64> {
    if reserve_in <= 0.0 || reserve_out <= 0.0 || p_ext <= 0.0 {
        return Err(UniswapError::InvalidInput);
    }
    let gamma = 1.0 - (fee_bps as f64 / 10_000.0);
    let p_dex = reserve_in / reserve_out;
    if (p_ext - p_dex).abs() < 1e-15 {
        return Ok(0.0);
    }
    if p_ext > p_dex {
        // Case A: buy Y on DEX (input X)
        let dx = ((gamma * reserve_in * reserve_out * p_ext).sqrt() - reserve_in) / gamma;
        if dx > 0.0 { Ok(dx) } else { Ok(0.0) }
    } else {
        // Case B: sell Y on DEX (input Y)
        let dy = ((gamma * reserve_in * reserve_out / p_ext).sqrt() - reserve_out) / gamma;
        if dy > 0.0 { Ok(-dy) } else { Ok(0.0) }
    }
}

/// Profit (in units of `X`) of an arbitrage of size `dx` on a CPM
/// against an external price `p_ext`. Negative if the trade is a loss.
pub fn arb_profit_in_x(
    reserve_in: f64,
    reserve_out: f64,
    fee_bps: u32,
    p_ext: f64,
    dx: f64,
) -> UniswapResult<f64> {
    if dx == 0.0 {
        return Ok(0.0);
    }
    if dx > 0.0 {
        // BUY Y on CPM with `dx` of X, SELL Y on CEX at p_ext
        let dy = amount_out_given_in(reserve_in, reserve_out, dx, fee_bps)?;
        Ok(dy * p_ext - dx)
    } else {
        // SELL Y on CPM (input = -dx of Y), BUY Y on CEX
        let dy_in = -dx;
        let dx_out = amount_out_given_in(reserve_out, reserve_in, dy_in, fee_bps)?;
        Ok(dx_out - dy_in * p_ext)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "expected {b}, got {a}");
    }

    #[test]
    fn amount_out_matches_uniswap_v2_invariant() {
        // R_x = 1000, R_y = 2000, fee = 30bps, dx = 100
        let dy = amount_out_given_in(1000.0, 2000.0, 100.0, 30).unwrap();
        // dx_eff = 100 * 0.997 = 99.7
        // dy = 2000 * 99.7 / (1000 + 99.7) = 199400 / 1099.7
        approx(dy, 199_400.0 / 1099.7, 1e-9);
    }

    #[test]
    fn spot_price_is_ratio() {
        approx(spot_price(1000.0, 2000.0).unwrap(), 0.5, 1e-12);
    }

    #[test]
    fn convexity_coefficient_matches_cartea_eq_24() {
        // S = 1000 / 2000 = 0.5, κ = S^2 / R_y = 0.25 / 2000 = 1.25e-4
        approx(
            convexity_cost_coefficient(1000.0, 2000.0).unwrap(),
            1.25e-4,
            1e-12,
        );
    }

    #[test]
    fn optimal_arb_zeros_at_no_discrepancy() {
        // No fee, p_ext == spot → dx* should be ~0
        let r_x = 1000.0;
        let r_y = 2000.0;
        let p = r_x / r_y; // 0.5
        let dx = optimal_arb_size_two_venues(r_x, r_y, 0, p).unwrap();
        approx(dx, 0.0, 1e-9);
    }

    #[test]
    fn optimal_arb_buys_when_cex_above_dex() {
        // DEX price = 0.5 (X per Y). CEX price = 0.6 → Y is "expensive" on CEX,
        // so BUY Y cheaply on DEX (positive dx of X input). Expect dx > 0.
        let dx = optimal_arb_size_two_venues(1000.0, 2000.0, 30, 0.6).unwrap();
        assert!(dx > 0.0, "expected buy-on-dex direction, got dx = {dx}");
        // Profit at optimal must be > 0
        let pnl = arb_profit_in_x(1000.0, 2000.0, 30, 0.6, dx).unwrap();
        assert!(pnl > 0.0, "optimal arb should be profitable, got {pnl}");
    }

    #[test]
    fn optimal_arb_sells_when_cex_below_dex() {
        // CEX price = 0.4 < DEX 0.5 → SELL Y on DEX (negative direction)
        let d = optimal_arb_size_two_venues(1000.0, 2000.0, 30, 0.4).unwrap();
        assert!(d < 0.0, "expected sell-on-dex direction, got {d}");
        let pnl = arb_profit_in_x(1000.0, 2000.0, 30, 0.4, d).unwrap();
        assert!(pnl > 0.0, "optimal arb should be profitable, got {pnl}");
    }

    #[test]
    fn arb_profit_concave_in_dx() {
        // Profit at optimal must beat profit at 0.5x and 1.5x of optimal.
        let dx_star = optimal_arb_size_two_venues(1000.0, 2000.0, 30, 0.7).unwrap();
        let pnl_star = arb_profit_in_x(1000.0, 2000.0, 30, 0.7, dx_star).unwrap();
        let pnl_lo = arb_profit_in_x(1000.0, 2000.0, 30, 0.7, 0.5 * dx_star).unwrap();
        let pnl_hi = arb_profit_in_x(1000.0, 2000.0, 30, 0.7, 1.5 * dx_star).unwrap();
        assert!(pnl_star >= pnl_lo - 1e-9, "{pnl_star} < {pnl_lo}");
        assert!(pnl_star >= pnl_hi - 1e-9, "{pnl_star} < {pnl_hi}");
    }
}
