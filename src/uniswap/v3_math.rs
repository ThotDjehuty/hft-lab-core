//! # Uniswap v3 — concentrated liquidity math (Echenim, Gobet, Maurice 2025)
//!
//! Implements the formulas from:
//! Echenim, Gobet, Maurice — "Uniswap v3: impermanent loss modeling and
//! swap fees asymptotic analysis" (HAL-04214315v4, Sep 2025).
//!
//! Conventions (matching the Uniswap v3 whitepaper §6):
//!   - Prices are quoted as `P = R_y / R_x` where token0 = X, token1 = Y.
//!     (NOTE: this is the inverse of `v2_math::spot_price`; v3 uses
//!     P = price of *token0 in token1*.)
//!   - The sqrt-price `s = sqrt(P)` lives in `[s_a, s_b]` for a position
//!     with tick range `[i_a, i_b]`. We expose `tick_to_sqrt_price` and
//!     `sqrt_price_to_tick` helpers.
//!   - Liquidity `L` is the constant such that within an active range:
//!         x = L · (1/s - 1/s_b)        (token0 amount, eq. 6.29 v3 wp)
//!         y = L · (s - s_a)            (token1 amount, eq. 6.30 v3 wp)
//!
//! Pure functions, no I/O.

use super::{UniswapError, UniswapResult};

/// Uniswap v3 base for tick→price: 1.0001.
const TICK_BASE: f64 = 1.0001;

/// Convert tick `i` to sqrt-price `sqrt(1.0001^i)`.
pub fn tick_to_sqrt_price(tick: i32) -> f64 {
    TICK_BASE.powf(tick as f64 / 2.0)
}

/// Convert sqrt-price to nearest integer tick `floor(2 · log_{1.0001}(s))`.
pub fn sqrt_price_to_tick(sqrt_price: f64) -> i32 {
    if sqrt_price <= 0.0 {
        return i32::MIN;
    }
    (2.0 * sqrt_price.ln() / TICK_BASE.ln()).floor() as i32
}

/// Token0 amount held by a position with liquidity `L` over
/// `[tick_lower, tick_upper]` when current sqrt-price is `s`.
///
/// Echenim-Gobet eq. (2.6):
/// ```text
///   x(s) = L · ( 1/max(s, s_a) - 1/min(s_b, max(s, s_a)) )
/// ```
pub fn amount0_in_range(liquidity: f64, sqrt_price: f64, tick_lower: i32, tick_upper: i32) -> UniswapResult<f64> {
    if liquidity <= 0.0 || sqrt_price <= 0.0 || tick_lower >= tick_upper {
        return Err(UniswapError::InvalidInput);
    }
    let s_a = tick_to_sqrt_price(tick_lower);
    let s_b = tick_to_sqrt_price(tick_upper);
    let s = sqrt_price.clamp(s_a, s_b);
    Ok(liquidity * (1.0 / s - 1.0 / s_b))
}

/// Token1 amount held by a position with liquidity `L` over
/// `[tick_lower, tick_upper]` when current sqrt-price is `s`.
///
/// Echenim-Gobet eq. (2.7):
/// ```text
///   y(s) = L · ( min(s_b, max(s, s_a)) - s_a )
/// ```
pub fn amount1_in_range(liquidity: f64, sqrt_price: f64, tick_lower: i32, tick_upper: i32) -> UniswapResult<f64> {
    if liquidity <= 0.0 || sqrt_price <= 0.0 || tick_lower >= tick_upper {
        return Err(UniswapError::InvalidInput);
    }
    let s_a = tick_to_sqrt_price(tick_lower);
    let s_b = tick_to_sqrt_price(tick_upper);
    let s = sqrt_price.clamp(s_a, s_b);
    Ok(liquidity * (s - s_a))
}

/// Result of an exact-input swap *within a single tick* (no tick crossing).
#[derive(Debug, Clone, Copy)]
pub struct V3SwapWithinTick {
    pub amount_in_used: f64,
    pub amount_out: f64,
    pub new_sqrt_price: f64,
}

/// Exact-input swap of token0 → token1 within the active tick.
///
/// Uniswap v3 whitepaper eq. (6.16):
/// ```text
///   s_new = (L · s) / (L + dx · s)
///   dy    = L · (s - s_new)
/// ```
///
/// `dx` is the input of token0 (assumed already net of swap fee).
pub fn swap_token0_for_token1_within_tick(
    liquidity: f64,
    sqrt_price: f64,
    amount_in: f64,
) -> UniswapResult<V3SwapWithinTick> {
    if liquidity <= 0.0 || sqrt_price <= 0.0 || amount_in <= 0.0 {
        return Err(UniswapError::InvalidInput);
    }
    let s_new = (liquidity * sqrt_price) / (liquidity + amount_in * sqrt_price);
    let dy = liquidity * (sqrt_price - s_new);
    Ok(V3SwapWithinTick {
        amount_in_used: amount_in,
        amount_out: dy,
        new_sqrt_price: s_new,
    })
}

/// Exact-input swap of token1 → token0 within the active tick.
///
/// Uniswap v3 whitepaper eq. (6.13):
/// ```text
///   s_new = s + dy / L
///   dx    = L · (1/s - 1/s_new)
/// ```
pub fn swap_token1_for_token0_within_tick(
    liquidity: f64,
    sqrt_price: f64,
    amount_in: f64,
) -> UniswapResult<V3SwapWithinTick> {
    if liquidity <= 0.0 || sqrt_price <= 0.0 || amount_in <= 0.0 {
        return Err(UniswapError::InvalidInput);
    }
    let s_new = sqrt_price + amount_in / liquidity;
    let dx = liquidity * (1.0 / sqrt_price - 1.0 / s_new);
    Ok(V3SwapWithinTick {
        amount_in_used: amount_in,
        amount_out: dx,
        new_sqrt_price: s_new,
    })
}

/// Impermanent loss of a v3 LP position over a single tick range
/// `[i_a, i_b]` between initial price `P0` and final price `P1`.
///
/// Following Echenim-Gobet §3 (eq. 3.5), IL is the *value loss vs the
/// HODL portfolio* of the same initial token0/token1 quantities:
///
/// ```text
///   IL(P0, P1) = V_pool(P1) / V_hodl(P1) - 1
/// ```
///
/// where value is measured in token1 (i.e. `V = x · P + y`).
///
/// `IL ∈ (-∞, 0]`. Returns 0 when prices coincide.
pub fn impermanent_loss_v3(
    liquidity: f64,
    p0: f64,
    p1: f64,
    tick_lower: i32,
    tick_upper: i32,
) -> UniswapResult<f64> {
    if liquidity <= 0.0 || p0 <= 0.0 || p1 <= 0.0 || tick_lower >= tick_upper {
        return Err(UniswapError::InvalidInput);
    }
    let s0 = p0.sqrt();
    let s1 = p1.sqrt();
    let x0 = amount0_in_range(liquidity, s0, tick_lower, tick_upper)?;
    let y0 = amount1_in_range(liquidity, s0, tick_lower, tick_upper)?;
    let x1 = amount0_in_range(liquidity, s1, tick_lower, tick_upper)?;
    let y1 = amount1_in_range(liquidity, s1, tick_lower, tick_upper)?;
    let v_pool = x1 * p1 + y1;
    let v_hodl = x0 * p1 + y0;
    if v_hodl <= 0.0 {
        return Err(UniswapError::InvalidInput);
    }
    Ok(v_pool / v_hodl - 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "expected {b}, got {a}");
    }

    #[test]
    fn tick_zero_is_unit_sqrt_price() {
        approx(tick_to_sqrt_price(0), 1.0, 1e-12);
    }

    #[test]
    fn tick_roundtrip_close() {
        for tick in [-20_000, -100, 0, 100, 20_000].iter() {
            let s = tick_to_sqrt_price(*tick);
            let t = sqrt_price_to_tick(s);
            assert!((t - *tick).abs() <= 1, "{} -> {} -> {}", *tick, s, t);
        }
    }

    #[test]
    fn amounts_at_lower_bound_are_pure_token0() {
        // At s = s_a, position holds only token0 (eq. 2.7 ⇒ y = 0).
        let l = 1_000.0;
        let (a, b) = (-1000, 1000);
        let s_a = tick_to_sqrt_price(a);
        let y = amount1_in_range(l, s_a, a, b).unwrap();
        approx(y, 0.0, 1e-9);
        let x = amount0_in_range(l, s_a, a, b).unwrap();
        assert!(x > 0.0);
    }

    #[test]
    fn amounts_at_upper_bound_are_pure_token1() {
        let l = 1_000.0;
        let (a, b) = (-1000, 1000);
        let s_b = tick_to_sqrt_price(b);
        let x = amount0_in_range(l, s_b, a, b).unwrap();
        approx(x, 0.0, 1e-9);
        let y = amount1_in_range(l, s_b, a, b).unwrap();
        assert!(y > 0.0);
    }

    #[test]
    fn swap0for1_preserves_invariant() {
        // After x0→y, sqrt-price decreases; check L·(s - s_new) = dy
        let l = 10_000.0;
        let s = 1.0;
        let r = swap_token0_for_token1_within_tick(l, s, 50.0).unwrap();
        approx(l * (s - r.new_sqrt_price), r.amount_out, 1e-9);
        assert!(r.new_sqrt_price < s, "price must move down");
    }

    #[test]
    fn swap1for0_preserves_invariant() {
        let l = 10_000.0;
        let s = 1.0;
        let r = swap_token1_for_token0_within_tick(l, s, 50.0).unwrap();
        // dx = L·(1/s - 1/s_new)
        approx(l * (1.0 / s - 1.0 / r.new_sqrt_price), r.amount_out, 1e-9);
        assert!(r.new_sqrt_price > s, "price must move up");
    }

    #[test]
    fn impermanent_loss_zero_at_no_move() {
        let il = impermanent_loss_v3(1_000.0, 1.0, 1.0, -1000, 1000).unwrap();
        approx(il, 0.0, 1e-12);
    }

    #[test]
    fn impermanent_loss_negative_when_price_moves() {
        // Move price up by 50% within the range
        let il = impermanent_loss_v3(1_000.0, 1.0, 1.5, -10_000, 10_000).unwrap();
        assert!(il < 0.0, "IL must be negative for any price move, got {il}");
        // And down
        let il2 = impermanent_loss_v3(1_000.0, 1.0, 0.7, -10_000, 10_000).unwrap();
        assert!(il2 < 0.0, "IL must be negative for any price move, got {il2}");
    }
}
