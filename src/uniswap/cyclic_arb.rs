//! # Cyclic (triangular) arbitrage on AMM DEXes
//!
//! Implements the closed-form optimal-size and detection algorithms from:
//! Wang, Chen, Wu, Zhou, Deng, Wattenhofer 2022 — "Cyclic Arbitrage in
//! Decentralized Exchanges" (arXiv:2105.02784v3).
//!
//! Model: a *cycle* is a sequence of CPMM pools
//! `(R_in_1, R_out_1, fee_1) → (R_in_2, R_out_2, fee_2) → … → (R_in_n, R_out_n, fee_n)`
//! such that token-out of pool i = token-in of pool i+1, and the final
//! token-out equals the initial token-in (closing the cycle).
//!
//! The composed swap `dx → dy` through the cycle is (Wang et al. eq. 4):
//!
//! ```text
//!   dy(dx) = dx · ∏ γ_i / ( ∏ R_in_i / ∏ R_out_i + dx · (correction term) )
//! ```
//!
//! Differentiating the profit `π(dx) = dy(dx) - dx` gives the closed-form
//! optimal `dx*` (Wang et al. eq. 5).
//!
//! Pure functions, no I/O.

use super::{UniswapError, UniswapResult};

/// One leg of a cyclic arb path: `(reserve_in, reserve_out, fee_bps)`.
#[derive(Debug, Clone, Copy)]
pub struct PoolLeg {
    pub reserve_in: f64,
    pub reserve_out: f64,
    pub fee_bps: u32,
}

/// Compose the effective `(a, b)` such that `dy(dx) = a · dx / (1 + b · dx)`
/// for a sequence of CPMM legs. This follows by induction from
/// `dy_i = (γ_i · R_out_i · dx_i) / (R_in_i + γ_i · dx_i)`.
///
/// Returns `(a, b)`.
fn compose_cycle(legs: &[PoolLeg]) -> UniswapResult<(f64, f64)> {
    if legs.is_empty() {
        return Err(UniswapError::InvalidInput);
    }
    // After leg 1: dy = γ1·R_out1·dx / (R_in1 + γ1·dx)
    //              = (γ1·R_out1/R_in1) · dx / (1 + (γ1/R_in1)·dx)
    //   so a1 = γ1·R_out1/R_in1, b1 = γ1/R_in1.
    //
    // Composition with leg 2 (input = a1·dx/(1+b1·dx)):
    //   dy2 = γ2·R_out2 · u / (R_in2 + γ2 · u),  u = a1·dx/(1+b1·dx)
    //       = (γ2·R_out2 · a1 / R_in2) · dx / (1 + (b1 + γ2·a1/R_in2) · dx)
    //   so a2 = a1 · γ2·R_out2/R_in2,
    //      b2 = b1 + γ2·a1/R_in2.
    let mut a = 1.0_f64;
    let mut b = 0.0_f64;
    for leg in legs {
        if leg.reserve_in <= 0.0 || leg.reserve_out <= 0.0 {
            return Err(UniswapError::InvalidInput);
        }
        let gamma = 1.0 - (leg.fee_bps as f64 / 10_000.0);
        let factor = gamma * leg.reserve_out / leg.reserve_in;
        // new_b = b + gamma * a / R_in
        b += gamma * a / leg.reserve_in;
        a *= factor;
    }
    Ok((a, b))
}

/// Composed output `dy(dx)` for an arbitrary cycle of CPMM legs.
pub fn cycle_amount_out(legs: &[PoolLeg], dx: f64) -> UniswapResult<f64> {
    if dx <= 0.0 {
        return Err(UniswapError::InvalidInput);
    }
    let (a, b) = compose_cycle(legs)?;
    Ok(a * dx / (1.0 + b * dx))
}

/// Closed-form optimal cycle input size `dx*` (Wang et al. eq. 5).
///
/// Setting `d/d(dx) [a·dx/(1+b·dx) - dx] = 0`:
/// ```text
///   a / (1 + b·dx)^2 = 1
///   ⇒  dx* = ( sqrt(a) - 1 ) / b
/// ```
///
/// The cycle is profitable iff `a > 1` (gross gain factor exceeds 1
/// after fees). Returns `Ok(0.0)` for unprofitable cycles.
pub fn optimal_cycle_size(legs: &[PoolLeg]) -> UniswapResult<f64> {
    let (a, b) = compose_cycle(legs)?;
    if a <= 1.0 || b <= 0.0 {
        return Ok(0.0);
    }
    Ok((a.sqrt() - 1.0) / b)
}

/// Profit (in same units as `dx`) of a cyclic arb of size `dx`.
/// Negative if the trade is a loss.
pub fn cycle_profit(legs: &[PoolLeg], dx: f64) -> UniswapResult<f64> {
    Ok(cycle_amount_out(legs, dx)? - dx)
}

/// Edge of a directed token graph used by Bellman-Ford cycle detection.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub from_token: usize,
    pub to_token: usize,
    pub leg: PoolLeg,
}

/// Bellman-Ford search for negative-weight cycles in
/// `weight = -log(γ · R_out / R_in)`. A negative cycle ⇒ ∏ γ·R_out/R_in > 1
/// ⇒ profitable cyclic arb at infinitesimal size.
///
/// Returns the first negative cycle found as a list of edge indices.
/// Returns `None` if no profitable cycle exists.
///
/// Note: this finds a cycle whose *infinitesimal* edge product is
/// favourable; the caller must then call [`optimal_cycle_size`] on the
/// composed legs to size the arb (large `dx` reduces effective rate).
pub fn find_negative_cycle(num_tokens: usize, edges: &[GraphEdge]) -> Option<Vec<usize>> {
    if num_tokens == 0 || edges.is_empty() {
        return None;
    }
    let mut dist = vec![0.0_f64; num_tokens];
    let mut pred_edge: Vec<Option<usize>> = vec![None; num_tokens];

    let weight = |e: &GraphEdge| -> f64 {
        let gamma = 1.0 - (e.leg.fee_bps as f64 / 10_000.0);
        let rate = gamma * e.leg.reserve_out / e.leg.reserve_in;
        if rate <= 0.0 { f64::INFINITY } else { -rate.ln() }
    };

    let mut last_updated: Option<usize> = None;
    for iter in 0..num_tokens {
        last_updated = None;
        for (idx, e) in edges.iter().enumerate() {
            let w = weight(e);
            if dist[e.from_token] + w < dist[e.to_token] - 1e-15 {
                dist[e.to_token] = dist[e.from_token] + w;
                pred_edge[e.to_token] = Some(idx);
                if iter == num_tokens - 1 {
                    last_updated = Some(e.to_token);
                }
            }
        }
        if last_updated.is_none() && iter == num_tokens - 1 {
            return None;
        }
    }

    // Reconstruct cycle from last_updated by walking predecessors.
    let mut node = last_updated?;
    for _ in 0..num_tokens {
        let edge_idx = pred_edge[node]?;
        node = edges[edge_idx].from_token;
    }
    // `node` is now guaranteed to be on the cycle.
    let start = node;
    let mut cycle_edges = Vec::new();
    let mut cur = start;
    loop {
        let edge_idx = pred_edge[cur]?;
        cycle_edges.push(edge_idx);
        cur = edges[edge_idx].from_token;
        if cur == start {
            break;
        }
        if cycle_edges.len() > num_tokens + 1 {
            return None;
        }
    }
    cycle_edges.reverse();
    Some(cycle_edges)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn approx(a: f64, b: f64, tol: f64) {
        assert!((a - b).abs() < tol, "expected {b}, got {a}");
    }

    #[test]
    fn single_leg_matches_v2_math() {
        let leg = PoolLeg { reserve_in: 1000.0, reserve_out: 2000.0, fee_bps: 30 };
        let dy = cycle_amount_out(&[leg], 100.0).unwrap();
        let dy_v2 = super::super::v2_math::amount_out_given_in(1000.0, 2000.0, 100.0, 30).unwrap();
        approx(dy, dy_v2, 1e-9);
    }

    #[test]
    fn closed_loop_no_arb_zero_profit() {
        // Two pools that exactly invert each other: A → B at rate 2, B → A at rate 0.5
        // With ZERO fees, dy(dx) = dx (closed loop). Profit = 0, optimal = 0.
        let legs = vec![
            PoolLeg { reserve_in: 1_000_000.0, reserve_out: 2_000_000.0, fee_bps: 0 },
            PoolLeg { reserve_in: 2_000_000.0, reserve_out: 1_000_000.0, fee_bps: 0 },
        ];
        // Tiny dx so price-impact negligible
        let dy = cycle_amount_out(&legs, 1.0).unwrap();
        approx(dy, 1.0, 1e-3); // small price impact even for dx=1
        let dx_star = optimal_cycle_size(&legs).unwrap();
        approx(dx_star, 0.0, 1e-9);
    }

    #[test]
    fn profitable_three_pool_cycle_has_positive_optimal() {
        // Build a triangle A → B → C → A with a deliberate price gap.
        // Spot rates: A→B = 2, B→C = 2, C→A = 0.30
        // Product = 1.2 > 1 ⇒ infinitesimal cycle profitable (no fees).
        let legs = vec![
            PoolLeg { reserve_in: 1_000.0,  reserve_out: 2_000.0,  fee_bps: 30 },
            PoolLeg { reserve_in: 2_000.0,  reserve_out: 4_000.0,  fee_bps: 30 },
            PoolLeg { reserve_in: 4_000.0,  reserve_out: 1_200.0,  fee_bps: 30 },
        ];
        let dx_star = optimal_cycle_size(&legs).unwrap();
        assert!(dx_star > 0.0, "expected profitable cycle, got dx*={dx_star}");
        let pnl = cycle_profit(&legs, dx_star).unwrap();
        assert!(pnl > 0.0, "expected profit, got {pnl}");
        // Suboptimal sizes earn less
        let pnl_lo = cycle_profit(&legs, 0.5 * dx_star).unwrap();
        let pnl_hi = cycle_profit(&legs, 1.5 * dx_star).unwrap();
        assert!(pnl >= pnl_lo - 1e-9);
        assert!(pnl >= pnl_hi - 1e-9);
    }

    #[test]
    fn unprofitable_cycle_returns_zero_size() {
        // Product of γ·R_out/R_in < 1 (e.g. fees eat the gap).
        let legs = vec![
            PoolLeg { reserve_in: 1_000.0,  reserve_out: 1_000.0, fee_bps: 30 },
            PoolLeg { reserve_in: 1_000.0,  reserve_out: 1_000.0, fee_bps: 30 },
            PoolLeg { reserve_in: 1_000.0,  reserve_out: 1_000.0, fee_bps: 30 },
        ];
        let dx = optimal_cycle_size(&legs).unwrap();
        approx(dx, 0.0, 1e-12);
    }

    #[test]
    fn bellman_ford_finds_triangular_arb() {
        // 3 tokens: 0=A, 1=B, 2=C. Build edges with the same favourable triangle.
        let edges = vec![
            GraphEdge { from_token: 0, to_token: 1,
                leg: PoolLeg { reserve_in: 1_000.0, reserve_out: 2_000.0, fee_bps: 30 } },
            GraphEdge { from_token: 1, to_token: 2,
                leg: PoolLeg { reserve_in: 2_000.0, reserve_out: 4_000.0, fee_bps: 30 } },
            GraphEdge { from_token: 2, to_token: 0,
                leg: PoolLeg { reserve_in: 4_000.0, reserve_out: 1_200.0, fee_bps: 30 } },
        ];
        let cycle = find_negative_cycle(3, &edges).expect("expected a negative cycle");
        assert_eq!(cycle.len(), 3, "expected a 3-edge cycle, got {cycle:?}");
    }

    #[test]
    fn bellman_ford_reports_none_when_no_arb() {
        let edges = vec![
            GraphEdge { from_token: 0, to_token: 1,
                leg: PoolLeg { reserve_in: 1_000.0, reserve_out: 1_000.0, fee_bps: 30 } },
            GraphEdge { from_token: 1, to_token: 0,
                leg: PoolLeg { reserve_in: 1_000.0, reserve_out: 1_000.0, fee_bps: 30 } },
        ];
        assert!(find_negative_cycle(2, &edges).is_none());
    }
}
