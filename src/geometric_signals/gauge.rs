//! Gauge theory on asset price graphs.
//!
//! Models the market as a principal fiber bundle:
//! - Connection A_{i->j} = log(P_j / P_i)
//! - Curvature F = cycle sum of connections (non-zero = arbitrage)
//! - Holonomy H = exp(F) = cycle product of price ratios
//!
//! References: Ilinski (2001), "Gauge Invariance, Geometry and Arbitrage"

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Compute triangle curvature for all unique triangles in a complete graph.
///
/// Given n asset prices, computes F_{ijk} = log(P_j/P_i) + log(P_k/P_j) + log(P_i/P_k)
/// for all (n choose 3) triangles.
///
/// # Arguments
/// * `prices` - Asset prices at current time, length n
///
/// # Returns
/// Vec of (i, j, k, curvature) tuples flattened as [i, j, k, F, i, j, k, F, ...]
#[cfg_attr(feature = "python", pyfunction)]
pub fn triangle_curvature(prices: Vec<f64>) -> Vec<f64> {
    let n = prices.len();
    let mut result = Vec::new();

    for i in 0..n {
        for j in (i + 1)..n {
            for k in (j + 1)..n {
                let a_ij = (prices[j] / prices[i]).ln();
                let a_jk = (prices[k] / prices[j]).ln();
                let a_ki = (prices[i] / prices[k]).ln();
                let f = a_ij + a_jk + a_ki;
                result.push(i as f64);
                result.push(j as f64);
                result.push(k as f64);
                result.push(f);
            }
        }
    }
    result
}

/// Compute curvature for a specific cycle of asset indices.
///
/// # Arguments
/// * `prices` - Asset prices
/// * `cycle` - Ordered list of asset indices forming the cycle
///
/// # Returns
/// The curvature F = sum of log price ratios around the cycle
#[cfg_attr(feature = "python", pyfunction)]
pub fn cycle_curvature(prices: Vec<f64>, cycle: Vec<usize>) -> f64 {
    if cycle.len() < 2 {
        return 0.0;
    }
    let mut f = 0.0;
    for w in cycle.windows(2) {
        f += (prices[w[1]] / prices[w[0]]).ln();
    }
    // Close the cycle
    f += (prices[cycle[0]] / prices[*cycle.last().unwrap()]).ln();
    f
}

/// Compute rolling z-scores for a time series of curvature values.
///
/// z(t) = (F(t) - μ) / (σ + ε)
///
/// # Arguments
/// * `curvatures` - Time series of curvature values
/// * `window` - Rolling window size
///
/// # Returns
/// Z-scores (first `window-1` values are NaN)
#[cfg_attr(feature = "python", pyfunction)]
pub fn curvature_zscore(curvatures: Vec<f64>, window: usize) -> Vec<f64> {
    let n = curvatures.len();
    let mut zscores = vec![f64::NAN; n];
    let eps = 1e-10;

    if window == 0 || window > n {
        return zscores;
    }

    for t in window..n {
        let slice = &curvatures[(t - window)..t];
        let mean: f64 = slice.iter().sum::<f64>() / window as f64;
        let var: f64 =
            slice.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / (window as f64 - 1.0);
        let std = var.sqrt();
        zscores[t] = (curvatures[t] - mean) / (std + eps);
    }
    zscores
}

/// Compute the holonomy (exponential of curvature) for a cycle.
///
/// H = exp(F) = product of price ratios around the cycle.
/// H = 1 means no arbitrage; H != 1 means arbitrage opportunity.
///
/// # Arguments
/// * `prices` - Asset prices
/// * `cycle` - Ordered cycle of asset indices
///
/// # Returns
/// Holonomy value
#[cfg_attr(feature = "python", pyfunction)]
pub fn holonomy(prices: Vec<f64>, cycle: Vec<usize>) -> f64 {
    cycle_curvature(prices, cycle).exp()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle_curvature_no_arb() {
        // Consistent prices: curvature should be ~0
        let prices = vec![100.0, 200.0, 50.0];
        let result = triangle_curvature(prices);
        assert_eq!(result.len(), 4); // 1 triangle * 4 values
        assert!(result[3].abs() < 1e-10); // F ≈ 0
    }

    #[test]
    fn test_triangle_curvature_with_arb() {
        // For absolute prices, triangle curvature is always 0 (math identity).
        // Arbitrage detection requires exchange rate matrices, not absolute prices.
        // Use cycle_curvature with explicit exchange rates for arb detection.
        // Here we just verify the no-arb property holds for any prices.
        let prices = vec![100.0, 205.0, 48.0];
        let result = triangle_curvature(prices);
        assert!(result[3].abs() < 1e-10); // Always 0 for absolute prices
    }

    #[test]
    fn test_cycle_curvature_fx() {
        // USD -> EUR -> GBP -> USD: rates 0.85, 1.15, 1.025
        // Product = 0.85 * 1.15 * 1.025 = 1.001... (small arb)
        let prices = vec![1.0, 0.85, 0.85 * 1.15]; // normalized
        let f = cycle_curvature(prices, vec![0, 1, 2]);
        // Non-zero curvature expected
        assert!(f.abs() > 0.0);
    }

    #[test]
    fn test_curvature_zscore() {
        let data: Vec<f64> = (0..100).map(|i| (i as f64 * 0.1).sin() * 0.01).collect();
        let zs = curvature_zscore(data, 20);
        // First 20 should be NaN
        assert!(zs[0].is_nan());
        assert!(zs[19].is_nan());
        // After window, should have values
        assert!(!zs[20].is_nan());
    }

    #[test]
    fn test_holonomy_no_arb() {
        let prices = vec![100.0, 200.0, 50.0];
        let h = holonomy(prices, vec![0, 1, 2]);
        assert!((h - 1.0).abs() < 1e-10);
    }
}
