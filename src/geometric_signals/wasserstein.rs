//! Wasserstein (Earth Mover's) distance and optimal transport.
//!
//! Measures how much "work" is needed to reshape one distribution into another.
//! Key for comparing order-book distributions across time or assets.
//!
//! Implements:
//! - W₁ (1D closed-form via CDF inversion)
//! - Sinkhorn divergence (entropic regularization of W₂)
//! - Rolling Wasserstein distance for regime detection
//!
//! References:
//! - Villani (2009), "Optimal Transport: Old and New"
//! - Cuturi (2013), "Sinkhorn Distances: Lightspeed Computation of Optimal Transport"

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// 1D Wasserstein-1 distance (Earth Mover's Distance).
///
/// For 1D distributions the W₁ distance has a closed-form solution:
/// W₁(P, Q) = ∫|F_P(x) - F_Q(x)| dx
///
/// When given samples, this reduces to sorting both arrays and computing
/// the L¹ distance between sorted values (divided by n).
///
/// # Arguments
/// * `a` - First sample (will be sorted internally)
/// * `b` - Second sample (must have same length)
///
/// # Returns
/// W₁ distance (non-negative)
#[cfg_attr(feature = "python", pyfunction)]
pub fn wasserstein_1d(a: Vec<f64>, b: Vec<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Samples must have equal length");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let mut a_sorted = a;
    let mut b_sorted = b;
    a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let sum: f64 = a_sorted
        .iter()
        .zip(b_sorted.iter())
        .map(|(&ai, &bi)| (ai - bi).abs())
        .sum();

    sum / n as f64
}

/// 1D Wasserstein-2 distance.
///
/// W₂² = ∫(F_P⁻¹(u) - F_Q⁻¹(u))² du
///
/// For empirical distributions: W₂² = (1/n) Σ (a[σ(i)] - b[σ(i)])²
///
/// # Arguments
/// * `a` - First sample
/// * `b` - Second sample (same length)
///
/// # Returns
/// W₂ distance (non-negative)
#[cfg_attr(feature = "python", pyfunction)]
pub fn wasserstein_2_1d(a: Vec<f64>, b: Vec<f64>) -> f64 {
    assert_eq!(a.len(), b.len(), "Samples must have equal length");
    let n = a.len();
    if n == 0 {
        return 0.0;
    }

    let mut a_sorted = a;
    let mut b_sorted = b;
    a_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    b_sorted.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));

    let sum_sq: f64 = a_sorted
        .iter()
        .zip(b_sorted.iter())
        .map(|(&ai, &bi)| {
            let diff = ai - bi;
            diff * diff
        })
        .sum();

    (sum_sq / n as f64).sqrt()
}

/// Sinkhorn distance (entropic regularized optimal transport).
///
/// Computes an approximation to W₂ using Sinkhorn-Knopp iterations:
///   min_{P ∈ U(a,b)} <P, C> - ε H(P)
///
/// # Arguments
/// * `cost` - Flat n×m cost matrix C_{ij} (row-major)
/// * `a` - Source distribution (length n, sums to 1)
/// * `b` - Target distribution (length m, sums to 1)
/// * `n_source` - Number of source points
/// * `n_target` - Number of target points
/// * `epsilon` - Regularization strength (typically 0.01–0.1)
/// * `max_iter` - Maximum Sinkhorn iterations
///
/// # Returns
/// Sinkhorn divergence (approximation to OT cost)
#[cfg_attr(feature = "python", pyfunction)]
#[allow(clippy::too_many_arguments)]
pub fn sinkhorn_distance(
    cost: Vec<f64>,
    a: Vec<f64>,
    b: Vec<f64>,
    n_source: usize,
    n_target: usize,
    epsilon: f64,
    max_iter: usize,
) -> f64 {
    assert_eq!(cost.len(), n_source * n_target);
    assert_eq!(a.len(), n_source);
    assert_eq!(b.len(), n_target);

    // Gibbs kernel K = exp(-C/ε)
    let k: Vec<f64> = cost.iter().map(|&c| (-c / epsilon).exp()).collect();

    // Initialize dual variables
    let mut u = vec![1.0 / n_source as f64; n_source];
    let mut v = vec![1.0 / n_target as f64; n_target];

    let tol = 1e-9;

    for _ in 0..max_iter {
        // u = a / (K v)
        let mut u_new = vec![0.0; n_source];
        for i in 0..n_source {
            let mut kv = 0.0;
            for j in 0..n_target {
                kv += k[i * n_target + j] * v[j];
            }
            u_new[i] = if kv > 1e-300 { a[i] / kv } else { a[i] };
        }

        // v = b / (Kᵀ u)
        let mut v_new = vec![0.0; n_target];
        for j in 0..n_target {
            let mut ku = 0.0;
            for i in 0..n_source {
                ku += k[i * n_target + j] * u_new[i];
            }
            v_new[j] = if ku > 1e-300 { b[j] / ku } else { b[j] };
        }

        // Check convergence
        let max_diff: f64 = u_new
            .iter()
            .zip(u.iter())
            .map(|(&un, &uo)| (un - uo).abs())
            .fold(0.0, f64::max);

        u = u_new;
        v = v_new;

        if max_diff < tol {
            break;
        }
    }

    // Compute transport cost: Σᵢⱼ uᵢ Kᵢⱼ vⱼ Cᵢⱼ
    let mut transport_cost = 0.0;
    for i in 0..n_source {
        for j in 0..n_target {
            transport_cost += u[i] * k[i * n_target + j] * v[j] * cost[i * n_target + j];
        }
    }
    transport_cost
}

/// Rolling Wasserstein-1 distance for regime detection.
///
/// Computes W₁ between consecutive windows of a time series.
/// Spikes indicate distribution shifts (regime changes).
///
/// # Arguments
/// * `series` - Time series data
/// * `window` - Window size for each distribution sample
///
/// # Returns
/// Vector of W₁ distances between consecutive windows (length N - 2*window + 1)
#[cfg_attr(feature = "python", pyfunction)]
pub fn rolling_wasserstein(series: Vec<f64>, window: usize) -> Vec<f64> {
    let n = series.len();
    if n < 2 * window {
        return vec![];
    }

    let n_out = n - 2 * window + 1;
    let mut distances = Vec::with_capacity(n_out);

    for t in 0..n_out {
        let win1 = series[t..t + window].to_vec();
        let win2 = series[t + window..t + 2 * window].to_vec();
        distances.push(wasserstein_1d(win1, win2));
    }
    distances
}

/// Compare two order-book distributions using Wasserstein distance.
///
/// Models bid/ask levels as 1D distributions weighted by volume.
/// Useful for detecting microstructure regime changes.
///
/// # Arguments
/// * `levels_a` - Price levels of first snapshot
/// * `volumes_a` - Volumes at each level (weights)
/// * `levels_b` - Price levels of second snapshot
/// * `volumes_b` - Volumes at each level (weights)
///
/// # Returns
/// Wasserstein-1 distance between the two LOB distributions
#[cfg_attr(feature = "python", pyfunction)]
pub fn lob_wasserstein(
    levels_a: Vec<f64>,
    volumes_a: Vec<f64>,
    levels_b: Vec<f64>,
    volumes_b: Vec<f64>,
) -> f64 {
    assert_eq!(levels_a.len(), volumes_a.len());
    assert_eq!(levels_b.len(), volumes_b.len());

    // Expand into weighted samples (repeat each level proportionally)
    // Using CDF approach for weighted 1D Wasserstein
    let sum_a: f64 = volumes_a.iter().sum();
    let sum_b: f64 = volumes_b.iter().sum();
    if sum_a < 1e-15 || sum_b < 1e-15 {
        return 0.0;
    }

    // Normalize weights
    let wa: Vec<f64> = volumes_a.iter().map(|&v| v / sum_a).collect();
    let wb: Vec<f64> = volumes_b.iter().map(|&v| v / sum_b).collect();

    // Sort by price level
    let mut pairs_a: Vec<(f64, f64)> = levels_a.into_iter().zip(wa).collect();
    let mut pairs_b: Vec<(f64, f64)> = levels_b.into_iter().zip(wb).collect();
    pairs_a.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    pairs_b.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    // Merge events and compute integral of |CDF_A - CDF_B|
    let mut events: Vec<(f64, f64, f64)> = Vec::new(); // (price, delta_cdf_a, delta_cdf_b)
    for (p, w) in &pairs_a {
        events.push((*p, *w, 0.0));
    }
    for (p, w) in &pairs_b {
        events.push((*p, 0.0, *w));
    }
    events.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));

    let mut integral: f64 = 0.0;
    let mut cdf_a: f64 = 0.0;
    let mut cdf_b: f64 = 0.0;
    let mut prev_price = if !events.is_empty() {
        events[0].0
    } else {
        0.0
    };

    for (price, da, db) in events {
        integral += (cdf_a - cdf_b).abs() * (price - prev_price);
        cdf_a += da;
        cdf_b += db;
        prev_price = price;
    }
    integral
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wasserstein_1d_identical() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let d = wasserstein_1d(a.clone(), a);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_wasserstein_1d_shifted() {
        let a = vec![0.0, 1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let d = wasserstein_1d(a, b);
        assert!((d - 1.0).abs() < 1e-10); // Shift by 1
    }

    #[test]
    fn test_wasserstein_2_1d() {
        let a = vec![0.0, 1.0, 2.0, 3.0];
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let d = wasserstein_2_1d(a, b);
        assert!((d - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_sinkhorn_uniform() {
        // Two identical uniform distributions → cost ≈ 0
        let n = 3;
        let a = vec![1.0 / 3.0; 3];
        let b = vec![1.0 / 3.0; 3];
        let cost = vec![
            0.0, 1.0, 2.0, // row 0
            1.0, 0.0, 1.0, // row 1
            2.0, 1.0, 0.0, // row 2
        ];
        let d = sinkhorn_distance(cost, a, b, n, n, 0.1, 100);
        // Self-transport with metric cost should be small but not exactly 0 due to regularization
        assert!(d < 0.5);
    }

    #[test]
    fn test_rolling_wasserstein() {
        // Stationary series → distances should be small
        let series: Vec<f64> = (0..20).map(|i| (i as f64 * 0.1).sin()).collect();
        let dists = rolling_wasserstein(series, 5);
        assert!(!dists.is_empty());
        // All distances should be finite
        assert!(dists.iter().all(|&d| d.is_finite()));
    }

    #[test]
    fn test_lob_wasserstein_identical() {
        let levels = vec![100.0, 101.0, 102.0];
        let volumes = vec![10.0, 20.0, 15.0];
        let d = lob_wasserstein(
            levels.clone(),
            volumes.clone(),
            levels,
            volumes,
        );
        assert!(d.abs() < 1e-10);
    }
}
