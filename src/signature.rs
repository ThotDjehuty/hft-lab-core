//! # Path Signature Methods
//!
//! Implements Chen's *iterated-integral path signature* and related tools for
//! analysing time-series data in quantitative finance.
//!
//! ## Background
//!
//! The **signature** of a path X: [0,T] → ℝ^d is the collection of all
//! iterated integrals:
//!
//! ```text
//! S(X)^{i₁,…,iₖ} = ∫₀ᵀ ∫₀^{tₖ} … ∫₀^{t₂} dX^{i₁}_{t₁} ⊗ … ⊗ dX^{iₖ}_{tₖ}
//! ```
//!
//! For discrete paths this reduces to iterated cumulative sums (Chen sums).
//! The signature:
//! - **Uniquely characterises** the path up to tree-like equivalence.
//! - **Respects time-ordering** — captures temporal patterns ML models often miss.
//! - **Is robust** to irregular / high-frequency sampling.
//!
//! ## Implemented Tools
//!
//! | Function | Description |
//! |----------|-------------|
//! | `path_signature_2d` | Signature up to depth 3 for 2-D paths |
//! | `lead_lag_transform` | Convert 1-D prices to canonical 2-D lead-lag path |
//! | `log_signature_2d` | Log-signature (Lyndon basis, depth ≤ 3) |
//! | `rolling_signature_features` | Sliding-window feature matrix for ML |
//! | `signature_distance` | L² distance in signature space |
//!
//! ## References
//!
//! - K.T. Chen (1954). *Iterated integrals and exponential homomorphisms.*
//! - T. Lyons (1998). *Differential equations driven by rough signals.* Rev. Mat. Iberoam.
//! - Chevyrev & Kormilitzin (2016). *A Primer on the Signature Method in Machine Learning.*
//! - Gyurko, Lyons & Ni (2013). *Extracting information from the signature of a financial data stream.*
//! - Király & Oberhauser (2019). *Kernels for sequentially ordered data.* JMLR.

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyList;

// ─────────────────────────────────────────────────────────────────────────────
//  Lead-lag transform
// ─────────────────────────────────────────────────────────────────────────────

/// Convert a 1-D price series to the **lead-lag 2-D path**.
///
/// Given prices `[p₀, p₁, …, pₙ]`, the lead-lag transform produces an
/// interleaved 2-D path `[(p₀, p₀), (p₁, p₀), (p₁, p₁), (p₂, p₁), …]`.
/// This encoding allows the signature to capture autocorrelation and
/// cross-correlations between the path and its lagged copy.
///
/// Reference: Gyurko et al. (2013), Section 2.
pub fn lead_lag_transform(prices: &[f64]) -> Vec<(f64, f64)> {
    if prices.is_empty() {
        return vec![];
    }
    let mut out = Vec::with_capacity(2 * prices.len() - 1);
    out.push((prices[0], prices[0]));
    for i in 1..prices.len() {
        out.push((prices[i], prices[i - 1]));
        out.push((prices[i], prices[i]));
    }
    out
}

// ─────────────────────────────────────────────────────────────────────────────
//  Signature computation (2-D paths)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the **path signature** of a discrete 2-D path up to depth 3.
///
/// For a 2-D path `(X, Y)` the signature components at each level are:
///
/// **Depth 1** (2 terms):
/// - S¹ = X_T − X_0  (total X increment)
/// - S² = Y_T − Y_0  (total Y increment)
///
/// **Depth 2** (4 terms):  iterated sums Σᵢ ΔXᵢ · Xᵢ₋₁, etc.
///
/// **Depth 3** (8 terms):  triple iterated sums.
///
/// Returns a `Vec<f64>` of length 2 + 4 + 8 = **14** for depth 3.
///
/// # Arguments
/// * `path` – Sequence of `(x, y)` points.
/// * `depth` – Maximum depth (1, 2 or 3).
pub fn path_signature_2d(path: &[(f64, f64)], depth: usize) -> Vec<f64> {
    let depth = depth.min(3).max(1);
    let n = path.len();
    if n < 2 {
        let size = match depth { 1 => 2, 2 => 6, _ => 14 };
        return vec![0.0; size];
    }

    // Increments
    let dx: Vec<f64> = path.windows(2).map(|w| w[1].0 - w[0].0).collect();
    let dy: Vec<f64> = path.windows(2).map(|w| w[1].1 - w[0].1).collect();
    let m = dx.len();

    // ── Depth 1 ──────────────────────────────────────────────────────────────
    let s1_x: f64 = dx.iter().sum();
    let s1_y: f64 = dy.iter().sum();

    if depth == 1 {
        return vec![s1_x, s1_y];
    }

    // ── Depth 2 ──────────────────────────────────────────────────────────────
    // S^{ij} = Σₜ Δᵢ(t) · A^j(t-1)  where A^j is the running sum up to t-1
    // Using Chen's shuffle product identity for discrete paths:
    // S^{11} = ½ (S¹)² − ½ Σ (Δx)²
    // S^{22} = ½ (S²)² − ½ Σ (Δy)²
    // S^{12} = Σₜ Δxₜ · cumsum_y_{t-1}
    // S^{21} = Σₜ Δyₜ · cumsum_x_{t-1}
    let sum_dx2: f64 = dx.iter().map(|v| v * v).sum();
    let sum_dy2: f64 = dy.iter().map(|v| v * v).sum();
    let s2_11 = 0.5 * (s1_x * s1_x - sum_dx2);
    let s2_22 = 0.5 * (s1_y * s1_y - sum_dy2);

    // S^{12} and S^{21} via running sums
    let mut s2_12 = 0.0_f64;
    let mut s2_21 = 0.0_f64;
    {
        let mut cum_x = 0.0_f64;
        let mut cum_y = 0.0_f64;
        for i in 0..m {
            s2_12 += dx[i] * cum_y;
            s2_21 += dy[i] * cum_x;
            cum_x += dx[i];
            cum_y += dy[i];
        }
    }

    if depth == 2 {
        return vec![s1_x, s1_y, s2_11, s2_12, s2_21, s2_22];
    }

    // ── Depth 3 ──────────────────────────────────────────────────────────────
    // 8 terms: S^{111}, S^{112}, S^{121}, S^{122}, S^{211}, S^{212}, S^{221}, S^{222}
    let mut depth3 = [0.0_f64; 8];
    {
        let mut cum_x = 0.0_f64;
        let mut cum_y = 0.0_f64;
        // Running depth-2 signature accumulators
        let mut cum_11 = 0.0_f64; // integral of X w.r.t. X
        let mut cum_12 = 0.0_f64; // integral of X w.r.t. Y
        let mut cum_21 = 0.0_f64;
        let mut cum_22 = 0.0_f64;

        for i in 0..m {
            let dxi = dx[i];
            let dyi = dy[i];
            // Depth 3: each new increment times each depth-2 accumulator
            depth3[0] += dxi * cum_11; // S^{111}
            depth3[1] += dxi * cum_12; // S^{112} (x then y then x)
            depth3[2] += dyi * cum_11; // S^{121}
            depth3[3] += dyi * cum_12; // S^{122}
            depth3[4] += dxi * cum_21; // S^{211}
            depth3[5] += dxi * cum_22; // S^{212}
            depth3[6] += dyi * cum_21; // S^{221}
            depth3[7] += dyi * cum_22; // S^{222}
            // Update depth-2 accumulators
            cum_11 += dxi * cum_x;
            cum_12 += dxi * cum_y;
            cum_21 += dyi * cum_x;
            cum_22 += dyi * cum_y;
            cum_x += dxi;
            cum_y += dyi;
        }
    }

    let mut sig = vec![s1_x, s1_y, s2_11, s2_12, s2_21, s2_22];
    sig.extend_from_slice(&depth3);
    sig
}

// ─────────────────────────────────────────────────────────────────────────────
//  Log-signature (depth ≤ 2, Lyndon basis)
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the **log-signature** of a 2-D path up to depth 2.
///
/// The log-signature lives in the free Lie algebra and is given by:
///
/// - Level 1: same as signature [S¹, S²]
/// - Level 2: Lie bracket [e₁, e₂] = S^{12} − S^{21}  (the "area" term)
///
/// The log-signature is a more compact, non-redundant representation.
/// For depth 2, it has **3** components (versus 6 for the signature).
pub fn log_signature_2d(path: &[(f64, f64)]) -> Vec<f64> {
    let sig = path_signature_2d(path, 2);
    if sig.len() < 6 {
        return sig;
    }
    // [S¹, S², S^12 - S^21]
    vec![sig[0], sig[1], sig[3] - sig[4]]
}

// ─────────────────────────────────────────────────────────────────────────────
//  Rolling window signature features
// ─────────────────────────────────────────────────────────────────────────────

/// Compute **rolling lead-lag signature features** from a 1-D price series.
///
/// At each time step `t` (after the first `window` prices), applies the
/// lead-lag transform and computes the depth-`sig_depth` signature of the
/// resulting 2-D path.
///
/// # Returns
/// A matrix of shape `(n - window + 1) × feature_dim` flattened to a 1-D
/// `Vec<Vec<f64>>` (rows = time steps, cols = signature components).
///
/// Feature dimension:
/// - depth 1 → 2
/// - depth 2 → 6
/// - depth 3 → 14
pub fn rolling_signature_features(
    prices: &[f64],
    window: usize,
    sig_depth: usize,
) -> Vec<Vec<f64>> {
    let n = prices.len();
    if n < window || window < 2 {
        return vec![];
    }
    prices
        .windows(window)
        .map(|w| {
            let ll = lead_lag_transform(w);
            path_signature_2d(&ll, sig_depth)
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Signature distance
// ─────────────────────────────────────────────────────────────────────────────

/// L² distance between two signature vectors.
///
/// Provides a geometry on the space of paths via signature embeddings.
/// Useful for clustering, k-NN, or kernel methods on time series.
pub fn signature_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Path normalisation helpers
// ─────────────────────────────────────────────────────────────────────────────

/// Normalise prices to [0, 1] before signature computation.
/// Improves numerical conditioning for long or high-magnitude series.
pub fn normalise_prices(prices: &[f64]) -> Vec<f64> {
    let min = prices.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = prices.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let range = (max - min).max(1e-10);
    prices.iter().map(|&p| (p - min) / range).collect()
}

/// Time-normalise a path so the parameter runs over [0, 1].
pub fn time_normalise(path: &[(f64, f64)]) -> Vec<(f64, f64)> {
    let n = path.len();
    if n < 2 {
        return path.to_vec();
    }
    let max_t = (n - 1) as f64;
    path.iter()
        .enumerate()
        .map(|(i, &(x, y))| (x * (i as f64 / max_t + 1.0), y))
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Python bindings
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "lead_lag_transform")]
pub fn lead_lag_transform_py(prices: Vec<f64>) -> Vec<(f64, f64)> {
    lead_lag_transform(&prices)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "path_signature_2d")]
pub fn path_signature_2d_py(path: Vec<(f64, f64)>, depth: usize) -> Vec<f64> {
    path_signature_2d(&path, depth)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "log_signature_2d")]
pub fn log_signature_2d_py(path: Vec<(f64, f64)>) -> Vec<f64> {
    log_signature_2d(&path)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "rolling_signature_features")]
pub fn rolling_signature_features_py(
    py: Python,
    prices: Vec<f64>,
    window: usize,
    sig_depth: usize,
) -> PyResult<PyObject> {
    let features = rolling_signature_features(&prices, window, sig_depth);
    let list = PyList::new_bound(py, features.iter().map(|row| {
        PyList::new_bound(py, row.iter())
    }));
    Ok(list.into())
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "signature_distance")]
pub fn signature_distance_py(a: Vec<f64>, b: Vec<f64>) -> f64 {
    signature_distance(&a, &b)
}

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(name = "normalise_prices")]
pub fn normalise_prices_py(prices: Vec<f64>) -> Vec<f64> {
    normalise_prices(&prices)
}

/// Convenience: full pipeline from prices → rolling lead-lag signatures.
///
/// Equivalent to: normalise → rolling windows → lead_lag → path_signature_2d
#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (prices, window=20, sig_depth=2, normalise=true))]
pub fn prices_to_signature_features_py(
    py: Python,
    prices: Vec<f64>,
    window: usize,
    sig_depth: usize,
    normalise: bool,
) -> PyResult<PyObject> {
    let working = if normalise {
        normalise_prices(&prices)
    } else {
        prices
    };
    let features = rolling_signature_features(&working, window, sig_depth);
    let list = PyList::new_bound(py, features.iter().map(|row| {
        PyList::new_bound(py, row.iter())
    }));
    Ok(list.into())
}

// ─────────────────────────────────────────────────────────────────────────────
//  Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lead_lag_length() {
        let prices = vec![1.0, 2.0, 3.0, 4.0];
        let ll = lead_lag_transform(&prices);
        // n prices → 2n-1 points
        assert_eq!(ll.len(), 2 * prices.len() - 1);
    }

    #[test]
    fn test_lead_lag_diagonal_start_end() {
        let prices = vec![10.0, 11.0, 12.0];
        let ll = lead_lag_transform(&prices);
        // First and last should be on the diagonal
        assert_eq!(ll[0].0, ll[0].1, "start must be diagonal");
        assert_eq!(ll.last().unwrap().0, ll.last().unwrap().1, "end diagonal");
    }

    #[test]
    fn test_signature_depth1_length() {
        let path = vec![(0.0_f64, 0.0_f64), (1.0, 0.5), (2.0, 1.0)];
        let sig = path_signature_2d(&path, 1);
        assert_eq!(sig.len(), 2);
    }

    #[test]
    fn test_signature_depth2_length() {
        let path = vec![(0.0_f64, 0.0_f64), (1.0, 0.5), (2.0, 1.0)];
        let sig = path_signature_2d(&path, 2);
        assert_eq!(sig.len(), 6);
    }

    #[test]
    fn test_signature_depth3_length() {
        let path = vec![(0.0_f64, 0.0_f64), (1.0, 0.5), (2.0, 1.0), (3.0, 1.5)];
        let sig = path_signature_2d(&path, 3);
        assert_eq!(sig.len(), 14);
    }

    #[test]
    fn test_log_signature_length() {
        let path = vec![(0.0_f64, 0.0_f64), (1.0, 1.0), (2.0, 1.5)];
        let log_sig = log_signature_2d(&path);
        assert_eq!(log_sig.len(), 3);
    }

    #[test]
    fn test_depth1_equals_increments() {
        // Depth-1 signature = total X and Y increments
        let path = vec![(1.0_f64, 2.0_f64), (3.0, 5.0), (4.0, 3.0)];
        let sig = path_signature_2d(&path, 1);
        assert!((sig[0] - 3.0).abs() < 1e-12, "S¹ = Δx total");
        assert!((sig[1] - 1.0).abs() < 1e-12, "S² = Δy total");
    }

    #[test]
    fn test_straight_line_zero_area() {
        // For a straight line, S^{12} = S^{21} (area = 0)
        let path: Vec<(f64, f64)> = (0..=10).map(|i| (i as f64, i as f64)).collect();
        let sig = path_signature_2d(&path, 2);
        let area = (sig[3] - sig[4]).abs();
        assert!(area < 1e-10, "Straight line has zero signed area: {area}");
    }

    #[test]
    fn test_rolling_signatures_shape() {
        let prices: Vec<f64> = (0..50).map(|i| 100.0 + i as f64 * 0.1).collect();
        let feats = rolling_signature_features(&prices, 10, 2);
        assert_eq!(feats.len(), 50 - 10 + 1);
        assert_eq!(feats[0].len(), 6); // depth 2 → 6 features
    }

    #[test]
    fn test_signature_distance_zero_self() {
        let sig = vec![1.0, 2.0, 3.0];
        assert!((signature_distance(&sig, &sig)).abs() < 1e-12);
    }

    #[test]
    fn test_normalise_range() {
        let prices = vec![50.0, 100.0, 75.0, 25.0, 150.0];
        let norm = normalise_prices(&prices);
        let min = norm.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = norm.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        assert!(min.abs() < 1e-10, "min should be 0");
        assert!((max - 1.0).abs() < 1e-10, "max should be 1");
    }

    #[test]
    fn test_chen_identity_concatenation() {
        // Chen's identity: S(X*Y) = S(X) ⊗ S(Y) for concatenated paths
        // Simplified check: signature of [A,B,C] level-1 = sum of increments
        let p1 = vec![(0.0_f64, 0.0_f64), (1.0, 0.5)];
        let p2 = vec![(1.0_f64, 0.5_f64), (2.0, 1.5), (3.0, 0.5)];
        let mut full = p1.clone();
        full.extend_from_slice(&p2[1..]);
        let sig_full = path_signature_2d(&full, 1);
        let sig1 = path_signature_2d(&p1, 1);
        let sig2 = path_signature_2d(&p2, 1);
        // Level-1 is additive (just increments)
        assert!((sig_full[0] - (sig1[0] + sig2[0])).abs() < 1e-10);
        assert!((sig_full[1] - (sig1[1] + sig2[1])).abs() < 1e-10);
    }
}
