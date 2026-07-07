//! Spectral analysis of market correlation graphs.
//!
//! The "shape of the market" is captured by eigenvalues of the graph Laplacian:
//! - Fiedler value λ₂: algebraic connectivity / fragmentation
//! - Spectral entropy: eigenvalue concentration (systemic risk)
//! - Heat trace: multi-scale shape descriptor for regime detection
//!
//! References: "Can You Hear the Shape of a Market?"

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Compute the (unnormalized) graph Laplacian L = D - W.
///
/// Input: flat row-major weight matrix W (n×n), output: flat L (n×n).
///
/// # Arguments
/// * `weights` - Flat n×n adjacency/weight matrix (row-major)
/// * `n` - Number of vertices
///
/// # Returns
/// Flat n×n Laplacian matrix
#[cfg_attr(feature = "python", pyfunction)]
pub fn graph_laplacian(weights: Vec<f64>, n: usize) -> Vec<f64> {
    assert_eq!(weights.len(), n * n);
    let mut lap = vec![0.0; n * n];

    for i in 0..n {
        let mut degree = 0.0;
        for j in 0..n {
            if i != j {
                let w = weights[i * n + j];
                lap[i * n + j] = -w;
                degree += w;
            }
        }
        lap[i * n + i] = degree;
    }
    lap
}

/// Compute the normalized graph Laplacian L_sym = I - D^{-1/2} W D^{-1/2}.
///
/// # Arguments
/// * `weights` - Flat n×n adjacency/weight matrix (row-major)
/// * `n` - Number of vertices
///
/// # Returns
/// Flat n×n normalized Laplacian matrix
#[cfg_attr(feature = "python", pyfunction)]
pub fn normalized_laplacian(weights: Vec<f64>, n: usize) -> Vec<f64> {
    assert_eq!(weights.len(), n * n);

    // Compute degrees
    let mut degrees = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                degrees[i] += weights[i * n + j];
            }
        }
    }

    // D^{-1/2}
    let inv_sqrt_d: Vec<f64> = degrees
        .iter()
        .map(|&d| if d > 1e-12 { 1.0 / d.sqrt() } else { 0.0 })
        .collect();

    let mut lap = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            if i == j {
                lap[i * n + j] = if degrees[i] > 1e-12 { 1.0 } else { 0.0 };
            } else {
                lap[i * n + j] = -inv_sqrt_d[i] * weights[i * n + j] * inv_sqrt_d[j];
            }
        }
    }
    lap
}

/// Compute eigenvalues of a symmetric matrix using Jacobi iteration.
///
/// Simple but robust O(n³) eigensolver suitable for small-to-medium matrices.
fn symmetric_eigenvalues(matrix: &[f64], n: usize) -> Vec<f64> {
    // Copy matrix
    let mut a = matrix.to_vec();
    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
        // Find largest off-diagonal element
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let v = a[i * n + j].abs();
                if v > max_val {
                    max_val = v;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < tol {
            break;
        }

        // Compute rotation angle
        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < tol {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        // Apply rotation
        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[i * n + p] = c * a[i * n + p] + s * a[i * n + q];
                new_a[p * n + i] = new_a[i * n + p];
                new_a[i * n + q] = -s * a[i * n + p] + c * a[i * n + q];
                new_a[q * n + i] = new_a[i * n + q];
            }
        }
        new_a[p * n + p] = c * c * app + 2.0 * c * s * apq + s * s * aqq;
        new_a[q * n + q] = s * s * app - 2.0 * c * s * apq + c * c * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;

        a = new_a;
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Compute eigenvalues and eigenvectors of a symmetric matrix.
///
/// Returns (eigenvalues, eigenvectors_flat) where eigenvectors are column-major.
fn symmetric_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; n * n]; // eigenvector matrix (columns)
    for i in 0..n {
        v[i * n + i] = 1.0; // identity
    }
    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
        let mut max_val = 0.0;
        let mut p = 0;
        let mut q = 1;
        for i in 0..n {
            for j in (i + 1)..n {
                let val = a[i * n + j].abs();
                if val > max_val {
                    max_val = val;
                    p = i;
                    q = j;
                }
            }
        }
        if max_val < tol {
            break;
        }

        let app = a[p * n + p];
        let aqq = a[q * n + q];
        let apq = a[p * n + q];

        let theta = if (app - aqq).abs() < tol {
            std::f64::consts::FRAC_PI_4
        } else {
            0.5 * (2.0 * apq / (app - aqq)).atan()
        };

        let c = theta.cos();
        let s = theta.sin();

        let mut new_a = a.clone();
        for i in 0..n {
            if i != p && i != q {
                new_a[i * n + p] = c * a[i * n + p] + s * a[i * n + q];
                new_a[p * n + i] = new_a[i * n + p];
                new_a[i * n + q] = -s * a[i * n + p] + c * a[i * n + q];
                new_a[q * n + i] = new_a[i * n + q];
            }
        }
        new_a[p * n + p] = c * c * app + 2.0 * c * s * apq + s * s * aqq;
        new_a[q * n + q] = s * s * app - 2.0 * c * s * apq + c * c * aqq;
        new_a[p * n + q] = 0.0;
        new_a[q * n + p] = 0.0;
        a = new_a;

        // Update eigenvectors
        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip + s * viq;
            v[i * n + q] = -s * vip + c * viq;
        }
    }

    let mut eigen_pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|j| {
            let eigenvalue = a[j * n + j];
            let eigenvector: Vec<f64> = (0..n).map(|i| v[i * n + j]).collect();
            (eigenvalue, eigenvector)
        })
        .collect();

    eigen_pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = eigen_pairs.iter().map(|(e, _)| *e).collect();
    let eigenvectors: Vec<f64> = eigen_pairs
        .iter()
        .flat_map(|(_, v)| v.iter())
        .copied()
        .collect();

    (eigenvalues, eigenvectors)
}

/// Compute the Fiedler value (second-smallest eigenvalue of the Laplacian).
///
/// Measures algebraic connectivity: λ₂ ≈ 0 means market is fragmented.
///
/// # Arguments
/// * `laplacian` - Flat n×n Laplacian matrix (from `graph_laplacian`)
/// * `n` - Number of vertices
///
/// # Returns
/// λ₂ (second smallest eigenvalue)
#[cfg_attr(feature = "python", pyfunction)]
pub fn fiedler_value(laplacian: Vec<f64>, n: usize) -> f64 {
    let eigenvalues = symmetric_eigenvalues(&laplacian, n);
    if eigenvalues.len() >= 2 {
        eigenvalues[1]
    } else {
        0.0
    }
}

/// Get the Fiedler vector (eigenvector of λ₂).
///
/// Useful for spectral clustering.
#[cfg_attr(feature = "python", pyfunction)]
pub fn fiedler_vector(laplacian: Vec<f64>, n: usize) -> Vec<f64> {
    let (_, eigenvectors) = symmetric_eigen(&laplacian, n);
    if n >= 2 {
        // Second eigenvector (column 1)
        eigenvectors[n..(2 * n)].to_vec()
    } else {
        vec![0.0; n]
    }
}

/// Compute spectral entropy of the Laplacian eigenvalues.
///
/// H = -Σ p_m log(p_m + ε), where p_m = λ_m / Σλ_r
///
/// Low entropy ⇒ few dominant modes (systemic risk).
/// High entropy ⇒ diversified eigenvalue spectrum.
///
/// # Arguments
/// * `eigenvalues` - Eigenvalues of the Laplacian
///
/// # Returns
/// Spectral entropy
#[cfg_attr(feature = "python", pyfunction)]
pub fn spectral_entropy(eigenvalues: Vec<f64>) -> f64 {
    let eps = 1e-15;
    // Filter out near-zero eigenvalues (the first one is always ~0 for connected graphs)
    let positive: Vec<f64> = eigenvalues.iter().filter(|&&v| v > eps).copied().collect();

    if positive.is_empty() {
        return 0.0;
    }

    let total: f64 = positive.iter().sum();
    if total < eps {
        return 0.0;
    }

    -positive
        .iter()
        .map(|&v| {
            let p = v / total;
            p * (p + eps).ln()
        })
        .sum::<f64>()
}

/// Compute the heat trace at specified scales.
///
/// HT(s) = Tr(exp(-sL)) = Σ exp(-s λ_m)
///
/// Multi-scale shape descriptor for the correlation graph.
///
/// # Arguments
/// * `eigenvalues` - Eigenvalues of the Laplacian
/// * `scales` - Scale parameters s to evaluate at
///
/// # Returns
/// Heat trace values at each scale
#[cfg_attr(feature = "python", pyfunction)]
pub fn heat_trace(eigenvalues: Vec<f64>, scales: Vec<f64>) -> Vec<f64> {
    scales
        .iter()
        .map(|&s| eigenvalues.iter().map(|&lam| (-s * lam).exp()).sum::<f64>())
        .collect()
}

/// Compute all Laplacian eigenvalues of a weight matrix.
///
/// Convenience function: builds Laplacian then computes eigenvalues.
///
/// # Arguments
/// * `weights` - Flat n×n adjacency matrix
/// * `n` - Number of vertices
///
/// # Returns
/// Sorted eigenvalues
#[cfg_attr(feature = "python", pyfunction)]
pub fn laplacian_eigenvalues(weights: Vec<f64>, n: usize) -> Vec<f64> {
    let lap = graph_laplacian(weights, n);
    symmetric_eigenvalues(&lap, n)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_laplacian_simple() {
        // 3-node complete graph with uniform weights
        let w = vec![
            0.0, 1.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 1.0, 0.0, //
        ];
        let lap = graph_laplacian(w, 3);
        // Diagonal should be 2, off-diagonal -1
        assert_eq!(lap[0], 2.0);
        assert_eq!(lap[1], -1.0);
        assert_eq!(lap[4], 2.0);
    }

    #[test]
    fn test_fiedler_value_connected() {
        let w = vec![
            0.0, 1.0, 1.0, //
            1.0, 0.0, 1.0, //
            1.0, 1.0, 0.0, //
        ];
        let lap = graph_laplacian(w, 3);
        let fv = fiedler_value(lap, 3);
        // For K3 with unit weights, λ₂ = 3
        assert!((fv - 3.0).abs() < 0.01);
    }

    #[test]
    fn test_spectral_entropy() {
        // Uniform eigenvalues => max entropy
        let eigs = vec![1.0, 1.0, 1.0, 1.0];
        let h = spectral_entropy(eigs);
        assert!((h - (4.0_f64).ln()).abs() < 0.01);
    }

    #[test]
    fn test_heat_trace() {
        let eigs = vec![0.0, 1.0, 2.0];
        let ht = heat_trace(eigs, vec![0.0, 1.0]);
        assert!((ht[0] - 3.0).abs() < 1e-10); // HT(0) = n
        // HT(1) = exp(0) + exp(-1) + exp(-2)
        let expected = 1.0 + (-1.0_f64).exp() + (-2.0_f64).exp();
        assert!((ht[1] - expected).abs() < 1e-10);
    }
}
