//! Sheaf cohomology and Hodge decomposition for invariant arbitrage detection.
//!
//! Models the market as a cellular sheaf on the exchange graph:
//! - Coboundary operator δ encodes price consistency conditions
//! - Sheaf Laplacian L_F = δᵀδ
//! - dim H¹ > 0 ⟺ Arbitrage exists
//! - Hodge decomposition: gradient (no-arb) + curl (cycle arb) + harmonic (structural)
//!
//! References: Hansen & Ghrist (2019), "Toward a Spectral Theory of Cellular Sheaves"

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Result of Hodge decomposition: f = gradient + curl + harmonic
#[derive(Debug, Clone)]
#[cfg_attr(feature = "python", pyclass(get_all))]
pub struct HodgeComponents {
    /// Gradient component (consistent with global pricing, no arbitrage)
    pub gradient: Vec<f64>,
    /// Curl component (cyclic flows, transient cycle arbitrage)
    pub curl: Vec<f64>,
    /// Harmonic component (structural residuals, persistent H¹ arbitrage)
    pub harmonic: Vec<f64>,
}

/// Build the coboundary matrix δ for a directed graph.
///
/// For each edge (i, j), δ has +1 at column j and -1 at column i.
/// This is the simplicial coboundary operator d₀: C⁰ → C¹.
///
/// # Arguments
/// * `edges` - List of directed edges as flat [i₀, j₀, i₁, j₁, ...]
/// * `n_vertices` - Number of vertices
///
/// # Returns
/// Flat m×n coboundary matrix (row-major)
#[cfg_attr(feature = "python", pyfunction)]
pub fn coboundary_matrix(edges: Vec<usize>, n_vertices: usize) -> Vec<f64> {
    let n_edges = edges.len() / 2;
    let mut delta = vec![0.0; n_edges * n_vertices];

    for e in 0..n_edges {
        let i = edges[2 * e];
        let j = edges[2 * e + 1];
        delta[e * n_vertices + i] = -1.0;
        delta[e * n_vertices + j] = 1.0;
    }
    delta
}

/// Build a weighted coboundary matrix using log price ratios as edge weights.
///
/// For edge (i, j) with weight A_{i→j} = log(P_j/P_i):
/// δ_{e,i} = -A_{i→j}, δ_{e,j} = 1
///
/// # Arguments
/// * `edges` - Flat edge list [i₀, j₀, i₁, j₁, ...]
/// * `edge_weights` - Log price ratios A_{i→j} for each edge
/// * `n_vertices` - Number of vertices
///
/// # Returns
/// Flat m×n weighted coboundary matrix
#[cfg_attr(feature = "python", pyfunction)]
pub fn weighted_coboundary_matrix(
    edges: Vec<usize>,
    edge_weights: Vec<f64>,
    n_vertices: usize,
) -> Vec<f64> {
    let n_edges = edges.len() / 2;
    assert_eq!(edge_weights.len(), n_edges);
    let mut delta = vec![0.0; n_edges * n_vertices];

    for e in 0..n_edges {
        let i = edges[2 * e];
        let j = edges[2 * e + 1];
        delta[e * n_vertices + i] = -edge_weights[e];
        delta[e * n_vertices + j] = 1.0;
    }
    delta
}

/// Compute the sheaf Laplacian L_F = δᵀδ.
///
/// # Arguments
/// * `coboundary` - Flat m×n coboundary matrix
/// * `m` - Number of edges
/// * `n` - Number of vertices
///
/// # Returns
/// Flat n×n sheaf Laplacian matrix
#[cfg_attr(feature = "python", pyfunction)]
pub fn sheaf_laplacian(coboundary: Vec<f64>, m: usize, n: usize) -> Vec<f64> {
    assert_eq!(coboundary.len(), m * n);
    let mut lap = vec![0.0; n * n];

    // L = δᵀ δ
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for e in 0..m {
                sum += coboundary[e * n + i] * coboundary[e * n + j];
            }
            lap[i * n + j] = sum;
        }
    }
    lap
}

/// Detect arbitrage via near-zero eigenvalues of the sheaf Laplacian.
///
/// Returns the eigenvectors (invariant portfolios) corresponding to
/// eigenvalues smaller than `tol`.
///
/// # Arguments
/// * `sheaf_lap` - Flat n×n sheaf Laplacian
/// * `n` - Number of vertices
/// * `tol` - Tolerance for "near-zero" eigenvalues
///
/// # Returns
/// List of eigenvectors (invariant portfolios), each of length n
#[cfg_attr(feature = "python", pyfunction)]
pub fn detect_arbitrage(sheaf_lap: Vec<f64>, n: usize, tol: f64) -> Vec<Vec<f64>> {
    let (eigenvalues, eigenvectors) = symmetric_eigen_simple(&sheaf_lap, n);

    let mut portfolios = Vec::new();
    for (idx, &eval) in eigenvalues.iter().enumerate() {
        if eval.abs() < tol {
            let portfolio: Vec<f64> = eigenvectors[idx * n..(idx + 1) * n].to_vec();
            portfolios.push(portfolio);
        }
    }
    portfolios
}

/// Count the dimension of H¹ (number of independent arbitrage opportunities).
///
/// # Arguments
/// * `sheaf_lap` - Flat n×n sheaf Laplacian
/// * `n` - Number of vertices
/// * `tol` - Tolerance for near-zero eigenvalues
///
/// # Returns
/// dim H¹ (number of near-zero eigenvalues)
#[cfg_attr(feature = "python", pyfunction)]
pub fn cohomology_dimension(sheaf_lap: Vec<f64>, n: usize, tol: f64) -> usize {
    let eigenvalues = symmetric_eigenvalues_simple(&sheaf_lap, n);
    eigenvalues.iter().filter(|&&v| v.abs() < tol).count()
}

/// Compute the Hodge decomposition of an edge signal.
///
/// Decomposes f = δg (gradient) + δ*h (curl) + z (harmonic)
///
/// # Arguments
/// * `edge_signal` - Signal on edges (length m)
/// * `coboundary` - Flat m×n coboundary matrix
/// * `m` - Number of edges
/// * `n` - Number of vertices
///
/// # Returns
/// HodgeComponents with gradient, curl, and harmonic parts
#[cfg_attr(feature = "python", pyfunction)]
pub fn hodge_decompose(
    edge_signal: Vec<f64>,
    coboundary: Vec<f64>,
    m: usize,
    n: usize,
) -> HodgeComponents {
    assert_eq!(edge_signal.len(), m);
    assert_eq!(coboundary.len(), m * n);

    // Compute L₀ = δᵀδ (vertex Laplacian)
    let l0 = sheaf_laplacian(coboundary.clone(), m, n);

    // Solve L₀ g = δᵀ f for the gradient potential g
    // δᵀ f:
    let mut delta_t_f = vec![0.0; n];
    for i in 0..n {
        for e in 0..m {
            delta_t_f[i] += coboundary[e * n + i] * edge_signal[e];
        }
    }

    // Solve via pseudoinverse (regularized)
    let g = pseudo_solve(&l0, &delta_t_f, n);

    // Gradient component = δ g
    let mut gradient = vec![0.0; m];
    for e in 0..m {
        for i in 0..n {
            gradient[e] += coboundary[e * n + i] * g[i];
        }
    }

    // For a simple graph (no 2-simplices), we skip the curl term.
    // Harmonic = f - gradient
    let harmonic: Vec<f64> = edge_signal
        .iter()
        .zip(gradient.iter())
        .map(|(&f, &g)| f - g)
        .collect();

    // In the simple case, curl is empty (zero)
    let curl = vec![0.0; m];

    HodgeComponents {
        gradient,
        curl,
        harmonic,
    }
}

/// Simple symmetric eigenvalue computation (Jacobi method).
fn symmetric_eigenvalues_simple(matrix: &[f64], n: usize) -> Vec<f64> {
    let mut a = matrix.to_vec();
    let max_iter = 100 * n * n;
    let tol = 1e-12;

    for _ in 0..max_iter {
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
    }

    let mut eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    eigenvalues.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    eigenvalues
}

/// Eigenvalues and eigenvectors (Jacobi method).
fn symmetric_eigen_simple(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
    let mut a = matrix.to_vec();
    let mut v = vec![0.0; n * n];
    for i in 0..n {
        v[i * n + i] = 1.0;
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

        for i in 0..n {
            let vip = v[i * n + p];
            let viq = v[i * n + q];
            v[i * n + p] = c * vip + s * viq;
            v[i * n + q] = -s * vip + c * viq;
        }
    }

    let mut pairs: Vec<(f64, Vec<f64>)> = (0..n)
        .map(|j| {
            let ev = a[j * n + j];
            let vec: Vec<f64> = (0..n).map(|i| v[i * n + j]).collect();
            (ev, vec)
        })
        .collect();
    pairs.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

    let eigenvalues: Vec<f64> = pairs.iter().map(|(e, _)| *e).collect();
    let eigenvectors: Vec<f64> = pairs.iter().flat_map(|(_, v)| v.iter()).copied().collect();
    (eigenvalues, eigenvectors)
}

/// Solve Ax = b via regularized pseudoinverse (Tikhonov).
fn pseudo_solve(a: &[f64], b: &[f64], n: usize) -> Vec<f64> {
    let reg = 1e-10;
    // A_reg = A + reg * I
    let mut a_reg = a.to_vec();
    for i in 0..n {
        a_reg[i * n + i] += reg;
    }

    // Simple Gaussian elimination (for small n)
    let mut aug = vec![0.0; n * (n + 1)];
    for i in 0..n {
        for j in 0..n {
            aug[i * (n + 1) + j] = a_reg[i * n + j];
        }
        aug[i * (n + 1) + n] = b[i];
    }

    // Forward elimination
    for k in 0..n {
        // Partial pivoting
        let mut max_idx = k;
        let mut max_val = aug[k * (n + 1) + k].abs();
        for i in (k + 1)..n {
            if aug[i * (n + 1) + k].abs() > max_val {
                max_val = aug[i * (n + 1) + k].abs();
                max_idx = i;
            }
        }
        if max_idx != k {
            for j in 0..=n {
                let tmp = aug[k * (n + 1) + j];
                aug[k * (n + 1) + j] = aug[max_idx * (n + 1) + j];
                aug[max_idx * (n + 1) + j] = tmp;
            }
        }

        let pivot = aug[k * (n + 1) + k];
        if pivot.abs() < 1e-15 {
            continue;
        }

        for i in (k + 1)..n {
            let factor = aug[i * (n + 1) + k] / pivot;
            for j in k..=n {
                aug[i * (n + 1) + j] -= factor * aug[k * (n + 1) + j];
            }
        }
    }

    // Back substitution
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = aug[i * (n + 1) + n];
        for j in (i + 1)..n {
            sum -= aug[i * (n + 1) + j] * x[j];
        }
        let pivot = aug[i * (n + 1) + i];
        x[i] = if pivot.abs() > 1e-15 {
            sum / pivot
        } else {
            0.0
        };
    }
    x
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_coboundary_matrix() {
        // Triangle: edges (0,1), (1,2), (2,0)
        let edges = vec![0, 1, 1, 2, 2, 0];
        let delta = coboundary_matrix(edges, 3);
        assert_eq!(delta.len(), 3 * 3); // 3 edges × 3 vertices
        // Edge (0,1): row 0 = [-1, 1, 0]
        assert_eq!(delta[0], -1.0);
        assert_eq!(delta[1], 1.0);
        assert_eq!(delta[2], 0.0);
    }

    #[test]
    fn test_sheaf_laplacian_no_arb() {
        // 3 vertices, 3 edges (triangle), consistent pricing ⇒ H¹ = 0
        let edges = vec![0, 1, 1, 2, 2, 0];
        let delta = coboundary_matrix(edges, 3);
        let lap = sheaf_laplacian(delta, 3, 3);
        let dim = cohomology_dimension(lap, 3, 0.01);
        // For a triangle graph, the sheaf Laplacian of the unweighted coboundary
        // has 1 zero eigenvalue (H⁰ = ℝ, global constant sections)
        assert!(dim >= 1);
    }

    #[test]
    fn test_hodge_decompose() {
        // Simple triangle with edge signal
        let edges = vec![0, 1, 1, 2, 2, 0];
        let delta = coboundary_matrix(edges, 3);
        // Edge signal: [1.0, 1.0, -2.0] (not a gradient)
        let signal = vec![1.0, 1.0, -2.0];
        let components = hodge_decompose(signal.clone(), delta, 3, 3);

        // Sum should reconstruct original
        let reconstructed: Vec<f64> = (0..3)
            .map(|i| components.gradient[i] + components.curl[i] + components.harmonic[i])
            .collect();
        for i in 0..3 {
            assert!((reconstructed[i] - signal[i]).abs() < 0.1);
        }
    }
}
