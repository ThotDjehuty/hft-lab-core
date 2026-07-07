//! Symmetric Positive Definite (SPD) manifold operations.
//!
//! Covariance matrices live on the SPD manifold P(n).
//! Standard Euclidean operations (mean, distance) are ill-suited
//! because SPD(n) is a curved Riemannian manifold.
//!
//! This module implements:
//! - Log-Euclidean metric: d(Σ₁, Σ₂) = ‖log(Σ₁) - log(Σ₂)‖_F
//! - Fréchet mean in log-Euclidean space
//! - Matrix logarithm via eigendecomposition
//! - Geodesic interpolation on SPD(n)
//!
//! References:
//! - Arsigny et al. (2007), "Geometric Means in a Novel Vector Space Structure on SPD Matrices"
//! - Pennec et al. (2006), "A Riemannian Framework for Tensor Computing"

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Compute the matrix logarithm of a symmetric positive definite matrix.
///
/// Uses eigendecomposition: log(Σ) = V · diag(log λᵢ) · Vᵀ
///
/// # Arguments
/// * `matrix` - Flat n×n SPD matrix (row-major)
/// * `n` - Matrix dimension
///
/// # Returns
/// Flat n×n matrix logarithm
#[cfg_attr(feature = "python", pyfunction)]
pub fn matrix_log_sym(matrix: Vec<f64>, n: usize) -> Vec<f64> {
    let (eigenvalues, eigenvectors) = jacobi_eigen(&matrix, n);

    // log(Σ) = V · diag(log λᵢ) · Vᵀ
    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                let lam_k = eigenvalues[k].max(1e-15).ln();
                sum += eigenvectors[i * n + k] * lam_k * eigenvectors[j * n + k];
            }
            result[i * n + j] = sum;
        }
    }
    result
}

/// Compute the matrix exponential of a symmetric matrix.
///
/// Uses eigendecomposition: exp(M) = V · diag(exp λᵢ) · Vᵀ
///
/// # Arguments
/// * `matrix` - Flat n×n symmetric matrix (row-major)
/// * `n` - Matrix dimension
///
/// # Returns
/// Flat n×n matrix exponential
#[cfg_attr(feature = "python", pyfunction)]
pub fn matrix_exp_sym(matrix: Vec<f64>, n: usize) -> Vec<f64> {
    let (eigenvalues, eigenvectors) = jacobi_eigen(&matrix, n);

    let mut result = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            let mut sum = 0.0;
            for k in 0..n {
                sum += eigenvectors[i * n + k] * eigenvalues[k].exp() * eigenvectors[j * n + k];
            }
            result[i * n + j] = sum;
        }
    }
    result
}

/// Log-Euclidean distance between two SPD matrices.
///
/// d_LE(Σ₁, Σ₂) = ‖log(Σ₁) - log(Σ₂)‖_F
///
/// This is a proper Riemannian distance on SPD(n), computationally
/// cheaper than the affine-invariant distance while sharing
/// many of its desirable properties.
///
/// # Arguments
/// * `sigma1` - Flat n×n SPD matrix
/// * `sigma2` - Flat n×n SPD matrix
/// * `n` - Matrix dimension
///
/// # Returns
/// Log-Euclidean distance (non-negative)
#[cfg_attr(feature = "python", pyfunction)]
pub fn log_euclidean_distance(sigma1: Vec<f64>, sigma2: Vec<f64>, n: usize) -> f64 {
    let log1 = matrix_log_sym(sigma1, n);
    let log2 = matrix_log_sym(sigma2, n);

    // Frobenius norm of difference
    let mut sum_sq = 0.0;
    for i in 0..n * n {
        let diff = log1[i] - log2[i];
        sum_sq += diff * diff;
    }
    sum_sq.sqrt()
}

/// Fréchet mean in log-Euclidean space.
///
/// μ_LE = exp(1/K Σₖ log(Σₖ))
///
/// The log-Euclidean Fréchet mean is unique and has a closed-form
/// solution (unlike the affine-invariant Fréchet mean which requires
/// iterative optimization).
///
/// # Arguments
/// * `matrices` - Flat array of K concatenated n×n SPD matrices
/// * `n` - Matrix dimension
/// * `count` - Number of matrices K
///
/// # Returns
/// Flat n×n Fréchet mean SPD matrix
#[cfg_attr(feature = "python", pyfunction)]
pub fn frechet_mean_le(matrices: Vec<f64>, n: usize, count: usize) -> Vec<f64> {
    assert_eq!(matrices.len(), count * n * n);

    // Mean of log matrices
    let mut mean_log = vec![0.0; n * n];
    for k in 0..count {
        let mat = &matrices[k * n * n..(k + 1) * n * n];
        let log_mat = matrix_log_sym(mat.to_vec(), n);
        for i in 0..n * n {
            mean_log[i] += log_mat[i] / count as f64;
        }
    }

    // Exponentiate back
    matrix_exp_sym(mean_log, n)
}

/// Geodesic interpolation between two SPD matrices in log-Euclidean space.
///
/// γ(t) = exp((1-t) · log(Σ₁) + t · log(Σ₂)), t ∈ [0, 1]
///
/// # Arguments
/// * `sigma1` - Start SPD matrix (flat n×n)
/// * `sigma2` - End SPD matrix (flat n×n)
/// * `n` - Matrix dimension
/// * `t` - Interpolation parameter in [0, 1]
///
/// # Returns
/// Interpolated SPD matrix on the geodesic
#[cfg_attr(feature = "python", pyfunction)]
pub fn spd_geodesic(sigma1: Vec<f64>, sigma2: Vec<f64>, n: usize, t: f64) -> Vec<f64> {
    let log1 = matrix_log_sym(sigma1, n);
    let log2 = matrix_log_sym(sigma2, n);

    let mut interp = vec![0.0; n * n];
    for i in 0..n * n {
        interp[i] = (1.0 - t) * log1[i] + t * log2[i];
    }

    matrix_exp_sym(interp, n)
}

/// Compute pairwise log-Euclidean distance matrix.
///
/// # Arguments
/// * `matrices` - Flat array of K concatenated n×n SPD matrices
/// * `n` - Matrix dimension
/// * `count` - Number of matrices K
///
/// # Returns
/// Flat K×K distance matrix
#[cfg_attr(feature = "python", pyfunction)]
pub fn spd_distance_matrix(matrices: Vec<f64>, n: usize, count: usize) -> Vec<f64> {
    assert_eq!(matrices.len(), count * n * n);

    // Pre-compute log matrices
    let logs: Vec<Vec<f64>> = (0..count)
        .map(|k| {
            let mat = &matrices[k * n * n..(k + 1) * n * n];
            matrix_log_sym(mat.to_vec(), n)
        })
        .collect();

    let mut dist = vec![0.0; count * count];
    for i in 0..count {
        for j in (i + 1)..count {
            let mut sum_sq = 0.0;
            for idx in 0..n * n {
                let diff = logs[i][idx] - logs[j][idx];
                sum_sq += diff * diff;
            }
            let d = sum_sq.sqrt();
            dist[i * count + j] = d;
            dist[j * count + i] = d;
        }
    }
    dist
}

/// Jacobi eigendecomposition for symmetric matrices.
/// Returns (eigenvalues, eigenvectors) where eigenvectors is row-major
/// with eigenvectors[i * n + k] = component i of eigenvector k.
fn jacobi_eigen(matrix: &[f64], n: usize) -> (Vec<f64>, Vec<f64>) {
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

    let eigenvalues: Vec<f64> = (0..n).map(|i| a[i * n + i]).collect();
    (eigenvalues, v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_matrix_log_identity() {
        // log(I) = 0
        let eye = vec![1.0, 0.0, 0.0, 1.0];
        let log_eye = matrix_log_sym(eye, 2);
        for &v in &log_eye {
            assert!(v.abs() < 1e-10);
        }
    }

    #[test]
    fn test_log_exp_roundtrip() {
        // exp(log(Σ)) = Σ
        let sigma = vec![2.0, 0.5, 0.5, 3.0]; // SPD 2×2
        let log_s = matrix_log_sym(sigma.clone(), 2);
        let recovered = matrix_exp_sym(log_s, 2);
        for i in 0..4 {
            assert!(
                (recovered[i] - sigma[i]).abs() < 1e-6,
                "Mismatch at {}: {} vs {}",
                i,
                recovered[i],
                sigma[i]
            );
        }
    }

    #[test]
    fn test_log_euclidean_distance() {
        let sigma1 = vec![2.0, 0.0, 0.0, 2.0];
        let sigma2 = vec![3.0, 0.0, 0.0, 3.0];
        let d = log_euclidean_distance(sigma1, sigma2, 2);
        // d = ||log(2I) - log(3I)||_F = ||diag(log2, log2) - diag(log3, log3)||_F
        // = sqrt(2) * |log2 - log3|
        let expected = (2.0_f64).sqrt() * (2.0_f64.ln() - 3.0_f64.ln()).abs();
        assert!((d - expected).abs() < 1e-6);
    }

    #[test]
    fn test_frechet_mean_single() {
        // Mean of a single matrix is itself
        let sigma = vec![2.0, 0.5, 0.5, 3.0];
        let mean = frechet_mean_le(sigma.clone(), 2, 1);
        for i in 0..4 {
            assert!((mean[i] - sigma[i]).abs() < 1e-6);
        }
    }

    #[test]
    fn test_spd_geodesic_endpoints() {
        let s1 = vec![1.0, 0.0, 0.0, 1.0];
        let s2 = vec![4.0, 0.0, 0.0, 4.0];

        // t=0 → s1
        let g0 = spd_geodesic(s1.clone(), s2.clone(), 2, 0.0);
        for i in 0..4 {
            assert!((g0[i] - s1[i]).abs() < 1e-6);
        }

        // t=1 → s2
        let g1 = spd_geodesic(s1.clone(), s2.clone(), 2, 1.0);
        for i in 0..4 {
            assert!((g1[i] - s2[i]).abs() < 1e-6);
        }
    }
}
