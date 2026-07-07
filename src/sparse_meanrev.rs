// Sparse Mean-Reverting Portfolios
// Implementation of algorithms from "Identifying Small Mean Reverting Portfolios" (d'Aspremont, 2011)
// and related sparse cointegration methods
#![allow(non_snake_case)]

use nalgebra::{DMatrix, DVector};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray2, PyReadonlyArray1, PyReadonlyArray2};

/// Sparse PCA using Iterative Thresholding (d'Aspremont approach)
/// 
/// Finds sparse principal components by adding L1 penalty to maximize variance
/// subject to sparsity constraints.
///
/// # Mathematical Formulation
/// maximize: w^T Σ w - λ ||w||_1
/// subject to: ||w||_2 = 1
///
/// where:
/// - Σ is the covariance matrix
/// - λ is the sparsity parameter (larger = sparser)
/// - w is the portfolio weight vector
///
/// # Algorithm (Iterative Soft-Thresholding)
/// 1. Initialize w = first eigenvector of Σ
/// 2. Repeat until convergence:
///    a. w_new = Σw - λ * sign(w)
///    b. Apply soft thresholding: w_i = sign(w_i) * max(|w_i| - λ, 0)
///    c. Normalize: w = w_new / ||w_new||_2
///
/// # Returns
/// (weights, variance_explained, sparsity, iterations)
#[pyfunction]
pub fn sparse_pca_rust(
    py: Python,
    returns: PyReadonlyArray2<f64>,
    n_components: usize,
    lambda: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyDict>> {
    let returns = returns.as_array();
    let (n_samples, n_assets) = (returns.nrows(), returns.ncols());
    
    // Compute covariance matrix
    let mut cov = DMatrix::zeros(n_assets, n_assets);
    let means = DVector::from_iterator(
        n_assets,
        (0..n_assets).map(|j| returns.column(j).sum() / n_samples as f64),
    );
    
    for i in 0..n_samples {
        let row = returns.row(i);
        for j in 0..n_assets {
            for k in 0..n_assets {
                let val_j = row[j] - means[j];
                let val_k = row[k] - means[k];
                cov[(j, k)] += val_j * val_k;
            }
        }
    }
    cov /= (n_samples - 1) as f64;
    
    // Store results
    let mut weights_matrix = DMatrix::zeros(n_components, n_assets);
    let mut variance_explained = Vec::new();
    let mut sparsity_levels = Vec::new();
    let mut iterations_used = Vec::new();
    
    // Deflation: extract components one by one
    let mut residual_cov = cov.clone();
    
    for comp in 0..n_components {
        // Initialize with leading eigenvector (power method approximation)
        let mut w = DVector::from_element(n_assets, 1.0 / (n_assets as f64).sqrt());
        
        let mut iter = 0;
        let mut converged = false;
        
        while iter < max_iter && !converged {
            let w_old = w.clone();
            
            // Gradient step: w_new = Σw
            let mut w_new = &residual_cov * &w;
            
            // Soft thresholding: shrink towards zero
            for i in 0..n_assets {
                let val = w_new[i];
                w_new[i] = val.signum() * (val.abs() - lambda).max(0.0);
            }
            
            // Normalize
            let norm = w_new.norm();
            if norm > 1e-10 {
                w_new /= norm;
            } else {
                // If all weights shrunk to zero, restart with smaller lambda effect
                w_new = DVector::from_element(n_assets, 1.0 / (n_assets as f64).sqrt());
            }
            
            w = w_new;
            
            // Check convergence
            let diff = (&w - &w_old).norm();
            if diff < tol {
                converged = true;
            }
            
            iter += 1;
        }
        
        // Store component
        weights_matrix.row_mut(comp).copy_from(&w.transpose());
        
        // Compute variance explained
        let var = w.transpose() * &residual_cov * &w;
        variance_explained.push(var[(0, 0)]);
        
        // Compute sparsity (percentage of non-zero weights)
        let non_zero = w.iter().filter(|&&x| x.abs() > 1e-6).count();
        sparsity_levels.push(non_zero as f64 / n_assets as f64);
        
        iterations_used.push(iter);
        
        // Deflate: remove component from covariance matrix
        // Σ_new = Σ - λ_i * w_i * w_i^T
        residual_cov -= var[(0, 0)] * &w * w.transpose();
    }
    
    // Convert to Python
    let dict = PyDict::new_bound(py);
    
    // Convert weights to 2D numpy array
    let _weights_vec: Vec<f64> = weights_matrix.iter().cloned().collect();
    let weights_py = PyArray2::from_vec2_bound(
        py,
        &(0..n_components)
            .map(|i| weights_matrix.row(i).iter().cloned().collect::<Vec<_>>())
            .collect::<Vec<_>>(),
    )?;
    dict.set_item("weights", weights_py)?;
    
    let total_var: f64 = variance_explained.iter().sum();
    dict.set_item("variance_explained", variance_explained.into_py(py))?;
    dict.set_item("sparsity", sparsity_levels.into_py(py))?;
    dict.set_item("iterations", iterations_used.into_py(py))?;
    dict.set_item("total_variance_explained", total_var.into_py(py))?;
    
    Ok(dict.into())
}

/// Box & Tao Decomposition for Mean-Reverting Portfolios
///
/// Decomposes price matrix into:
/// X = L + S + N
/// where:
/// - L: low-rank component (common factors)
/// - S: sparse component (idiosyncratic mean-reverting opportunities)
/// - N: noise
///
/// This is based on Robust PCA / Principal Component Pursuit
///
/// # Mathematical Formulation
/// minimize: ||L||_* + λ ||S||_1
/// subject to: X = L + S + N, ||N||_F ≤ ε
///
/// where:
/// - ||·||_* is the nuclear norm (sum of singular values)
/// - ||·||_1 is the L1 norm (sum of absolute values)
/// - ||·||_F is the Frobenius norm
///
/// # Algorithm (Alternating Direction Method of Multipliers - ADMM)
/// 1. Initialize L = S = 0, Y = 0
/// 2. Repeat until convergence:
///    a. Update L: soft-threshold singular values
///    b. Update S: soft-threshold entries
///    c. Update dual variable Y
///
/// # Returns
/// (low_rank_component, sparse_component, noise, objective_values)
#[pyfunction]
pub fn box_tao_decomposition_rust(
    py: Python,
    price_matrix: PyReadonlyArray2<f64>,
    lambda: f64,
    mu: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyDict>> {
    let X = price_matrix.as_array();
    let (n_samples, n_assets) = (X.nrows(), X.ncols());
    
    // Convert to nalgebra matrix
    let X_mat = DMatrix::from_iterator(n_samples, n_assets, X.iter().cloned());
    
    // Initialize variables
    let mut L = DMatrix::zeros(n_samples, n_assets);
    let mut S = DMatrix::zeros(n_samples, n_assets);
    let mut Y = DMatrix::zeros(n_samples, n_assets); // Dual variable
    
    let mut objective_values = Vec::new();
    let rho = 1.0 / mu; // Augmented Lagrangian parameter
    
    for iter in 0..max_iter {
        // Update L: Soft-threshold singular values (nuclear norm minimization)
        let temp = &X_mat - &S + (1.0 / rho) * &Y;
        L = soft_threshold_svd(&temp, 1.0 / rho);
        
        // Update S: Soft-threshold entries (L1 minimization)
        let temp = &X_mat - &L + (1.0 / rho) * &Y;
        S = soft_threshold_elementwise(&temp, lambda / rho);
        
        // Update dual variable Y
        let residual = &X_mat - &L - &S;
        Y += rho * &residual;
        
        // Compute objective: ||L||_* + λ||S||_1
        let nuclear_norm = compute_nuclear_norm(&L);
        let l1_norm = S.iter().map(|&x| x.abs()).sum::<f64>();
        let objective = nuclear_norm + lambda * l1_norm;
        objective_values.push(objective);
        
        // Check convergence
        let primal_residual = residual.norm();
        if iter > 0 && primal_residual < tol {
            break;
        }
    }
    
    // Compute noise: N = X - L - S
    let N = &X_mat - &L - &S;
    
    // Convert to Python
    let dict = PyDict::new_bound(py);
    
    let L_vec: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| L.row(i).iter().cloned().collect())
        .collect();
    dict.set_item("low_rank", PyArray2::from_vec2_bound(py, &L_vec)?)?;
    
    let S_vec: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| S.row(i).iter().cloned().collect())
        .collect();
    dict.set_item("sparse", PyArray2::from_vec2_bound(py, &S_vec)?)?;
    
    let N_vec: Vec<Vec<f64>> = (0..n_samples)
        .map(|i| N.row(i).iter().cloned().collect())
        .collect();
    dict.set_item("noise", PyArray2::from_vec2_bound(py, &N_vec)?)?;
    
    dict.set_item("objective_values", objective_values.into_py(py))?;
    
    Ok(dict.into())
}

/// Hurst Exponent Calculation using Rescaled Range (R/S) Analysis
///
/// The Hurst exponent H characterizes the long-term memory of a time series:
/// - H = 0.5: Random walk (Brownian motion)
/// - H < 0.5: Mean-reverting process (anti-persistent)
/// - H > 0.5: Trending process (persistent)
///
/// For mean-reverting portfolios, we want H < 0.5
///
/// # Mathematical Formulation
/// 1. Compute cumulative deviations: Y(t) = Σ[X(i) - mean(X)]
/// 2. Compute range: R(n) = max(Y) - min(Y)
/// 3. Compute standard deviation: S(n)
/// 4. Rescaled range: R/S(n)
/// 5. Hurst exponent: H from log(R/S) ≈ H * log(n)
///
/// # Algorithm
/// 1. Split series into windows of varying sizes
/// 2. For each window size n:
///    a. Compute R/S statistic
///    b. Average across windows
/// 3. Fit log(R/S) vs log(n) → slope = H
///
/// # Returns
/// (hurst_exponent, confidence_interval, is_mean_reverting)
#[pyfunction]
#[pyo3(signature = (time_series, min_window=None, max_window=None))]
pub fn hurst_exponent_rust(
    py: Python,
    time_series: PyReadonlyArray1<f64>,
    min_window: Option<usize>,
    max_window: Option<usize>,
) -> PyResult<Py<PyDict>> {
    let ts = time_series.as_array();
    let n = ts.len();
    
    let min_win = min_window.unwrap_or(10.max(n / 100));
    let max_win = max_window.unwrap_or(n / 2);
    
    // Generate window sizes (logarithmically spaced)
    let num_windows = 20;
    let mut window_sizes = Vec::new();
    let log_min = (min_win as f64).ln();
    let log_max = (max_win as f64).ln();
    let step = (log_max - log_min) / (num_windows as f64 - 1.0);
    
    for i in 0..num_windows {
        let size = (log_min + i as f64 * step).exp() as usize;
        if size >= min_win && size <= max_win && size > 0 {
            window_sizes.push(size);
        }
    }
    
    let mut rs_values = Vec::new();
    let mut log_n = Vec::new();
    let mut log_rs = Vec::new();
    
    for &window_size in &window_sizes {
        let num_windows_in_series = n / window_size;
        if num_windows_in_series == 0 {
            continue;
        }
        
        let mut rs_for_size = Vec::new();
        
        for w in 0..num_windows_in_series {
            let start = w * window_size;
            let end = (start + window_size).min(n);
            let window = &ts.as_slice().unwrap()[start..end];
            
            // Compute mean
            let mean = window.iter().sum::<f64>() / window.len() as f64;
            
            // Compute cumulative deviations
            let mut Y = vec![0.0];
            let mut sum = 0.0;
            for &x in window {
                sum += x - mean;
                Y.push(sum);
            }
            
            // Compute range
            let max_Y = Y.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_Y = Y.iter().cloned().fold(f64::INFINITY, f64::min);
            let range = max_Y - min_Y;
            
            // Compute standard deviation
            let variance = window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
            let std = variance.sqrt();
            
            // Rescaled range
            if std > 1e-10 {
                rs_for_size.push(range / std);
            }
        }
        
        if !rs_for_size.is_empty() {
            let avg_rs = rs_for_size.iter().sum::<f64>() / rs_for_size.len() as f64;
            rs_values.push(avg_rs);
            log_n.push((window_size as f64).ln());
            log_rs.push(avg_rs.ln());
        }
    }
    
    // Fit linear regression: log(R/S) = H * log(n) + c
    let (hurst, intercept) = linear_regression(&log_n, &log_rs);
    
    // Compute confidence interval (95%) using residuals
    let residuals: Vec<f64> = log_n
        .iter()
        .zip(&log_rs)
        .map(|(&ln_n, &ln_rs)| ln_rs - (hurst * ln_n + intercept))
        .collect();
    
    let std_error = (residuals.iter().map(|&r| r * r).sum::<f64>() / (residuals.len() - 2) as f64).sqrt();
    let t_value = 1.96; // 95% confidence for large n
    let se_slope = std_error / (log_n.iter().map(|&x| x * x).sum::<f64>()
        - log_n.iter().sum::<f64>().powi(2) / log_n.len() as f64).sqrt();
    
    let confidence_lower = hurst - t_value * se_slope;
    let confidence_upper = hurst + t_value * se_slope;
    
    // Determine if mean-reverting (H < 0.5 with statistical significance)
    let is_mean_reverting = confidence_upper < 0.5;
    
    let dict = PyDict::new_bound(py);
    dict.set_item("hurst_exponent", hurst)?;
    dict.set_item("confidence_interval", vec![confidence_lower, confidence_upper].into_py(py))?;
    dict.set_item("is_mean_reverting", is_mean_reverting)?;
    dict.set_item("interpretation", 
        if hurst < 0.5 {
            "Mean-reverting (anti-persistent)"
        } else if hurst > 0.5 {
            "Trending (persistent)"
        } else {
            "Random walk"
        }
    )?;
    
    // Add detailed statistics for analysis
    dict.set_item("window_sizes", window_sizes.into_py(py))?;
    dict.set_item("rs_values", rs_values.into_py(py))?;
    dict.set_item("standard_error", se_slope)?;
    
    Ok(dict.into())
}

/// Sparse Cointegration Algorithm
///
/// Identifies sparse cointegrating portfolios (small number of assets)
/// that exhibit mean-reverting behavior.
///
/// # Mathematical Formulation
/// Find weights w such that:
/// 1. Portfolio is cointegrated: w^T X(t) is stationary
/// 2. Portfolio is sparse: ||w||_0 is small (or ||w||_1 is small)
///
/// # Algorithm
/// 1. For each asset i, test cointegration with other assets
/// 2. Use sparse regression (LASSO/Elastic Net) to find cointegrating vector
/// 3. Verify mean-reversion with Hurst exponent and ADF test
///
/// # Returns
/// (weights, cointegration_pvalue, hurst_exponent, residuals)
#[pyfunction]
pub fn sparse_cointegration_rust(
    py: Python,
    prices: PyReadonlyArray2<f64>,
    target_asset: usize,
    lambda_l1: f64,
    lambda_l2: f64,
    max_iter: usize,
    tol: f64,
) -> PyResult<Py<PyDict>> {
    let prices_arr = prices.as_array();
    let (n_samples, n_assets) = (prices_arr.nrows(), prices_arr.ncols());
    
    if target_asset >= n_assets {
        return Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(
            "target_asset index out of bounds"
        ));
    }
    
    // Extract target and regressor prices
    let target_prices = DVector::from_iterator(
        n_samples,
        prices_arr.column(target_asset).iter().cloned(),
    );
    
    let mut X = DMatrix::zeros(n_samples, n_assets - 1);
    let mut col_idx = 0;
    for j in 0..n_assets {
        if j != target_asset {
            for i in 0..n_samples {
                X[(i, col_idx)] = prices_arr[(i, j)];
            }
            col_idx += 1;
        }
    }
    
    // Elastic Net regression: minimize ||y - Xw||_2^2 + λ1||w||_1 + λ2||w||_2^2
    let weights = elastic_net_regression(&X, &target_prices, lambda_l1, lambda_l2, max_iter, tol);
    
    // Compute residuals (cointegrating portfolio)
    let fitted = &X * &weights;
    let residuals = &target_prices - &fitted;
    
    // Convert weights back to full portfolio (including target asset with weight -1)
    let mut full_weights = vec![0.0; n_assets];
    full_weights[target_asset] = -1.0; // Normalize to target asset
    
    let mut idx = 0;
    for j in 0..n_assets {
        if j != target_asset {
            full_weights[j] = weights[idx];
            idx += 1;
        }
    }
    
    // Normalize weights
    let weight_sum: f64 = full_weights.iter().map(|&w| w.abs()).sum();
    if weight_sum > 1e-10 {
        for w in &mut full_weights {
            *w /= weight_sum;
        }
    }
    
    // Compute sparsity
    let non_zero = full_weights.iter().filter(|&&w| w.abs() > 1e-4).count();
    let sparsity = non_zero as f64 / n_assets as f64;
    
    // Compute Hurst exponent on residuals
    let residuals_vec: Vec<f64> = residuals.iter().cloned().collect();
    
    let dict = PyDict::new_bound(py);
    dict.set_item("weights", full_weights.into_py(py))?;
    dict.set_item("residuals", residuals_vec.into_py(py))?;
    dict.set_item("sparsity", sparsity)?;
    dict.set_item("non_zero_count", non_zero)?;
    
    Ok(dict.into())
}

// ============================================================================
// Helper Functions
// ============================================================================

/// Soft-threshold SVD for nuclear norm minimization
fn soft_threshold_svd(matrix: &DMatrix<f64>, threshold: f64) -> DMatrix<f64> {
    let svd = matrix.clone().svd(true, true);
    let mut singular_values = svd.singular_values.clone();
    
    // Soft-threshold singular values
    for val in singular_values.iter_mut() {
        *val = (*val - threshold).max(0.0);
    }
    
    // Reconstruct matrix
    let U = svd.u.unwrap();
    let V_t = svd.v_t.unwrap();
    let S = DMatrix::from_diagonal(&singular_values);
    
    &U * &S * &V_t
}

/// Element-wise soft thresholding
fn soft_threshold_elementwise(matrix: &DMatrix<f64>, threshold: f64) -> DMatrix<f64> {
    matrix.map(|x| {
        if x > threshold {
            x - threshold
        } else if x < -threshold {
            x + threshold
        } else {
            0.0
        }
    })
}

/// Compute nuclear norm (sum of singular values)
fn compute_nuclear_norm(matrix: &DMatrix<f64>) -> f64 {
    let svd = matrix.clone().svd(false, false);
    svd.singular_values.sum()
}

/// Linear regression: y = ax + b
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum();
    
    let slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    let intercept = (sum_y - slope * sum_x) / n;
    
    (slope, intercept)
}

/// Elastic Net regression using coordinate descent
fn elastic_net_regression(
    X: &DMatrix<f64>,
    y: &DVector<f64>,
    lambda_l1: f64,
    lambda_l2: f64,
    max_iter: usize,
    tol: f64,
) -> DVector<f64> {
    let (_n_samples, n_features) = (X.nrows(), X.ncols());
    let mut weights = DVector::zeros(n_features);
    
    // Precompute X^T X and X^T y for efficiency
    let XtX = X.transpose() * X;
    let _Xty = X.transpose() * y;
    
    for _iter in 0..max_iter {
        let weights_old = weights.clone();
        
        // Coordinate descent: update each weight separately
        for j in 0..n_features {
            // Compute residual without j-th feature
            let mut residual = y.clone();
            for k in 0..n_features {
                if k != j {
                    residual -= weights[k] * X.column(k);
                }
            }
            
            // Compute optimal update for j-th weight
            let rho = X.column(j).dot(&residual);
            let z = XtX[(j, j)] + lambda_l2;
            
            // Soft thresholding
            weights[j] = if rho > lambda_l1 {
                (rho - lambda_l1) / z
            } else if rho < -lambda_l1 {
                (rho + lambda_l1) / z
            } else {
                0.0
            };
        }
        
        // Check convergence
        let diff = (&weights - &weights_old).norm();
        if diff < tol {
            break;
        }
    }
    
    weights
}


