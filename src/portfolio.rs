//! Public portfolio construction and time-series analysis primitives.
//!
//! Academic algorithms for portfolio weight optimisation (CARA utility,
//! maximum Sharpe ratio) and the Hurst exponent — exposed unconditionally
//! to all users of hfthot-lab-core (no feature gate).
//!
//! These functions were originally part of the proprietary-gated `meanrev`
//! and `sparse_meanrev` modules. Because the underlying mathematics
//! (Markowitz mean-variance, rescaled-range analysis) is textbook material,
//! they are now available in the public API starting from v1.6.0.

#[cfg(feature = "python")]
use nalgebra::{DMatrix, DVector};
#[cfg(feature = "python")]
use numpy::PyReadonlyArray1;
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

// ─────────────────────────────────────────────────────────────
// CARA optimal weights
// ─────────────────────────────────────────────────────────────

/// Compute optimal portfolio weights using CARA utility maximisation.
///
/// `w* = (1/γ) · Σ⁻¹ · μ`
///
/// Returns dict with `"weights"`, `"expected_return"`, `"expected_variance"`.
#[cfg(feature = "python")]
#[pyfunction]
pub fn cara_optimal_weights_rust(
    py: Python,
    expected_returns: Vec<f64>,
    covariance_matrix: Vec<Vec<f64>>,
    gamma: f64,
) -> PyResult<PyObject> {
    let n = expected_returns.len();

    if n == 0 || covariance_matrix.len() != n || gamma <= 0.0 {
        let result = PyDict::new_bound(py);
        result.set_item("weights", vec![0.0; n])?;
        result.set_item("expected_return", 0.0)?;
        result.set_item("expected_variance", 0.0)?;
        return Ok(result.into());
    }

    let mu = DVector::from_vec(expected_returns.clone());
    let mut cov_data = Vec::with_capacity(n * n);
    for row in &covariance_matrix {
        cov_data.extend_from_slice(row);
    }
    let sigma = DMatrix::from_row_slice(n, n, &cov_data);

    let epsilon = 1e-8;
    let mut sigma_reg = sigma.clone();
    for i in 0..n {
        sigma_reg[(i, i)] += epsilon;
    }

    let weights = match sigma_reg.try_inverse() {
        Some(sigma_inv) => {
            let w = sigma_inv * mu.clone();
            let scaled = w.scale(1.0 / gamma);
            scaled.as_slice().to_vec()
        }
        None => vec![1.0 / n as f64; n],
    };

    let w_vec = DVector::from_vec(weights.clone());
    let expected_return = w_vec.dot(&mu);
    let sigma_w = &sigma * &w_vec;
    let expected_variance = w_vec.dot(&sigma_w);

    let result = PyDict::new_bound(py);
    result.set_item("weights", weights)?;
    result.set_item("expected_return", expected_return)?;
    result.set_item("expected_variance", expected_variance)?;

    Ok(result.into())
}

// ─────────────────────────────────────────────────────────────
// Sharpe-optimal weights
// ─────────────────────────────────────────────────────────────

/// Compute risk-adjusted portfolio weights via Sharpe ratio maximisation.
///
/// `w* ∝ Σ⁻¹ · (μ − r_f)`, normalised to sum to 1.
///
/// Returns dict with `"weights"`, `"sharpe_ratio"`, `"expected_return"`, `"expected_std"`.
#[cfg(feature = "python")]
#[pyfunction]
pub fn sharpe_optimal_weights_rust(
    py: Python,
    expected_returns: Vec<f64>,
    covariance_matrix: Vec<Vec<f64>>,
    risk_free_rate: f64,
) -> PyResult<PyObject> {
    let n = expected_returns.len();

    if n == 0 || covariance_matrix.len() != n {
        let result = PyDict::new_bound(py);
        result.set_item("weights", vec![0.0; n])?;
        result.set_item("sharpe_ratio", 0.0)?;
        result.set_item("expected_return", 0.0)?;
        result.set_item("expected_std", 0.0)?;
        return Ok(result.into());
    }

    let mu = DVector::from_vec(expected_returns.clone());
    let mut cov_data = Vec::with_capacity(n * n);
    for row in &covariance_matrix {
        cov_data.extend_from_slice(row);
    }
    let sigma = DMatrix::from_row_slice(n, n, &cov_data);

    let excess_returns: Vec<f64> = expected_returns
        .iter()
        .map(|&r| r - risk_free_rate)
        .collect();
    let mu_excess = DVector::from_vec(excess_returns);

    let epsilon = 1e-8;
    let mut sigma_reg = sigma.clone();
    for i in 0..n {
        sigma_reg[(i, i)] += epsilon;
    }

    let weights = match sigma_reg.try_inverse() {
        Some(sigma_inv) => {
            let w = sigma_inv * mu_excess.clone();
            let sum = w.sum();
            if sum.abs() > 1e-10 {
                w.scale(1.0 / sum).as_slice().to_vec()
            } else {
                vec![1.0 / n as f64; n]
            }
        }
        None => vec![1.0 / n as f64; n],
    };

    let w_vec = DVector::from_vec(weights.clone());
    let expected_return = w_vec.dot(&mu);
    let sigma_w = &sigma * &w_vec;
    let expected_variance = w_vec.dot(&sigma_w);
    let expected_std = expected_variance.sqrt();

    let sharpe_ratio = if expected_std > 1e-10 {
        (expected_return - risk_free_rate) / expected_std
    } else {
        0.0
    };

    let result = PyDict::new_bound(py);
    result.set_item("weights", weights)?;
    result.set_item("sharpe_ratio", sharpe_ratio)?;
    result.set_item("expected_return", expected_return)?;
    result.set_item("expected_std", expected_std)?;

    Ok(result.into())
}

// ─────────────────────────────────────────────────────────────
// Hurst exponent (rescaled-range R/S method)
// ─────────────────────────────────────────────────────────────

/// Simple OLS: `y = a·x + b`. Returns `(slope, intercept)`.
fn linear_regression(x: &[f64], y: &[f64]) -> (f64, f64) {
    let n = x.len() as f64;
    let sum_x: f64 = x.iter().sum();
    let sum_y: f64 = y.iter().sum();
    let sum_xx: f64 = x.iter().map(|&xi| xi * xi).sum();
    let sum_xy: f64 = x.iter().zip(y).map(|(&xi, &yi)| xi * yi).sum();

    let denom = n * sum_xx - sum_x * sum_x;
    if denom.abs() < 1e-15 {
        return (0.0, sum_y / n);
    }
    let slope = (n * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / n;
    (slope, intercept)
}

/// Estimate the Hurst exponent via the rescaled-range (R/S) method.
///
/// * H < 0.5 → mean-reverting (anti-persistent)
/// * H = 0.5 → random walk
/// * H > 0.5 → trending (persistent)
///
/// Returns dict with `"hurst_exponent"`, `"confidence_interval"`,
/// `"is_mean_reverting"`, `"interpretation"`, plus diagnostic arrays.
#[cfg(feature = "python")]
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

    // Logarithmically-spaced window sizes
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

        let ts_slice = ts.as_slice().unwrap();
        let mut rs_for_size = Vec::new();

        for w in 0..num_windows_in_series {
            let start = w * window_size;
            let end = (start + window_size).min(n);
            let window = &ts_slice[start..end];

            let mean = window.iter().sum::<f64>() / window.len() as f64;

            // Cumulative deviations
            let mut cumdev = vec![0.0];
            let mut sum = 0.0;
            for &x in window {
                sum += x - mean;
                cumdev.push(sum);
            }

            let max_y = cumdev.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_y = cumdev.iter().cloned().fold(f64::INFINITY, f64::min);
            let range = max_y - min_y;

            let variance =
                window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;
            let std = variance.sqrt();

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

    let (hurst, intercept) = linear_regression(&log_n, &log_rs);

    // 95 % confidence interval via residuals
    let residuals: Vec<f64> = log_n
        .iter()
        .zip(&log_rs)
        .map(|(&ln_n, &ln_rs)| ln_rs - (hurst * ln_n + intercept))
        .collect();

    let n_pts = residuals.len();
    let std_error = if n_pts > 2 {
        (residuals.iter().map(|&r| r * r).sum::<f64>() / (n_pts - 2) as f64).sqrt()
    } else {
        0.0
    };
    let t_value = 1.96;
    let ss_x = log_n.iter().map(|&x| x * x).sum::<f64>()
        - log_n.iter().sum::<f64>().powi(2) / log_n.len() as f64;
    let se_slope = if ss_x > 1e-15 {
        std_error / ss_x.sqrt()
    } else {
        0.0
    };

    let confidence_lower = hurst - t_value * se_slope;
    let confidence_upper = hurst + t_value * se_slope;
    let is_mean_reverting = confidence_upper < 0.5;

    let dict = PyDict::new_bound(py);
    dict.set_item("hurst_exponent", hurst)?;
    dict.set_item(
        "confidence_interval",
        vec![confidence_lower, confidence_upper].into_py(py),
    )?;
    dict.set_item("is_mean_reverting", is_mean_reverting)?;
    dict.set_item(
        "interpretation",
        if hurst < 0.5 {
            "Mean-reverting (anti-persistent)"
        } else if hurst > 0.5 {
            "Trending (persistent)"
        } else {
            "Random walk"
        },
    )?;
    dict.set_item("window_sizes", window_sizes.into_py(py))?;
    dict.set_item("rs_values", rs_values.into_py(py))?;
    dict.set_item("standard_error", se_slope)?;

    Ok(dict.into())
}
