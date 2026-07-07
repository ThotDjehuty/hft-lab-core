/// Mean-reversion portfolio discovery and analysis functions
/// Exposed to Python via PyO3 for performance-critical operations.

#[cfg(feature = "python")]
use nalgebra::{DMatrix, DVector, SVD};
#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

/// Compute PCA on returns matrix using nalgebra SVD
/// Returns dict with "components" and "explained_variance"
#[cfg(feature = "python")]
#[pyfunction]
pub fn compute_pca_rust(
    py: Python,
    prices: Vec<Vec<f64>>,
    n_components: usize,
) -> PyResult<PyObject> {
    if prices.is_empty() || prices[0].is_empty() {
        let result = PyDict::new_bound(py);
        result.set_item("components", Vec::<Vec<f64>>::new())?;
        result.set_item("explained_variance", Vec::<f64>::new())?;
        return Ok(result.into());
    }

    let n_samples = prices.len();
    let n_features = prices[0].len();
    
    // Convert to nalgebra matrix
    let mut data = Vec::with_capacity(n_samples * n_features);
    for row in &prices {
        data.extend_from_slice(row);
    }
    let matrix = DMatrix::from_row_slice(n_samples, n_features, &data);
    
    // Center the data (subtract mean)
    let means = matrix.column_mean();
    let mut centered = matrix.clone();
    for i in 0..n_samples {
        for j in 0..n_features {
            centered[(i, j)] -= means[j];
        }
    }
    
    // Compute covariance matrix: C = (1/n) * X^T * X
    let cov = (&centered.transpose() * &centered) / (n_samples as f64);
    
    // SVD decomposition
    let svd = SVD::new(cov, true, true);
    
    let singular_values = svd.singular_values;
    let v_matrix = svd.v_t.unwrap().transpose();
    
    // Extract top n_components
    let n_comp = n_components.min(n_features);
    
    // Components (eigenvectors)
    let mut components = Vec::with_capacity(n_comp);
    for i in 0..n_comp {
        let col = v_matrix.column(i);
        components.push(col.iter().copied().collect::<Vec<f64>>());
    }
    
    // Explained variance (eigenvalues)
    let total_variance: f64 = singular_values.iter().sum();
    let explained_variance: Vec<f64> = singular_values
        .iter()
        .take(n_comp)
        .map(|&v| v / total_variance)
        .collect();
    
    // Return as Python dict
    let result = PyDict::new_bound(py);
    result.set_item("components", components)?;
    result.set_item("explained_variance", explained_variance)?;
    
    Ok(result.into())
}

/// Estimate Ornstein-Uhlenbeck process parameters
/// Returns dict with "theta", "mu", "sigma", "half_life"
#[cfg(feature = "python")]
#[pyfunction]
pub fn estimate_ou_process_rust(py: Python, prices: Vec<f64>) -> PyResult<PyObject> {
    if prices.len() < 3 {
        let result = PyDict::new_bound(py);
        result.set_item("theta", 0.0)?;
        result.set_item("mu", 0.0)?;
        result.set_item("sigma", 0.0)?;
        result.set_item("half_life", f64::INFINITY)?;
        return Ok(result.into());
    }
    
    let n = prices.len();
    
    // Compute differences: dX = X[t] - X[t-1]
    let mut diffs = Vec::with_capacity(n - 1);
    let mut lags = Vec::with_capacity(n - 1);
    
    for i in 1..n {
        diffs.push(prices[i] - prices[i-1]);
        lags.push(prices[i-1]);
    }
    
    // Estimate mu as mean of prices
    let mu = prices.iter().sum::<f64>() / (n as f64);
    
    // Regression: dX = a + b * X[t-1] + epsilon
    // Where b approximates -theta
    let mean_lag = lags.iter().sum::<f64>() / lags.len() as f64;
    let mean_diff = diffs.iter().sum::<f64>() / diffs.len() as f64;
    
    let mut num = 0.0;
    let mut den = 0.0;
    
    for i in 0..lags.len() {
        let lag_dev = lags[i] - mean_lag;
        let diff_dev = diffs[i] - mean_diff;
        num += lag_dev * diff_dev;
        den += lag_dev * lag_dev;
    }
    
    let slope = if den.abs() > 1e-10 { num / den } else { 0.0 };
    let theta = -slope;
    
    // Estimate sigma from residuals
    let intercept = mean_diff - slope * mean_lag;
    let mut residuals_sq = 0.0;
    for i in 0..lags.len() {
        let predicted = intercept + slope * lags[i];
        let residual = diffs[i] - predicted;
        residuals_sq += residual * residual;
    }
    
    let sigma = (residuals_sq / (lags.len() as f64)).sqrt();
    
    // Half-life: ln(2) / theta
    let half_life = if theta > 1e-10 {
        (2.0_f64).ln() / theta
    } else {
        f64::INFINITY
    };
    
    let result = PyDict::new_bound(py);
    result.set_item("theta", theta)?;
    result.set_item("mu", mu)?;
    result.set_item("sigma", sigma)?;
    result.set_item("half_life", half_life)?;
    
    Ok(result.into())
}

/// Cointegration test between two price series
/// Returns dict with "statistic", "p_value", "is_cointegrated"
#[cfg(feature = "python")]
#[pyfunction]
pub fn cointegration_test_rust(
    py: Python,
    prices1: Vec<f64>,
    prices2: Vec<f64>,
) -> PyResult<PyObject> {
    if prices1.len() != prices2.len() || prices1.len() < 3 {
        let result = PyDict::new_bound(py);
        result.set_item("statistic", 0.0)?;
        result.set_item("p_value", 1.0)?;
        result.set_item("is_cointegrated", false)?;
        return Ok(result.into());
    }
    
    let n = prices1.len();
    
    // Compute linear regression: prices2 = a + b * prices1
    let mean1 = prices1.iter().sum::<f64>() / n as f64;
    let mean2 = prices2.iter().sum::<f64>() / n as f64;
    
    let mut num = 0.0;
    let mut den = 0.0;
    
    for i in 0..n {
        let dev1 = prices1[i] - mean1;
        let dev2 = prices2[i] - mean2;
        num += dev1 * dev2;
        den += dev1 * dev1;
    }
    
    let beta = if den.abs() > 1e-10 { num / den } else { 1.0 };
    let alpha = mean2 - beta * mean1;
    
    // Compute spread: spread = prices2 - (alpha + beta * prices1)
    let spread: Vec<f64> = prices2
        .iter()
        .enumerate()
        .map(|(i, &p2)| p2 - (alpha + beta * prices1[i]))
        .collect();
    
    // Augmented Dickey-Fuller test on spread
    // Simplified: test if spread is mean-reverting
    let spread_mean = spread.iter().sum::<f64>() / spread.len() as f64;
    let mut spread_var = 0.0;
    for &s in &spread {
        spread_var += (s - spread_mean).powi(2);
    }
    spread_var /= spread.len() as f64;
    
    // Compute first-order autocorrelation
    let mut auto_cov = 0.0;
    for i in 1..spread.len() {
        auto_cov += (spread[i] - spread_mean) * (spread[i-1] - spread_mean);
    }
    auto_cov /= (spread.len() - 1) as f64;
    
    let rho = auto_cov / spread_var;
    
    // Test statistic (simplified ADF)
    let adf_stat = (rho - 1.0) * ((spread.len() as f64).sqrt());
    
    // Simplified p-value estimation (approximation)
    let p_value = if adf_stat < -3.5 {
        0.01
    } else if adf_stat < -2.9 {
        0.05
    } else if adf_stat < -2.6 {
        0.10
    } else {
        0.20
    };
    
    let is_cointegrated = p_value < 0.05;
    
    let result = PyDict::new_bound(py);
    result.set_item("statistic", adf_stat)?;
    result.set_item("p_value", p_value)?;
    result.set_item("is_cointegrated", is_cointegrated)?;
    
    Ok(result.into())
}

/// Backtest mean-reversion strategy
/// Returns dict with "returns", "positions", "pnl", "sharpe", "max_drawdown"
#[cfg(feature = "python")]
#[pyfunction]
pub fn backtest_strategy_rust(
    py: Python,
    prices: Vec<f64>,
    entry_z: f64,
    exit_z: f64,
) -> PyResult<PyObject> {
    if prices.len() < 3 {
        let result = PyDict::new_bound(py);
        result.set_item("returns", Vec::<f64>::new())?;
        result.set_item("positions", Vec::<i32>::new())?;
        result.set_item("pnl", Vec::<f64>::new())?;
        result.set_item("sharpe", 0.0)?;
        result.set_item("max_drawdown", 0.0)?;
        return Ok(result.into());
    }
    
    let n = prices.len();
    
    // Compute rolling mean and std
    let window = 20.min(n / 4);
    let mut positions = vec![0; n];
    let mut pnl = vec![0.0; n];
    let mut returns = vec![0.0; n];
    
    let mut current_position = 0;
    let cash = 100000.0;
    let mut portfolio_value = cash;
    let mut peak_value = cash;
    let mut max_dd = 0.0;
    
    for i in window..n {
        // Compute rolling statistics
        let window_prices = &prices[i-window..i];
        let mean = window_prices.iter().sum::<f64>() / window as f64;
        let variance = window_prices
            .iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / window as f64;
        let std = variance.sqrt();
        
        if std < 1e-10 {
            positions[i] = current_position;
            continue;
        }
        
        // Compute z-score
        let z_score = (prices[i] - mean) / std;
        
        // Trading logic
        if z_score < -entry_z && current_position == 0 {
            // Buy signal
            current_position = 1;
        } else if z_score > entry_z && current_position == 0 {
            // Short signal
            current_position = -1;
        } else if z_score.abs() < exit_z && current_position != 0 {
            // Exit signal
            current_position = 0;
        }
        
        positions[i] = current_position;
        
        // Compute returns
        if i > 0 {
            let price_return = (prices[i] - prices[i-1]) / prices[i-1];
            returns[i] = price_return * current_position as f64;
            
            // Update portfolio value
            portfolio_value *= 1.0 + returns[i];
            pnl[i] = portfolio_value - cash;
            
            // Track drawdown
            if portfolio_value > peak_value {
                peak_value = portfolio_value;
            }
            let drawdown = (peak_value - portfolio_value) / peak_value;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }
    }
    
    // Compute Sharpe ratio
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let return_variance = returns
        .iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>() / returns.len() as f64;
    let return_std = return_variance.sqrt();
    
    let sharpe = if return_std > 1e-10 {
        (mean_return / return_std) * (252.0_f64).sqrt() // Annualized
    } else {
        0.0
    };
    
    let result = PyDict::new_bound(py);
    result.set_item("returns", returns)?;
    result.set_item("positions", positions)?;
    result.set_item("pnl", pnl)?;
    result.set_item("sharpe", sharpe)?;
    result.set_item("max_drawdown", max_dd)?;
    
    Ok(result.into())
}

/// Compute optimal portfolio weights using CARA utility maximization (Appendix A)
/// gamma: risk aversion parameter (higher = more risk averse)
/// Returns dict with "weights", "expected_return", "expected_variance"
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
    
    // Convert to nalgebra
    let mu = DVector::from_vec(expected_returns.clone());
    let mut cov_data = Vec::with_capacity(n * n);
    for row in &covariance_matrix {
        cov_data.extend_from_slice(row);
    }
    let sigma = DMatrix::from_row_slice(n, n, &cov_data);
    
    // CARA optimal weights: w* = (1/gamma) * Sigma^{-1} * mu
    // Handle singular matrix by adding small regularization
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
        None => {
            // Fallback: equal weights
            vec![1.0 / n as f64; n]
        }
    };
    
    // Compute expected return and variance
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

/// Compute risk-adjusted portfolio weights using Sharpe ratio maximization
/// target_return: optional target return (if None, maximize Sharpe)
/// Returns dict with "weights", "sharpe_ratio", "expected_return", "expected_std"
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
    
    // Convert to nalgebra
    let mu = DVector::from_vec(expected_returns.clone());
    let mut cov_data = Vec::with_capacity(n * n);
    for row in &covariance_matrix {
        cov_data.extend_from_slice(row);
    }
    let sigma = DMatrix::from_row_slice(n, n, &cov_data);
    
    // Excess returns
    let excess_returns: Vec<f64> = expected_returns.iter()
        .map(|&r| r - risk_free_rate)
        .collect();
    let mu_excess = DVector::from_vec(excess_returns);
    
    // Sharpe optimal weights: w* = Sigma^{-1} * (mu - rf)
    // Normalized to sum to 1
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
        None => {
            vec![1.0 / n as f64; n]
        }
    };
    
    // Compute metrics
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

/// Backtest with transaction costs
/// transaction_cost: proportional cost per trade (e.g., 0.001 = 0.1%)
/// Returns dict with "returns", "positions", "pnl", "sharpe", "max_drawdown", "total_costs"
#[cfg(feature = "python")]
#[pyfunction]
pub fn backtest_with_costs_rust(
    py: Python,
    prices: Vec<f64>,
    entry_z: f64,
    exit_z: f64,
    transaction_cost: f64,
) -> PyResult<PyObject> {
    if prices.len() < 3 {
        let result = PyDict::new_bound(py);
        result.set_item("returns", Vec::<f64>::new())?;
        result.set_item("positions", Vec::<i32>::new())?;
        result.set_item("pnl", Vec::<f64>::new())?;
        result.set_item("sharpe", 0.0)?;
        result.set_item("max_drawdown", 0.0)?;
        result.set_item("total_costs", 0.0)?;
        return Ok(result.into());
    }
    
    let n = prices.len();
    let window = 20.min(n / 4);
    let mut positions = vec![0; n];
    let mut pnl = vec![0.0; n];
    let mut returns = vec![0.0; n];
    
    let mut current_position = 0;
    let cash = 100000.0;
    let mut portfolio_value = cash;
    let mut peak_value = cash;
    let mut max_dd = 0.0;
    let mut total_costs = 0.0;
    
    for i in window..n {
        // Compute rolling statistics
        let window_prices = &prices[i-window..i];
        let mean = window_prices.iter().sum::<f64>() / window as f64;
        let variance = window_prices
            .iter()
            .map(|&p| (p - mean).powi(2))
            .sum::<f64>() / window as f64;
        let std = variance.sqrt();
        
        if std < 1e-10 {
            positions[i] = current_position;
            continue;
        }
        
        let z_score = (prices[i] - mean) / std;
        let prev_position = current_position;
        
        // Trading logic
        if z_score < -entry_z && current_position == 0 {
            current_position = 1;
        } else if z_score > entry_z && current_position == 0 {
            current_position = -1;
        } else if z_score.abs() < exit_z && current_position != 0 {
            current_position = 0;
        }
        
        positions[i] = current_position;
        
        // Compute returns with transaction costs
        if i > 0 {
            let price_return = (prices[i] - prices[i-1]) / prices[i-1];
            
            // Apply transaction cost if position changed
            let mut cost = 0.0;
            if prev_position != current_position {
                // Cost proportional to trade size
                let position_change = ((prev_position - current_position) as f64).abs();
                cost = transaction_cost * prices[i] * position_change;
                total_costs += cost;
            }
            
            returns[i] = price_return * prev_position as f64 - (cost / portfolio_value);
            portfolio_value *= 1.0 + returns[i];
            pnl[i] = portfolio_value - cash;
            
            if portfolio_value > peak_value {
                peak_value = portfolio_value;
            }
            let drawdown = (peak_value - portfolio_value) / peak_value;
            if drawdown > max_dd {
                max_dd = drawdown;
            }
        }
    }
    
    // Compute Sharpe ratio
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let return_variance = returns
        .iter()
        .map(|&r| (r - mean_return).powi(2))
        .sum::<f64>() / returns.len() as f64;
    let return_std = return_variance.sqrt();
    
    let sharpe = if return_std > 1e-10 {
        (mean_return / return_std) * (252.0_f64).sqrt()
    } else {
        0.0
    };
    
    let result = PyDict::new_bound(py);
    result.set_item("returns", returns)?;
    result.set_item("positions", positions)?;
    result.set_item("pnl", pnl)?;
    result.set_item("sharpe", sharpe)?;
    result.set_item("max_drawdown", max_dd)?;
    result.set_item("total_costs", total_costs)?;
    
    Ok(result.into())
}

/// Compute optimal entry/exit thresholds based on OU process parameters
/// Uses expected return and half-life to determine optimal stopping times
/// Returns dict with "optimal_entry", "optimal_exit", "expected_holding_period"
#[cfg(feature = "python")]
#[pyfunction]
pub fn optimal_thresholds_rust(
    py: Python,
    theta: f64,
    _mu: f64,
    sigma: f64,
    transaction_cost: f64,
) -> PyResult<PyObject> {
    if theta <= 0.0 || sigma <= 0.0 {
        let result = PyDict::new_bound(py);
        result.set_item("optimal_entry", 2.0)?;
        result.set_item("optimal_exit", 0.5)?;
        result.set_item("expected_holding_period", 10.0)?;
        return Ok(result.into());
    }
    
    // Half-life of mean reversion
    let half_life = (2.0_f64).ln() / theta;
    
    // Optimal entry: balance signal strength vs transaction costs
    // Higher costs => wait for stronger signals
    let cost_adjustment = (1.0 + 100.0 * transaction_cost).sqrt();
    let optimal_entry = 1.5 * cost_adjustment;
    
    // Optimal exit: exit when profit potential diminishes
    // Based on expected time to mean revert
    let optimal_exit = 0.3 * cost_adjustment.sqrt();
    
    // Expected holding period based on OU process
    // E[time to cross threshold] ≈ 1/(2*theta) for small threshold
    let expected_holding_period = half_life * 0.5;
    
    let result = PyDict::new_bound(py);
    result.set_item("optimal_entry", optimal_entry)?;
    result.set_item("optimal_exit", optimal_exit)?;
    result.set_item("expected_holding_period", expected_holding_period)?;
    
    Ok(result.into())
}

/// Multi-period portfolio optimization with dynamic rebalancing
/// Optimizes portfolio over multiple periods considering transaction costs
/// Returns dict with "weights_sequence", "rebalance_times", "expected_utility"
#[cfg(feature = "python")]
#[pyfunction]
pub fn multiperiod_optimize_rust(
    py: Python,
    returns_history: Vec<Vec<f64>>,  // T x N matrix
    covariances: Vec<Vec<f64>>,       // N x N matrix
    gamma: f64,
    transaction_cost: f64,
    n_periods: usize,
) -> PyResult<PyObject> {
    if returns_history.is_empty() || n_periods == 0 {
        let result = PyDict::new_bound(py);
        result.set_item("weights_sequence", Vec::<Vec<f64>>::new())?;
        result.set_item("rebalance_times", Vec::<usize>::new())?;
        result.set_item("expected_utility", 0.0)?;
        return Ok(result.into());
    }
    
    let n_assets = returns_history[0].len();
    let t_total = returns_history.len();
    
    // Simple dynamic programming approach
    // Divide time into n_periods and optimize each period
    let period_length = (t_total / n_periods).max(1);
    
    let mut weights_sequence = Vec::new();
    let mut rebalance_times = Vec::new();
    let mut expected_utility = 0.0;
    
    for p in 0..n_periods {
        let start_idx = p * period_length;
        let end_idx = ((p + 1) * period_length).min(t_total);
        
        if start_idx >= t_total {
            break;
        }
        
        // Compute expected returns for this period
        let period_returns = &returns_history[start_idx..end_idx];
        let mut avg_returns = vec![0.0; n_assets];
        
        for returns in period_returns {
            for (j, &r) in returns.iter().enumerate() {
                avg_returns[j] += r;
            }
        }
        
        let n_obs = period_returns.len() as f64;
        for r in &mut avg_returns {
            *r /= n_obs;
        }
        
        // Compute CARA optimal weights for this period
        let mu = DVector::from_vec(avg_returns.clone());
        let mut cov_data = Vec::with_capacity(n_assets * n_assets);
        for row in &covariances {
            cov_data.extend_from_slice(row);
        }
        let sigma = DMatrix::from_row_slice(n_assets, n_assets, &cov_data);
        
        let epsilon = 1e-8;
        let mut sigma_reg = sigma.clone();
        for i in 0..n_assets {
            sigma_reg[(i, i)] += epsilon;
        }
        
        let weights = match sigma_reg.try_inverse() {
            Some(sigma_inv) => {
                let w = sigma_inv * mu.clone();
                let scaled = w.scale(1.0 / gamma);
                
                // Adjust for transaction costs if not first period
                let mut w_vec = scaled.as_slice().to_vec();
                if p > 0 && !weights_sequence.is_empty() {
                    // Penalize large changes from previous weights
                    let prev_weights: &Vec<f64> = &weights_sequence[weights_sequence.len() - 1];
                    for (i, w) in w_vec.iter_mut().enumerate() {
                        let change = (*w - prev_weights[i]).abs();
                        *w -= transaction_cost * change.signum() * change;
                    }
                }
                w_vec
            }
            None => {
                vec![1.0 / n_assets as f64; n_assets]
            }
        };
        
        // Compute period utility (negative exponential)
        let w_vec = DVector::from_vec(weights.clone());
        let period_return = w_vec.dot(&mu);
        let sigma_w = &sigma * &w_vec;
        let period_variance = w_vec.dot(&sigma_w);
        
        // CARA utility: U = -exp(-gamma * (return - 0.5 * gamma * variance))
        let utility = (-gamma * (period_return - 0.5 * gamma * period_variance)).exp();
        expected_utility += utility;
        
        weights_sequence.push(weights);
        rebalance_times.push(start_idx);
    }
    
    expected_utility /= n_periods as f64;
    
    let result = PyDict::new_bound(py);
    result.set_item("weights_sequence", weights_sequence)?;
    result.set_item("rebalance_times", rebalance_times)?;
    result.set_item("expected_utility", expected_utility)?;
    
    Ok(result.into())
}
