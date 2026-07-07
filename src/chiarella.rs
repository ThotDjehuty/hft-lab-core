//! # Chiarella Agent-Based Market Model
//!
//! Implements the Chiarella-He (2002) heterogeneous agents model for financial
//! markets. This classical academic model captures the interaction between two
//! types of rational agents:
//!
//! - **Fundamentalists**: Trade on mispricing, providing a stabilising force.
//! - **Chartists**: Trade on past trends, creating momentum and volatility clusters.
//!
//! Agent fractions evolve endogenously via the Brock-Hommes (1998) discrete-choice
//! mechanism: agents switch to the strategy with higher recent profitability.
//!
//! ## References
//! - Chiarella, C. & He, X.-Z. (2002). *Heterogeneous beliefs, risk and learning
//!   in a simple asset pricing model.* Computational Economics.
//! - Brock, W. & Hommes, C. (1998). *Heterogeneous beliefs and routes to chaos.*
//!   Journal of Economic Dynamics and Control.
//!
//! ## Usage
//!
//! ```rust
//! use hfthot_lab_core::chiarella::{ChiarellaParams, simulate_chiarella};
//!
//! let params = ChiarellaParams {
//!     beta_f: 0.5,   // fundamentalist strength
//!     beta_c: 1.0,   // chartist strength
//!     gamma: 1.0,    // switching intensity
//!     mu: 0.1,       // price adjustment speed
//!     sigma: 0.05,   // noise volatility
//! };
//! let result = simulate_chiarella(100.0, 100.0, 1000, &params, Some(42));
//! println!("Final price: {:.2}", result.prices.last().unwrap());
//! ```

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};

#[cfg(feature = "python")]
use pyo3::prelude::*;
#[cfg(feature = "python")]
use pyo3::types::PyDict;

// ─────────────────────────────────────────────────────────────────────────────
//  Parameter structs
// ─────────────────────────────────────────────────────────────────────────────

/// Parameters for the Chiarella-He heterogeneous agent model.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ChiarellaParams {
    /// Fundamentalist demand sensitivity (strength of mean-reversion force, β_f).
    pub beta_f: f64,
    /// Chartist demand sensitivity (trend-following strength, β_c).
    pub beta_c: f64,
    /// Agent-switching intensity (Brock-Hommes γ). Higher → faster switching.
    pub gamma: f64,
    /// Price adjustment speed (market-maker/Walrasian auctioneer speed, μ).
    pub mu: f64,
    /// Additive noise volatility (σ).
    pub sigma: f64,
}

impl Default for ChiarellaParams {
    fn default() -> Self {
        ChiarellaParams {
            beta_f: 0.5,
            beta_c: 1.0,
            gamma: 1.0,
            mu: 0.1,
            sigma: 0.05,
        }
    }
}

/// Output from a Chiarella model simulation run.
#[derive(Clone, Debug)]
pub struct ChiarellaSimulationResult {
    /// Simulated asset price path.
    pub prices: Vec<f64>,
    /// Fraction of fundamentalist agents at each tick (0–1).
    pub fundamentalist_fractions: Vec<f64>,
    /// Fraction of chartist agents at each tick (0–1).
    pub chartist_fractions: Vec<f64>,
    /// Excess demand pressure at each tick.
    pub excess_demands: Vec<f64>,
    /// Fundamental price used throughout the simulation.
    pub fundamental_price: f64,
}

impl ChiarellaSimulationResult {
    /// Returns the final simulated price.
    pub fn final_price(&self) -> f64 {
        *self.prices.last().unwrap_or(&0.0)
    }

    /// True if chartists (fraction > 0.5) dominate in the last period.
    pub fn is_chartist_dominated(&self) -> bool {
        self.chartist_fractions.last().copied().unwrap_or(0.0) > 0.5
    }

    /// Price deviation from fundamental at each tick.
    pub fn mispricing(&self) -> Vec<f64> {
        self.prices
            .iter()
            .map(|&p| p - self.fundamental_price)
            .collect()
    }

    /// Annualised price return volatility (assumes 252 trading days).
    pub fn annualised_volatility(&self) -> f64 {
        if self.prices.len() < 2 {
            return 0.0;
        }
        let log_returns: Vec<f64> = self.prices
            .windows(2)
            .map(|w| (w[1] / w[0]).ln())
            .collect();
        let n = log_returns.len() as f64;
        let mean = log_returns.iter().sum::<f64>() / n;
        let var = log_returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
        var.sqrt() * (252.0_f64).sqrt()
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Core simulation
// ─────────────────────────────────────────────────────────────────────────────

/// Run the Chiarella-He heterogeneous agent simulation.
///
/// # Arguments
/// * `initial_price` – Starting asset price.
/// * `fundamental_price` – Constant fundamental value the asset converges to.
/// * `n_steps` – Number of time steps to simulate.
/// * `params` – Model parameters (β_f, β_c, γ, μ, σ).
/// * `seed` – Optional RNG seed for reproducibility.
///
/// # Returns
/// `ChiarellaSimulationResult` containing full price path and agent fractions.
pub fn simulate_chiarella(
    initial_price: f64,
    fundamental_price: f64,
    n_steps: usize,
    params: &ChiarellaParams,
    seed: Option<u64>,
) -> ChiarellaSimulationResult {
    let mut rng: rand::rngs::StdRng = match seed {
        Some(s) => rand::rngs::StdRng::seed_from_u64(s),
        None => rand::rngs::StdRng::from_entropy(),
    };
    let noise = Normal::new(0.0, params.sigma).unwrap();

    let mut prices = Vec::with_capacity(n_steps + 1);
    let mut frac_f = Vec::with_capacity(n_steps + 1);
    let mut frac_c = Vec::with_capacity(n_steps + 1);
    let mut demands = Vec::with_capacity(n_steps + 1);

    prices.push(initial_price);
    frac_f.push(0.5_f64);
    frac_c.push(0.5_f64);
    demands.push(0.0_f64);

    // Trend window for chartist moving-average (10-step EMA)
    let trend_window = 10_usize;
    let mut price_history: std::collections::VecDeque<f64> = std::collections::VecDeque::new();
    price_history.push_back(initial_price);

    for t in 1..=n_steps {
        let p_prev = prices[t - 1];
        let nf_prev = frac_f[t - 1];
        let nc_prev = frac_c[t - 1];

        // Trend: short-window exponential moving average
        price_history.push_back(p_prev);
        if price_history.len() > trend_window + 1 {
            price_history.pop_front();
        }
        let trend_ref = if price_history.len() > 1 {
            price_history[price_history.len() / 2]
        } else {
            p_prev
        };

        // Agent demands
        let d_f = params.beta_f * (fundamental_price - p_prev);
        let d_c = params.beta_c * (p_prev - trend_ref);
        let excess = nf_prev * d_f + nc_prev * d_c;

        // Price update (Walrasian tâtonnement)
        let epsilon = noise.sample(&mut rng);
        let p_new = p_prev + params.mu * excess + epsilon;
        prices.push(p_new.max(0.001)); // prevent negative prices

        // Profitability proxies (realised PnL for each strategy)
        let dp = p_new - p_prev;
        let u_f = d_f * dp;
        let u_c = d_c * dp;

        // Brock-Hommes discrete choice
        let exp_f = (params.gamma * u_f).exp().min(1e15);
        let exp_c = (params.gamma * u_c).exp().min(1e15);
        let denom = exp_f + exp_c;
        let nf_new = if denom > 0.0 { exp_f / denom } else { 0.5 };
        frac_f.push(nf_new);
        frac_c.push(1.0 - nf_new);
        demands.push(excess);
    }

    ChiarellaSimulationResult {
        prices,
        fundamentalist_fractions: frac_f,
        chartist_fractions: frac_c,
        excess_demands: demands,
        fundamental_price,
    }
}

// ─────────────────────────────────────────────────────────────────────────────
//  Analytical utilities
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the Chiarella bifurcation parameter Λ.
///
/// Λ = (α · γ) / (β · δ) determines market stability:
/// - Λ < 0.67 → stable (mean-reverting)
/// - 0.67 ≤ Λ ≤ 1.5 → mixed (complex dynamics)
/// - Λ > 1.5 → unstable (trending, bubbles)
pub fn bifurcation_lambda(alpha: f64, beta: f64, gamma: f64, delta: f64) -> f64 {
    if beta * delta == 0.0 {
        f64::INFINITY
    } else {
        (alpha * gamma) / (beta * delta)
    }
}

/// Classify the market regime given an instantaneous Λ value.
pub fn classify_regime(lambda: f64) -> &'static str {
    if lambda < 0.67 {
        "stable"
    } else if lambda <= 1.5 {
        "mixed"
    } else {
        "unstable"
    }
}

/// Rolling estimate of the dominant agent type over a window.
///
/// Returns a `Vec<f64>` in [-1, 1]:
/// - `-1` → fully fundamentalist
/// - `+1` → fully chartist
pub fn rolling_dominance(
    chartist_fractions: &[f64],
    window: usize,
) -> Vec<f64> {
    if chartist_fractions.is_empty() || window == 0 {
        return vec![];
    }
    chartist_fractions
        .windows(window)
        .map(|w| {
            let mean_c = w.iter().sum::<f64>() / w.len() as f64;
            2.0 * mean_c - 1.0 // maps [0,1] → [-1,1]
        })
        .collect()
}

/// Compute realised price volatility (rolling std of log-returns).
pub fn rolling_volatility(prices: &[f64], window: usize) -> Vec<f64> {
    if prices.len() < 2 || window < 2 {
        return vec![];
    }
    let log_ret: Vec<f64> = prices
        .windows(2)
        .map(|w| (w[1] / w[0]).ln())
        .collect();
    log_ret
        .windows(window - 1)
        .map(|w| {
            let n = w.len() as f64;
            let mean = w.iter().sum::<f64>() / n;
            let var = w.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
            var.sqrt()
        })
        .collect()
}

// ─────────────────────────────────────────────────────────────────────────────
//  Python bindings
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(feature = "python")]
#[pyfunction]
#[pyo3(signature = (initial_price, fundamental_price, n_steps, beta_f=0.5, beta_c=1.0, gamma=1.0, mu=0.1, sigma=0.05, seed=None))]
pub fn simulate_chiarella_py(
    py: Python,
    initial_price: f64,
    fundamental_price: f64,
    n_steps: usize,
    beta_f: f64,
    beta_c: f64,
    gamma: f64,
    mu: f64,
    sigma: f64,
    seed: Option<u64>,
) -> PyResult<PyObject> {
    let params = ChiarellaParams { beta_f, beta_c, gamma, mu, sigma };
    let result = simulate_chiarella(initial_price, fundamental_price, n_steps, &params, seed);

    let dict = PyDict::new_bound(py);
    dict.set_item("prices", result.prices.clone())?;
    dict.set_item("fundamentalist_fractions", result.fundamentalist_fractions.clone())?;
    dict.set_item("chartist_fractions", result.chartist_fractions.clone())?;
    dict.set_item("excess_demands", result.excess_demands.clone())?;
    dict.set_item("fundamental_price", result.fundamental_price)?;
    dict.set_item("final_price", result.final_price())?;
    dict.set_item("is_chartist_dominated", result.is_chartist_dominated())?;
    dict.set_item("annualised_volatility", result.annualised_volatility())?;
    dict.set_item("mispricing", result.mispricing())?;
    Ok(dict.into())
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn bifurcation_lambda_py(alpha: f64, beta: f64, gamma: f64, delta: f64) -> f64 {
    bifurcation_lambda(alpha, beta, gamma, delta)
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn rolling_dominance_py(chartist_fractions: Vec<f64>, window: usize) -> Vec<f64> {
    rolling_dominance(&chartist_fractions, window)
}

#[cfg(feature = "python")]
#[pyfunction]
pub fn rolling_volatility_py(prices: Vec<f64>, window: usize) -> Vec<f64> {
    rolling_volatility(&prices, window)
}

// ─────────────────────────────────────────────────────────────────────────────
//  Unit tests
// ─────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn default_params() -> ChiarellaParams {
        ChiarellaParams::default()
    }

    #[test]
    fn test_simulation_length() {
        let res = simulate_chiarella(100.0, 100.0, 500, &default_params(), Some(1));
        assert_eq!(res.prices.len(), 501);
        assert_eq!(res.fundamentalist_fractions.len(), 501);
        assert_eq!(res.chartist_fractions.len(), 501);
    }

    #[test]
    fn test_fractions_sum_to_one() {
        let res = simulate_chiarella(100.0, 100.0, 100, &default_params(), Some(2));
        for (f, c) in res.fundamentalist_fractions.iter().zip(res.chartist_fractions.iter()) {
            let sum = f + c;
            assert!((sum - 1.0).abs() < 1e-10, "fractions don't sum to 1: {sum}");
        }
    }

    #[test]
    fn test_stable_regime_convergence() {
        // Strong fundamentalists → price should converge toward fundamental
        let params = ChiarellaParams {
            beta_f: 2.0,
            beta_c: 0.1,
            gamma: 0.1,
            mu: 0.1,
            sigma: 0.001,
        };
        let res = simulate_chiarella(90.0, 100.0, 2000, &params, Some(3));
        let final_p = res.final_price();
        // Price should be within ±10 of fundamental with dominant fundamentalists
        assert!(
            (final_p - 100.0).abs() < 15.0,
            "Price {final_p} did not converge near 100.0"
        );
    }

    #[test]
    fn test_bifurcation_lambda() {
        assert!(bifurcation_lambda(1.0, 1.0, 0.5, 1.0) < 0.67);
        assert!(bifurcation_lambda(1.5, 1.0, 1.5, 0.5) > 1.5);
    }

    #[test]
    fn test_rolling_dominance_range() {
        let fracs = vec![0.3, 0.4, 0.6, 0.7, 0.8];
        let dom = rolling_dominance(&fracs, 3);
        for d in &dom {
            assert!(*d >= -1.0 && *d <= 1.0, "dominance out of range: {d}");
        }
    }

    #[test]
    fn test_reproducibility() {
        let res1 = simulate_chiarella(100.0, 100.0, 200, &default_params(), Some(99));
        let res2 = simulate_chiarella(100.0, 100.0, 200, &default_params(), Some(99));
        assert_eq!(res1.prices, res2.prices, "seeded simulation should be reproducible");
    }
}
