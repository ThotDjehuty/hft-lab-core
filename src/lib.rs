//! HFT Lab Core — Professional-grade quantitative finance toolkit
//!
//! An open-source Rust library providing composable building blocks for
//! high-frequency trading research, statistical arbitrage, and
//! portfolio construction.
//!
//! # Architecture
//!
//! The crate is organised into three layers:
//!
//! ```text
//! ┌────────────────────────────────────────────────────────┐
//! │                    prelude (re-exports)                │
//! ├────────────────────────────────────────────────────────┤
//! │  functional/          │  pipeline/                     │
//! │  ├─ compose           │  ├─ Stage / Pipeline           │
//! │  └─ result_ext        │  └─ parallel (Rayon)           │
//! ├────────────────────────────────────────────────────────┤
//! │  Backtesting & analytics                               │
//! │  backtest · analytics                                  │
//! ├────────────────────────────────────────────────────────┤
//! │  Domain modules                                        │
//! │  lob · meanrev · sparse_meanrev · optimization        │
//! │  geometric_signals · chiarella · signature             │
//! └────────────────────────────────────────────────────────┘
//! ```
//!
//! ## Railway-oriented programming
//!
//! Every fallible function returns [`error::HftResult<T>`].  The
//! [`functional`] module provides composable `pipe` / `chain` combinators
//! and [`ResultExt`] helpers so errors propagate automatically without
//! manual `match` or `unwrap`.
//!
//! ## Parallelism
//!
//! Pure functions compose naturally with Rayon.  The [`pipeline::parallel`]
//! module offers `par_map`, `par_fold`, and `par_chunks` that distribute
//! work across all available cores with zero synchronisation overhead.
//!
//! # Modules
//!
//! | Module | Purpose |
//! |--------|---------|
//! | [`error`] | Unified error types (`HftError`, `HftResult`) |
//! | [`functional`] | `pipe`, `chain`, `retry`, `ResultExt` combinators |
//! | [`pipeline`] | Railway-oriented data-processing stages |
//! | [`prelude`] | Ergonomic one-line import |
//! | [`backtest`] | Signal-driven backtesting framework |
//! | [`analytics`] | Performance metrics (Sharpe, drawdown, …) |
//! | [`lob`] | Limit Order Book analytics |
//! | [`meanrev`] | Mean-reversion portfolio construction |
//! | [`sparse_meanrev`] | Sparse PCA, Box-Tiao, Hurst exponent |
//! | [`pricing`] | Rough Heston MC pricer (options, Greeks, vol smile) |
//! | [`portfolio`] | CARA/Sharpe optimal weights, Hurst exponent |
//! | [`optimization`] | HMM, MCMC, Differential Evolution |
//! | [`geometric_signals`] | Gauge, spectral, cohomology, Wasserstein |
//! | [`chiarella`] | Heterogeneous agent market simulation |
//! | [`signature`] | Path signatures (Chen iterated integrals) |
//!
//! # Python bindings
//!
//! When built with the `python` feature (default), all modules are accessible
//! from Python via PyO3:  `import hfthot_lab_core as hft`.

// ── Foundation layer ─────────────────────────────────────────────────────────
pub mod error;
pub mod functional;
pub mod pipeline;
pub mod prelude;

// ── Backtesting & analytics ─────────────────────────────────────────────────────────
pub mod backtest;
pub mod analytics;

// ── Domain modules ───────────────────────────────────────────────────────────
pub mod lob;
#[cfg(feature = "proprietary")]
pub mod meanrev;
pub mod optimization;
pub mod geometric_signals;
pub mod chiarella;
pub mod signature;

pub mod uniswap;
#[cfg(all(feature = "python", feature = "proprietary"))]
pub mod sparse_meanrev;
#[cfg(feature = "python")]
pub mod pricing;
pub mod portfolio;

// Re-export key types from lob
pub use lob::{OrderBookLevel, OrderBookSnapshot, OrderBookUpdate, LOBAnalytics};
pub use lob::{calculate_lob_analytics, apply_orderbook_update};
#[cfg(feature = "python")]
pub use lob::parse_binance_orderbook;

// Re-export error types
pub use error::{HftError, HftResult};


// Re-export backtesting
pub use backtest::{BacktestConfig, BacktestResult, Trade, run_backtest};

// Re-export functional combinators
pub use functional::compose::{pipe, compose, chain, identity};
pub use functional::result_ext::ResultExt;

// Re-export pipeline types
pub use pipeline::{Pipeline, Stage, chain_stages, chain_stages3};

// Re-export meanrev functions (proprietary, Python-only)
#[cfg(all(feature = "python", feature = "proprietary"))]
pub use meanrev::{
    compute_pca_rust,
    estimate_ou_process_rust,
    cointegration_test_rust,
    backtest_strategy_rust,
    cara_optimal_weights_rust,
    sharpe_optimal_weights_rust,
    backtest_with_costs_rust,
    optimal_thresholds_rust,
    multiperiod_optimize_rust,
};

// Re-export optimization
pub use optimization::HMMParams;
pub use optimization::{
    fit_hmm,
    viterbi_decode,
    mutual_information,
    shannon_entropy,
};
#[cfg(feature = "python")]
pub use optimization::{
    mcmc_sample,
    differential_evolution,
    grid_search,
};

// Re-export pricing (Rough Heston)
#[cfg(feature = "python")]
pub use pricing::{
    rh_mc_call, rh_mc_put, rh_greeks, rh_vol_smile,
    rh_simulate_paths, rh_atm_skew_mc, rh_variance_swap, rh_leverage_swap,
};

// Re-export portfolio (public academic algorithms)
#[cfg(feature = "python")]
pub use portfolio::{
    hurst_exponent_rust,
};

// Re-export chiarella
pub use chiarella::{
    ChiarellaParams, ChiarellaSimulationResult,
    simulate_chiarella, bifurcation_lambda, classify_regime,
    rolling_dominance, rolling_volatility,
};

// Re-export signature
pub use signature::{
    lead_lag_transform, path_signature_2d, log_signature_2d,
    rolling_signature_features, signature_distance, normalise_prices,
};

// Re-export geometric signals
pub use geometric_signals::{
    triangle_curvature, cycle_curvature, curvature_zscore, holonomy,
    graph_laplacian, normalized_laplacian, fiedler_value, fiedler_vector,
    spectral_entropy, heat_trace, laplacian_eigenvalues,
    coboundary_matrix, weighted_coboundary_matrix, sheaf_laplacian,
    detect_arbitrage, cohomology_dimension, hodge_decompose,
    mirror_descent_step, natural_gradient, bregman_kl,
    functionally_generated_portfolio, relative_entropy_rate,
    matrix_log_sym, matrix_exp_sym, log_euclidean_distance,
    frechet_mean_le, spd_geodesic, spd_distance_matrix,
    wasserstein_1d, wasserstein_2_1d, sinkhorn_distance,
    rolling_wasserstein, lob_wasserstein,
};

#[cfg(feature = "python")]
use pyo3::prelude::*;

// ── Python wrappers for analytics (thin PyO3 glue) ──────────────────────────
#[cfg(feature = "python")]
mod analytics_py {
    use pyo3::prelude::*;

    #[pyfunction]
    #[pyo3(name = "sharpe_ratio")]
    pub fn sharpe_ratio_py(returns: Vec<f64>, risk_free: f64) -> f64 {
        crate::analytics::sharpe_ratio(&returns, risk_free)
    }

    #[pyfunction]
    #[pyo3(name = "sortino_ratio")]
    pub fn sortino_ratio_py(returns: Vec<f64>, risk_free: f64) -> f64 {
        crate::analytics::sortino_ratio(&returns, risk_free)
    }

    #[pyfunction]
    #[pyo3(name = "max_drawdown")]
    pub fn max_drawdown_py(returns: Vec<f64>) -> f64 {
        crate::analytics::max_drawdown_from_returns(&returns)
    }

    #[pyfunction]
    #[pyo3(name = "win_rate")]
    pub fn win_rate_py(returns: Vec<f64>) -> f64 {
        crate::analytics::win_rate(&returns)
    }

    #[pyfunction]
    #[pyo3(name = "profit_factor")]
    pub fn profit_factor_py(returns: Vec<f64>) -> f64 {
        crate::analytics::profit_factor(&returns)
    }

    #[pyfunction]
    #[pyo3(name = "annualised_return")]
    pub fn annualised_return_py(returns: Vec<f64>) -> f64 {
        crate::analytics::annualised_return(&returns)
    }

    #[pyfunction]
    #[pyo3(name = "annualised_volatility")]
    pub fn annualised_volatility_py(returns: Vec<f64>) -> f64 {
        crate::analytics::annualised_volatility(&returns)
    }
}

/// Python module definition
#[cfg(feature = "python")]
#[pymodule]
fn hfthot_lab_core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    // LOB classes
    m.add_class::<lob::OrderBookLevel>()?;
    m.add_class::<lob::OrderBookSnapshot>()?;
    m.add_class::<lob::OrderBookUpdate>()?;
    m.add_class::<lob::LOBAnalytics>()?;
    
    // LOB functions
    m.add_function(wrap_pyfunction!(lob::calculate_lob_analytics, m)?)?;
    m.add_function(wrap_pyfunction!(lob::apply_orderbook_update, m)?)?;
    m.add_function(wrap_pyfunction!(lob::parse_binance_orderbook, m)?)?;
    
    // Mean reversion functions (proprietary — only available with `proprietary` feature)
    #[cfg(feature = "proprietary")]
    {
        m.add_function(wrap_pyfunction!(meanrev::compute_pca_rust, m)?)?;
        m.add_function(wrap_pyfunction!(meanrev::estimate_ou_process_rust, m)?)?;
        m.add_function(wrap_pyfunction!(meanrev::cointegration_test_rust, m)?)?;
        m.add_function(wrap_pyfunction!(meanrev::backtest_strategy_rust, m)?)?;
        m.add_function(wrap_pyfunction!(meanrev::cara_optimal_weights_rust, m)?)?;
        m.add_function(wrap_pyfunction!(meanrev::sharpe_optimal_weights_rust, m)?)?;
        m.add_function(wrap_pyfunction!(meanrev::backtest_with_costs_rust, m)?)?;
        m.add_function(wrap_pyfunction!(meanrev::optimal_thresholds_rust, m)?)?;
        m.add_function(wrap_pyfunction!(meanrev::multiperiod_optimize_rust, m)?)?;
    }
    
    // Sparse mean reversion functions (proprietary)
    #[cfg(feature = "proprietary")]
    {
        m.add_function(wrap_pyfunction!(sparse_meanrev::sparse_pca_rust, m)?)?;
        m.add_function(wrap_pyfunction!(sparse_meanrev::box_tao_decomposition_rust, m)?)?;
        m.add_function(wrap_pyfunction!(sparse_meanrev::hurst_exponent_rust, m)?)?;
        m.add_function(wrap_pyfunction!(sparse_meanrev::sparse_cointegration_rust, m)?)?;
    }
    
    // Optimization
    m.add_class::<optimization::HMMParams>()?;
    m.add_function(wrap_pyfunction!(optimization::fit_hmm, m)?)?;
    m.add_function(wrap_pyfunction!(optimization::viterbi_decode, m)?)?;
    m.add_function(wrap_pyfunction!(optimization::mcmc_sample, m)?)?;
    m.add_function(wrap_pyfunction!(optimization::mutual_information, m)?)?;
    m.add_function(wrap_pyfunction!(optimization::differential_evolution, m)?)?;
    m.add_function(wrap_pyfunction!(optimization::grid_search, m)?)?;
    m.add_function(wrap_pyfunction!(optimization::shannon_entropy, m)?)?;
    
    // Geometric signals - Gauge theory
    m.add_function(wrap_pyfunction!(geometric_signals::gauge::triangle_curvature, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::gauge::cycle_curvature, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::gauge::curvature_zscore, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::gauge::holonomy, m)?)?;
    
    // Geometric signals - Spectral topology
    m.add_function(wrap_pyfunction!(geometric_signals::spectral::graph_laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::spectral::normalized_laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::spectral::fiedler_value, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::spectral::fiedler_vector, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::spectral::spectral_entropy, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::spectral::heat_trace, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::spectral::laplacian_eigenvalues, m)?)?;
    
    // Geometric signals - Sheaf cohomology
    m.add_class::<geometric_signals::cohomology::HodgeComponents>()?;
    m.add_function(wrap_pyfunction!(geometric_signals::cohomology::coboundary_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::cohomology::weighted_coboundary_matrix, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::cohomology::sheaf_laplacian, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::cohomology::detect_arbitrage, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::cohomology::cohomology_dimension, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::cohomology::hodge_decompose, m)?)?;
    
    // Geometric signals - Information geometry
    m.add_function(wrap_pyfunction!(geometric_signals::information::mirror_descent_step, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::information::natural_gradient, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::information::bregman_kl, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::information::functionally_generated_portfolio, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::information::relative_entropy_rate, m)?)?;
    
    // Geometric signals - SPD manifold
    m.add_function(wrap_pyfunction!(geometric_signals::spd::matrix_log_sym, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::spd::matrix_exp_sym, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::spd::log_euclidean_distance, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::spd::frechet_mean_le, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::spd::spd_geodesic, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::spd::spd_distance_matrix, m)?)?;
    
    // Geometric signals - Wasserstein / Optimal transport
    m.add_function(wrap_pyfunction!(geometric_signals::wasserstein::wasserstein_1d, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::wasserstein::wasserstein_2_1d, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::wasserstein::sinkhorn_distance, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::wasserstein::rolling_wasserstein, m)?)?;
    m.add_function(wrap_pyfunction!(geometric_signals::wasserstein::lob_wasserstein, m)?)?;

    // ── Chiarella heterogeneous agent model ───────────────────────────────────
    m.add_function(wrap_pyfunction!(chiarella::simulate_chiarella_py, m)?)?;
    m.add_function(wrap_pyfunction!(chiarella::bifurcation_lambda_py, m)?)?;
    m.add_function(wrap_pyfunction!(chiarella::rolling_dominance_py, m)?)?;
    m.add_function(wrap_pyfunction!(chiarella::rolling_volatility_py, m)?)?;

    // ── Path signature methods ────────────────────────────────────────────────
    m.add_function(wrap_pyfunction!(signature::lead_lag_transform_py, m)?)?;
    m.add_function(wrap_pyfunction!(signature::path_signature_2d_py, m)?)?;
    m.add_function(wrap_pyfunction!(signature::log_signature_2d_py, m)?)?;
    m.add_function(wrap_pyfunction!(signature::rolling_signature_features_py, m)?)?;
    m.add_function(wrap_pyfunction!(signature::signature_distance_py, m)?)?;
    m.add_function(wrap_pyfunction!(signature::normalise_prices_py, m)?)?;
    m.add_function(wrap_pyfunction!(signature::prices_to_signature_features_py, m)?)?;

    // ── Analytics functions (pure math, no tracking needed) ───────────────────
    m.add_function(wrap_pyfunction!(analytics_py::sharpe_ratio_py, m)?)?;
    m.add_function(wrap_pyfunction!(analytics_py::sortino_ratio_py, m)?)?;
    m.add_function(wrap_pyfunction!(analytics_py::max_drawdown_py, m)?)?;
    m.add_function(wrap_pyfunction!(analytics_py::win_rate_py, m)?)?;
    m.add_function(wrap_pyfunction!(analytics_py::profit_factor_py, m)?)?;
    m.add_function(wrap_pyfunction!(analytics_py::annualised_return_py, m)?)?;
    m.add_function(wrap_pyfunction!(analytics_py::annualised_volatility_py, m)?)?;

    // ── Rough Heston pricing ─────────────────────────────────────────────────
    pricing::register(m)?;

    // ── Portfolio construction (public academic algorithms) ──────────────────
    m.add_function(wrap_pyfunction!(portfolio::cara_optimal_weights_rust, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio::sharpe_optimal_weights_rust, m)?)?;
    m.add_function(wrap_pyfunction!(portfolio::hurst_exponent_rust, m)?)?;


    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_orderbook_level_creation() {
        let level = OrderBookLevel::new(50000.0, 1.5);
        assert_eq!(level.price, 50000.0);
        assert_eq!(level.quantity, 1.5);
    }

    #[test]
    fn test_orderbook_snapshot() {
        let snapshot = OrderBookSnapshot::new(
            "2024-01-01T00:00:00Z".to_string(),
            "BTC-USD".to_string(),
            12345,
            "binance".to_string(),
            vec![(50000.0, 1.0), (49999.0, 2.0)],
            vec![(50001.0, 1.0), (50002.0, 2.0)],
        );
        assert_eq!(snapshot.symbol, "BTC-USD");
        assert_eq!(snapshot.mid_price(), 50000.5);
    }
}
