//! Geometric Signals Module
//!
//! Implements geometric and topological tools for financial signal generation:
//! - **Gauge**: Connection, curvature, holonomy on price graphs (arbitrage detection)
//! - **Spectral**: Graph Laplacian, Fiedler value, spectral entropy, heat trace
//! - **Cohomology**: Sheaf cohomology, Hodge decomposition, invariant portfolios
//! - **Information**: Mirror descent, Bregman divergences, KL-regularized portfolio updates
//! - **SPD**: Log-Euclidean distance, Fréchet mean on positive-definite manifold
//! - **Wasserstein**: 1D Wasserstein distance, Sinkhorn entropic OT

pub mod gauge;
pub mod spectral;
pub mod cohomology;
pub mod information;
pub mod spd;
pub mod wasserstein;

// Re-exports for convenience
pub use gauge::{triangle_curvature, cycle_curvature, curvature_zscore, holonomy};
pub use spectral::{graph_laplacian, normalized_laplacian, fiedler_value, fiedler_vector, spectral_entropy, heat_trace, laplacian_eigenvalues};
pub use cohomology::{coboundary_matrix, weighted_coboundary_matrix, sheaf_laplacian, detect_arbitrage, cohomology_dimension, hodge_decompose, HodgeComponents};
pub use information::{mirror_descent_step, natural_gradient, bregman_kl, functionally_generated_portfolio, relative_entropy_rate};
pub use spd::{log_euclidean_distance, frechet_mean_le, matrix_log_sym, matrix_exp_sym, spd_geodesic, spd_distance_matrix};
pub use wasserstein::{wasserstein_1d, wasserstein_2_1d, sinkhorn_distance, rolling_wasserstein, lob_wasserstein};
