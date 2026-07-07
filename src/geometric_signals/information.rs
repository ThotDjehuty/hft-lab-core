//! Information geometry for portfolio optimization.
//!
//! The portfolio simplex Δₙ is a statistical manifold with the Fisher metric.
//! Natural gradient and mirror descent exploit this geometry for stable updates
//! that respect the simplex constraint.
//!
//! Key operations:
//! - Mirror descent with neg-entropy (entropic proximal)
//! - Natural gradient on the simplex
//! - Bregman divergence (KL)
//! - Functionally generated portfolios (FGP)
//!
//! References:
//! - Amari (2016), "Information Geometry and Its Applications"
//! - Fernholz (2002), "Stochastic Portfolio Theory"

#[cfg(feature = "python")]
use pyo3::prelude::*;

/// Mirror descent step with negative entropy generating function.
///
/// The entropic mirror map is Φ(w) = Σ wᵢ log wᵢ.
/// The update rule is:
///   w̃ᵢ = wᵢ · exp(-η · ∇ᵢ)
///   wᵢ⁺ = w̃ᵢ / Σⱼ w̃ⱼ   (re-normalize)
///
/// This is equivalent to exponentiated gradient descent and stays on the simplex.
///
/// # Arguments
/// * `weights` - Current portfolio weights (must sum to 1)
/// * `gradients` - Gradient of loss w.r.t. weights
/// * `eta` - Learning rate
///
/// # Returns
/// Updated weights on the simplex
#[cfg_attr(feature = "python", pyfunction)]
pub fn mirror_descent_step(weights: Vec<f64>, gradients: Vec<f64>, eta: f64) -> Vec<f64> {
    assert_eq!(weights.len(), gradients.len());
    let n = weights.len();

    // Multiplicative update: w̃ᵢ = wᵢ × exp(-η × ∇ᵢ)
    let mut w_tilde: Vec<f64> = (0..n)
        .map(|i| {
            let v = weights[i] * (-eta * gradients[i]).exp();
            v.max(1e-15) // prevent underflow
        })
        .collect();

    // Re-normalize to simplex
    let sum: f64 = w_tilde.iter().sum();
    for w in &mut w_tilde {
        *w /= sum;
    }
    w_tilde
}

/// Natural gradient on the simplex manifold.
///
/// The Fisher information matrix on Δₙ is G = diag(1/w₁, ..., 1/wₙ).
/// The natural gradient is G⁻¹ ∇f = diag(w₁,...,wₙ) · ∇f,
/// then projected to the tangent space of the simplex.
///
/// # Arguments
/// * `weights` - Current portfolio weights
/// * `gradient` - Euclidean gradient
///
/// # Returns
/// Natural gradient projected onto simplex tangent space
#[cfg_attr(feature = "python", pyfunction)]
pub fn natural_gradient(weights: Vec<f64>, gradient: Vec<f64>) -> Vec<f64> {
    assert_eq!(weights.len(), gradient.len());
    let n = weights.len();

    // G⁻¹ ∇f = w ⊙ ∇f
    let mut nat_grad: Vec<f64> = (0..n).map(|i| weights[i] * gradient[i]).collect();

    // Project onto tangent space of simplex: subtract mean
    let mean: f64 = nat_grad.iter().sum::<f64>() / n as f64;
    for v in &mut nat_grad {
        *v -= mean;
    }
    nat_grad
}

/// Bregman divergence D_KL(w || w_ref) = KL divergence.
///
/// D_KL(w || w_ref) = Σᵢ wᵢ log(wᵢ / w_refᵢ)
///
/// This is the Bregman divergence associated with the negative entropy
/// generating function Φ(w) = Σ wᵢ log wᵢ.
///
/// # Arguments
/// * `w` - First distribution
/// * `w_ref` - Reference distribution
///
/// # Returns
/// KL divergence (non-negative)
#[cfg_attr(feature = "python", pyfunction)]
pub fn bregman_kl(w: Vec<f64>, w_ref: Vec<f64>) -> f64 {
    assert_eq!(w.len(), w_ref.len());
    w.iter()
        .zip(w_ref.iter())
        .map(|(&wi, &wri)| {
            if wi > 1e-15 && wri > 1e-15 {
                wi * (wi / wri).ln()
            } else {
                0.0
            }
        })
        .sum()
}

/// Functionally Generated Portfolio (FGP) weights from a log-concave
/// generating function.
///
/// Given market weights μ and generating function Φ(μ) = Σ μᵢ^p (p < 1),
/// the FGP weights are:
///   wᵢ = ∂Φ/∂μᵢ · μᵢ / Σⱼ (∂Φ/∂μⱼ · μⱼ)
///
/// Fernholz's diversity-weighted portfolio uses Φ(μ) = (Σ μᵢ^p)^{1/p}.
///
/// # Arguments
/// * `market_weights` - Market capitalization weights
/// * `p` - Diversity exponent (0 < p < 1, typically 0.5-0.76)
///
/// # Returns
/// FGP portfolio weights
#[cfg_attr(feature = "python", pyfunction)]
pub fn functionally_generated_portfolio(market_weights: Vec<f64>, p: f64) -> Vec<f64> {
    let n = market_weights.len();
    assert!(p > 0.0 && p < 1.0, "p must be in (0, 1)");

    // ∂Φ/∂μᵢ = μᵢ^{p-1}
    let derivatives: Vec<f64> = market_weights.iter().map(|&mu| mu.powf(p - 1.0)).collect();

    // w̃ᵢ = ∂Φ/∂μᵢ · μᵢ = μᵢ^p
    let w_tilde: Vec<f64> = (0..n).map(|i| derivatives[i] * market_weights[i]).collect();

    // Normalize
    let sum: f64 = w_tilde.iter().sum();
    w_tilde.iter().map(|&w| w / sum).collect()
}

/// Relative entropy rate for tracking a portfolio against a benchmark.
///
/// Computes the rolling KL divergence between portfolio weights and benchmark,
/// useful for detecting style drift.
///
/// # Arguments
/// * `portfolio_history` - Flat T×n matrix of portfolio weights over time
/// * `benchmark` - Reference weights (length n)
/// * `n_assets` - Number of assets
///
/// # Returns
/// Vector of KL divergences at each time step
#[cfg_attr(feature = "python", pyfunction)]
pub fn relative_entropy_rate(
    portfolio_history: Vec<f64>,
    benchmark: Vec<f64>,
    n_assets: usize,
) -> Vec<f64> {
    let t = portfolio_history.len() / n_assets;
    assert_eq!(benchmark.len(), n_assets);

    (0..t)
        .map(|i| {
            let w = &portfolio_history[i * n_assets..(i + 1) * n_assets];
            w.iter()
                .zip(benchmark.iter())
                .map(|(&wi, &bi)| {
                    if wi > 1e-15 && bi > 1e-15 {
                        wi * (wi / bi).ln()
                    } else {
                        0.0
                    }
                })
                .sum()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_mirror_descent_stays_on_simplex() {
        let weights = vec![0.4, 0.3, 0.3];
        let gradients = vec![0.1, -0.2, 0.1];
        let result = mirror_descent_step(weights, gradients, 0.5);

        let sum: f64 = result.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);
        assert!(result.iter().all(|&w| w > 0.0));
    }

    #[test]
    fn test_natural_gradient_tangent() {
        let weights = vec![0.5, 0.3, 0.2];
        let gradient = vec![1.0, 2.0, 3.0];
        let nat_grad = natural_gradient(weights, gradient);

        // Natural gradient should sum to ~0 (tangent to simplex)
        let sum: f64 = nat_grad.iter().sum();
        assert!(sum.abs() < 1e-10);
    }

    #[test]
    fn test_bregman_kl_non_negative() {
        let w = vec![0.5, 0.3, 0.2];
        let w_ref = vec![0.33, 0.33, 0.34];
        let kl = bregman_kl(w.clone(), w_ref);
        assert!(kl >= 0.0);

        // KL(w || w) = 0
        let kl_self = bregman_kl(w.clone(), w.clone());
        assert!(kl_self.abs() < 1e-10);
    }

    #[test]
    fn test_fgp_weights_sum_to_one() {
        let market = vec![0.5, 0.3, 0.15, 0.05];
        let weights = functionally_generated_portfolio(market, 0.76);
        let sum: f64 = weights.iter().sum();
        assert!((sum - 1.0).abs() < 1e-10);

        // FGP should overweight small caps (more equal than market)
        // weights[3] / market[3] should be > weights[0] / market[0]
        assert!(weights[3] / 0.05 > weights[0] / 0.5);
    }
}
