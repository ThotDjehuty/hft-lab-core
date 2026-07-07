//! # Performance Analytics — risk and return metrics for trading strategies
//!
//! Pure mathematical functions — no usage tracking (not billable).
//! These are intentionally public: they help evaluate backtest results
//! and drive adoption of the paid tier.
//!
//! ## Quick start
//!
//! ```rust
//! use hfthot_lab_core::analytics;
//!
//! let returns = vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005];
//! let sharpe  = analytics::sharpe_ratio(&returns, 0.0);
//! let mdd     = analytics::max_drawdown_from_returns(&returns);
//! let sortino = analytics::sortino_ratio(&returns, 0.0);
//!
//! assert!(sharpe > 0.0);
//! assert!(mdd >= 0.0 && mdd <= 1.0);
//! ```

// ── Return-based metrics ─────────────────────────────────────────────────────

/// Annualised Sharpe ratio from a series of period returns.
///
/// `risk_free` is the per-period risk-free rate (e.g. 0.0 for zero).
pub fn sharpe_ratio(returns: &[f64], risk_free: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let n = returns.len() as f64;
    let excess: Vec<f64> = returns.iter().map(|r| r - risk_free).collect();
    let mean = excess.iter().sum::<f64>() / n;
    let var = excess.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let std = var.sqrt();
    if std < 1e-15 {
        return 0.0;
    }
    (mean / std) * 252.0_f64.sqrt()
}

/// Annualised Sortino ratio (downside risk only).
pub fn sortino_ratio(returns: &[f64], risk_free: f64) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let n = returns.len() as f64;
    let excess: Vec<f64> = returns.iter().map(|r| r - risk_free).collect();
    let mean = excess.iter().sum::<f64>() / n;
    let downside_var = excess
        .iter()
        .map(|r| if *r < 0.0 { r.powi(2) } else { 0.0 })
        .sum::<f64>()
        / (n - 1.0);
    let downside_std = downside_var.sqrt();
    if downside_std < 1e-15 {
        return 0.0;
    }
    (mean / downside_std) * 252.0_f64.sqrt()
}

/// Calmar ratio = annualised return / max drawdown.
pub fn calmar_ratio(annual_return: f64, max_dd: f64) -> f64 {
    if max_dd < 1e-15 {
        return 0.0;
    }
    annual_return / max_dd
}

/// Maximum drawdown from a series of period returns.
/// Returns a value in `[0, 1]`.
pub fn max_drawdown_from_returns(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let mut equity = 1.0;
    let mut peak = 1.0;
    let mut max_dd = 0.0_f64;
    for &r in returns {
        equity *= 1.0 + r;
        if equity > peak {
            peak = equity;
        }
        let dd = (peak - equity) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Maximum drawdown from an equity curve.
pub fn max_drawdown_from_equity(equity: &[f64]) -> f64 {
    if equity.is_empty() {
        return 0.0;
    }
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &e in equity {
        if e > peak {
            peak = e;
        }
        let dd = (peak - e) / peak.abs().max(1e-15);
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

/// Win rate: fraction of positive returns.
pub fn win_rate(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let wins = returns.iter().filter(|r| **r > 0.0).count();
    wins as f64 / returns.len() as f64
}

/// Profit factor: gross profit / gross loss.
///
/// Returns `f64::INFINITY` when there are no losses.
pub fn profit_factor(returns: &[f64]) -> f64 {
    let gross_profit: f64 = returns.iter().filter(|r| **r > 0.0).sum();
    let gross_loss: f64 = returns.iter().filter(|r| **r < 0.0).map(|r| r.abs()).sum();
    if gross_loss < 1e-15 {
        return f64::INFINITY;
    }
    gross_profit / gross_loss
}

/// Annualised return from a series of period returns.
///
/// Assumes 252 trading days per year.
pub fn annualised_return(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    let cumulative: f64 = returns.iter().fold(1.0, |acc, r| acc * (1.0 + r));
    let n = returns.len() as f64;
    cumulative.powf(252.0 / n) - 1.0
}

/// Annualised volatility from a series of period returns.
pub fn annualised_volatility(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0);
    var.sqrt() * 252.0_f64.sqrt()
}

/// Average return per period.
pub fn mean_return(returns: &[f64]) -> f64 {
    if returns.is_empty() {
        return 0.0;
    }
    returns.iter().sum::<f64>() / returns.len() as f64
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1e-6;

    fn sample_returns() -> Vec<f64> {
        vec![0.01, -0.005, 0.02, -0.01, 0.015, 0.005, -0.002, 0.008]
    }

    #[test]
    fn sharpe_positive_trend() {
        let s = sharpe_ratio(&sample_returns(), 0.0);
        assert!(s > 0.0, "Expected positive Sharpe, got {}", s);
    }

    #[test]
    fn sharpe_empty() {
        assert!((sharpe_ratio(&[], 0.0)).abs() < EPS);
        assert!((sharpe_ratio(&[0.01], 0.0)).abs() < EPS);
    }

    #[test]
    fn sortino_higher_than_sharpe_for_skewed() {
        // Mostly positive returns → Sortino should be higher than Sharpe
        let rets = vec![0.02, 0.01, 0.015, -0.002, 0.01, 0.005];
        let s = sharpe_ratio(&rets, 0.0);
        let so = sortino_ratio(&rets, 0.0);
        assert!(so > s, "Sortino {} should exceed Sharpe {}", so, s);
    }

    #[test]
    fn max_drawdown_simple() {
        let rets = vec![0.10, -0.15, 0.05];
        let dd = max_drawdown_from_returns(&rets);
        // 1.0 → 1.10 → 0.935 → 0.98175
        // Peak 1.10, trough 0.935 → dd = 0.15
        assert!(dd > 0.14 && dd < 0.16, "dd = {}", dd);
    }

    #[test]
    fn max_drawdown_equity() {
        let eq = vec![100.0, 110.0, 95.0, 105.0];
        let dd = max_drawdown_from_equity(&eq);
        // 110 → 95 = 15/110 ≈ 0.1364
        assert!((dd - 15.0 / 110.0).abs() < EPS);
    }

    #[test]
    fn drawdown_no_loss() {
        let rets = vec![0.01, 0.02, 0.03];
        assert!(max_drawdown_from_returns(&rets) < EPS);
    }

    #[test]
    fn win_rate_works() {
        let rets = vec![0.01, -0.01, 0.02, 0.0, -0.005];
        let w = win_rate(&rets);
        assert!((w - 2.0 / 5.0).abs() < EPS);
    }

    #[test]
    fn profit_factor_works() {
        let rets = vec![0.10, -0.05, 0.20];
        let pf = profit_factor(&rets);
        assert!((pf - (0.30 / 0.05)).abs() < EPS);
    }

    #[test]
    fn profit_factor_no_losses() {
        let rets = vec![0.01, 0.02, 0.03];
        assert!(profit_factor(&rets).is_infinite());
    }

    #[test]
    fn annualised_return_works() {
        let rets = vec![0.001; 252]; // ~0.1% daily for a year
        let ar = annualised_return(&rets);
        // (1.001)^252 - 1 ≈ 0.2856
        assert!(ar > 0.20 && ar < 0.40, "ar = {}", ar);
    }

    #[test]
    fn annualised_volatility_works() {
        let rets = sample_returns();
        let vol = annualised_volatility(&rets);
        assert!(vol > 0.0);
    }

    #[test]
    fn calmar_ratio_works() {
        let c = calmar_ratio(0.20, 0.10);
        assert!((c - 2.0).abs() < EPS);
    }

    #[test]
    fn calmar_zero_dd() {
        assert!((calmar_ratio(0.20, 0.0)).abs() < EPS);
    }

    #[test]
    fn mean_return_works() {
        let rets = vec![0.01, 0.02, 0.03];
        assert!((mean_return(&rets) - 0.02).abs() < EPS);
    }
}
