//! # Backtesting Framework — signal-driven backtest engine
//!
//! A public, configurable backtesting engine that processes price series
//! and trading signals to produce performance reports.
//!
//! ## Quick start
//!
//! ```rust
//! use hfthot_lab_core::backtest::{BacktestConfig, run_backtest};
//!
//! let prices  = vec![100.0, 102.0, 101.0, 105.0, 103.0, 107.0];
//! let signals = vec![  1.0,   1.0,   0.0,   1.0,  -1.0,   0.0];
//! let config  = BacktestConfig::default();
//!
//! let result = run_backtest(&prices, &signals, &config).unwrap();
//! assert!(result.num_trades > 0);
//! ```

use crate::error::{HftError, HftResult};

// ── Configuration ────────────────────────────────────────────────────────────

/// Backtest parameters.
#[derive(Debug, Clone)]
pub struct BacktestConfig {
    /// Starting capital in base currency.
    pub initial_capital: f64,
    /// One-way commission in basis points (e.g. 10 → 0.10 %).
    pub commission_bps: f64,
    /// Estimated slippage in basis points.
    pub slippage_bps: f64,
}

impl Default for BacktestConfig {
    fn default() -> Self {
        Self {
            initial_capital: 10_000.0,
            commission_bps: 10.0,
            slippage_bps: 5.0,
        }
    }
}

// ── Individual trade record ──────────────────────────────────────────────────

/// One completed round-trip trade.
#[derive(Debug, Clone)]
pub struct Trade {
    pub entry_idx: usize,
    pub exit_idx: usize,
    pub entry_price: f64,
    pub exit_price: f64,
    pub direction: f64,
    pub pnl: f64,
}

// ── Backtest result ──────────────────────────────────────────────────────────

/// Full result of a backtest run.
#[derive(Debug, Clone)]
pub struct BacktestResult {
    /// Final P&L as a fraction of initial capital (e.g. 0.12 → +12 %).
    pub total_return: f64,
    /// Annualised Sharpe ratio (assuming 252 trading days).
    pub sharpe_ratio: f64,
    /// Maximum peak-to-trough drawdown (0.0 – 1.0).
    pub max_drawdown: f64,
    /// Number of completed round-trip trades.
    pub num_trades: usize,
    /// Fraction of trades that were profitable (0.0 – 1.0).
    pub win_rate: f64,
    /// Gross profit / gross loss.  `f64::INFINITY` if no losing trades.
    pub profit_factor: f64,
    /// Equity curve (one value per price bar).
    pub equity_curve: Vec<f64>,
    /// Individual trade records.
    pub trades: Vec<Trade>,
}

// ── Engine ───────────────────────────────────────────────────────────────────

/// Run a signal-based backtest with mandatory usage tracking.
///
/// **Signals convention:**
/// * `> 0` → long (buy / hold)
/// * `< 0` → short (sell / hold short)
/// * `= 0` → flat (close any position)
///
/// Returns [`BacktestResult`] on success.
///
/// # Errors
///
/// * Empty or mismatched input lengths.
pub fn run_backtest(
    prices: &[f64],
    signals: &[f64],
    config: &BacktestConfig,
) -> HftResult<BacktestResult> {
    // ── input validation ─────────────────────────────────────────────────
    if prices.len() < 2 {
        return Err(HftError::invalid_input("prices must have at least 2 bars"));
    }
    if prices.len() != signals.len() {
        return Err(HftError::invalid_input(
            "prices and signals must have the same length",
        ));
    }

    let n = prices.len();
    let cost_mult = (config.commission_bps + config.slippage_bps) / 10_000.0;

    let mut equity = config.initial_capital;
    let mut equity_curve = Vec::with_capacity(n);
    let mut position: f64 = 0.0; // current position direction (-1, 0, +1)
    let mut entry_price = 0.0_f64;
    let mut entry_idx = 0_usize;
    let mut trades: Vec<Trade> = Vec::new();

    for i in 0..n {
        let sig = signals[i].clamp(-1.0, 1.0);

        // Detect position change
        if (sig - position).abs() > f64::EPSILON {
            // Close existing position
            if position.abs() > f64::EPSILON && i > 0 {
                let gross_pnl = position * (prices[i] - entry_price);
                let cost = (prices[i] + entry_price) * cost_mult;
                let net_pnl = gross_pnl - cost;
                equity += net_pnl;
                trades.push(Trade {
                    entry_idx,
                    exit_idx: i,
                    entry_price,
                    exit_price: prices[i],
                    direction: position,
                    pnl: net_pnl,
                });
            }

            // Open new position
            position = sig;
            if position.abs() > f64::EPSILON {
                entry_price = prices[i];
                entry_idx = i;
            }
        }

        equity_curve.push(equity);
    }

    // Close any open position at end
    if position.abs() > f64::EPSILON {
        let last = prices[n - 1];
        let gross_pnl = position * (last - entry_price);
        let cost = (last + entry_price) * cost_mult;
        let net_pnl = gross_pnl - cost;
        equity += net_pnl;
        trades.push(Trade {
            entry_idx,
            exit_idx: n - 1,
            entry_price,
            exit_price: last,
            direction: position,
            pnl: net_pnl,
        });
        if let Some(last_eq) = equity_curve.last_mut() {
            *last_eq = equity;
        }
    }

    // ── Analytics ────────────────────────────────────────────────────────
    let total_return = (equity - config.initial_capital) / config.initial_capital;
    let max_drawdown = compute_max_drawdown(&equity_curve);
    let (win_rate, profit_factor) = trade_stats(&trades);
    let sharpe_ratio = compute_sharpe(&equity_curve);


    Ok(BacktestResult {
        total_return,
        sharpe_ratio,
        max_drawdown,
        num_trades: trades.len(),
        win_rate,
        profit_factor,
        equity_curve,
        trades,
    })
}

// ── Helpers ──────────────────────────────────────────────────────────────────

fn compute_max_drawdown(equity: &[f64]) -> f64 {
    let mut peak = f64::NEG_INFINITY;
    let mut max_dd = 0.0_f64;
    for &e in equity {
        if e > peak {
            peak = e;
        }
        let dd = (peak - e) / peak;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

fn trade_stats(trades: &[Trade]) -> (f64, f64) {
    if trades.is_empty() {
        return (0.0, 0.0);
    }
    let wins = trades.iter().filter(|t| t.pnl > 0.0).count();
    let win_rate = wins as f64 / trades.len() as f64;

    let gross_profit: f64 = trades.iter().filter(|t| t.pnl > 0.0).map(|t| t.pnl).sum();
    let gross_loss: f64 = trades
        .iter()
        .filter(|t| t.pnl <= 0.0)
        .map(|t| t.pnl.abs())
        .sum();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else {
        f64::INFINITY
    };

    (win_rate, profit_factor)
}

fn compute_sharpe(equity: &[f64]) -> f64 {
    if equity.len() < 2 {
        return 0.0;
    }
    let returns: Vec<f64> = equity
        .windows(2)
        .map(|w| (w[1] - w[0]) / w[0].abs().max(1e-12))
        .collect();
    let n = returns.len() as f64;
    let mean = returns.iter().sum::<f64>() / n;
    let var = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (n - 1.0).max(1.0);
    let std = var.sqrt();
    if std < 1e-12 {
        return 0.0;
    }
    // Annualise (252 trading days)
    (mean / std) * 252.0_f64.sqrt()
}

// ── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_backtest() {
        let prices = vec![100.0, 102.0, 101.0, 105.0, 103.0, 107.0];
        let signals = vec![1.0, 1.0, 0.0, 1.0, -1.0, 0.0];
        let result = run_backtest(&prices, &signals, &BacktestConfig::default()).unwrap();

        assert!(result.num_trades > 0);
        assert!(!result.equity_curve.is_empty());
        assert!(result.max_drawdown >= 0.0);
        assert!(result.max_drawdown <= 1.0);
    }

    #[test]
    fn buy_and_hold() {
        let prices = vec![100.0, 110.0, 120.0, 130.0];
        let signals = vec![1.0, 1.0, 1.0, 1.0];
        let result = run_backtest(&prices, &signals, &BacktestConfig::default()).unwrap();

        // Should have positive return from 100→130
        assert!(result.total_return > 0.0);
        assert_eq!(result.num_trades, 1); // one open→close at end
    }

    #[test]
    fn empty_prices_fails() {
        let result = run_backtest(&[], &[], &BacktestConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn mismatched_lengths_fails() {
        let result = run_backtest(&[1.0, 2.0], &[1.0], &BacktestConfig::default());
        assert!(result.is_err());
    }

    #[test]
    fn flat_signal_no_trades() {
        let prices = vec![100.0, 102.0, 101.0];
        let signals = vec![0.0, 0.0, 0.0];
        let result = run_backtest(&prices, &signals, &BacktestConfig::default()).unwrap();
        assert_eq!(result.num_trades, 0);
        assert!((result.total_return).abs() < 1e-10);
    }

    #[test]
    fn short_trade() {
        let prices = vec![100.0, 95.0, 90.0]; // falling market
        let signals = vec![-1.0, -1.0, -1.0];
        let result = run_backtest(&prices, &signals, &BacktestConfig::default()).unwrap();
        // Short in falling market → positive return (minus costs)
        assert!(result.total_return > 0.0);
    }

    #[test]
    fn compute_max_drawdown_works() {
        let eq = vec![100.0, 105.0, 95.0, 110.0];
        let dd = compute_max_drawdown(&eq);
        // Peak was 105, trough was 95 → dd = 10/105 ≈ 0.0952
        assert!((dd - 10.0 / 105.0).abs() < 1e-6);
    }
}
