//! # Crate prelude — single import for common types and traits
//!
//! ```rust
//! use hfthot_lab_core::prelude::*;
//! ```
//!
//! This brings into scope the most frequently used items without
//! polluting the namespace with implementation details.

// ── Error types ──────────────────────────────────────────────────────────────
pub use crate::error::{HftError, HftResult};

// ── Functional combinators ───────────────────────────────────────────────────
pub use crate::functional::compose::{pipe, compose, chain, identity};
pub use crate::functional::result_ext::ResultExt;

// ── Pipeline ─────────────────────────────────────────────────────────────────
pub use crate::pipeline::{Pipeline, Stage};

// ── Core domain types ────────────────────────────────────────────────────────
pub use crate::lob::{OrderBookLevel, OrderBookSnapshot, OrderBookUpdate, LOBAnalytics};
pub use crate::chiarella::{ChiarellaParams, ChiarellaSimulationResult};
pub use crate::optimization::HMMParams;

// ── Backtesting ──────────────────────────────────────────────────────────────────
pub use crate::backtest::{BacktestConfig, BacktestResult, Trade, run_backtest};
