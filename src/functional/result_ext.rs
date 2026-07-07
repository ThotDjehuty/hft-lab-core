//! # Extension methods for `Result<T, E>`
//!
//! Railway-oriented combinators that enrich Rust's built-in `Result`
//! with context-aware error wrapping, logging hooks, and recovery
//! strategies.
//!
//! All methods are provided via the [`ResultExt`] trait, which is
//! blanket-implemented for every `Result<T, E>`.
//!
//! # Example — contextual error chaining
//!
//! ```rust
//! use hfthot_lab_core::functional::result_ext::ResultExt;
//! use hfthot_lab_core::error::{HftError, HftResult};
//!
//! fn parse_price(raw: &str) -> HftResult<f64> {
//!     raw.parse::<f64>()
//!         .map_err(|e| HftError::invalid_input(format!("{e}")))
//!         .with_context("parsing price from exchange feed")
//! }
//! ```

use crate::error::{HftError, HftResult};

// ── Trait definition ─────────────────────────────────────────────────────────

/// Extension methods for `Result<T, HftError>`.
///
/// Imported via `use hfthot_lab_core::functional::prelude::*`.
pub trait ResultExt<T> {
    /// Wrap the error with additional context (like `anyhow::Context`
    /// but typed).
    ///
    /// The context string is prepended to the existing error message,
    /// separated by `: `.
    fn with_context(self, ctx: &str) -> HftResult<T>;

    /// Like [`and_then`](Result::and_then) but wraps any error with
    /// a contextual label.
    fn and_then_with_context<U, F>(self, f: F, ctx: &str) -> HftResult<U>
    where
        F: FnOnce(T) -> HftResult<U>;

    /// If `self` is `Err`, try the fallback function instead.
    ///
    /// The fallback receives the original error and may itself fail.
    fn or_recover<F>(self, fallback: F) -> HftResult<T>
    where
        F: FnOnce(HftError) -> HftResult<T>;

    /// Run a side-effect closure on the `Ok` value (for logging /
    /// metrics) without consuming it.
    fn inspect_ok<F>(self, f: F) -> HftResult<T>
    where
        F: FnOnce(&T);

    /// Run a side-effect closure on the `Err` variant.
    fn inspect_err_hft<F>(self, f: F) -> HftResult<T>
    where
        F: FnOnce(&HftError);

    /// Convert an `Err` to a default value, consuming the error.
    fn unwrap_or_log(self, default: T, label: &str) -> T;

    /// Ensure a predicate holds on the `Ok` value, otherwise produce
    /// an [`HftError::InvalidInput`].
    fn ensure<F>(self, predicate: F, msg: &str) -> HftResult<T>
    where
        F: FnOnce(&T) -> bool;
}

// ── Blanket implementation ───────────────────────────────────────────────────

impl<T> ResultExt<T> for HftResult<T> {
    #[inline]
    fn with_context(self, ctx: &str) -> HftResult<T> {
        self.map_err(|e| {
            let inner = e.to_string();
            HftError::Other(format!("{ctx}: {inner}"))
        })
    }

    #[inline]
    fn and_then_with_context<U, F>(self, f: F, ctx: &str) -> HftResult<U>
    where
        F: FnOnce(T) -> HftResult<U>,
    {
        self.and_then(f).with_context(ctx)
    }

    #[inline]
    fn or_recover<F>(self, fallback: F) -> HftResult<T>
    where
        F: FnOnce(HftError) -> HftResult<T>,
    {
        match self {
            Ok(val) => Ok(val),
            Err(e) => fallback(e),
        }
    }

    #[inline]
    fn inspect_ok<F>(self, f: F) -> HftResult<T>
    where
        F: FnOnce(&T),
    {
        if let Ok(ref val) = self {
            f(val);
        }
        self
    }

    #[inline]
    fn inspect_err_hft<F>(self, f: F) -> HftResult<T>
    where
        F: FnOnce(&HftError),
    {
        if let Err(ref e) = self {
            f(e);
        }
        self
    }

    #[inline]
    fn unwrap_or_log(self, default: T, label: &str) -> T {
        match self {
            Ok(val) => val,
            Err(e) => {
                log::warn!("[{label}] recovered from error: {e}");
                default
            }
        }
    }

    #[inline]
    fn ensure<F>(self, predicate: F, msg: &str) -> HftResult<T>
    where
        F: FnOnce(&T) -> bool,
    {
        self.and_then(|val| {
            if predicate(&val) {
                Ok(val)
            } else {
                Err(HftError::invalid_input(msg))
            }
        })
    }
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn with_context_wraps_error() {
        let r: HftResult<i32> = Err(HftError::numerical("singular matrix"));
        let r2 = r.with_context("computing PCA");
        let msg = r2.unwrap_err().to_string();
        assert!(msg.contains("computing PCA"));
        assert!(msg.contains("singular matrix"));
    }

    #[test]
    fn with_context_passes_ok() {
        let r: HftResult<i32> = Ok(42);
        assert_eq!(r.with_context("should not matter"), Ok(42));
    }

    #[test]
    fn or_recover_uses_fallback() {
        let r: HftResult<i32> = Err(HftError::numerical("bad"));
        let recovered = r.or_recover(|_| Ok(99));
        assert_eq!(recovered, Ok(99));
    }

    #[test]
    fn or_recover_passes_ok() {
        let r: HftResult<i32> = Ok(42);
        let same = r.or_recover(|_| Ok(99));
        assert_eq!(same, Ok(42));
    }

    #[test]
    fn ensure_passes_valid() {
        let r: HftResult<f64> = Ok(5.0);
        assert!(r.ensure(|x| *x > 0.0, "must be positive").is_ok());
    }

    #[test]
    fn ensure_rejects_invalid() {
        let r: HftResult<f64> = Ok(-1.0);
        let err = r.ensure(|x| *x > 0.0, "must be positive").unwrap_err();
        assert!(err.to_string().contains("must be positive"));
    }

    #[test]
    fn inspect_ok_runs_closure() {
        let mut seen = false;
        let r: HftResult<i32> = Ok(42);
        let _ = r.inspect_ok(|_| seen = true);
        assert!(seen);
    }

    #[test]
    fn inspect_ok_skips_on_err() {
        let mut seen = false;
        let r: HftResult<i32> = Err(HftError::numerical("x"));
        let _ = r.inspect_ok(|_| seen = true);
        assert!(!seen);
    }

    #[test]
    fn and_then_with_context_chains() {
        let r: HftResult<f64> = Ok(4.0);
        let out = r.and_then_with_context(
            |x| {
                if x > 0.0 {
                    Ok(x.sqrt())
                } else {
                    Err(HftError::invalid_input("negative"))
                }
            },
            "sqrt stage",
        );
        assert_eq!(out.unwrap(), 2.0);
    }
}
