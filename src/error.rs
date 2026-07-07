//! Domain-specific error types for HFT Lab Core.
//!
//! Provides a unified error hierarchy using [`thiserror`] with automatic
//! conversion from common upstream errors. Every fallible function in this
//! crate returns `Result<T, HftError>` (aliased as [`HftResult`]).
//!
//! # Railway semantics
//!
//! Errors are *values*, not exceptions.  They propagate along the error
//! rail of a [`Result`] and can be mapped, recovered, or accumulated
//! using the combinators in [`crate::functional`].

use thiserror::Error;

// ── Core error enum ──────────────────────────────────────────────────────────

/// Unified error type for the `hfthot-lab-core` crate.
///
/// Each variant carries enough context to produce an actionable diagnostic
/// without requiring a backtrace in release builds.
#[derive(Debug, Error)]
pub enum HftError {
    /// Linear-algebra or numerical computation failure.
    #[error("numerical error: {message}")]
    Numerical {
        message: String,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    /// Input violates a documented precondition (e.g. empty vector, NaN).
    #[error("invalid input: {0}")]
    InvalidInput(String),

    /// Requested dimension, index, or size is out of range.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },

    /// An iterative algorithm did not converge within the allowed budget.
    #[error("convergence failure after {iterations} iterations (tolerance: {tolerance})")]
    Convergence { iterations: usize, tolerance: f64 },

    /// I/O or serialisation failure.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Catch-all for errors that don't fit the categories above.
    #[error("{0}")]
    Other(String),
}

// ── PartialEq (manual — `source` is not comparable) ──────────────────────────

impl PartialEq for HftError {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::Numerical { message: a, .. }, Self::Numerical { message: b, .. }) => a == b,
            (Self::InvalidInput(a), Self::InvalidInput(b)) => a == b,
            (
                Self::DimensionMismatch { expected: e1, actual: a1 },
                Self::DimensionMismatch { expected: e2, actual: a2 },
            ) => e1 == e2 && a1 == a2,
            (
                Self::Convergence { iterations: i1, tolerance: t1 },
                Self::Convergence { iterations: i2, tolerance: t2 },
            ) => i1 == i2 && t1 == t2,
            (Self::Serialization(a), Self::Serialization(b)) => a == b,
            (Self::Other(a), Self::Other(b)) => a == b,
            _ => false,
        }
    }
}

// ── Convenience alias ────────────────────────────────────────────────────────

/// Crate-wide result alias.
pub type HftResult<T> = Result<T, HftError>;

// ── Constructors ─────────────────────────────────────────────────────────────

impl HftError {
    /// Create a [`Numerical`](HftError::Numerical) error without a source.
    #[inline]
    pub fn numerical(msg: impl Into<String>) -> Self {
        Self::Numerical {
            message: msg.into(),
            source: None,
        }
    }

    /// Create a [`Numerical`](HftError::Numerical) error that wraps an upstream cause.
    #[inline]
    pub fn numerical_with_source(
        msg: impl Into<String>,
        src: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Numerical {
            message: msg.into(),
            source: Some(Box::new(src)),
        }
    }

    /// Create an [`InvalidInput`](HftError::InvalidInput) error.
    #[inline]
    pub fn invalid_input(msg: impl Into<String>) -> Self {
        Self::InvalidInput(msg.into())
    }

    /// Create a [`DimensionMismatch`](HftError::DimensionMismatch) error.
    #[inline]
    pub fn dimension_mismatch(expected: usize, actual: usize) -> Self {
        Self::DimensionMismatch { expected, actual }
    }

    /// Create a [`Convergence`](HftError::Convergence) error.
    #[inline]
    pub fn convergence(iterations: usize, tolerance: f64) -> Self {
        Self::Convergence {
            iterations,
            tolerance,
        }
    }
}

// ── PyO3 conversion ──────────────────────────────────────────────────────────

#[cfg(feature = "python")]
impl From<HftError> for pyo3::PyErr {
    fn from(err: HftError) -> pyo3::PyErr {
        match &err {
            HftError::InvalidInput(_) | HftError::DimensionMismatch { .. } => {
                pyo3::exceptions::PyValueError::new_err(err.to_string())
            }
            HftError::Convergence { .. } => {
                pyo3::exceptions::PyRuntimeError::new_err(err.to_string())
            }
            _ => pyo3::exceptions::PyRuntimeError::new_err(err.to_string()),
        }
    }
}

// ── Display helpers ──────────────────────────────────────────────────────────

impl HftError {
    /// Return a structured diagnostic suitable for logging.
    pub fn diagnostic(&self) -> String {
        match self {
            Self::Numerical { message, source } => {
                let mut s = format!("[NUMERICAL] {message}");
                if let Some(src) = source {
                    s.push_str(&format!("\n  caused by: {src}"));
                }
                s
            }
            Self::DimensionMismatch { expected, actual } => {
                format!("[DIM] expected {expected}, got {actual}")
            }
            Self::Convergence {
                iterations,
                tolerance,
            } => {
                format!("[CONVERGENCE] {iterations} iters, tol={tolerance:.2e}")
            }
            other => format!("[ERROR] {other}"),
        }
    }
}
