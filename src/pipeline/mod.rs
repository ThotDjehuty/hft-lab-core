//! # Pipeline — composable data-processing stages
//!
//! Railway-oriented, *parallelism-ready* data pipelines built on top of
//! the [`crate::functional`] abstractions.
//!
//! ## Design
//!
//! A [`Pipeline`] is a chain of **stages** — pure functions that transform
//! a value or return an error.  Stages compose via `.then()` (like
//! monadic bind) and can be executed sequentially or in parallel via
//! Rayon.
//!
//! ```text
//!  ┌───────────┐     ┌───────────┐     ┌───────────┐
//!  │  Stage A   │────▶│  Stage B   │────▶│  Stage C   │──▶ Ok(output)
//!  └───────────┘     └───────────┘     └───────────┘
//!        │                 │                 │
//!        ▼                 ▼                 ▼
//!     Err(e) ──────────▶ Err(e) ──────────▶ Err(e)   (error rail)
//! ```
//!
//! ## Parallelism
//!
//! The [`parallel`] sub-module provides `par_map`, `par_filter_map`,
//! and `par_fold` that distribute work across Rayon's thread pool.
//! Since all pipeline functions are pure, this is data-race free
//! by construction.
//!
//! ## Example
//!
//! ```rust
//! use hfthot_lab_core::pipeline::{Pipeline, Stage};
//! use hfthot_lab_core::error::HftResult;
//!
//! let pipeline = Pipeline::new()
//!     .then(Stage::new("validate", |x: f64| -> HftResult<f64> {
//!         if x.is_nan() { Err(hfthot_lab_core::error::HftError::invalid_input("NaN")) }
//!         else { Ok(x) }
//!     }))
//!     .then(Stage::map("double", |x: f64| x * 2.0))
//!     .then(Stage::map("offset", |x: f64| x + 1.0));
//!
//! assert_eq!(pipeline.run(5.0).unwrap(), 11.0);
//! assert!(pipeline.run(f64::NAN).is_err());
//! ```

pub mod parallel;

use crate::error::{HftError, HftResult};
use std::fmt;

// ── Stage ────────────────────────────────────────────────────────────────────

/// A named, fallible transformation step.
///
/// Each stage carries a human-readable label (for diagnostics) and
/// a function `T → HftResult<U>`.
pub struct Stage<T, U> {
    /// Human-readable label for this stage (appears in error context).
    pub label: &'static str,
    func: Box<dyn Fn(T) -> HftResult<U> + Send + Sync>,
}

impl<T, U> Stage<T, U> {
    /// Create a stage from a fallible function.
    pub fn new<F>(label: &'static str, f: F) -> Self
    where
        F: Fn(T) -> HftResult<U> + Send + Sync + 'static,
    {
        Stage {
            label,
            func: Box::new(f),
        }
    }

    /// Create a stage from an infallible function (always succeeds).
    pub fn map<F>(label: &'static str, f: F) -> Self
    where
        F: Fn(T) -> U + Send + Sync + 'static,
    {
        Stage {
            label,
            func: Box::new(move |x| Ok(f(x))),
        }
    }

    /// Create a validation stage that checks a predicate.
    ///
    /// Passes the value through on success; returns
    /// [`HftError::InvalidInput`] on failure.
    pub fn guard<F>(label: &'static str, predicate: F, msg: &'static str) -> Stage<T, T>
    where
        F: Fn(&T) -> bool + Send + Sync + 'static,
        T: 'static,
    {
        Stage {
            label,
            func: Box::new(move |x| {
                if predicate(&x) {
                    Ok(x)
                } else {
                    Err(HftError::invalid_input(msg))
                }
            }),
        }
    }

    /// Execute this stage, wrapping any error with the stage label.
    #[inline]
    pub fn run(&self, input: T) -> HftResult<U> {
        (self.func)(input).map_err(|e| {
            HftError::Other(format!("[{}] {}", self.label, e))
        })
    }
}

impl<T, U> fmt::Debug for Stage<T, U> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Stage(\"{}\")", self.label)
    }
}

// ── Pipeline ─────────────────────────────────────────────────────────────────

/// A composable chain of [`Stage`]s operating on a single value type.
///
/// The pipeline is type-erased internally so that stages can be
/// appended dynamically.  For heterogeneous pipelines (different
/// types between stages), use the [`Stage`] combinators directly
/// with `.run()`.
pub struct Pipeline<T> {
    stages: Vec<Stage<T, T>>,
}

impl<T: 'static> Pipeline<T> {
    /// Create an empty pipeline (identity).
    pub fn new() -> Self {
        Pipeline { stages: Vec::new() }
    }

    /// Append a stage to the pipeline.
    pub fn then(mut self, stage: Stage<T, T>) -> Self {
        self.stages.push(stage);
        self
    }

    /// Execute all stages in order, short-circuiting on the first error.
    pub fn run(&self, mut input: T) -> HftResult<T> {
        for stage in &self.stages {
            input = stage.run(input)?;
        }
        Ok(input)
    }

    /// Return the number of stages in this pipeline.
    pub fn len(&self) -> usize {
        self.stages.len()
    }

    /// Is this pipeline empty (zero stages)?
    pub fn is_empty(&self) -> bool {
        self.stages.is_empty()
    }

    /// Return the labels of all stages, in order.
    pub fn stage_labels(&self) -> Vec<&str> {
        self.stages.iter().map(|s| s.label).collect()
    }
}

impl<T: 'static> Default for Pipeline<T> {
    fn default() -> Self {
        Self::new()
    }
}

impl<T: 'static> fmt::Debug for Pipeline<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Pipeline({} stages: {:?})", self.stages.len(), self.stage_labels())
    }
}

// ── Heterogeneous stage chaining ─────────────────────────────────────────────

/// Chain two stages with different intermediate types.
///
/// ```rust
/// use hfthot_lab_core::pipeline::{Stage, chain_stages};
///
/// let parse = Stage::new("parse", |s: &str| {
///     s.parse::<f64>().map_err(|_| hfthot_lab_core::error::HftError::invalid_input("parse"))
/// });
/// let double = Stage::map("double", |x: f64| x * 2.0);
///
/// let result = chain_stages("42.0", &parse, &double);
/// assert_eq!(result.unwrap(), 84.0);
/// ```
pub fn chain_stages<A, B, C>(
    input: A,
    first: &Stage<A, B>,
    second: &Stage<B, C>,
) -> HftResult<C> {
    first.run(input).and_then(|mid| second.run(mid))
}

/// Chain three stages with different intermediate types.
pub fn chain_stages3<A, B, C, D>(
    input: A,
    s1: &Stage<A, B>,
    s2: &Stage<B, C>,
    s3: &Stage<C, D>,
) -> HftResult<D> {
    s1.run(input)
        .and_then(|b| s2.run(b))
        .and_then(|c| s3.run(c))
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_pipeline_is_identity() {
        let p: Pipeline<f64> = Pipeline::new();
        assert_eq!(p.run(42.0).unwrap(), 42.0);
    }

    #[test]
    fn pipeline_chains_stages() {
        let p = Pipeline::new()
            .then(Stage::map("double", |x: f64| x * 2.0))
            .then(Stage::map("add1", |x: f64| x + 1.0));
        assert_eq!(p.run(5.0).unwrap(), 11.0);
    }

    #[test]
    fn pipeline_short_circuits_on_error() {
        let p = Pipeline::new()
            .then(Stage::<f64, f64>::guard("positive", |x: &f64| *x > 0.0, "must be > 0"))
            .then(Stage::map("double", |x: f64| x * 2.0));
        assert!(p.run(-1.0).is_err());
    }

    #[test]
    fn stage_labels_preserved() {
        let p = Pipeline::new()
            .then(Stage::map("a", |x: i32| x))
            .then(Stage::map("b", |x: i32| x))
            .then(Stage::map("c", |x: i32| x));
        assert_eq!(p.stage_labels(), vec!["a", "b", "c"]);
    }

    #[test]
    fn chain_stages_heterogeneous() {
        let parse = Stage::new("parse", |s: &str| {
            s.parse::<f64>().map_err(|_| HftError::invalid_input("not a number"))
        });
        let double = Stage::map("double", |x: f64| x * 2.0);
        assert_eq!(chain_stages("21.0", &parse, &double).unwrap(), 42.0);
    }

    #[test]
    fn guard_stage_passes_valid() {
        let guard = Stage::<f64, f64>::guard("non_nan", |x: &f64| !x.is_nan(), "NaN detected");
        assert_eq!(guard.run(5.0).unwrap(), 5.0);
    }

    #[test]
    fn guard_stage_rejects_invalid() {
        let guard = Stage::<f64, f64>::guard("non_nan", |x: &f64| !x.is_nan(), "NaN detected");
        assert!(guard.run(f64::NAN).is_err());
    }
}
