//! # Functional Combinators
//!
//! Practical, zero-overhead composition tools for railway-oriented
//! programming.  Values travel on the **success rail** ([`Ok`]) or the
//! **error rail** ([`Err`]) and transformations compose without manual
//! `match` or `unwrap`.
//!
//! Rust already provides `Result::map`, `Result::and_then`, and the `?`
//! operator.  This module extends those with domain-specific helpers:
//!
//! | Module | Provides |
//! |--------|----------|
//! | [`compose`] | `pipe`, `chain`, `retry`, `tap` — function combinators |
//! | [`result_ext`] | `with_context`, `ensure`, `or_recover` — [`HftResult`] extensions |
//!
//! ## Quick start
//!
//! ```rust
//! use hfthot_lab_core::functional::prelude::*;
//! use hfthot_lab_core::error::{HftError, HftResult};
//!
//! fn validate(x: f64) -> HftResult<f64> {
//!     if x.is_nan() { Err(HftError::invalid_input("NaN")) }
//!     else { Ok(x) }
//! }
//!
//! let result = Ok(42.0_f64)
//!     .and_then_with_context(|v| validate(v), "validation")
//!     .map(|v| v * 2.0)
//!     .unwrap_or(0.0);
//! ```

pub mod compose;
pub mod result_ext;

/// Convenience re-exports for `use hfthot_lab_core::functional::prelude::*`.
pub mod prelude {
    pub use super::compose::{pipe, compose, chain, identity};
    pub use super::result_ext::ResultExt;
    pub use crate::error::{HftError, HftResult};
}
