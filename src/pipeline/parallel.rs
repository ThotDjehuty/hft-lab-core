//! # Parallel combinators for batch data processing
//!
//! Rayon-backed parallel `map`, `filter_map`, and `fold` that integrate
//! with the crate's [`HftResult`] error type.
//!
//! ## Thread safety
//!
//! All functions in this module require `Send + Sync` bounds on the
//! closure and data.  Because the functional pipeline is pure (no
//! shared mutable state), these bounds are satisfied automatically
//! for any closure that captures only owned or `Arc`-wrapped data.
//!
//! ## GIL interaction
//!
//! When called from Python (via PyO3), wrap calls in
//! `py.allow_threads(|| ...)` to release the GIL and allow true
//! multi-core execution.
//!
//! ## Examples
//!
//! ```rust
//! use hfthot_lab_core::pipeline::parallel::*;
//! use hfthot_lab_core::error::{HftError, HftResult};
//!
//! let prices = vec![100.0, 200.0, 300.0, 0.0, 500.0];
//!
//! // par_map — parallel transform (fails fast on error)
//! let log_prices = par_map(&[100.0_f64, 200.0, 300.0], |p| {
//!     if *p <= 0.0 { Err(HftError::invalid_input("non-positive price")) }
//!     else { Ok(f64::ln(*p)) }
//! });
//! assert!(log_prices.is_ok());
//!
//! // par_filter_map — parallel filter + transform
//! let positive = par_filter_map(&prices, |p| {
//!     if *p > 0.0 { Some(f64::ln(*p)) } else { None }
//! });
//! assert_eq!(positive.len(), 4);
//! ```

use crate::error::{HftError, HftResult};
use rayon::prelude::*;

// ── par_map ──────────────────────────────────────────────────────────────────

/// Apply a fallible function to every element in parallel.
///
/// Returns `Ok(Vec<U>)` if all invocations succeed, or the first
/// `Err` encountered (Rayon will still complete in-flight work but
/// additional errors are discarded).
///
/// The output order matches the input order.
pub fn par_map<T, U, F>(items: &[T], f: F) -> HftResult<Vec<U>>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> HftResult<U> + Sync,
{
    items
        .par_iter()
        .map(|item| f(item))
        .collect::<Result<Vec<_>, _>>()
}

/// Apply an infallible function to every element in parallel.
///
/// This is the parallel equivalent of `items.iter().map(f).collect()`.
pub fn par_map_ok<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> U + Sync,
{
    items.par_iter().map(|item| f(item)).collect()
}

// ── par_filter_map ───────────────────────────────────────────────────────────

/// Parallel filter + transform.  Elements for which `f` returns `None`
/// are dropped; the rest are collected.
///
/// Output order matches the input order.
pub fn par_filter_map<T, U, F>(items: &[T], f: F) -> Vec<U>
where
    T: Sync,
    U: Send,
    F: Fn(&T) -> Option<U> + Sync,
{
    items.par_iter().filter_map(|item| f(item)).collect()
}

// ── par_fold ─────────────────────────────────────────────────────────────────

/// Parallel fold with an associative binary operation.
///
/// `identity` is the neutral element (e.g. `0.0` for sum, `1.0` for
/// product).  `fold_fn` combines two accumulators; it *must* be
/// associative and commutative for deterministic results.
///
/// ```rust
/// use hfthot_lab_core::pipeline::parallel::par_fold;
///
/// let data = vec![1.0_f64, 2.0, 3.0, 4.0, 5.0];
/// let sum = par_fold(&data, 0.0, |acc, x| acc + x, |a, b| a + b);
/// assert!((sum - 15.0).abs() < 1e-10);
/// ```
pub fn par_fold<T, U, FoldFn, ReduceFn>(
    items: &[T],
    identity: U,
    fold_fn: FoldFn,
    reduce_fn: ReduceFn,
) -> U
where
    T: Sync,
    U: Send + Sync + Clone,
    FoldFn: Fn(U, &T) -> U + Sync,
    ReduceFn: Fn(U, U) -> U + Sync,
{
    items
        .par_iter()
        .fold(
            || identity.clone(),
            |acc, item| fold_fn(acc, item),
        )
        .reduce(|| identity.clone(), |a, b| reduce_fn(a, b))
}

// ── par_chunks ───────────────────────────────────────────────────────────────

/// Process data in parallel chunks, returning one result per chunk.
///
/// Useful for windowed operations (rolling statistics, block
/// decompositions) where each chunk can be processed independently.
pub fn par_chunks<T, U, F>(items: &[T], chunk_size: usize, f: F) -> HftResult<Vec<U>>
where
    T: Sync,
    U: Send,
    F: Fn(&[T]) -> HftResult<U> + Sync,
{
    if chunk_size == 0 {
        return Err(HftError::invalid_input("chunk_size must be > 0"));
    }
    items
        .par_chunks(chunk_size)
        .map(|chunk| f(chunk))
        .collect::<Result<Vec<_>, _>>()
}

// ── par_zip_map ──────────────────────────────────────────────────────────────

/// Apply a binary function element-wise to two slices in parallel.
///
/// Returns `Err` if the slices have different lengths.
pub fn par_zip_map<T, U, V, F>(a: &[T], b: &[U], f: F) -> HftResult<Vec<V>>
where
    T: Sync,
    U: Sync,
    V: Send,
    F: Fn(&T, &U) -> V + Sync,
{
    if a.len() != b.len() {
        return Err(HftError::dimension_mismatch(a.len(), b.len()));
    }
    let out: Vec<V> = a
        .par_iter()
        .zip(b.par_iter())
        .map(|(ai, bi)| f(ai, bi))
        .collect();
    Ok(out)
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn par_map_all_ok() {
        let data = vec![1.0, 4.0, 9.0, 16.0];
        let roots = par_map(&data, |x| Ok(f64::sqrt(*x))).unwrap();
        assert_eq!(roots, vec![1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn par_map_propagates_error() {
        let data = vec![1.0, -1.0, 4.0];
        let result = par_map(&data, |x| {
            if *x < 0.0 {
                Err(HftError::invalid_input("negative"))
            } else {
                Ok(f64::sqrt(*x))
            }
        });
        assert!(result.is_err());
    }

    #[test]
    fn par_map_ok_works() {
        let data = vec![1, 2, 3];
        let doubled = par_map_ok(&data, |x| x * 2);
        assert_eq!(doubled, vec![2, 4, 6]);
    }

    #[test]
    fn par_filter_map_drops_nones() {
        let data = vec![1, 2, 3, 4, 5];
        let evens: Vec<i32> = par_filter_map(&data, |x| {
            if x % 2 == 0 { Some(*x) } else { None }
        });
        assert_eq!(evens, vec![2, 4]);
    }

    #[test]
    fn par_fold_sum() {
        let data: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let sum = par_fold(&data, 0.0, |acc, x| acc + x, |a, b| a + b);
        assert!((sum - 5050.0).abs() < 1e-10);
    }

    #[test]
    fn par_chunks_windows() {
        let data: Vec<f64> = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let means = par_chunks(&data, 3, |chunk| {
            let sum: f64 = chunk.iter().sum();
            Ok(sum / chunk.len() as f64)
        })
        .unwrap();
        assert!((means[0] - 2.0).abs() < 1e-10);
        assert!((means[1] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn par_chunks_zero_size_errors() {
        let data = vec![1.0];
        assert!(par_chunks(&data, 0, |_| Ok(0.0)).is_err());
    }

    #[test]
    fn par_zip_map_adds() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![10.0, 20.0, 30.0];
        let c = par_zip_map(&a, &b, |x, y| x + y).unwrap();
        assert_eq!(c, vec![11.0, 22.0, 33.0]);
    }

    #[test]
    fn par_zip_map_length_mismatch() {
        let a = vec![1.0, 2.0];
        let b = vec![1.0];
        assert!(par_zip_map(&a, &b, |x, y| x + y).is_err());
    }
}
