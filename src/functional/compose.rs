//! # Function composition and piping combinators
//!
//! Pure, zero-cost combinators for building data-processing pipelines
//! from small, reusable functions.
//!
//! ## Railway orientation
//!
//! All combinators propagate errors automatically: if any stage
//! returns `Err`, subsequent stages are skipped and the error is
//! forwarded to the caller.
//!
//! ## Parallelism note
//!
//! Because every function here is pure (no shared mutable state),
//! composed pipelines can be safely distributed across Rayon threads
//! via the [`crate::pipeline`] module.
//!
//! # Examples
//!
//! ```rust
//! use hfthot_lab_core::functional::compose::*;
//!
//! let double = |x: f64| x * 2.0;
//! let inc    = |x: f64| x + 1.0;
//!
//! // Left-to-right: double then increment
//! let f = pipe(double, inc);
//! assert_eq!(f(5.0), 11.0);
//!
//! // Right-to-left (mathematical): inc ∘ double
//! let g = compose(inc, double);
//! assert_eq!(g(5.0), 11.0);
//! ```

// ── Identity ─────────────────────────────────────────────────────────────────

/// The identity function — returns its argument unchanged.
///
/// Useful as a default "no-op" stage in generic pipeline construction.
#[inline]
pub fn identity<T>(x: T) -> T {
    x
}

// ── Pipe (left-to-right composition) ─────────────────────────────────────────

/// Compose two functions left-to-right: `pipe(f, g)(x) == g(f(x))`.
///
/// This is the natural reading order for data pipelines:
/// *first* `f`, *then* `g`.
#[inline]
pub fn pipe<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> C
where
    F: Fn(A) -> B,
    G: Fn(B) -> C,
{
    move |a| g(f(a))
}

/// Compose three functions left-to-right: `pipe3(f, g, h)(x) == h(g(f(x)))`.
#[inline]
pub fn pipe3<A, B, C, D, F, G, H>(f: F, g: G, h: H) -> impl Fn(A) -> D
where
    F: Fn(A) -> B,
    G: Fn(B) -> C,
    H: Fn(C) -> D,
{
    move |a| h(g(f(a)))
}

// ── Compose (right-to-left, mathematical) ────────────────────────────────────

/// Compose two functions right-to-left: `compose(f, g)(x) == f(g(x))`.
///
/// Matches the mathematical convention `(f ∘ g)(x) = f(g(x))`.
#[inline]
pub fn compose<A, B, C, F, G>(f: F, g: G) -> impl Fn(A) -> C
where
    G: Fn(A) -> B,
    F: Fn(B) -> C,
{
    move |a| f(g(a))
}

// ── Chain (Result-aware left-to-right composition) ───────────────────────────

/// Chain two fallible functions on the success rail.
///
/// `chain(f, g)(x)` is equivalent to `f(x).and_then(g)`.
/// If `f` returns `Err`, `g` is never called.
///
/// ```rust
/// use hfthot_lab_core::functional::compose::chain;
/// use hfthot_lab_core::error::{HftError, HftResult};
///
/// let parse = |s: &str| -> HftResult<f64> {
///     s.parse::<f64>().map_err(|_| HftError::invalid_input("not a number"))
/// };
/// let validate = |x: f64| -> HftResult<f64> {
///     if x > 0.0 { Ok(x) } else { Err(HftError::invalid_input("must be positive")) }
/// };
///
/// let pipeline = chain(parse, validate);
/// assert!(pipeline("42.0").is_ok());
/// assert!(pipeline("-1.0").is_err());
/// assert!(pipeline("abc").is_err());
/// ```
#[inline]
pub fn chain<A, B, C, E, F, G>(f: F, g: G) -> impl Fn(A) -> Result<C, E>
where
    F: Fn(A) -> Result<B, E>,
    G: Fn(B) -> Result<C, E>,
{
    move |a| f(a).and_then(|b| g(b))
}

/// Chain three fallible functions on the success rail.
#[inline]
pub fn chain3<A, B, C, D, E, F, G, H>(f: F, g: G, h: H) -> impl Fn(A) -> Result<D, E>
where
    F: Fn(A) -> Result<B, E>,
    G: Fn(B) -> Result<C, E>,
    H: Fn(C) -> Result<D, E>,
{
    move |a| f(a).and_then(|b| g(b)).and_then(|c| h(c))
}

// ── Tap (side-effect without modifying the pipeline) ─────────────────────────

/// Run a side-effect on the success value without consuming it.
///
/// Useful for logging or metrics in the middle of a pipeline:
///
/// ```rust
/// use hfthot_lab_core::functional::compose::tap;
///
/// let logged = tap(|x: &f64| println!("value = {x}"));
/// let result: Result<f64, String> = Ok(42.0).map(|x| { logged(&x); x });
/// ```
#[inline]
pub fn tap<T, F>(f: F) -> impl Fn(&T)
where
    F: Fn(&T),
{
    move |x| f(x)
}

/// Run a side-effect on the `Ok` branch of a `Result`, returning
/// the same `Result` unchanged.
pub fn tap_ok<T: Clone, E, F>(result: Result<T, E>, f: F) -> Result<T, E>
where
    F: FnOnce(&T),
{
    if let Ok(ref val) = result {
        f(val);
    }
    result
}

/// Run a side-effect on the `Err` branch of a `Result`, returning
/// the same `Result` unchanged.
pub fn tap_err<T, E: Clone, F>(result: Result<T, E>, f: F) -> Result<T, E>
where
    F: FnOnce(&E),
{
    if let Err(ref err) = result {
        f(err);
    }
    result
}

// ── Retry combinator ─────────────────────────────────────────────────────────

/// Retry a fallible operation up to `max_attempts` times.
///
/// Returns the first `Ok`, or the last `Err` if all attempts fail.
/// The closure receives the 0-based attempt index.
///
/// ```rust
/// use hfthot_lab_core::functional::compose::retry;
///
/// let mut counter = 0u32;
/// let result: Result<u32, &str> = retry(3, |attempt| {
///     if attempt < 2 { Err("not yet") } else { Ok(42) }
/// });
/// assert_eq!(result, Ok(42));
/// ```
pub fn retry<T, E, F>(max_attempts: usize, mut f: F) -> Result<T, E>
where
    F: FnMut(usize) -> Result<T, E>,
{
    let mut last_err = None;
    for attempt in 0..max_attempts {
        match f(attempt) {
            Ok(val) => return Ok(val),
            Err(e) => last_err = Some(e),
        }
    }
    Err(last_err.expect("max_attempts must be > 0"))
}

// ══════════════════════════════════════════════════════════════════════════════
// Tests
// ══════════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pipe_two_functions() {
        let double = |x: i32| x * 2;
        let inc = |x: i32| x + 1;
        let f = pipe(double, inc);
        assert_eq!(f(5), 11);
    }

    #[test]
    fn pipe3_three_functions() {
        let f = pipe3(|x: i32| x + 1, |x| x * 2, |x| x - 3);
        assert_eq!(f(5), 9); // (5+1)*2 - 3 = 9
    }

    #[test]
    fn compose_mathematical_order() {
        let double = |x: i32| x * 2;
        let inc = |x: i32| x + 1;
        // compose(f, g)(x) = f(g(x)) = double(inc(5)) = 12
        let f = compose(double, inc);
        assert_eq!(f(5), 12);
    }

    #[test]
    fn chain_success_path() {
        let f = |x: i32| -> Result<i32, &str> { Ok(x + 1) };
        let g = |x: i32| -> Result<i32, &str> { Ok(x * 2) };
        let pipeline = chain(f, g);
        assert_eq!(pipeline(5), Ok(12));
    }

    #[test]
    fn chain_error_path() {
        let f = |_x: i32| -> Result<i32, &str> { Err("boom") };
        let g = |x: i32| -> Result<i32, &str> { Ok(x * 2) };
        let pipeline = chain(f, g);
        assert!(pipeline(5).is_err());
    }

    #[test]
    fn identity_returns_input() {
        assert_eq!(identity(42), 42);
        assert_eq!(identity("hello"), "hello");
    }

    #[test]
    fn retry_succeeds_on_third_attempt() {
        let result: Result<i32, &str> = retry(5, |attempt| {
            if attempt < 2 {
                Err("not yet")
            } else {
                Ok(100)
            }
        });
        assert_eq!(result, Ok(100));
    }

    #[test]
    fn retry_fails_after_exhausting_attempts() {
        let result: Result<i32, &str> = retry(2, |_| Err("always fails"));
        assert_eq!(result, Err("always fails"));
    }

    #[test]
    fn tap_ok_runs_side_effect() {
        let mut observed = 0;
        let r: Result<i32, &str> = Ok(42);
        let _ = tap_ok(r, |v| observed = *v);
        assert_eq!(observed, 42);
    }

    #[test]
    fn tap_ok_skips_on_err() {
        let mut observed = 0;
        let r: Result<i32, &str> = Err("oops");
        let _ = tap_ok(r, |v| observed = *v);
        assert_eq!(observed, 0);
    }
}
