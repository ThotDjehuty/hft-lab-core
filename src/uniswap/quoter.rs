//! Uniswap quoter helper — wraps the on-chain Quoter contract.
//!
//! Phase C0 skeleton.

use super::{UniswapError, UniswapResult};

pub fn quote_exact_input(
    _token_in: &str,
    _token_out: &str,
    _amount_in: u128,
    _fee_tier: u32,
) -> UniswapResult<u128> {
    Err(UniswapError::NotImplemented)
}
