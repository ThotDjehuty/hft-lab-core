//! Uniswap v3/v4 pool state reader (`slot0`, liquidity, tick bitmap).
//!
//! Phase C0 skeleton.

use super::{UniswapError, UniswapResult};

#[derive(Debug, Clone)]
pub struct PoolSnapshot {
    pub pool_address: String,
    pub sqrt_price_x96: u128,
    pub tick: i32,
    pub liquidity: u128,
}

pub fn read_snapshot(_pool_address: &str) -> UniswapResult<PoolSnapshot> {
    Err(UniswapError::NotImplemented)
}
