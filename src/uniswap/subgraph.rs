//! Uniswap subgraph helpers (GraphQL over HTTPS).
//!
//! Phase C0 skeleton.

use super::{UniswapError, UniswapResult};

pub fn top_pools_by_volume(_limit: usize) -> UniswapResult<Vec<String>> {
    Err(UniswapError::NotImplemented)
}
