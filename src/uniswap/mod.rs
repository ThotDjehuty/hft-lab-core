//! # Uniswap read-only connector primitives (PUBLIC)
//!
//! Public Rust primitives for querying Uniswap v2/v3/v4 pool state and
//! quoter contracts, plus subgraph helpers. **Read-only** — no order
//! placement (the MEV-protected router lives in
//! `hfthot-lab-strategies/app/services/quant/dex/`).
//!
//! Phase C0: skeleton.

pub mod quoter;
pub mod pool_state;
pub mod subgraph;
pub mod v2_math;
pub mod v3_math;
pub mod cyclic_arb;
pub mod fee_asymptotics;

#[derive(Debug, thiserror::Error)]
pub enum UniswapError {
    #[error("not implemented (Phase C0 skeleton)")]
    NotImplemented,
    #[error("rpc error: {0}")]
    Rpc(String),
    #[error("decode error: {0}")]
    Decode(String),
    #[error("invalid input (non-positive reserves or amount)")]
    InvalidInput,
}

pub type UniswapResult<T> = Result<T, UniswapError>;
