<p align="center">
  <img src="docs/logo_transparent.png" alt="HFThot Research" width="160"/>
</p>

# HFT Lab Core

[![CI](https://github.com/ThotDjehuty/hfthot-lab-core/actions/workflows/ci.yml/badge.svg)](https://github.com/ThotDjehuty/hfthot-lab-core/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

**Professional-grade quantitative finance primitives in Rust with Python bindings.**

A high-performance library for HFT research providing limit order book analytics, mean-reversion portfolio construction, optimization algorithms, geometric/topological market signals, **agent-based market models (Chiarella-He)**, and **path signature features (Chen / rough paths)** — all usable from both Rust and Python.

## Overview

| Module | Purpose | Key Functions |
|--------|---------|---------------|
| **LOB** | Limit Order Book analytics | Snapshot management, spread/depth/imbalance metrics, market impact |
| **Pricing** | Rough Heston Monte Carlo pricer | `rh_mc_call`, `rh_mc_put`, `rh_greeks`, `rh_vol_smile`, `rh_simulate_paths` |
| **Portfolio** | Portfolio construction & time-series | `cara_optimal_weights_rust`, `sharpe_optimal_weights_rust`, `hurst_exponent_rust` |
| **Optimization** | Model fitting | HMM Baum-Welch, Viterbi, MCMC, differential evolution |
| **Geometric Signals** | Topological market analysis | Gauge curvature, spectral topology, sheaf cohomology, Wasserstein distances |
| **Chiarella** | Agent-based market simulation | `simulate_chiarella_py`, `bifurcation_lambda`, `classify_regime`, rolling dominance |
| **Signature** | Path signature features | `lead_lag_transform`, `path_signature_2d`, `log_signature_2d`, `rolling_signature_features` |
| **Mean Reversion** | Portfolio construction *(proprietary)* | OU estimation, cointegration, backtesting |
| **Sparse Mean Rev** | High-dimensional portfolios *(proprietary)* | Sparse PCA, Box-Tiao decomposition |

## Installation

### Python (via PyO3)

```bash
# From source (recommended)
git clone https://github.com/ThotDjehuty/hfthot-lab-core.git
cd hfthot-lab-core
pip install maturin
maturin develop --release --features python
```

### From Source

```bash
git clone https://github.com/ThotDjehuty/hfthot-lab-core.git
cd hfthot-lab-core

# Pure Rust library
cargo build --release

# Python bindings
pip install maturin
maturin develop --release --features python
```

### Rust (as a dependency)

```toml
# Cargo.toml — pure Rust, no Python dependency
[dependencies]
hfthot-lab-core = { version = "1.5", default-features = false }
```

## Quick Start

### Python

```python
import hfthot_lab_core as hft

# --- Order Book Analytics ---
snapshot = hft.OrderBookSnapshot(
    "2024-01-15T10:30:00Z", "BTC-USD", 12345, "binance",
    bids=[(50000.0, 1.0), (49999.0, 2.0), (49998.0, 3.0)],
    asks=[(50001.0, 1.0), (50002.0, 2.0), (50003.0, 3.0)],
)
analytics = hft.calculate_lob_analytics(snapshot)
print(f"Spread: {analytics.spread_bps:.2f} bps, Imbalance: {analytics.volume_imbalance:.2%}")

# --- HMM Regime Detection ---
params = hft.fit_hmm(returns, n_states=2, n_iterations=100, tolerance=1e-6)
states = hft.viterbi_decode(returns, params)

# --- Chiarella Agent-Based Model ---
result = hft.simulate_chiarella_py(
    initial_price=95.0, fundamental_price=100.0, n_steps=2000,
    beta_f=0.5, beta_c=1.0, gamma=1.5, mu=0.1, sigma=0.05, seed=42,
)
print(f"Regime: {hft.classify_regime(hft.bifurcation_lambda(1.0, 1.5, 0.5, 0.2))}")

# --- Arbitrage Detection via Sheaf Cohomology ---
dim = hft.cohomology_dimension(n_nodes=4, edges=edges, edge_values=rates)
if dim > 0:
    print("Arbitrage opportunity detected!")
    hodge = hft.hodge_decompose(n_nodes=4, edges=edges, edge_values=rates)
    print(f"Cycle component: {hodge.curl}")

# --- Rough Heston Pricing (new in v1.6) ---
call = hft.rh_mc_call(spot=100.0, k=105.0, t=0.25, r=0.02,
                       h=0.1, nu=0.3, rho=-0.7, lambda_=1.5,
                       theta=0.04, v0=0.04, n_paths=50000, n_steps=200)
greeks = hft.rh_greeks(spot=100.0, k=105.0, t=0.25, r=0.02,
                        h=0.1, nu=0.3, rho=-0.7, kappa=1.5,
                        theta=0.04, v0=0.04, is_call=True)
delta, gamma, vega, theta_g, rho_g = greeks
print(f"Call: {call:.4f}  Δ={delta:.4f}  Γ={gamma:.6f}  V={vega:.4f}")

# --- Portfolio Optimisation (new in v1.6) ---
weights = hft.cara_optimal_weights_rust(
    expected_returns=[0.08, 0.12, 0.06],
    covariance_matrix=[[0.04, 0.01, 0.005],
                       [0.01, 0.09, 0.02],
                       [0.005, 0.02, 0.03]],
    gamma=3.0,
)
print(f"CARA weights: {weights['weights']}")

# --- Hurst Exponent (new in v1.6) ---
import numpy as np
hurst = hft.hurst_exponent_rust(np.random.randn(1000))
print(f"H = {hurst['hurst_exponent']:.3f} → {hurst['interpretation']}")

# --- Path Signatures ---
path = hft.lead_lag_transform(prices)
sig = hft.path_signature_2d(path, depth=2)
log_sig = hft.log_signature_2d(path)
```

### Rust

```rust
use hfthot_lab_core::{OrderBookSnapshot, OrderBookLevel, calculate_lob_analytics};
use hfthot_lab_core::{HMMParams, fit_hmm, viterbi_decode, mutual_information};

// Order book analytics
let snapshot = OrderBookSnapshot::new(
    "2024-01-15T10:30:00Z".into(), "BTC-USD".into(), 12345, "binance".into(),
    vec![(50000.0, 1.0), (49999.0, 2.0)],
    vec![(50001.0, 1.0), (50002.0, 2.0)],
);
let analytics = calculate_lob_analytics(&snapshot);
println!("Spread: {:.2} bps", analytics.spread_bps);

// HMM fitting (pure Rust, no Python needed)
let params = fit_hmm(observations, 2, 100, 1e-6);
let states = viterbi_decode(observations, params);
```

## Architecture

```
hfthot-lab-core/
├── src/
│   ├── lib.rs                      # Module re-exports + Python bindings
│   ├── lob.rs                      # Limit Order Book (4 structs, analytics, updates)
│   ├── optimization.rs             # HMM Baum-Welch, Viterbi, MCMC, differential evolution
│   ├── chiarella.rs                # Chiarella-He agent-based model + Python bindings
│   ├── signature.rs                # Chen path signatures, lead-lag, log-signature
│   ├── pricing/                     # Rough Heston MC pricer (options, Greeks, smile)
│   │   └── mod.rs
│   ├── portfolio.rs                 # CARA/Sharpe optimal weights, Hurst exponent
│   ├── meanrev.rs                  # [proprietary] Mean reversion (OU, cointegration, backtesting)
│   ├── sparse_meanrev.rs           # [proprietary] Sparse PCA, Box-Tiao
│   └── geometric_signals/
│       ├── mod.rs                   # Sub-module exports
│       ├── gauge.rs                 # Curvature & holonomy on price graphs
│       ├── spectral.rs              # Graph Laplacian, Fiedler value, spectral entropy
│       ├── cohomology.rs            # Sheaf Laplacian, Hodge decomposition, arbitrage detection
│       ├── information.rs           # Mirror descent, natural gradient, Bregman divergence
│       ├── spd.rs                   # SPD manifold (matrix log/exp, Fréchet mean, geodesics)
│       └── wasserstein.rs           # Optimal transport (W₁, W₂, Sinkhorn, rolling LOB distances)
├── tests/                           # Unit + integration tests
├── Cargo.toml
├── pyproject.toml                   # Maturin build config
└── README.md
```

## Python Connectors

Zero-dependency data connectors for researchers — no API keys required:

```python
# Polymarket CLOB (prediction markets real-time order book)
from hfthot_lab_core.connectors import PolymarketCLOBClient
client = PolymarketCLOBClient()
markets = client.get_markets(limit=10)
book    = client.get_order_book(token_id="0xabc...")
print(f"Mid: {book.mid_price:.4f}  Spread: {book.spread:.6f}  Imbalance: {book.book_imbalance:.3f}")

# Yahoo Finance OHLCV
from hfthot_lab_core.connectors import YahooFinanceClient
yf = YahooFinanceClient()
bars = yf.get_ohlcv("AAPL", interval="1d", range_str="1y")
prices = yf.get_close_prices("BTC-USD", interval="1h", range_str="1mo")
```

## Chiarella Agent-Based Model

Simulate heterogeneous-agent markets (Chiarella & He 2002, Brock & Hommes 1998):

```python
import hfthot_lab_core as hft

# Simulate: 2000 steps, initial=95, fundamental=100
result = hft.simulate_chiarella_py(
    initial_price=95.0,
    fundamental_price=100.0,
    n_steps=2000,
    beta_f=0.5,    # fundamentalist demand sensitivity
    beta_c=1.0,    # chartist trend sensitivity
    gamma=1.5,     # agent switching rate
    mu=0.1,        # market impact
    sigma=0.05,    # noise volatility
    seed=42,
)
prices = result["prices"]
n_fund = result["fundamentalist_fractions"]

# Stability analysis
lam    = hft.bifurcation_lambda(alpha=1.0, gamma=1.5, beta=0.5, delta=0.2)
regime = hft.classify_regime(lam)   # "stable" | "mixed" | "unstable"
print(f"Λ = {lam:.3f} → {regime}")
```

See [`examples/notebooks/chiarella_agent_based.ipynb`](examples/notebooks/chiarella_agent_based.ipynb) for a full tutorial.

## Path Signature Methods

Universal features from rough paths theory (Chen 1954, Lyons 1998):

```python
import hfthot_lab_core as hft
import numpy as np

prices = np.array([100., 101.2, 99.8, 102.1, 101.5])  # or any price series

# Lead-lag transform (Chevyrev & Kormilitzin 2016)
path = hft.lead_lag_transform(prices)               # shape (2n-1, 2)

# Path signature up to depth 2 (6 terms for 2D path)
sig  = hft.path_signature_2d(path, depth=2)         # [S¹, S², S¹¹, S¹², S²¹, S²²]

# Log-signature (Lévy area = S¹²-S²¹)/2 encodes quadratic covariation
log_sig = hft.log_signature_2d(path)                # [l¹, l², Lévy area]

# Rolling feature matrix for ML
feats = hft.prices_to_signature_features_py(prices, window=30, sig_depth=2, normalise=True)
print(f"Feature matrix: {feats.shape}")
```

See [`examples/notebooks/path_signatures_trading.ipynb`](examples/notebooks/path_signatures_trading.ipynb) for a full tutorial.

## Feature Flags

| Flag | Default | Description |
|------|---------|-------------|
| `python` | **yes** | PyO3 bindings for Python interop |
| `proprietary` | **no** | Enables `meanrev` and `sparse_meanrev` modules (alpha-generation code) |

Without `python`, the library compiles as a pure Rust crate with zero Python dependencies — ideal for embedding in Rust trading systems.

Without `proprietary`, all mean-reversion and sparse portfolio modules are excluded from the build — the public release contains only infrastructure, academic math, agent-based models, **Rough Heston pricing**, **CARA/Sharpe portfolio optimisation**, and **Hurst exponent analysis**.

## Geometric Signals

A unique module applying differential geometry and algebraic topology to financial markets:

- **Gauge Theory** — Model price ratios as connections on a fiber bundle. Non-zero curvature = statistical arbitrage signal.
- **Spectral Topology** — Graph Laplacian eigenvectors reveal market clustering. Fiedler value tracks connectivity.
- **Sheaf Cohomology** — Hodge decomposition separates price flows into gradient (no-arb), curl (cycle arbitrage), and harmonic (structural) components. `dim H¹ > 0` ⟺ arbitrage exists.
- **Information Geometry** — Mirror descent on the simplex, natural gradient for portfolio optimization, Bregman divergences.
- **SPD Manifold** — Log-Euclidean metrics on covariance matrices, Fréchet means, geodesic interpolation.
- **Optimal Transport** — Wasserstein distances between order book distributions, Sinkhorn regularization for efficient computation.

## Performance

Built in Rust for maximum throughput:

- LOB snapshot analytics in **< 1 μs**
- HMM Baum-Welch convergence **10-100× faster** than Python statsmodels
- Wasserstein/Sinkhorn distances computed without Python overhead
- Zero-copy integration with NumPy arrays via PyO3

## Development

```bash
# Build & test (pure Rust)
cargo test --no-default-features

# Build & test (with Python bindings)
cargo test --features python

# Lint
cargo clippy --all-targets --all-features

# Format
cargo fmt
```

## Streamlit Lab Integration

Every Rust module has a corresponding interactive lab in the **[HFThot Research Lab](https://hfthot-lab.eu)** Streamlit platform. Notebooks and lab pages share parameters through a bridge documented in [`docs/notebook_streamlit_sync.md`](docs/notebook_streamlit_sync.md).

| Notebook (`examples/notebooks/`) | Streamlit Lab page | Rust module |
|---|---|---|
| `kalman_filter_market_making.ipynb` | Market Making Lab | `kalman` |
| `chiarella_agent_based.ipynb` | Chiarella Dynamics Lab | `chiarella` |
| `path_signatures_trading.ipynb` | Path Signatures Lab | `signature` |
| `mean_field_games_portfolio.ipynb` | MFG Portfolio Lab | `geometric_signals` |
| `geometric_arbitrage_polymarket.ipynb` | Per-Market Arbitrage Lab | `lob` / `geometric_signals` |
| `mev_defi_signal_feed.ipynb` | MEV / DeFi Signal Lab | `lob` |

> **Sync protocol:** Run `jupyter nbconvert --execute examples/notebooks/<name>.ipynb --to notebook --inplace` to validate a notebook before any platform release. CI enforces this for all notebooks on every push.

## Contributing

Contributions welcome! See [CONTRIBUTING.md](CONTRIBUTING.md).

Areas of interest:
- Additional geometric signal indicators
- More exchange-specific LOB parsers
- Performance benchmarks and optimization
- Documentation and examples

## License

MIT — see [LICENSE](LICENSE).

## Ecosystem

| Project | Description |
|---------|-------------|
| **[Polarway](https://github.com/ThotDjehuty/polarway)** | High-performance time-series storage (18× compression, Parquet + DuckDB) |
| **[Optimiz-R](https://github.com/ThotDjehuty/optimiz-r)** | Rust optimization library (DE, HMM, MCMC — 50-100× faster than SciPy) |
| **HFT Lab Core** (this repo) | Quantitative finance primitives |

## Platform

The open-source library powers the **[HFThot Research Lab](https://hfthot-lab.eu)** platform — a Streamlit-based quantitative research environment built on top of the Rust core. See the [live demo](https://hfthot-lab.eu/app/) with guest access.

---

*Built by the HFThot Research Lab team*
