# Changelog

All notable changes to `hfthot-lab-core` are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).  
Versioning follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [3.0.0] — 2026-07-06

**Open-core split (BREAKING).** The public repo now contains only generic
applied finance & mathematics. All platform/commercial components moved to
the private platform crate.

### Removed (moved to private `hfthot-lab-platform`)
- **`session`** — lab session persistence (platform glue, not quant math).
- **`usage`** — API-key usage metering & demo quotas (commercial).
- **`lms`** — LMS prerequisite graph (platform feature).
- **`client_sdk`** — dark-pool client SDK skeletons (operator-specific).
- `docs/COMMERCIALISATION_CHECKLIST.md`.

### Changed
- **`run_backtest(prices, signals, config)`** — the `tracker` parameter and
  quota gate are gone; the backtest engine is now a pure function (BREAKING).
- `pyo3` built with `abi3-py38`: one wheel covers CPython 3.8–3.13.
- Python module no longer exports `save_lab_state` / `load_lab_state` /
  `PrereqGraph` — import them from the platform wheel instead.

---

## [2.0.0] — 2026-05-18

Major release: public Plan-C v2 modules (LMS prereq-graph, Polymarket Gamma client, Uniswap v2/v3 quant primitives), full test green.

### Added
- **`lms`** — public `PrereqGraph` for the LMS lesson DAG (Phase 2: cycle detection, topological order, ready-set queries).
- **`client_sdk`** — thin shared HTTP scaffolding for Polymarket / Uniswap clients (Plan C v2).
- **`polymarket`** — `PolymarketGammaClient` with search / events / markets endpoints.
- **`uniswap`** — v2 constant-product + v3 concentrated-liquidity math, cyclic-arb detection, Echenim-Gobet fee asymptotics, MEV public quant primitives.
- **Notebooks**: `gauge_theory_cross_asset_arbitrage.ipynb`, `chiarella_agent_based.ipynb` refreshed; `mev_defi_signal_feed.ipynb` (new), `path_signatures_trading.ipynb` (new), `mean_field_games_portfolio.ipynb` (uses OptimizR Rust solver).
- **`docs/notebook_streamlit_sync.md`** — documents the public notebook ↔ Streamlit-page sync model.

### Changed
- **`mean_field_games_portfolio.ipynb`** — numpy reimplementation removed; now uses OptimizR Rust solver throughout.
- Repository cleanup: stale `.bak` / `.REMOVED` notebook files removed.
- Version bump: 1.6.0 → 2.0.0.

### Tests
- **153 / 153 passing** (`cargo test --release --lib --no-fail-fast`).

---

## [1.6.0] — 2026-06-08

### Added

- **`pricing` module** — Rough Heston Monte Carlo engine (El Euch & Rosenbaum 2019) with 8 Python-callable functions:
  - `rh_mc_call` / `rh_mc_put` — European option pricing via parallel MC (Rayon)
  - `rh_greeks` — Delta, Gamma, Vega, Theta, Rho via bump-and-reprice
  - `rh_vol_smile` — Volatility smile across strikes at fixed maturity
  - `rh_simulate_paths` — Terminal (S_T, V_T) pairs for path analysis
  - `rh_atm_skew_mc` — ATM implied skew via finite-difference on MC
  - `rh_variance_swap` — Analytical variance swap rate (first-order)
  - `rh_leverage_swap` — Analytical leverage swap proxy (ρ·ν·VarSwap)

- **`portfolio` module** — Public academic portfolio construction and time-series primitives (no feature gate):
  - `cara_optimal_weights_rust` — CARA utility-maximising portfolio weights (w* = (1/γ)·Σ⁻¹·μ)
  - `sharpe_optimal_weights_rust` — Maximum Sharpe ratio portfolio (w* ∝ Σ⁻¹·(μ−r_f))
  - `hurst_exponent_rust` — Rescaled-range Hurst exponent with 95% confidence interval

### Changed

- **Promoted `cara_optimal_weights_rust`, `sharpe_optimal_weights_rust`, `hurst_exponent_rust`** from proprietary-gated modules to the public `portfolio` module — these are standard academic algorithms (Markowitz mean-variance, rescaled-range R/S analysis) and belong in the open-source API
- Updated README with new module table, quick-start examples, and architecture diagram
- Version bump: 1.5.0 → 1.6.0

---

## [1.4.0] — 2026-03-23

### Added

- **Functional programming foundation** — Composable data pipeline primitives with Railway-Oriented error handling
- **Usage tracking & analytics** — Built-in usage counters, quota enforcement, and API key analytics
- **Backtesting framework** — End-to-end backtesting engine for strategy evaluation
- **Real market data notebooks** — All example notebooks now use real Yahoo Finance data instead of synthetic data

### Changed

- **Refactored functional modules** — Removed academic functor/validated modules in favor of practical composable patterns

### Security

- **Jupyter token auth** — Added `JUPYTER_TOKEN` env var for server-side notebook authentication

### Documentation

- Added commercialisation go-live checklist
- Updated README with logo and commercialisation checklist

---

## [1.3.0] — 2026-03-09

### Added

- **`docs/notebook_streamlit_sync.md`** — Canonical reference documenting how every
  research notebook in `examples/notebooks/` maps to its corresponding Streamlit lab page:
  - Notebook → Lab page mapping table with direct links
  - Sync protocol: how cell parameters propagate to Streamlit sidebar widgets
  - Parameter bridge pattern (`st.session_state` ↔ notebook variables)
  - Reproducibility checklist: kernel version, library versions, dataset snapshot
  - CI guidance: `jupyter nbconvert --execute` gate per notebook in `ci.yml`

- **`RELEASE_NOTES_v1.3.0.md`** — Standalone release notes for distribution to early
  adopters, beta testers and university lab partners

### Changed

#### Platform (Streamlit App — `hfthot-lab.eu`)

- **Home page — Quick Start** — Replaced single-line `st.info()` step list with a
  4-column visual step-card grid (icons, colour-coded step badges, short descriptions)
  matching the onboarding flow: Load Data → Choose Lab → Backtest → Trade Live

- **Home page — sidebar** — Tier/role/badge widgets hidden for admin users on
  private self-hosted instances (not relevant for single-user deployments)

- **Home page — noise reduction** — Removed redundant RUST ACCELERATION success
  banner duplicating the System Status row directly below it

- **Arbitrage Lab — Overview tab** — Added collapsible `💡 How to interpret these
  metrics` expander with a full KPI reference table (metric, definition, healthy range)
  for all 9 metrics in the 3×3 command-center grid

- **Arbitrage Lab — no-data gate** — Replaced plain `st.info()` dead-end with a
  rich dark-gradient card (icon + description + primary `🚀 Go to Data Loader` CTA button)

- **Navigation** — All `st.switch_page("HFT_Arbitrage_Lab.py")` references replaced
  with `st.switch_page("pages/home.py")` across 7 files; eliminates `NavigationError`
  caused by routing to the entry-point script instead of a navigable page

#### Documentation

- Internal architecture diagrams converted from ASCII art to clean Mermaid diagrams
  (9 diagrams) and markdown tables

### Test Results

```
37 passed, 8 skipped in 6.90s
```
*(full test suite: `tests/test_arb_mechanics.py` + `tests/test_styler_safety.py`)*

---

## [1.2.1] — 2026-02-26

### Added

- **`docs/kalman.md`** — Full mathematical reference for Kalman filter market making  
  - Linear state-space model derivation (transition + observation equations)  
  - Kalman predict/update recursion with Joseph-form covariance  
  - Microstructure noise model and efficient price decomposition  
  - Gaussian log-likelihood for parameter estimation (MLE / EM)  
  - Avellaneda-Stoikov optimal spread and reservation price  
  - Order-flow imbalance AR(1) state and Bayesian interpretation  
  - RTS smoother equations  
  - Full API reference for `LinearKalmanFilter` and `KalmanEnhancedMarketMaker`  
  - Data pipeline diagram and performance metrics formulae  
  - References: Kalman (1960), Avellaneda-Stoikov (2008), Roll (1984), Kyle (1985), Cont et al. (2014)

### Changed

- **`kalman_filter_market_making.ipynb`** — Replaced synthetic `MarketMicrostructureSimulator`  
  with **real Binance ETH/USDT 1-minute OHLCV bars** (3 days ≈ 4 320 bars)  
  - Efficient price: EWM(span=50) of close  
  - Spread proxy: `clip((H−L)/C × 10 000, 1, 80)` bps → dollar units  
  - Imbalance: cumulative signed-volume, normalised to [−1, 1]  
  - `ccxt_helper.py` loaded via `importlib.util.spec_from_file_location` to bypass  
    `fetchers/__init__.py` circular imports  
  - Market-maker parameters tuned for ETH (~$3 000): `base_spread=1.50`, `max_inventory=10`  
  - All 8 cells execute cleanly with real market output and verified outputs committed

- All example notebooks execute with zero errors (verified via `jupyter nbconvert --execute`):  
  `chiarella_agent_based.ipynb`, `kalman_filter_market_making.ipynb`,  
  `mean_field_games_portfolio.ipynb`, `path_signatures_trading.ipynb`

---

## [1.2.0] — 2026-02

### Added

#### Rust Modules

- **`chiarella`** — Chiarella-He (2002) heterogeneous-agent market simulation  
  - `simulate_chiarella()` — full Brock-Hommes discrete-choice agent dynamics  
  - Python binding: `simulate_chiarella_py(initial_price, fundamental_price, n_steps, ...)`  
  - `bifurcation_lambda(alpha, beta, gamma, delta)` — stability parameter $\Lambda$  
  - `classify_regime(lambda)` → `"stable"` / `"mixed"` / `"unstable"`  
  - `rolling_dominance()` — rolling agent-fraction dominance window  
  - `rolling_volatility()` — rolling annualised volatility  
  - 6 unit tests (all passing)  
  - *References: Chiarella & He (2002), Brock & Hommes (1998)*

- **`signature`** — Chen path signatures and lead-lag transforms  
  - `lead_lag_transform(prices)` — Chevyrev-Kormilitzin 2D interleaved embedding (2n-1 steps)  
  - `path_signature_2d(path, depth)` — Chen iterated-integral signature up to depth 3  
    - Depth 1: 2 terms; depth 2: 6 terms; depth 3: 14 terms  
  - `log_signature_2d(path)` — 3-component Lie algebra representation  
  - `rolling_signature_features(prices, window, depth)` → feature matrix  
  - `signature_distance(a, b)` — $L^2$ distance in signature space  
  - `normalise_prices(prices)`, `time_normalise(path)` — path preprocessing utilities  
  - Python binding: `prices_to_signature_features_py(prices, window, sig_depth, normalise)`  
  - 11 unit tests (all passing)  
  - *References: Chen (1954), Lyons (1998), Chevyrev & Kormilitzin (2016), Gyurko et al. (2013)*

#### Python Connectors (`python/connectors/`)

- **`PolymarketCLOBClient`** — public Polymarket CLOB REST client  
  - Zero external dependencies (stdlib `urllib` only)  
  - `get_markets(limit)`, `get_order_book(token_id)`, `get_last_trade_price(token_id)`  
  - `get_recent_trades(token_id)`, `get_multiple_books(token_ids)`  
  - `OrderBookData` with `best_bid`, `best_ask`, `spread`, `mid_price`, `book_imbalance`, `vwap()`

- **`YahooFinanceClient`** — Yahoo Finance v8 OHLCV adapter  
  - Zero external dependencies  
  - `get_ohlcv(symbol, interval, range_str)` → `list[OHLCVBar]`  
  - `get_close_prices(symbol, interval, range_str)` → `np.ndarray`

#### Example Notebooks (`examples/notebooks/`)

- **`chiarella_agent_based.ipynb`** — 25-cell research notebook  
  - Full Chiarella model derivation with LaTeX mathematics  
  - Agent-based simulation (Rust extension + pure-Python fallback)  
  - Bifurcation diagram (sweep $\beta_c$ 0.1→2.5)  
  - Phase portrait & 2D stability heatmap  
  - Rolling regime detection with background shading  
  - Data connector interface demo (`SyntheticConnector`, `PolymarketCLOBClient` stub)  
  - Regime-aware trading signals (fade / momentum / blended)  
  - Path signature feature extraction  
  - Performance analytics (Sharpe, Sortino, Calmar, drawdown)  
  - Auto-generated release notes & API surface validator

- **`path_signatures_trading.ipynb`** — 9-cell deep-dive notebook  
  - Lead-lag transform theory + visualisation  
  - Path signature (depth 1-3) with term-by-term interpretation  
  - Log-signature (Lévy area)  
  - Rolling signature feature matrix + GBM demo  
  - Signature momentum signal + backtest equity curve  
  - Performance comparison (Sharpe vs buy-and-hold)

### Changed

- `Cargo.toml`: version bumped `1.0.0` → `1.2.0`
- `src/lib.rs`: added `pub mod chiarella` and `pub mod signature`; registered Python bindings for both new modules

### Test Results

```
running 48 tests
test result: ok. 48 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```
*(includes 6 new Chiarella tests + 11 new Signature tests + 1 doc-test)*

---

## [1.1.0] — 2026-02-19

### Added
- CI/CD: GitHub Actions workflow (`ci.yml`) — runs `cargo test` on push/PR
- Docker: production `Dockerfile` (multi-stage, musl static binary target)
- Example notebook: `deribit_market_data.ipynb` — Deribit WebSocket data ingestion demo
- CONTRIBUTING.md

### Changed
- PyO3 stable ABI (`abi3-py38`) configured in `pyproject.toml`

---

## [1.0.0] — 2026-02-08

### Initial Release

#### Modules
- **`lob`** — Limit Order Book: snapshot management, spread / depth / imbalance / market impact
- **`meanrev`** — Mean reversion: OU process estimation, cointegration, CARA/Sharpe weights, backtesting
- **`sparse_meanrev`** — Sparse PCA, Box-Tiao decomposition, Hurst exponent
- **`optimization`** — HMM Baum-Welch / Viterbi / MCMC / differential evolution
- **`geometric_signals`** — Gauge curvature, spectral topology, sheaf cohomology, information geometry, SPD manifold, Wasserstein distances

#### Example Notebooks
- `limit_order_book_analytics.ipynb`
- `mean_reversion_portfolio.ipynb`
- `geometric_arbitrage.ipynb`
- `regime_switching.ipynb`
- `sparse_portfolio_construction.ipynb`

---

*hfthot-lab-core is part of the [HFThot Research Lab](https://hfthot-lab.eu) open-source ecosystem.*
