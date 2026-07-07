# TODO & Known Issues

## Compilation Status

**Compilation: FIXED** - Successfully builds on Python 3.13

Changes made:
- Updated PyO3 from 0.21 to 0.22 for Python 3.13 support
- Updated numpy from 0.21 to 0.22
- Replaced deprecated `PyDict::new()` with `PyDict::new_bound()`
- Replaced deprecated `PyArray2::from_vec2()` with `from_vec2_bound()`
- Fixed lib.rs to export actual module contents

## Action Items

- [x] Add `numpy` to Cargo.toml dependencies
- [x] Update PyO3 to 0.22 for Python 3.13 support
- [x] Fix deprecated PyO3 API calls
- [x] Fix lib.rs exports to match actual module contents
- [x] Setup GitHub Actions CI/CD
- [ ] Add comprehensive unit tests
- [ ] Publish to crates.io
- [ ] Publish to PyPI

## v1.3.0 End-to-End Testing Checklist

### Rust Core (48 tests + 1 doc-test)
- [x] `cargo test --features python` — 48 passed, 0 failed
- [x] `chiarella` module — bifurcation, fractions, reproducibility, stable regime (6 tests)
- [x] `geometric_signals` — cohomology, gauge, information, SPD, spectral, Wasserstein (22 tests)
- [x] `signature` module — Chen identity, lead-lag, log/depth, normalise, rolling (11 tests)
- [x] `lob` module — orderbook level creation, snapshot (2 tests)
- [x] Doc-tests — Chiarella example (1 test)

### Notebooks (headless execution)
- [ ] `chiarella_agent_based.ipynb` — run all cells clean
- [ ] `geometric_arbitrage_polymarket.ipynb` — run all cells clean
- [ ] `kalman_filter_market_making.ipynb` — run all cells clean
- [ ] `mean_field_games_portfolio.ipynb` — run all cells clean
- [ ] `mev_defi_signal_feed.ipynb` — run all cells clean
- [ ] `path_signatures_trading.ipynb` — run all cells clean

### Streamlit Lab Pages (manual verification)
- [ ] Home page — 4-column step cards render, sidebar shows only WASM status (private)
- [ ] Data Loader — CSV/Parquet/API ingestion works
- [ ] Arbitrage Lab — KPI expander visible, Data Gate CTA navigates correctly
- [ ] MEV Signal Feed Lab — signals display, GeckoTerminal data loads
- [ ] Navigation — all `st.switch_page("pages/home.py")` links work (7 files)
- [ ] ThotBook Research page — ArXiv search functional

### Bridge & Integration
- [ ] `LabIntegrationBridge` — DuckDB strategy load/leaderboard/export
- [ ] Parameter bridge — notebook defaults match Streamlit sidebar defaults
- [ ] ThotBook MCP — `thotbook_generate` produces valid notebook

### Public Repo Hygiene
- [x] No business plan files tracked
- [x] No Stripe/pricing/monetization code tracked
- [x] No internal documentation tracked
- [x] MIT LICENSE file present
- [x] .gitignore blocks re-addition of commercial files
- [x] No stale backup files tracked
- [x] README contains no pricing table

### Private Lab Sync (http://lab.thotprivatecloud.mel/)
- [x] Tier badges hidden for admin user
- [x] Pricing banner removed from login page
- [x] DEPLOYMENT_TYPE=private set in docker-compose

## Current Status

**Phase 1 Complete:**
- ✅ Repository structure created
- ✅ MIT license added
- ✅ README documentation
- ✅ Source files extracted from hfthot-lab-strategies
- ✅ Examples and notebooks copied

**Phase 2 In Progress:**
- ✅ Fix compilation errors
- ✅ CI/CD GitHub Actions
- ⏳ Add Python bindings tests
- ⏳ Complete documentation (missing: lob, meanrev, sparse_meanrev, optimization, geometric_signals, wasserstein)
- ⏳ Publish to crates.io
- ⏳ Publish to PyPI

## Documentation Gaps

The following core modules lack dedicated docs in `docs/`:

| Module | Status | Notes |
|--------|--------|-------|
| `lob` | ❌ Missing | Order book operations — needs doc |
| `meanrev` | ❌ Missing | Mean-reversion signals — needs doc |
| `sparse_meanrev` | ❌ Missing | Sparse mean-reversion — needs doc |
| `optimization` | ❌ Missing | Differential evolution — needs doc |
| `geometric_signals` | ❌ Missing | Full geometric finance suite — needs doc |
| `wasserstein` | ❌ Missing | Wasserstein distance — needs doc (part of geometric_signals) |
| `chiarella` | ✅ Done | `docs/chiarella.md` |
| `signature` | ✅ Done | `docs/signature.md` |
| `kalman` | ✅ Done | `docs/kalman.md` |

## Contributing

This is a work in progress. Contributions are welcome!

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.
