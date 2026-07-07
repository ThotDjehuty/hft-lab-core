# hfthot-lab-core — Release Notes

## Unreleased — Polymarket Gamma Client + MFG Notebook Refactor

### Highlights
- **Polymarket Gamma client** (`PolymarketGammaClient`): search / events / markets API.
- **Mean-Field Games notebook** refactored to use the OptimizR Rust solver instead of a numpy reimplementation.

### Maintenance
- Removed 4 stale notebook artifacts: `*.ipynb.bak` and `*.ipynb.REMOVED` from `examples/notebooks/`.

---

## v1.6.0 — Rough Heston + Academic Portfolio
- Rough Heston pricer.
- Un-gated academic portfolio functions.

## v1.5.0 — Proprietary Feature Gating
- Proprietary modules gated behind a feature flag for OSS/commercial split.

## v1.4.0 — Functional Programming + Real Data
- Functional programming foundation, backtesting framework, usage tracking & analytics.
- All notebooks switched from synthetic data to real Yahoo Finance feeds.
- Jupyter token auth via `JUPYTER_TOKEN`.
