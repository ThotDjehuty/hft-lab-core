# HFT Lab Core v1.3.0 — Release Notes

**Date:** 9 mars 2026  
**Type:** Minor release — platform polish + documentation  
**Upgrade:** `pip install --upgrade hfthot-lab-core`

---

## Highlights

> v1.3.0 focuses on **professional developer experience** — the Rust computation core is unchanged, but the platform surrounding it (Streamlit labs, onboarding flow, documentation) reaches a new level of coherence and polish. Every notebook now has a documented path to its Streamlit counterpart.

---

## What's New

### 🖥️ Platform — Streamlit App (`hfthot-lab.eu`)

#### Home Page — Onboarding Redesign
The "Quick Start" section previously showed a one-line info banner:
```
1. Load Data → 2. Choose a Lab → 3. Backtest → 4. Trade Live
```
It is now a **4-column visual step-card grid** with icons, colour-coded step badges, and context-specific descriptions for each phase of the research workflow. New users understand the platform in 5 seconds without reading text.

#### Admin Sidebar — Tier UI Hidden
On private self-hosted instances (ThotCloud), the admin user no longer sees role/tier/RBAC widgets in the sidebar. These are commercial widgets relevant for multi-tenant SaaS deployments — not for the platform owner. The sidebar now shows only the username and WASM compute status.

#### Arbitrage Lab — KPI Help Expander
A collapsed `💡 How to interpret these metrics` expander appears directly below the Overview tab's 3×3 command-center grid. It contains:

| Metric | Definition | Healthy range |
|--------|-----------|---------------|
| Spread obs | Data-points across all pairs | More = richer analysis |
| Signals | Entry/exit events above threshold | 5–15 % of observations |
| Win Rate | Fraction of trades with positive P&L | > 55 % on mean-rev |
| Sharpe Ratio | Risk-adjusted annualised return | > 1.0 acceptable, > 2.0 strong |
| Max Drawdown | Largest equity peak-to-trough | < 15 % conservative |

#### Arbitrage Lab — Data Gate CTA
When no dataset is loaded, the lab previously showed a plain `st.info()` message with no next step. It now shows a styled card with a primary **🚀 Go to Data Loader** button that navigates directly to the data ingestion page. Eliminates the dead-end experience for new users.

#### Navigation Fixes
Seven files had `st.switch_page("HFT_Arbitrage_Lab.py")` — routing to the entry-point script instead of a navigable page, causing `NavigationError` on every click. All fixed to `st.switch_page("pages/home.py")`.

#### MEV Signal Feed Lab — New
A dedicated Streamlit lab page (`lab_mev_signal_feed.py`) + companion notebook (`mev_defi_signal_feed.ipynb`) for detecting Maximal Extractable Value opportunities in real-time:
- **Sandwich attacks**, **cross-DEX arbitrage**, and **liquidation monitoring** using live GeckoTerminal on-chain data
- Composite MEV/arb signal scoring engine with gas-cost estimation baked into the per-market arbitrage dashboard
- Prefect pipeline integration (`mev_signal_feed`) scheduled every 1 hour for continuous monitoring
- Output persisted to `data/mev_signals/` for historical analysis

#### ThotBook Research — Notebook Generator
The ThotBook Research MCP server (`@thotcloud/thotbook-research-mcp`) provides an AI-driven notebook generation pipeline:
- `thotbook_arxiv_search` / `thotbook_arxiv_fetch` / `thotbook_arxiv_trending` — ArXiv paper discovery
- `thotbook_paper_explain` / `thotbook_paper_prerequisites` — paper comprehension tools
- `thotbook_generate` — **automated notebook generation** from research papers with backtest scaffolding, visualisations, and Rust-accelerated compute cells
- Delta Lake historisation via `ThotBookLakehouse` — every generation is versioned with time-travel support
- SQLite + MeiliSearch index for fast search across papers, equations, and generated notebooks
- Runs on MCP port 3002, integrated into the wiki at `hfthot-lab.eu/wiki.html`

#### LabIntegrationBridge — Notebook ↔ Lab Strategy Bridge
The `LabIntegrationBridge` class provides a DuckDB-backed strategy pipeline connecting notebooks to the Streamlit lab:
- `bridge.load_strategies_iter()` — stream strategies from DuckDB `lab_strategies.duckdb`
- `bridge.leaderboard()` — rank strategies by Sharpe, drawdown, win rate
- `bridge.export_leaderboard()` — export to CSV/Parquet for external analysis
- Used by `mean_field_games_portfolio.ipynb` and `path_signatures_trading.ipynb` to persist research results directly into the lab's strategy store

---

### 📚 Documentation

#### `docs/notebook_streamlit_sync.md` — New
Full reference for the notebook ↔ Streamlit bridge:
- Notebook → Lab page mapping table
- Parameter bridge pattern (Python variables → sidebar widgets)
- Pre-release sync checklist (headless notebook execution + tests)
- Dataset format contract (OHLCV dict)
- Kernel/environment consistency table
- Step-by-step guide for adding new labs

#### `CHANGELOG.md` — Updated
v1.3.0 entry added following Keep-a-Changelog format.

#### `README.md` — Updated
- Module table: removed stale "new v1.2" markers (Chiarella, Signature now mature)
- Added **Streamlit Lab Integration** section with notebook→lab mapping table
- Added nbconvert CI guidance

---

## Upgrade Notes

No breaking API changes. The Rust ABI is identical to v1.2.1.

```bash
pip install --upgrade hfthot-lab-core==1.3.0
```

For self-hosted Streamlit deployments using the bind-mount pattern, changes are live-reloaded automatically without container restart.

---

## Test Results

```
48 passed; 0 failed; 0 ignored (unit tests)
 1 passed; 0 failed; 0 ignored (doc tests)
```

All Rust unit tests and doc-tests pass. Python integration tests validated via Streamlit lab E2E.

---

## Acknowledgements

Platform improvements developed as part of the HFThot Research Lab project.

---

*[hfthot-lab-core](https://github.com/ThotDjehuty/hfthot-lab-core) · [hfthot-lab.eu](https://hfthot-lab.eu) · MIT License*
