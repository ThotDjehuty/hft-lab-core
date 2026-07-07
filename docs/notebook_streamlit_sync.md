# Notebook ↔ Streamlit Lab Sync

> **Version:** `hfthot-lab-core v1.3.0` · **Date:** 2026-03-09  
> **Purpose:** Canonical reference ensuring research notebooks and Streamlit lab pages stay consistent in parameters, methodology, and outputs.

---

## 1. Mapping: Notebook → Lab Page

| Notebook | Streamlit Lab | Core Rust Module | Key Parameters |
|----------|--------------|-----------------|----------------|
| `kalman_filter_market_making.ipynb` | Market Making Lab | `kalman`, `meanrev` | `base_spread`, `max_inventory`, `gamma` (risk-aversion), `q_max` |
| `chiarella_agent_based.ipynb` | Chiarella Dynamics Lab | `chiarella` | `n_steps`, `beta_c`, `beta_f`, `alpha`, `fundamental_price` |
| `path_signatures_trading.ipynb` | Path Signatures Lab | `signature` | `window`, `sig_depth`, `entry_threshold`, `normalise` |
| `mean_field_games_portfolio.ipynb` | MFG Portfolio Lab | `geometric_signals` | `n_assets`, `dt`, `lambda_mfg`, `risk_free` |
| `geometric_arbitrage_polymarket.ipynb` | Per-Market Arbitrage Lab | `lob`, `geometric_signals` | `rolling_window`, `z_score_threshold`, `min_spread_bps` |
| `mev_defi_signal_feed.ipynb` | MEV / DeFi Signal Lab | `lob` | `pool_ids`, `block_lag`, `signal_decay` |

---

## 2. Parameter Bridge Pattern

Notebooks use plain Python variables. Streamlit labs expose the same parameters via `st.session_state` and sidebar widgets. The mapping is:

```python
# === Notebook variable ===
base_spread = 1.50
max_inventory = 10

# === Streamlit sidebar equivalent ===
base_spread    = st.sidebar.slider("Base spread (bps)", 0.1, 10.0, 1.5, 0.1)
max_inventory  = st.sidebar.number_input("Max inventory (units)", 1, 100, 10)

# Both write to the same Rust call:
hft.kalman_mm_quote(base_spread=base_spread, max_inventory=max_inventory, ...)
```

**Rule:** When a notebook default changes, the corresponding Streamlit widget default **must** be updated in the same PR.

---

## 3. Sync Checklist (pre-release)

Run before every minor version bump:

```bash
# 1. Execute all notebooks headlessly
cd examples/notebooks
for nb in *.ipynb; do
    jupyter nbconvert --to notebook --execute "$nb" --inplace \
        --ExecutePreprocessor.timeout=300
    echo "✅ $nb"
done

# 2. Compare parameter defaults  (manual review)
# Check CHANGELOG for any parameter renames

# 3. Run Rust tests
cargo test --all-features

# 4. Run Python tests
python -m pytest tests/ -v --tb=short
```

CI enforces step 1 via `.github/workflows/ci.yml` (`execute-notebooks` job).

---

## 4. Dataset Compatibility

Notebooks assume the following data format (OHLCV) unless noted:

```python
# Standard OHLCV dict used by all labs
ohlcv: dict[str, pd.DataFrame] = {
    "BTC/USDT": pd.DataFrame(
        columns=["open", "high", "low", "close", "volume"],
        index=pd.DatetimeIndex(...),
    )
}
```

The Streamlit Data Loader (`pages/data_loader.py`) exports this exact format into `st.session_state.persisted_datasets`. Notebooks simulate the same with:

```python
# Notebook bootstrap cell — mimics Streamlit session
import pandas as pd
ohlcv_map = {"ETH/USDT": pd.read_parquet("data/eth_usdt_1m.parquet")}
```

---

## 5. Kernel / Environment Consistency

| Requirement | Notebook | Streamlit |
|-------------|----------|-----------|
| Python | ≥ 3.11 | ≥ 3.11 (conda `rhftlab`) |
| `hfthot_lab_core` | PyO3 build from source or PyPI | Same — Docker bind mount |
| `polars` | ≥ 1.0 | ≥ 1.0 |
| `plotly` | ≥ 5.0 | ≥ 5.0 |
| `numpy` | ≥ 1.26 | ≥ 1.26 |

Lock versions in `pyproject.toml` (`[project.optional-dependencies] notebooks`).

---

## 6. Adding a New Lab (checklist)

1. **Create notebook** `examples/notebooks/<name>.ipynb` with:
   - LaTeX derivation cell (markdown)
   - Data bootstrap cell (see §4)
   - Core Rust call demonstration
   - Output: annotated Plotly charts
   - Performance cell: wall clock vs pure-Python baseline

2. **Create Streamlit page** `app/pages/lab_<name>.py` with:
   - Sidebar widgets matching notebook parameters (see §2)
   - Empty-state CTA card navigating to Data Loader
   - Collapsible `💡 How to interpret these metrics` expander
   - All plots via `st.plotly_chart(..., width='stretch')`

3. **Register in `page_registry.py`** under the appropriate RBAC tier.

4. **Add row to mapping table** (§1 of this file).

5. **Run sync checklist** (§3).

---

*Part of the [HFThot Research Lab](https://hfthot-lab.eu) open-source ecosystem.*
