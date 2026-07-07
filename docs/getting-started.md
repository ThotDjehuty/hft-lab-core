# Getting Started with HFT Lab Core

This guide will help you get started with HFT Lab Core.

## Installation

### From PyPI (coming soon)

```bash
pip install hfthot-lab-core
```

### From Source

```bash
# Prerequisites
# - Rust 1.75+ (install from https://rustup.rs)
# - Python 3.8+
# - pip and maturin

# Clone the repository
git clone https://github.com/ThotDjehuty/hfthot-lab-core.git
cd hfthot-lab-core

# Build and install
pip install maturin
maturin develop --release
```

## Quick Start

### 1. Limit Order Book (LOB)

```python
import hfthot_lab_core as hft

# Create an order book
lob = hft.LimitOrderBook("BTC-USD")

# Add orders
lob.add_bid(price=50000.0, quantity=1.5, order_id=1)
lob.add_ask(price=50100.0, quantity=2.0, order_id=2)

# Get best bid/ask
best_bid = lob.best_bid()
best_ask = lob.best_ask()
spread = lob.spread()

print(f"Best Bid: ${best_bid:.2f}")
print(f"Best Ask: ${best_ask:.2f}")
print(f"Spread: ${spread:.2f}")
```

### 2. Mean Reversion Strategy

```python
import hfthot_lab_core as hft
import numpy as np

# Generate sample price data
prices = np.random.randn(1000).cumsum() + 100

# Create mean reversion strategy
strategy = hft.MeanReversionStrategy(
    window=20,     # Moving average window
    threshold=2.0  # Z-score threshold
)

# Generate signals
signals = strategy.generate_signals(prices)

# signals will be:
#  1.0 for buy signal (price below threshold)
# -1.0 for sell signal (price above threshold)
#  0.0 for no signal
```

### 3. Sparse Mean Reversion

```python
import hfthot_lab_core as hft

# Sparse mean reversion with L1 regularization
strategy = hft.SparseMeanReversionStrategy(
    lookback=50,
    sparsity_lambda=0.1  # L1 regularization parameter
)

# This strategy performs automatic feature selection
# by applying L1 regularization to mean reversion
signals = strategy.generate_signals(prices)
```

### 4. Chiarella Agent-Based Model *(v1.2)*

```python
import hfthot_lab_core as hft

result = hft.simulate_chiarella_py(
    initial_price=95.0, fundamental_price=100.0, n_steps=1000,
    beta_f=0.5, beta_c=1.0, gamma=1.5, mu=0.1, sigma=0.05, seed=42
)
prices  = result["prices"]
regimes = [hft.classify_regime(hft.bifurcation_lambda(1.0, 1.5, 0.5, 0.2))]
print(f"Final price: {prices[-1]:.2f}, regime: {regimes[0]}")
```

See [docs/chiarella.md](chiarella.md) for full documentation.

### 5. Path Signature Features *(v1.2)*

```python
import hfthot_lab_core as hft
import numpy as np

prices = np.random.randn(200).cumsum() + 100
feats  = hft.prices_to_signature_features_py(prices, window=30, sig_depth=2, normalise=True)
print(f"Feature matrix: {feats.shape}")   # (170, 6)
```

See [docs/signature.md](signature.md) for full documentation.

## Next Steps

- [Chiarella Agent-Based Model](chiarella.md) — heterogeneous-agent simulation
- [Path Signature Methods](signature.md) — rough paths features for ML
- [Strategy Reference](strategies.md) — available strategies
- [API Reference](api-reference.md) — complete API documentation
- [Examples](../examples/) — Jupyter notebooks and Python examples

## Community

Need help? Join our community:

- Website: [hfthot-lab.eu](https://hfthot-lab.eu)
- GitHub Issues: [Report a bug](https://github.com/ThotDjehuty/hfthot-lab-core/issues)
- Discord: Coming soon

## What's Next?

Want more features? Check out the [HFThot Platform](https://hfthot-lab.eu):

- **Free Tier**: What you're using now (open source)
- **Hobbyist** (€9/month): Cloud hosting, historical data, basic backtesting
- **Pioneer** (€29/month): Real-time data, regime detection, Silent P2P beta
- **Professional** (€199/month): Silent P2P full access, live trading, API

[View Pricing](https://hfthot-lab.eu/pricing.html)
