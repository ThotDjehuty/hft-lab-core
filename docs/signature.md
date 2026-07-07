# Path Signature Methods

*Module:* `hfthot_lab_core::signature` | *Python:* `hfthot_lab_core.prices_to_signature_features_py`  
*Added in:* v1.2.0

---

## Overview

**Path signatures** are a principled, universal feature extraction framework for sequential data, rooted in **rough paths theory** (Lyons 1998). For financial time series they provide:

- **Depth-$N$ features** capturing up to $N$-th order interactions of price increments
- **Lévy area** — a "missing" statistic unavailable from endpoints; encodes quadratic covariation
- **Parameterization invariance** — only the shape of the path matters, not its speed
- **Universal approximation** — any continuous functional of a path can be approximated by a linear functional of its signature (Chen–Lyons theorem)

Gyurko et al. (2013) demonstrated these properties empirically on financial data.  
Chevyrev & Kormilitzin (2016) provided a practical primer including the lead-lag construction used here.

---

## Mathematical Background

### Path Signature

For a path $X = (X^1, \ldots, X^d): [0,T] \to \mathbb{R}^d$, the **signature up to depth $N$** is:

$$S(X) = \Bigl(1,\ S(X)^{i},\ S(X)^{i_1 i_2},\ \ldots,\ S(X)^{i_1 \cdots i_N}\Bigr)$$

where each multi-index term is the iterated integral:

$$S(X)^{i_1, \ldots, i_k} = \int_{0 < t_1 < \cdots < t_k < T} dX^{i_1}_{t_1} \cdots dX^{i_k}_{t_k}$$

**Dimension count** for $d$-dimensional path, depth $N$:

$$\dim S_N = \sum_{k=1}^{N} d^k = d \cdot \frac{d^N - 1}{d - 1}$$

For $d=2$ (lead-lag path): depth 1 → 2 terms; depth 2 → 6 terms; depth 3 → 14 terms.

### Lead-Lag Transform

The **Chevyrev-Kormilitzin lead-lag embedding** converts a 1D discrete price path $(X_1, \ldots, X_n)$ into a 2D path:

$$\tilde{X} = \{(X_{t_k}, X_{t_{k+1}}) : k = 1, \ldots, n-1\}$$

implemented as a staircase interpolation:

$$\tilde{X}_{2k} = (X_k, X_k), \quad \tilde{X}_{2k+1} = (X_k, X_{k+1})$$

**Why this matters:** The Lévy area of $\tilde{X}$ equals the **quadratic variation** of $X$, which is not captured by the endpoint or mean alone.

### Log-Signature and Lévy Area

The **log-signature** (depth 2, $d=2$) has only 3 components:

$$\ell^1 = S^1, \quad \ell^2 = S^2, \quad \ell^{12} = \frac{S^{12} - S^{21}}{2}$$

where $\ell^{12}$ is the **Lévy area** — a signed measure of the area enclosed by the 2D path.  
For the lead-lag path this directly measures realised variance / quadratic covariation.

---

## API Reference

### Python

```python
import hfthot_lab_core as hft
import numpy as np

prices = np.array([100., 101.2, 99.8, 102.1, 101.5, 103.0])

# ── Step 1: lead-lag embedding ───────────────────────────────────────────
path = hft.lead_lag_transform(prices)
# Returns ndarray of shape (2n-1, 2) = (11, 2) for n=6

# ── Step 2: path signature ───────────────────────────────────────────────
sig2 = hft.path_signature_2d(path, depth=2)   # 6 terms:  [S¹, S², S¹¹, S¹², S²¹, S²²]
sig3 = hft.path_signature_2d(path, depth=3)   # 14 terms

# ── Step 3: log-signature (Lévy area) ───────────────────────────────────
log_sig = hft.log_signature_2d(path)          # [l¹, l², Lévy area]
print(f"Lévy area: {log_sig[2]:.6f}")         # = realised quadratic covariation proxy

# ── Rolling feature matrix ───────────────────────────────────────────────
prices_long = np.random.randn(500).cumsum() + 100
feats = hft.prices_to_signature_features_py(
    prices_long,
    window=30,          # rolling window size
    sig_depth=2,        # signature depth (1, 2, or 3)
    normalise=True,     # z-score normalise features
)
print(f"Feature matrix: {feats.shape}")   # (470, 6)

# ── Signature distance ───────────────────────────────────────────────────
sig_a = hft.path_signature_2d(path, depth=2)
sig_b = hft.path_signature_2d(path + 0.5, depth=2)
d = hft.signature_distance(sig_a.tolist(), sig_b.tolist())
print(f"Signature distance: {d:.6f}")
```

### Rust

```rust
use hfthot_lab_core::signature::{
    lead_lag_transform, path_signature_2d, log_signature_2d,
    rolling_signature_features, signature_distance, normalise_prices,
};

let prices = vec![100.0f64, 101.2, 99.8, 102.1, 101.5, 103.0];

// Lead-lag → 2D path
let path = lead_lag_transform(&prices);

// Signature (depth 2 → 6 terms)
let sig = path_signature_2d(&path, 2);
println!("Net log-return ≈ S²: {:.6}", sig[1]);

// Log-signature Lévy area
let log_sig = log_signature_2d(&path);
println!("Lévy area: {:.6}", log_sig[2]);

// Rolling feature matrix
let feats = rolling_signature_features(&prices, 30, 2);
println!("Feature matrix: {} × {}", feats.len(), feats[0].len());
```

---

## Term Interpretation

For a 2D lead-lag path $(X^{\text{lead}}, X^{\text{lag}})$ built from log-prices:

| Term | Symbol | Meaning |
|---|---|---|
| $S^1$ | `sig[0]` | Net displacement in lead dimension (≈ total log-return) |
| $S^2$ | `sig[1]` | Net displacement in lag dimension (≈ same, shifted) |
| $S^{11}$ | `sig[2]` | Quadratic variation of lead (≈ realised variance) |
| $S^{12}$ | `sig[3]` | Cross-term: area swept in (lead, lag) upper half |
| $S^{21}$ | `sig[4]` | Cross-term: area swept in (lead, lag) lower half |
| $S^{22}$ | `sig[5]` | Quadratic variation of lag |
| Lévy area | `(sig[3]-sig[4])/2` | Signed area enclosed; = realised covariation of lead-lag path |

**Key insight:** $S^{11} \approx$ realised variance; Lévy area $\approx$ signed quadratic covariation. These are not available from the endpoint $S^1 = S^2 =$ total log-return alone.

---

## Performance

The Rust implementation is **10–50× faster** than equivalent pure Python:

| Operation | Python (numpy) | Rust (hfthot-lab-core) |
|---|---|---|
| Lead-lag transform (n=1000) | ~0.8 ms | ~0.02 ms |
| Depth-2 signature (n=1000) | ~1.5 ms | ~0.05 ms |
| Rolling features (n=10000, window=30) | ~150 ms | ~5 ms |

---

## Example: Rolling Feature Extraction

```python
import hfthot_lab_core as hft
import numpy as np
import matplotlib.pyplot as plt

# Simulate GBM prices
rng = np.random.default_rng(42)
rets = np.exp(0.0002 + 0.015 * rng.standard_normal(500))
prices = 100.0 * rets.cumprod()

# Extract depth-2 rolling features
feats = hft.prices_to_signature_features_py(prices, window=30, sig_depth=2, normalise=True)
# feats shape: (470, 6)

# Correlation with next-bar return
next_rets = np.diff(prices)[30:] / prices[30:-1]
corrs = [np.corrcoef(feats[:, j], next_rets)[0, 1] for j in range(6)]

names = ["S¹(lead)", "S¹(lag)", "S²(ll)", "S²(llag)", "S²(lagl)", "S²(lag²)"]
for n, c in zip(names, corrs):
    print(f"  {n:12s}: r = {c:+.4f}")
```

Full tutorial: [`examples/notebooks/path_signatures_trading.ipynb`](../examples/notebooks/path_signatures_trading.ipynb)

---

## References

1. Chen, K.T. (1954). *Iterated integrals and exponential homomorphisms*. Proc. London Math. Soc.
2. Lyons, T. (1998). *Differential equations driven by rough signals*. Rev. Mat. Iberoamer. 14(2), 215–310.
3. Chevyrev, I. & Kormilitzin, A. (2016). *A primer on the signature method in machine learning*. arXiv:1603.03788.
4. Gyurko, L., Lyons, T. & Kontkowski, M. (2013). *Extracting information from the signature of a financial data stream*. arXiv:1307.7244.
5. Király, F. & Oberhauser, H. (2019). *Kernels for sequentially ordered data*. JMLR 20(31), 1–45.
