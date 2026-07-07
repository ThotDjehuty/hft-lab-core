# Chiarella Agent-Based Model

*Module:* `hfthot_lab_core::chiarella` | *Python:* `hfthot_lab_core.simulate_chiarella_py`  
*Added in:* v1.2.0

---

## Overview

The **Chiarella-He (2002)** model is the canonical heterogeneous-agent market model in quantitative finance. It describes a market populated by two types of boundedly-rational traders:

| Agent type | Demand rule | Motivation |
|---|---|---|
| **Fundamentalists** | $D_f = \beta_f (P^* - P_t)$ | Price reverts to intrinsic value |
| **Chartists** | $D_c = \beta_c \cdot \text{trend}(t)$ | Recent price trend continues |

Agents switch between strategies using a **discrete-choice (logit) switching mechanism** proportional to recent profitability, mimicking real investor behaviour.

This model reproduces stylised facts of real markets:
- Fat-tailed return distributions
- Volatility clustering
- Bubbles and crashes from endogenous dynamics (no exogenous shocks needed)
- Regime switching between fundamentalist and chartist dominance

---

## Mathematical Derivation

### 1. Agent demands

At time $t$, let $n_f(t)$ and $n_c(t) = 1 - n_f(t)$ denote the fractions of fundamentalists and chartists.

$$D_f(t) = \beta_f \bigl(P^* - P(t)\bigr)$$

$$D_c(t) = \beta_c \cdot \overline{\Delta P}(t)$$

where $\overline{\Delta P}(t)$ is the moving average of recent price changes (trend proxy).

### 2. Price update (market clearing)

$$P(t+1) = P(t) + \mu \bigl[n_f(t) D_f(t) + n_c(t) D_c(t)\bigr] + \sigma \varepsilon_t$$

where $\mu$ is market impact speed, $\sigma$ is noise volatility, and $\varepsilon_t \sim \mathcal{N}(0,1)$.

### 3. Agent switching (logit map)

$$n_f(t+1) = \frac{e^{\gamma \Pi_f(t)}}{e^{\gamma \Pi_f(t)} + e^{\gamma \Pi_c(t)}}$$

where $\Pi_k(t) = D_k(t) \cdot \Delta P(t)$ is the realized profit and $\gamma$ is the switching intensity.

### 4. Stability analysis

The **bifurcation parameter** $\Lambda$ determines the long-run regime:

$$\Lambda = \frac{\alpha \cdot \gamma}{\beta \cdot \delta}$$

| $\Lambda$ | Regime | Interpretation |
|---|---|---|
| $\Lambda < 0.67$ | **Stable** | Mean-reversion dominates; price converges to $P^*$ |
| $0.67 \le \Lambda \le 1.5$ | **Mixed** | Complex dynamics; regime switching possible |
| $\Lambda > 1.5$ | **Unstable** | Chartist dominance; trending, bubbles/crashes possible |

---

## API Reference

### Python

```python
import hfthot_lab_core as hft

result = hft.simulate_chiarella_py(
    initial_price: float,       # Starting price
    fundamental_price: float,   # Intrinsic value P*
    n_steps: int,               # Number of discrete time steps
    beta_f: float = 0.5,        # Fundamentalist demand sensitivity
    beta_c: float = 1.0,        # Chartist trend sensitivity
    gamma: float  = 1.0,        # Agent switching intensity
    mu: float     = 0.1,        # Market impact / price adjustment speed
    sigma: float  = 0.05,       # Noise volatility
    seed: int     = 42,         # Random seed for reproducibility
) -> dict
```

**Returns:**
```python
{
    "prices":                   list[float],  # Price trajectory, length n_steps
    "fundamentalist_fractions": list[float],  # n_f(t), in [0, 1]
    "chartist_fractions":       list[float],  # n_c(t) = 1 - n_f(t)
    "excess_demands":           list[float],  # β_f (P* - P(t))
}
```

#### Stability helpers

```python
# Compute bifurcation parameter Λ = (α·γ) / (β·δ)
lam = hft.bifurcation_lambda(alpha=1.0, gamma=1.5, beta=0.5, delta=0.2)

# Classify into regime string
regime = hft.classify_regime(lam)   # "stable" | "mixed" | "unstable"
```

### Rust

```rust
use hfthot_lab_core::chiarella::{ChiarellaParams, simulate_chiarella, bifurcation_lambda, classify_regime};

let params = ChiarellaParams {
    beta_f: 0.5,
    beta_c: 1.0,
    gamma:  1.5,
    mu:     0.1,
    sigma:  0.05,
};

let result = simulate_chiarella(
    /*initial_price*/    95.0,
    /*fundamental_price*/100.0,
    /*n_steps*/          2000,
    params,
    /*seed*/             42,
);

println!("Final price: {:.4}", result.final_price());
println!("Chartist dominated: {}", result.is_chartist_dominated());
println!("Ann. volatility: {:.4}", result.annualised_volatility());

let lam = bifurcation_lambda(1.0, 1.5, 0.5, 0.2);
println!("Λ = {lam:.4} → {}", classify_regime(lam));
```

---

## Parameter Guide

| Parameter | Symbol | Typical range | Effect |
|---|---|---|---|
| `beta_f` | $\beta_f$ | 0.1 – 2.0 | Higher → faster mean reversion |
| `beta_c` | $\beta_c$ | 0.1 – 2.5 | Higher → stronger trend following; raises $\Lambda$ |
| `gamma` | $\gamma$ | 0.5 – 5.0 | Higher → faster strategy switching; raises $\Lambda$ |
| `mu` | $\mu$ | 0.05 – 0.5 | Market impact speed |
| `sigma` | $\sigma$ | 0.01 – 0.2 | Background noise |

**Calibration tip:** To match real asset volatility, start with $\sigma = $ realized daily vol, then tune $\gamma$ to match regime persistence.

---

## Example

```python
import hfthot_lab_core as hft
import numpy as np
import matplotlib.pyplot as plt

result = hft.simulate_chiarella_py(
    initial_price=95.0, fundamental_price=100.0, n_steps=2000,
    beta_f=0.5, beta_c=1.2, gamma=2.0, mu=0.1, sigma=0.04, seed=0
)

prices   = np.array(result["prices"])
n_f_arr  = np.array(result["fundamentalist_fractions"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
ax1.plot(prices); ax1.axhline(100, ls="--", color="red", label="P*=100")
ax1.set_ylabel("Price"); ax1.legend()
ax2.stackplot(range(len(n_f_arr)), n_f_arr, 1 - n_f_arr,
              labels=["Fundamentalists", "Chartists"], colors=["green","orange"], alpha=0.7)
ax2.set_ylabel("Fraction"); ax2.set_xlabel("Step"); ax2.legend(loc="upper right")
plt.tight_layout(); plt.show()
```

Full tutorial: [`examples/notebooks/chiarella_agent_based.ipynb`](../examples/notebooks/chiarella_agent_based.ipynb)

---

## References

1. Chiarella, C. & He, X.-Z. (2002). *Heterogeneous Beliefs, Risk and Learning in a Simple Asset Pricing Model*. Computational Economics.
2. Brock, W. & Hommes, C. (1998). *Heterogeneous beliefs and routes to chaos in a simple asset pricing model*. Journal of Economic Dynamics and Control, 22(8-9), 1235–1274.
3. Hommes, C. (2006). *Heterogeneous Agent Models in Economics and Finance*. Handbook of Computational Economics, Vol. 2.
