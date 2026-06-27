# Random Walk Metropolis-Hastings Sampler

A research-oriented Python implementation of the **Random Walk Metropolis-Hastings (RWMH)** algorithm for Bayesian posterior sampling.

The sampler is decoupled from any specific model: any log-target function can be passed in, enabling rapid experimentation with different priors, likelihoods, and targets.

---

## Motivation

Metropolis-Hastings is the foundational MCMC algorithm underlying modern Bayesian inference. This repository demonstrates:

- How to implement RWMH from first principles (no black-box dependencies)
- How burn-in, thinning, and multiple chains affect inference quality
- How to diagnose convergence rigorously (ESS, R-hat, ACF)
- How proposal width controls the exploration-exploitation tradeoff — illustrated with a bimodal stress-test target

---

## Repository structure

```
rand_walk_metropolisHasting/
├── src/
│   ├── sampler.py       # Generic RWMH — accepts any log_target callable
│   ├── targets.py       # Log-posterior factories (Normal mean, bimodal mixture)
│   ├── diagnostics.py   # ESS, Gelman-Rubin R-hat, autocorrelation
│   └── plots.py         # Trace, ACF, posterior density, convergence plots
├── examples/
│   ├── normal_mean.py   # Bayesian inference on a normal mean (Cauchy prior)
│   └── mixture.py       # Bimodal Gaussian mixture — proposal-width study
├── notebooks/
│   └── demo.ipynb       # Step-by-step research walkthrough
├── tests/
│   └── test_sampler.py  # Pytest suite (shapes, convergence, diagnostics)
├── outputs/             # Generated plots (gitignored, directory kept)
├── main.py              # Original single-file script (reference)
└── requirements.txt
```

---

## Models

### Model 1 — Normal mean with Cauchy prior

$$
y_i \mid \mu \sim \mathcal{N}(\mu, 1), \quad i = 1,\ldots,n
\qquad
\mu \sim \text{Cauchy}(0, 1)
$$

Log-posterior (up to an additive constant):

$$
\log \pi(\mu \mid y) \propto n\!\left(\bar{y}\,\mu - \tfrac{1}{2}\mu^2\right) - \log(1 + \mu^2)
$$

The initial value `mu_init = 30` is placed far from the true mean to demonstrate how quickly the chain burns in and how many iterations are needed to reach stationarity.

### Model 2 — Bimodal Gaussian mixture (stress test)

$$
p(x) = 0.5\,\mathcal{N}(x;\,-2,\,0.49) + 0.5\,\mathcal{N}(x;\,+2,\,0.49)
$$

Used to study how the proposal standard deviation controls inter-mode mixing. A narrow proposal (σ = 0.5) gets stuck; a wide proposal (σ = 3.0) crosses the barrier between modes.

---

## Sampler API

```python
from src.sampler import metropolis_hastings

result = metropolis_hastings(
    log_target = my_log_posterior,  # callable: float -> float
    init       = 0.0,               # starting value (or array per chain)
    n_iter     = 5_000,             # total iterations
    cand_std   = 0.5,               # Normal proposal std deviation
    burn_in    = 500,               # samples to discard at the start
    thin       = 1,                 # keep every nth sample
    n_chains   = 4,                 # independent chains (for R-hat)
    seed       = 42,
)

chains = result["samples"]          # shape (n_chains, n_kept)
rates  = result["acceptance_rates"] # shape (n_chains,)
```

---

## Diagnostics

```python
from src.diagnostics import effective_sample_size, gelman_rubin, chain_summary

chain_summary(chains)          # prints per-chain mean, std, ESS, and R-hat

ess   = effective_sample_size(chains[0])   # Geyer's initial positive sequence
r_hat = gelman_rubin(chains)               # < 1.1 indicates convergence
```

| Diagnostic | What it measures | Threshold |
|---|---|---|
| Acceptance rate | Fraction of proposals accepted | 0.2–0.5 (1-D) |
| ESS | Effective independent draws | > 100 per chain |
| R-hat | Chain-to-chain variance ratio | < 1.1 |
| ACF | Autocorrelation decay | Fast decay = good mixing |

---

## Quickstart

```bash
pip install -r requirements.txt

# Example 1 — Normal mean model
python -m examples.normal_mean

# Example 2 — Bimodal mixture (proposal-width comparison)
python -m examples.mixture

# Tests
python -m pytest tests/ -v

# Notebook
jupyter notebook notebooks/demo.ipynb
```

All plots are saved to `outputs/`.

---

## Sample outputs

### Normal mean model

**Trace plot** — chain 1 initialised at μ = 30; converges to the posterior within ~100 iterations:

![Trace plot](https://github.com/mariaob1201/rand_walk_metropolisHasting/blob/main/trace_plot.jpg)

**Prior vs Posterior** — the likelihood concentrates the Cauchy prior around ȳ = 0.99:

![Posterior density](https://github.com/mariaob1201/rand_walk_metropolisHasting/blob/main/posterior_density_plot.jpg)

### Bimodal mixture (wide vs narrow proposal)

| Proposal | Acceptance rate | Both modes sampled | R-hat |
|---|---|---|---|
| Narrow (σ = 0.5) | ~0.85 | No — chain gets stuck | > 1.1 |
| Wide (σ = 3.0) | ~0.35 | Yes | < 1.05 |

---

## Adding a new target

1. Add a `my_log_target(x: float) -> float` function to `src/targets.py`
2. Pass it directly to `metropolis_hastings(log_target=my_log_target, ...)`
3. No other changes needed

---

## References

- Gelfand & Smith (1990). Sampling-based approaches to calculating marginal densities. *JASA*.
- Gelman et al. (2013). *Bayesian Data Analysis* (3rd ed.), Ch. 11–12.
- Geyer (1992). Practical Markov chain Monte Carlo. *Statistical Science*.
- Roberts, Gelman & Gilks (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *Annals of Applied Probability*.
