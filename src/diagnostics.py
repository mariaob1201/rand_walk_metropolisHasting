"""
MCMC convergence diagnostics.

All functions accept plain numpy arrays and are independent of the sampler
implementation, so they can be applied to any MCMC output.
"""
import numpy as np


# ---------------------------------------------------------------------------
# Autocorrelation
# ---------------------------------------------------------------------------

def autocorrelation(samples, max_lag=None):
    """
    Normalized autocorrelation at lags 0, 1, ..., max_lag via FFT.

    Parameters
    ----------
    samples : 1-D array-like
    max_lag : int, optional — defaults to len(samples) // 2

    Returns
    -------
    acf : np.ndarray of length max_lag + 1
    """
    x = np.asarray(samples, dtype=float)
    n = len(x)
    if max_lag is None:
        max_lag = n // 2

    x = x - x.mean()
    # Zero-pad to next power of 2 for FFT efficiency
    fft_len = 1
    while fft_len < 2 * n:
        fft_len <<= 1

    f = np.fft.rfft(x, n=fft_len)
    acf_raw = np.fft.irfft(f * np.conj(f))[:n]
    acf_raw /= acf_raw[0]           # normalise so lag-0 = 1
    return acf_raw[:max_lag + 1]


# ---------------------------------------------------------------------------
# Effective Sample Size
# ---------------------------------------------------------------------------

def effective_sample_size(samples):
    """
    Effective Sample Size (ESS) via Geyer's initial positive sequence estimator.

    ESS = n / tau,  where tau = 1 + 2 * sum_{k=1}^{K} rho_k
    and K is the first lag where rho_k < 0 (initial positive sequence).

    Parameters
    ----------
    samples : 1-D array-like — a single MCMC chain

    Returns
    -------
    ess : float
    """
    x = np.asarray(samples, dtype=float)
    n = len(x)
    acf = autocorrelation(x)

    # Sum positive autocorrelations (stop at first negative lag)
    tau = 1.0
    for k in range(1, len(acf)):
        if acf[k] < 0:
            break
        tau += 2.0 * acf[k]

    return n / tau


# ---------------------------------------------------------------------------
# Gelman-Rubin R-hat
# ---------------------------------------------------------------------------

def gelman_rubin(chains):
    """
    Gelman-Rubin potential scale reduction factor (R-hat).

    Requires at least 2 chains of equal length. Values < 1.1 indicate
    convergence; < 1.01 is the current Stan recommendation.

    Parameters
    ----------
    chains : array-like, shape (n_chains, n_samples)

    Returns
    -------
    r_hat : float
    """
    chains = np.asarray(chains, dtype=float)
    if chains.ndim != 2:
        raise ValueError("chains must be 2-D: (n_chains, n_samples)")
    m, n = chains.shape
    if m < 2:
        raise ValueError("Need at least 2 chains to compute R-hat")

    chain_means = chains.mean(axis=1)           # shape (m,)
    overall_mean = chain_means.mean()

    B = n * np.var(chain_means, ddof=1)         # between-chain variance
    W = np.mean(np.var(chains, ddof=1, axis=1)) # within-chain variance

    var_hat = (1 - 1 / n) * W + B / n
    return float(np.sqrt(var_hat / W))


# ---------------------------------------------------------------------------
# Summary helper
# ---------------------------------------------------------------------------

def chain_summary(chains):
    """
    Print a per-chain diagnostic table.

    Parameters
    ----------
    chains : array-like, shape (n_chains, n_samples)

    Returns
    -------
    dict with keys 'ess', 'r_hat', 'means', 'stds'
    """
    chains = np.asarray(chains, dtype=float)
    m = chains.shape[0]

    ess_vals = np.array([effective_sample_size(chains[i]) for i in range(m)])
    r_hat = gelman_rubin(chains) if m >= 2 else float("nan")

    header = f"{'Chain':>6}  {'Mean':>8}  {'Std':>8}  {'ESS':>8}"
    print(header)
    print("-" * len(header))
    for i in range(m):
        print(f"{i + 1:>6}  {chains[i].mean():>8.4f}  {chains[i].std():>8.4f}  {ess_vals[i]:>8.0f}")
    print(f"\nR-hat: {r_hat:.4f}  ({'converged' if r_hat < 1.1 else 'NOT converged'})")

    return {
        "ess": ess_vals,
        "r_hat": r_hat,
        "means": chains.mean(axis=1),
        "stds": chains.std(axis=1),
    }
