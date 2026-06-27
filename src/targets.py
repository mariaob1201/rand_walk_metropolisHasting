"""
Pluggable log-target functions for use with metropolis_hastings().

Each function (or factory) returns a callable with signature:
    log_target(x: float) -> float
representing log pi(x) up to an additive constant.
"""
import numpy as np
from scipy.stats import t as t_dist


# ---------------------------------------------------------------------------
# Model 1 — Normal mean with Cauchy prior  (original use case)
# ---------------------------------------------------------------------------

def make_normal_mean_target(n, y_bar):
    """
    Factory for the log-posterior of mu in:
        y_i | mu  ~ N(mu, 1)   i.i.d., known variance
        mu        ~ Cauchy(0, 1)  [t_1]

    Log-posterior (up to constant):
        log pi(mu | y) ∝ n*(y_bar*mu - 0.5*mu^2) - log(1 + mu^2)

    Parameters
    ----------
    n     : int   — number of observations
    y_bar : float — sample mean of the data
    """
    def log_target(mu):
        return float(n * (y_bar * mu - 0.5 * mu ** 2) - np.log(1 + mu ** 2))
    return log_target


def cauchy_prior_pdf(x):
    """Cauchy(0,1) density — for overlay on posterior plots."""
    return t_dist.pdf(x, df=1)


# ---------------------------------------------------------------------------
# Model 2 — Bimodal Gaussian mixture  (stress test for mixing)
# ---------------------------------------------------------------------------

def bimodal_log_target(x, w=0.5, mu1=-2.0, mu2=2.0, sigma=0.7):
    """
    Log of an equal-weight Gaussian mixture:
        p(x) = w * N(x; mu1, sigma^2) + (1-w) * N(x; mu2, sigma^2)

    Useful for diagnosing how proposal width affects inter-mode mixing.

    Parameters
    ----------
    x     : float — evaluation point
    w     : float — weight on first component (default 0.5)
    mu1   : float — mean of first component
    mu2   : float — mean of second component
    sigma : float — shared standard deviation
    """
    log_p1 = np.log(w)       - 0.5 * ((x - mu1) / sigma) ** 2
    log_p2 = np.log(1 - w)   - 0.5 * ((x - mu2) / sigma) ** 2
    # log-sum-exp for numerical stability
    m = max(log_p1, log_p2)
    return float(m + np.log(np.exp(log_p1 - m) + np.exp(log_p2 - m)))


def bimodal_true_pdf(x, w=0.5, mu1=-2.0, mu2=2.0, sigma=0.7):
    """Normalised density for the bimodal mixture (for plot overlays)."""
    from scipy.stats import norm
    return w * norm.pdf(x, mu1, sigma) + (1 - w) * norm.pdf(x, mu2, sigma)
