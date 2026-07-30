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


def make_normal_mean_grad(n, y_bar):
    """
    Factory for d/dmu log pi(mu | y), matching make_normal_mean_target.

        d/dmu log pi(mu | y) = n*(y_bar - mu) - 2*mu / (1 + mu^2)

    For use as `grad_log_target` with gradient-based samplers (e.g. HMC).
    """
    def grad_log_target(mu):
        return float(n * (y_bar - mu) - 2 * mu / (1 + mu ** 2))
    return grad_log_target


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


def grad_bimodal_log_target(x, w=0.5, mu1=-2.0, mu2=2.0, sigma=0.7):
    """
    d/dx log p(x) for the bimodal mixture, matching bimodal_log_target.

    p(x) is a mixture of two components with the same sigma, so the missing
    normalisation constant in bimodal_log_target is shared and does not affect
    the gradient. Computed as a softmax-weighted average of each component's
    own gradient, -(x - mu_k) / sigma^2.

    For use as `grad_log_target` with gradient-based samplers (e.g. HMC).
    """
    log_p1 = np.log(w)     - 0.5 * ((x - mu1) / sigma) ** 2
    log_p2 = np.log(1 - w) - 0.5 * ((x - mu2) / sigma) ** 2
    m = max(log_p1, log_p2)
    e1 = np.exp(log_p1 - m)
    e2 = np.exp(log_p2 - m)
    weight1 = e1 / (e1 + e2)
    weight2 = e2 / (e1 + e2)

    d1 = -(x - mu1) / sigma ** 2
    d2 = -(x - mu2) / sigma ** 2
    return float(weight1 * d1 + weight2 * d2)
