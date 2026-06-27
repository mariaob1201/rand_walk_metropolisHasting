"""
Visualization utilities for MCMC output.

All plot functions save to outputs/ by default and optionally display
interactively. Each returns the matplotlib Figure for further customisation.
"""
import os
import numpy as np
import matplotlib.pyplot as plt

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def _save(fig, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    return path


# ---------------------------------------------------------------------------
# Trace plot
# ---------------------------------------------------------------------------

def trace_plot(samples, title="Trace Plot", filename="trace_plot.png", show=False):
    """
    Plot the chain values over iterations.

    Parameters
    ----------
    samples  : 1-D array-like — a single chain
    title    : str
    filename : str — saved under outputs/
    show     : bool — call plt.show() if True

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    samples = np.asarray(samples)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(samples, lw=0.7, alpha=0.85, color="steelblue")
    ax.axhline(samples.mean(), color="crimson", lw=1.2, linestyle="--", label=f"mean = {samples.mean():.3f}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Sample value")
    ax.set_title(title)
    ax.legend()
    _save(fig, filename)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Autocorrelation (ACF) plot
# ---------------------------------------------------------------------------

def acf_plot(samples, max_lag=60, title="Autocorrelation Function",
             filename="acf_plot.png", show=False):
    """
    Bar plot of autocorrelation at lags 0..max_lag.

    Parameters
    ----------
    samples : 1-D array-like
    max_lag : int
    """
    from .diagnostics import autocorrelation
    acf = autocorrelation(samples, max_lag=max_lag)
    lags = np.arange(len(acf))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(lags, acf, color="steelblue", alpha=0.75, width=0.8)
    ax.axhline(0, color="black", lw=0.8)
    # 95 % significance bounds (white-noise reference)
    n = len(samples)
    bound = 1.96 / np.sqrt(n)
    ax.axhline(bound,  color="crimson", lw=1, linestyle="--", label="95% CI")
    ax.axhline(-bound, color="crimson", lw=1, linestyle="--")
    ax.set_xlabel("Lag")
    ax.set_ylabel("Autocorrelation")
    ax.set_title(title)
    ax.legend()
    _save(fig, filename)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Posterior density plot
# ---------------------------------------------------------------------------

def posterior_density_plot(samples, prior_pdf=None, true_value=None,
                           x_range=None, title="Prior vs Posterior",
                           filename="posterior_density.png", show=False):
    """
    KDE of the posterior with optional prior overlay and true-value line.

    Parameters
    ----------
    samples    : 1-D array-like — posterior samples (all chains pooled)
    prior_pdf  : callable x -> density, optional
    true_value : float, optional — vertical line at the data mean / true param
    x_range    : (lo, hi), optional — x-axis limits
    """
    import pandas as pd

    samples = np.asarray(samples, dtype=float)
    fig, ax = plt.subplots(figsize=(10, 5))

    # Posterior KDE
    pd.Series(samples).plot.density(ax=ax, color="seagreen", lw=2, label="Posterior")

    # Prior overlay
    if prior_pdf is not None:
        lo = samples.min() - 1 if x_range is None else x_range[0]
        hi = samples.max() + 1 if x_range is None else x_range[1]
        xs = np.linspace(lo, hi, 400)
        ax.plot(xs, prior_pdf(xs), color="royalblue", lw=1.5,
                linestyle="--", label="Prior")

    # True / reference value
    if true_value is not None:
        ax.axvline(true_value, color="crimson", lw=1.5,
                   linestyle="-", label=f"y̅ = {true_value:.3f}")

    if x_range is not None:
        ax.set_xlim(x_range)

    ax.set_xlabel("Value")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend()
    _save(fig, filename)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Running-mean convergence plot
# ---------------------------------------------------------------------------

def convergence_plot(chains, true_value=None, title="Running Mean Convergence",
                     filename="convergence_plot.png", show=False):
    """
    Running posterior mean for each chain. Useful for visual burn-in assessment.

    Parameters
    ----------
    chains     : array-like, shape (n_chains, n_samples)
    true_value : float, optional
    """
    chains = np.asarray(chains, dtype=float)
    n_chains, n = chains.shape
    iters = np.arange(1, n + 1)

    fig, ax = plt.subplots(figsize=(10, 4))
    for i, chain in enumerate(chains):
        running_mean = np.cumsum(chain) / iters
        ax.plot(iters, running_mean, lw=1.0, alpha=0.8, label=f"Chain {i + 1}")

    if true_value is not None:
        ax.axhline(true_value, color="crimson", lw=1.4,
                   linestyle="--", label=f"Reference = {true_value:.3f}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Running mean")
    ax.set_title(title)
    ax.legend(fontsize=8)
    _save(fig, filename)
    if show:
        plt.show()
    plt.close(fig)
    return fig


# ---------------------------------------------------------------------------
# Multi-chain trace (one panel per chain)
# ---------------------------------------------------------------------------

def multi_chain_trace(chains, title="Multi-chain Trace",
                      filename="multi_chain_trace.png", show=False):
    """
    Stacked trace panels, one per chain — standard in MCMC papers.

    Parameters
    ----------
    chains : array-like, shape (n_chains, n_samples)
    """
    chains = np.asarray(chains, dtype=float)
    n_chains = chains.shape[0]

    fig, axes = plt.subplots(n_chains, 1, figsize=(10, 2.5 * n_chains), sharex=True)
    if n_chains == 1:
        axes = [axes]

    for i, (ax, chain) in enumerate(zip(axes, chains)):
        ax.plot(chain, lw=0.6, color="steelblue", alpha=0.9)
        ax.axhline(chain.mean(), color="crimson", lw=1, linestyle="--")
        ax.set_ylabel(f"Chain {i + 1}")

    axes[-1].set_xlabel("Iteration (post burn-in)")
    fig.suptitle(title, y=1.01)
    fig.tight_layout()
    _save(fig, filename)
    if show:
        plt.show()
    plt.close(fig)
    return fig
