"""
Example 3 — Compare Random Walk Metropolis-Hastings vs Hamiltonian Monte Carlo.

Runs both samplers on the same two targets (Normal mean, bimodal mixture),
timing each run and reporting acceptance rate, ESS, ESS/sec, and R-hat
side by side, plus overlaid trace and posterior-density plots.

Run:
    python -m examples.compare_samplers
"""
import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt

from src.samplers import metropolis_hastings, hamiltonian_monte_carlo
from src.targets import (
    make_normal_mean_target, make_normal_mean_grad, cauchy_prior_pdf,
    bimodal_log_target, grad_bimodal_log_target, bimodal_true_pdf,
)
from src.diagnostics import effective_sample_size, gelman_rubin
from src.plots import OUTPUT_DIR


def run_timed(fn, **kwargs):
    start = time.perf_counter()
    result = fn(**kwargs)
    result["runtime_sec"] = time.perf_counter() - start
    return result


def summarize(name, result):
    chains = result["samples"]
    ess = np.array([effective_sample_size(chains[i]) for i in range(chains.shape[0])])
    r_hat = gelman_rubin(chains) if chains.shape[0] >= 2 else float("nan")
    runtime = result["runtime_sec"]
    return {
        "name": name,
        "chains": chains,
        "accept_rate": result["acceptance_rates"].mean(),
        "ess_mean": ess.mean(),
        "ess_per_sec": ess.mean() / runtime,
        "r_hat": r_hat,
        "runtime_sec": runtime,
    }


def print_table(rows):
    header = f"{'Sampler':<12} {'Accept':>8} {'ESS':>10} {'ESS/sec':>10} {'R-hat':>8} {'Time(s)':>9}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['name']:<12} {r['accept_rate']:>8.3f} {r['ess_mean']:>10.1f} "
              f"{r['ess_per_sec']:>10.1f} {r['r_hat']:>8.4f} {r['runtime_sec']:>9.3f}")


def plot_comparison(rows, true_pdf, x_range, title_prefix, filename_prefix):
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.5))

    # Overlaid trace (chain 1) -------------------------------------------------
    ax = axes[0]
    for r in rows:
        ax.plot(r["chains"][0], lw=0.6, alpha=0.8, label=r["name"])
    ax.set(xlabel="Iteration (post burn-in, thinned)", ylabel="Value",
           title=f"{title_prefix} — trace (chain 1)")
    ax.legend()

    # Overlaid posterior density -----------------------------------------------
    import pandas as pd
    ax = axes[1]
    xs = np.linspace(*x_range, 400)
    if true_pdf is not None:
        ax.plot(xs, true_pdf(xs), color="black", lw=1.5, linestyle=":", label="True/prior ref.")
    for r in rows:
        pd.Series(r["chains"].flatten()).plot.density(ax=ax, lw=2, alpha=0.85, label=r["name"])
    ax.set(xlabel="Value", ylabel="Density", xlim=x_range,
           title=f"{title_prefix} — posterior density")
    ax.legend()

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{filename_prefix}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Saved: {path}")
    plt.close(fig)


def compare_normal_mean():
    y = np.array([1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9])
    y_bar, n = y.mean(), len(y)
    log_target = make_normal_mean_target(n, y_bar)
    grad_target = make_normal_mean_grad(n, y_bar)

    common = dict(init=30.0, n_iter=5_000, burn_in=500, thin=1, n_chains=4, seed=42)

    rwm = run_timed(metropolis_hastings, log_target=log_target, cand_std=y.std(), **common)
    hmc = run_timed(hamiltonian_monte_carlo, log_target=log_target, grad_log_target=grad_target,
                    step_size=0.05, n_leapfrog=20, **common)

    rows = [summarize("RWM", rwm), summarize("HMC", hmc)]
    print("\n=== Normal mean model ===")
    print_table(rows)
    plot_comparison(rows, cauchy_prior_pdf, (-1, 3), "Normal Mean", "compare_normal_mean")
    return rows


def compare_bimodal():
    common = dict(init=0.0, n_iter=10_000, burn_in=1_000, thin=2, n_chains=4, seed=42)

    rwm = run_timed(metropolis_hastings, log_target=bimodal_log_target, cand_std=3.0, **common)
    hmc = run_timed(hamiltonian_monte_carlo, log_target=bimodal_log_target,
                    grad_log_target=grad_bimodal_log_target,
                    step_size=0.1, n_leapfrog=20, **common)

    rows = [summarize("RWM", rwm), summarize("HMC", hmc)]
    print("\n=== Bimodal mixture ===")
    print_table(rows)
    plot_comparison(rows, bimodal_true_pdf, (-5, 5), "Bimodal Mixture", "compare_bimodal")
    return rows


def main():
    compare_normal_mean()
    compare_bimodal()
    print("\nAll comparison plots saved to outputs/")


if __name__ == "__main__":
    main()
