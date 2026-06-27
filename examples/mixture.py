"""
Example 2 — Bimodal Gaussian mixture stress test.

Target:
    p(x) = 0.5 * N(x; -2, 0.7^2) + 0.5 * N(x; +2, 0.7^2)

This example demonstrates how the proposal standard deviation (cand_std)
controls the sampler's ability to mix between well-separated modes:

  * Narrow proposal (sigma=0.5): high acceptance rate, poor inter-mode mixing,
    chain can get stuck in one mode — ESS is low relative to n.
  * Wide proposal  (sigma=3.0): lower acceptance rate but crosses the valley
    between modes, both peaks appear in the posterior — ESS is higher.

Run:
    python -m examples.mixture
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.sampler import metropolis_hastings
from src.targets import bimodal_log_target, bimodal_true_pdf
from src.diagnostics import chain_summary
from src.plots import (
    trace_plot,
    acf_plot,
    posterior_density_plot,
    convergence_plot,
    multi_chain_trace,
)


CONFIGS = [
    ("narrow", 0.5),
    ("wide",   3.0),
]


def run_config(label, cand_std):
    print(f"\n{'='*55}")
    print(f"  Proposal: {label}  (sigma={cand_std})")
    print(f"{'='*55}")

    result = metropolis_hastings(
        log_target=bimodal_log_target,
        init=0.0,
        n_iter=10_000,
        cand_std=cand_std,
        burn_in=1_000,
        thin=2,
        n_chains=4,
        seed=42,
    )

    chains = result["samples"]
    all_samples = chains.flatten()

    print(f"Acceptance rates: {result['acceptance_rates'].round(3)}")
    stats = chain_summary(chains)

    trace_plot(
        chains[0],
        title=f"Trace — Bimodal mixture ({label} proposal, sigma={cand_std})",
        filename=f"mix_trace_{label}.png",
    )
    acf_plot(chains[0], filename=f"mix_acf_{label}.png")
    multi_chain_trace(chains, filename=f"mix_multi_{label}.png")
    convergence_plot(chains, filename=f"mix_convergence_{label}.png")
    posterior_density_plot(
        all_samples,
        prior_pdf=bimodal_true_pdf,
        x_range=(-5, 5),
        title=f"Posterior — Bimodal Mixture ({label} proposal, sigma={cand_std})",
        filename=f"mix_posterior_{label}.png",
    )


def main():
    for label, cand_std in CONFIGS:
        run_config(label, cand_std)
    print("\nAll plots saved to outputs/")


if __name__ == "__main__":
    main()
