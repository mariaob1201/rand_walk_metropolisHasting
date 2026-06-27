"""
Example 1 — Bayesian inference on a normal mean with a Cauchy prior.

Model:
    y_i | mu  ~ N(mu, 1)    i.i.d., known variance
    mu        ~ Cauchy(0,1) [t distribution, df=1]

The initial value mu_init=30 is intentionally far from the true mean to
show that 4 chains with burn-in of 500 iterations converge reliably.

Run:
    python -m examples.normal_mean
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from src.sampler import metropolis_hastings
from src.targets import make_normal_mean_target, cauchy_prior_pdf
from src.diagnostics import chain_summary
from src.plots import (
    trace_plot,
    acf_plot,
    posterior_density_plot,
    convergence_plot,
    multi_chain_trace,
)


def main():
    # ----- Data -----
    y = np.array([1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9])
    y_bar = y.mean()
    n = len(y)
    print(f"Data: n={n},  y_bar={y_bar:.4f},  std={y.std():.4f}")

    # ----- Sampler -----
    log_target = make_normal_mean_target(n, y_bar)

    result = metropolis_hastings(
        log_target=log_target,
        init=30.0,           # deliberately far from truth
        n_iter=5_000,
        cand_std=y.std(),
        burn_in=500,
        thin=1,
        n_chains=4,
        seed=42,
    )

    chains = result["samples"]          # shape (4, 4500)
    all_samples = chains.flatten()

    # ----- Diagnostics -----
    print("\n--- Chain diagnostics ---")
    stats = chain_summary(chains)
    print(f"\nPosterior mean  : {all_samples.mean():.4f}")
    print(f"Posterior std   : {all_samples.std():.4f}")
    print(f"95% credible interval: [{np.percentile(all_samples, 2.5):.4f}, "
          f"{np.percentile(all_samples, 97.5):.4f}]")

    # ----- Plots -----
    trace_plot(
        chains[0],
        title=f"Trace Plot — chain 1 (init=30, y̅={y_bar:.2f})",
        filename="nm_trace.png",
    )
    acf_plot(chains[0], filename="nm_acf.png")
    multi_chain_trace(chains, filename="nm_multi_chain.png")
    convergence_plot(chains, true_value=y_bar, filename="nm_convergence.png")
    posterior_density_plot(
        all_samples,
        prior_pdf=cauchy_prior_pdf,
        true_value=y_bar,
        x_range=(-1, 3),
        title="Prior vs Posterior — Normal Mean Model",
        filename="nm_posterior.png",
    )
    print("\nAll plots saved to outputs/")


if __name__ == "__main__":
    main()
