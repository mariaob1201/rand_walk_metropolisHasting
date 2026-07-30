"""
Original entry point for the Random Walk Metropolis-Hastings demo.

The MCMC engine and target definitions now live in src/ (see examples/normal_mean.py
for the full multi-chain workflow with diagnostics); this script keeps its original
plotting style and output paths but delegates sampling to the shared library instead
of carrying its own duplicate implementation.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t

from src.samplers import metropolis_hastings, hamiltonian_monte_carlo
from src.targets import make_normal_mean_target, make_normal_mean_grad


def trace_plot(samples, description):
    """
    trace plot on MC
    :param samples:
    :param description:
    :return:
    """

    # Plot the trace plot
    plt.figure(figsize=(10, 5))
    plt.plot(samples)
    plt.title(f"Trace Plot on {description}")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    # Save the plot as a JPG image
    plt.savefig("trace_plot.jpg", format="jpg", dpi=300)


def plot_t_density(df=1, lty='--', add=False):
    """
    Plot the density of the t-distribution with specified degrees of freedom.

    Parameters:
    - df: degrees of freedom (default is 1).
    - lty: line style (default is '--', which corresponds to lty=2 in R).
    - add: whether to add to the existing plot or create a new one (default is False).
    """
    x = np.linspace(-1, 3, 400)
    y = t.pdf(x, df)

    if not add:
        plt.figure()

    plt.plot(x, y, linestyle=lty, color='blue', label='Prior Distribution')
    plt.legend()
    plt.xlabel("Value")
    plt.ylabel("Density")


def density_estimate_plot(samples_by_sampler, description, x_range, prior_mean):
    """

    :param samples_by_sampler: dict of {sampler_label: posterior samples} — one density
        curve is drawn per entry, so multiple samplers can be overlaid on one plot
    :param description:
    :param x_range:
    :param prior_mean:
    :return:
    """

    plt.figure(figsize=(10, 5))
    # Posterior — one curve per sampler
    colors = ['green', 'darkorange', 'purple', 'brown']
    for (label, samples), color in zip(samples_by_sampler.items(), colors):
        data = pd.DataFrame(samples, columns=['sampling'])
        data.sampling.plot.density(color=color, label=label)

    plt.legend()
    plt.title(description)

    plt.xlim(x_range)
    # Prior mean
    plt.axvline(prior_mean, color='red', linestyle='-', label='y_bar on prior')

    plt.xlabel("Value")
    plt.ylabel("Density")
    # Save the plot as a JPG image
    plot_t_density(df=1, lty='--', add=True)
    plt.title(f'Prior and Posterior')

    filename = "posterior_density_plot.jpg"
    plt.savefig(filename, format="jpg", dpi=300)
    plt.show()


# set up
def main(y, mu, std):
    """

    :param y:
    :param mu:
    :param std:
    :return:
    """
    y = np.asarray(y, dtype=float)
    ybar = y.mean()
    n = len(y)

    log_target = make_normal_mean_target(n, ybar)
    grad_target = make_normal_mean_grad(n, ybar)

    rwm_result = metropolis_hastings(log_target, init=mu, n_iter=1000, cand_std=std, seed=42)
    hmc_result = hamiltonian_monte_carlo(log_target, grad_target, init=mu, n_iter=1000,
                                         step_size=0.05, n_leapfrog=20, seed=42)

    rwm_samples = rwm_result["samples"][0]
    hmc_samples = hmc_result["samples"][0]

    trace_plot(rwm_samples, f"Mean {ybar} and Std {std} -- Acceptance ratio {rwm_result['acceptance_rates'][0]}")
    density_estimate_plot(
        {
            "RWM posterior": rwm_samples,
            f"HMC posterior (accept={hmc_result['acceptance_rates'][0]:.2f})": hmc_samples,
        },
        "Density estimate on posterior distribution ", (-1, 3), ybar,
    )


if __name__ == '__main__':
    # Example
    y = [1.2, 1.4, -.5, .3, .9, 2.3, 1, .1, 1.3, 1.9]
    std = np.std(y)

    mu = 30  # crazy intial value to test how many iterations are needed to get close to the true mean
    main(y, mu, std)
