import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t


def log_g_fun(mu, n, y_bar):
    """
    Compute the log posterior distribution for mu.

    Parameters:
    - mu: The current value of mu.
    - n: The number of samples.
    - y_bar: The sample mean of y.

    Returns:
    - log_g_mu: The computed log posterior value.
    """
    log_g_mu = n * (y_bar * mu - 0.5 * mu * mu) - np.log(mu * mu + 1)
    return log_g_mu


def metropolis_hastings(n, y_bar, n_iter, mu_init, cand_std):
    """
    Implement the Metropolis-Hastings algorithm for MCMC sampling.

    Parameters:
    - n: The number of samples.
    - y_bar: The sample mean of y.
    - n_iter: The number of iterations for the algorithm.
    - mu_init: The initial value for mu.
    - cand_std: The standard deviation for the candidate distribution.

    Returns:
    - A list containing the sampled values of mu and the acceptance ratio.
    """
    mu_out = []
    accept = 0
    mu_now = mu_init
    lg_now = log_g_fun(mu_now, n, y_bar)

    for i in range(n_iter):
        mu_cand = np.random.normal(mu_now, cand_std, 1)
        lg_cand = log_g_fun(mu_cand, n, y_bar)
        lalpha = lg_cand - lg_now
        alpha = np.exp(lalpha[0])
        u = random.random()

        if u < alpha:
            mu_now = mu_cand[0]
            accept += 1
            lg_now = lg_cand

        mu_out.append(mu_now)

    return [mu_out, accept / n_iter]


def trace_plot(samples, description):
    """
    Plot the trace of MCMC samples.

    Parameters:
    - samples: The list of sampled values.
    - description: A string description for the plot title.
    """
    plt.figure(figsize=(10, 5))
    plt.plot(samples)
    plt.title(f"Trace Plot: {description}")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    plt.savefig("trace_plot.jpg", format="jpg", dpi=300)
    plt.show()


def plot_t_density(df=1, lty='--', add=False):
    """
    Plot the density of the t-distribution with specified degrees of freedom.

    Parameters:
    - df: Degrees of freedom (default is 1).
    - lty: Line style (default is '--').
    - add: Whether to add to the existing plot or create a new one (default is False).
    """
    x = np.linspace(-1, 3, 400)
    y = t.pdf(x, df)

    if not add:
        plt.figure()

    plt.plot(x, y, linestyle=lty, color='blue', label='Prior Distribution')


def density_estimate_plot(samples, description, x_range, prior_mean):
    """
    Plot the posterior density using kernel density estimation.

    Parameters:
    - samples: List of posterior samples.
    - description: Description for the plot title.
    - x_range: Tuple specifying the x-axis range.
    - prior_mean: Prior mean value for plotting a vertical line.
    """
    plt.figure(figsize=(10, 5))
    data = pd.DataFrame(samples, columns=['sampling'])
    data.sampling.plot.density(color='green', label='Posterior Distribution')
    plt.legend()
    plt.title(description)
    plt.xlim(x_range)
    plt.axvline(prior_mean, color='red', linestyle='-', label='y_bar on prior')
    plot_t_density(df=1, lty='--', add=True)
    plt.title(f'Prior and Posterior')
    plt.savefig("posterior_density_plot.jpg", format="jpg", dpi=300)
    plt.show()


def main(y, mu, std):
    """
    Main function to execute the MCMC sampling and plotting.

    Parameters:
    - y: List of data samples.
    - mu: Initial value for mu.
    - std: Standard deviation for the candidate distribution.
    """
    random.seed(42)
    ybar = np.mean(y)
    n = len(y)

    posterior_samples, acceptance_ratio = metropolis_hastings(n, ybar, 1000, mu, std)
    print(f'Acceptance Ratio: {acceptance_ratio}')

    trace_plot(posterior_samples, f"Mean {ybar} and Std {std}")
    density_estimate_plot(posterior_samples, "Density estimate on posterior distribution ", (-1, 3), ybar)


if __name__ == '__main__':
    y = [1.2, 1.4, -.5, .3, .9, 2.3, 1, .1, 1.3, 1.9]
    std = 0.9
    mu = 30  # Initial value for testing how many iterations are needed to be close to the real mean
    main(y, mu, std)
