# This is a sample Python script.
import matplotlib
import random
import numpy as np
import matplotlib.pyplot as plt

"""
So yi represents the percent change in personnel for company i, and given mu, the mean,
each of these ys is identically distributed and independent from this normal distribution
with mean mu and variance 1. Our prior distribution on mu is the t-distribution with location 0,
scale parameter 1 and degrees of freedom 1.

To get posterior samples, we're going to need to setup a Markov chain,
who's stationary distribution is the posterior distribution we want.
"""

def log_g_fun(mu, n, y_bar):
    """
    posterior distribution for mu, log(g(mu))
    :param y_hat:
    :param n:
    :param mu:
    :return:
    """
    log_g_mu = n*(y_bar*mu - .5*mu*mu)-np.log(mu*mu+1)
    return log_g_mu

def metropolis_hastings(n, y_bar, n_iter, mu_init, cand_std):
    """
    Metropolis hastings
    :param n:
    :param y_bar: the sample mean of y.
    :param n_iter: how many iterations
    :param mu_init: initial valur for mu
    :param cand_std: std deviation for candidate
    :return:
    """
    # Selecting an initial value
    mu_out = []
    accept = 0
    mu_now = mu_init
    lg_now = log_g_fun(mu_now, n, y_bar)

    # Iterations
    for i in range(0, n_iter):
        mu_cand = np.random.normal(mu_now, cand_std, 1)
        lg_cand = log_g_fun(mu_cand, n, y_bar)
        # Acceptance ratio
        lalpha = lg_cand - lg_now
        #print(f"lalpha {lalpha[0]}")
        alpha = np.exp(lalpha[0])

        u = random.random()
        if u<alpha:
            mu_now_ = mu_cand
            try:
                mu_now = mu_now_[0]
            except:
                mu_now = mu_now_

            accept+=1
            lg_now = lg_cand

        mu_out.append(mu_now)

    return [mu_out, accept/n_iter]

def trace_plot(samples, description):

    # Plot the trace plot
    plt.figure(figsize=(10, 5))
    plt.plot(samples)
    plt.title(f"Trace Plot on {description}")
    plt.xlabel("Iteration")
    plt.ylabel("Value")
    # Save the plot as a JPG image
    plt.savefig("trace_plot.jpg", format="jpg", dpi=300)

# set up
def main(y, mu, std):
    # seed
    random.seed(42)
    ybar = np.mean(y)
    n = len(y)

    posterior_samples = metropolis_hastings(n, ybar, 1000, mu, std)
    print('Posterior: ', posterior_samples[0])
    print('Acceptance Ratio: ', posterior_samples[-1])
    description = f"Mean {mu} and Std {std}"
    trace_plot([arr for arr in posterior_samples[0]], description)
    return posterior_samples

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    y = [1.2, 1.4, -.5, .3, .9, 2.3, 1, .1, 1.3, 1.9]
    std = 0.9
    mu = 30 # crazy intial value
    main(y, mu, std)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
