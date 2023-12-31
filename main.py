import random
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t


def log_pi_fun(mu, n, y_bar):
    """
    For computational reasons we take the log-posterior of the target function g, log(g(mu)), as follows
    :param y_hat:
    :param n:
    :param mu:
    :return:
    """
    log_g_mu = n*(y_bar*mu - .5*mu**2)-np.log(1+mu**2)
    return log_g_mu

def metropolis_hastings(n, y_bar, n_iter, mu_init, cand_std):
    """
    Metropolis hastings
    :param n: sample size
    :param y_bar: the sample mean of y the data.
    :param n_iter: how many iterations
    :param mu_init: initial valur for mu
    :param cand_std: std deviation for the candidate generating normal distribution
    :return: 
    """
    # Selecting an initial values and initializing the Random walk Metropolis sampling
    mu_out = [] #numeric vector of mu accepted values
    accept = 0 #how many times we accepted
    mu_now = mu_init #updating our value for mu, intitalized as the mu_init, our param
    lg_now = log_pi_fun(mu_now, n, y_bar) #evaluation of lg function

    # MC iterations under accepting ratio
    for i in range(0, n_iter):
        mu_cand = np.random.normal(mu_now, cand_std, 1) #draw a candidate from the proposal distribution, normal with mean mu_now and std =cand_std
        #by last candidate assumption, a symmetric one, we now turn into a Metropolis algorithm 
        lg_cand = log_pi_fun(mu_cand, n, y_bar) #the numerator of the acceptance ration, in a logarithm scale

        # Acceptance ratio
        lalpha = lg_cand - lg_now #this is the log ratio, for posterior on the candidate minus the current value lg_now
        alpha = np.exp(lalpha[0]) #turned it back into a non log-scale

        u = random.random()
        # Acceptance condition to update the candidate
        if u<alpha: #that gives us an event that happens with probability alpha
            mu_now_ = mu_cand
            try:
                mu_now = mu_now_[0]
            except:
                mu_now = mu_now_

            accept+=1
            lg_now = lg_cand #updated the lg_now too as the one evaluated in the candidate

        mu_out.append(mu_now)
    
    #returning the mean estimated together with acceptance ratio
    return [mu_out, accept/n_iter]

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


def density_estimate_plot(samples, description, x_range, prior_mean):
    """

    :param samples: list of posterior samples
    :param description:
    :param x_range:
    :param prior_mean:
    :return:
    """

    plt.figure(figsize=(10, 5))
    # Posterior
    data = pd.DataFrame(samples, columns=['sampling'])
    data.sampling.plot.density(color='green', label='Posterior Distribution')

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
    random.seed(42)
    ybar = np.mean(y)
    n = len(y)

    posterior_samples = metropolis_hastings(n, ybar, 1000, mu, std)
    samples = [arr for arr in posterior_samples[0]]

    trace_plot(samples, f"Mean {ybar} and Std {std} -- Acceptance ratio {posterior_samples[-1]}")
    density_estimate_plot(samples, "Density estimate on posterior distribution ", (-1,3), ybar)

if __name__ == '__main__':
    # Example
    y = [1.2, 1.4, -.5, .3, .9, 2.3, 1, .1, 1.3, 1.9]
    std = np.std(y)
    
    mu = 30 # crazy intial value to test how many iterations are needed to get close to the true mean
    main(y, mu, std)
