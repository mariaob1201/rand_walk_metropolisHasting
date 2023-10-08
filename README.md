# Random Walk by Metropolis-Hastings Sampler for Bayesian Inference

This Python script demonstrates the application of the Metropolis-Hastings algorithm, a Markov chain Monte Carlo (MCMC) method, to perform Bayesian inference on company personnel data.

## Overview
Given the percent change in personnel for a company 
y_i given the mean as random and identically distributed and independent variables,
with mean mu and variance 1. Our prior distribution on mu is the t-distribution with location 0, scale parameter 1, and degrees of freedom 1. 
The goal is to sample from the posterior distribution of the first using the Metropolis-Hastings algorithm.

Key Functions:
- log_g_fun(mu, n, y_bar): Computes the log posterior distribution for mu
- metropolis_hastings(n, y_bar, n_iter, mu_init, cand_std)
Implements the Metropolis-Hastings algorithm. Returns posterior samples and acceptance ratio.
- trace_plot(samples, description)
Generates and saves a trace plot for the provided samples.
