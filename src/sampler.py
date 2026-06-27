import numpy as np


def metropolis_hastings(log_target, init, n_iter, cand_std,
                        burn_in=0, thin=1, n_chains=1, seed=None):
    """
    Random Walk Metropolis-Hastings sampler.

    Parameters
    ----------
    log_target : callable
        Log of the (unnormalized) target density. Accepts a scalar and returns a scalar.
    init : float or array-like
        Starting value(s). If n_chains > 1 and init is a scalar, all chains share the
        same starting point. Pass an array of length n_chains for distinct starts.
    n_iter : int
        Total MCMC iterations per chain (before burn-in removal and thinning).
    cand_std : float
        Standard deviation of the symmetric Normal proposal: mu' ~ N(mu, cand_std^2).
    burn_in : int
        Number of initial samples to discard per chain. Must be < n_iter.
    thin : int
        Retain every `thin`-th sample after burn-in (1 = keep all).
    n_chains : int
        Number of independent chains (required for Gelman-Rubin R-hat).
    seed : int, optional
        Base random seed. Each chain gets seed + chain_index for independence.

    Returns
    -------
    dict with keys:
        'samples'          : np.ndarray, shape (n_chains, n_kept)
        'acceptance_rates' : np.ndarray, shape (n_chains,)
        'n_iter'           : int
        'burn_in'          : int
        'thin'             : int
    """
    if burn_in >= n_iter:
        raise ValueError("burn_in must be less than n_iter")

    inits = np.full(n_chains, float(init)) if np.isscalar(init) else np.asarray(init, dtype=float)

    all_samples = []
    acceptance_rates = []

    for chain_idx in range(n_chains):
        rng = np.random.default_rng(None if seed is None else seed + chain_idx)

        current = inits[chain_idx % len(inits)]
        log_current = log_target(current)

        raw = np.empty(n_iter)
        n_accept = 0

        for i in range(n_iter):
            candidate = rng.normal(current, cand_std)
            log_cand = log_target(candidate)

            log_alpha = log_cand - log_current
            if np.log(rng.uniform()) < log_alpha:
                current = candidate
                log_current = log_cand
                n_accept += 1

            raw[i] = current

        kept = raw[burn_in::thin]
        all_samples.append(kept)
        acceptance_rates.append(n_accept / n_iter)

    return {
        "samples": np.array(all_samples),
        "acceptance_rates": np.array(acceptance_rates),
        "n_iter": n_iter,
        "burn_in": burn_in,
        "thin": thin,
    }
