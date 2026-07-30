import numpy as np


def hamiltonian_monte_carlo(log_target, grad_log_target, init, n_iter, step_size,
                             n_leapfrog, burn_in=0, thin=1, n_chains=1, seed=None):
    """
    Hamiltonian Monte Carlo sampler (unit mass, leapfrog integrator).

    Mirrors the interface of `samplers.random_walk.metropolis_hastings` so the
    two are drop-in alternatives in examples/diagnostics/plots.

    Parameters
    ----------
    log_target : callable
        Log of the (unnormalized) target density. Accepts a scalar and returns a scalar.
    grad_log_target : callable
        Gradient of log_target w.r.t. the position. Accepts a scalar and returns a scalar.
        See `grad_*` companions in `src/targets.py`.
    init : float or array-like
        Starting value(s), as in metropolis_hastings.
    n_iter : int
        Total MCMC iterations per chain (before burn-in removal and thinning).
    step_size : float
        Leapfrog integrator step size (epsilon).
    n_leapfrog : int
        Number of leapfrog steps per proposal (L).
    burn_in : int
        Number of initial samples to discard per chain.
    thin : int
        Retain every `thin`-th sample after burn-in.
    n_chains : int
        Number of independent chains.
    seed : int, optional
        Base random seed. Each chain gets seed + chain_index.

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
            p0 = rng.normal()
            q = current
            p = p0 + 0.5 * step_size * grad_log_target(q)

            for step in range(n_leapfrog):
                q = q + step_size * p
                if step != n_leapfrog - 1:
                    p = p + step_size * grad_log_target(q)

            log_q = log_target(q)
            p = p + 0.5 * step_size * grad_log_target(q)

            if np.isfinite(log_q) and np.isfinite(p):
                log_alpha = (log_q - 0.5 * p ** 2) - (log_current - 0.5 * p0 ** 2)
            else:
                log_alpha = -np.inf  # divergent trajectory — reject

            if np.log(rng.uniform()) < log_alpha:
                current = q
                log_current = log_q
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
