"""
Unit tests for the HMC sampler.
Run with:  python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import numpy as np
import pytest
from src.samplers import hamiltonian_monte_carlo
from src.targets import make_normal_mean_target, make_normal_mean_grad, bimodal_log_target, grad_bimodal_log_target


# ---------------------------------------------------------------------------
# Sampler — output shapes and basic invariants
# ---------------------------------------------------------------------------

class TestHMCOutput:
    def test_single_chain_shape(self):
        log_t = make_normal_mean_target(10, 1.0)
        grad_t = make_normal_mean_grad(10, 1.0)
        result = hamiltonian_monte_carlo(log_t, grad_t, init=1.0, n_iter=200,
                                         step_size=0.05, n_leapfrog=10,
                                         burn_in=0, thin=1, n_chains=1, seed=0)
        assert result["samples"].shape == (1, 200)

    def test_burn_in_removes_samples(self):
        log_t = make_normal_mean_target(10, 1.0)
        grad_t = make_normal_mean_grad(10, 1.0)
        result = hamiltonian_monte_carlo(log_t, grad_t, init=1.0, n_iter=500,
                                         step_size=0.05, n_leapfrog=10,
                                         burn_in=100, thin=1, n_chains=1, seed=0)
        assert result["samples"].shape == (1, 400)

    def test_thinning(self):
        log_t = make_normal_mean_target(10, 1.0)
        grad_t = make_normal_mean_grad(10, 1.0)
        result = hamiltonian_monte_carlo(log_t, grad_t, init=1.0, n_iter=500,
                                         step_size=0.05, n_leapfrog=10,
                                         burn_in=0, thin=5, n_chains=1, seed=0)
        assert result["samples"].shape == (1, 100)

    def test_multiple_chains(self):
        log_t = make_normal_mean_target(10, 1.0)
        grad_t = make_normal_mean_grad(10, 1.0)
        result = hamiltonian_monte_carlo(log_t, grad_t, init=1.0, n_iter=300,
                                         step_size=0.05, n_leapfrog=10,
                                         burn_in=50, thin=1, n_chains=4, seed=0)
        assert result["samples"].shape == (4, 250)
        assert result["acceptance_rates"].shape == (4,)

    def test_acceptance_rates_in_range(self):
        log_t = make_normal_mean_target(10, 1.0)
        grad_t = make_normal_mean_grad(10, 1.0)
        result = hamiltonian_monte_carlo(log_t, grad_t, init=1.0, n_iter=500,
                                         step_size=0.05, n_leapfrog=10,
                                         n_chains=4, seed=7)
        rates = result["acceptance_rates"]
        assert np.all(rates >= 0) and np.all(rates <= 1)

    def test_burn_in_too_large_raises(self):
        log_t = make_normal_mean_target(10, 1.0)
        grad_t = make_normal_mean_grad(10, 1.0)
        with pytest.raises(ValueError):
            hamiltonian_monte_carlo(log_t, grad_t, init=1.0, n_iter=100,
                                    step_size=0.05, n_leapfrog=10, burn_in=100)

    def test_same_seed_reproducible(self):
        log_t = make_normal_mean_target(10, 1.0)
        grad_t = make_normal_mean_grad(10, 1.0)
        r1 = hamiltonian_monte_carlo(log_t, grad_t, init=1.0, n_iter=200,
                                     step_size=0.05, n_leapfrog=10, n_chains=1, seed=42)
        r2 = hamiltonian_monte_carlo(log_t, grad_t, init=1.0, n_iter=200,
                                     step_size=0.05, n_leapfrog=10, n_chains=1, seed=42)
        np.testing.assert_array_equal(r1["samples"], r2["samples"])


# ---------------------------------------------------------------------------
# Sampler — statistical correctness (large-n checks)
# ---------------------------------------------------------------------------

class TestHMCConvergence:
    def test_normal_mean_posterior_mean(self):
        """Posterior mean should be close to y_bar for a concentrated likelihood."""
        y = np.array([1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9])
        y_bar = y.mean()
        log_t = make_normal_mean_target(len(y), y_bar)
        grad_t = make_normal_mean_grad(len(y), y_bar)
        result = hamiltonian_monte_carlo(log_t, grad_t, init=0.0, n_iter=5_000,
                                         step_size=0.02, n_leapfrog=20,
                                         burn_in=500, thin=1, n_chains=1, seed=42)
        post_mean = result["samples"].mean()
        assert abs(post_mean - y_bar) < 0.15, f"posterior mean {post_mean:.3f} far from y_bar {y_bar:.3f}"

    def test_bimodal_gradient_matches_finite_difference(self):
        """Sanity-check the analytic gradient against a numerical one."""
        xs = np.linspace(-4, 4, 9)
        eps = 1e-5
        for x in xs:
            numeric = (bimodal_log_target(x + eps) - bimodal_log_target(x - eps)) / (2 * eps)
            analytic = grad_bimodal_log_target(x)
            assert abs(numeric - analytic) < 1e-3, f"gradient mismatch at x={x}: {numeric} vs {analytic}"
