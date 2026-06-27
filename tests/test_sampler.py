"""
Unit tests for the RWMH sampler and diagnostics.
Run with:  python -m pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest
from src.sampler import metropolis_hastings
from src.targets import make_normal_mean_target, bimodal_log_target
from src.diagnostics import effective_sample_size, gelman_rubin, autocorrelation


# ---------------------------------------------------------------------------
# Sampler — output shapes and basic invariants
# ---------------------------------------------------------------------------

class TestSamplerOutput:
    def test_single_chain_shape(self):
        log_t = make_normal_mean_target(10, 1.0)
        result = metropolis_hastings(log_t, init=1.0, n_iter=200, cand_std=1.0,
                                     burn_in=0, thin=1, n_chains=1, seed=0)
        assert result["samples"].shape == (1, 200)

    def test_burn_in_removes_samples(self):
        log_t = make_normal_mean_target(10, 1.0)
        result = metropolis_hastings(log_t, init=1.0, n_iter=500, cand_std=1.0,
                                     burn_in=100, thin=1, n_chains=1, seed=0)
        assert result["samples"].shape == (1, 400)

    def test_thinning(self):
        log_t = make_normal_mean_target(10, 1.0)
        result = metropolis_hastings(log_t, init=1.0, n_iter=500, cand_std=1.0,
                                     burn_in=0, thin=5, n_chains=1, seed=0)
        assert result["samples"].shape == (1, 100)

    def test_burn_in_and_thinning(self):
        log_t = make_normal_mean_target(10, 1.0)
        result = metropolis_hastings(log_t, init=1.0, n_iter=600, cand_std=1.0,
                                     burn_in=100, thin=5, n_chains=1, seed=0)
        # (600 - 100) / 5 = 100
        assert result["samples"].shape == (1, 100)

    def test_multiple_chains(self):
        log_t = make_normal_mean_target(10, 1.0)
        result = metropolis_hastings(log_t, init=1.0, n_iter=300, cand_std=1.0,
                                     burn_in=50, thin=1, n_chains=4, seed=0)
        assert result["samples"].shape == (4, 250)
        assert result["acceptance_rates"].shape == (4,)

    def test_acceptance_rates_in_range(self):
        log_t = make_normal_mean_target(10, 1.0)
        result = metropolis_hastings(log_t, init=1.0, n_iter=1000, cand_std=1.0,
                                     n_chains=4, seed=7)
        rates = result["acceptance_rates"]
        assert np.all(rates >= 0) and np.all(rates <= 1)

    def test_burn_in_too_large_raises(self):
        log_t = make_normal_mean_target(10, 1.0)
        with pytest.raises(ValueError):
            metropolis_hastings(log_t, init=1.0, n_iter=100, cand_std=1.0, burn_in=100)

    def test_distinct_seeds_give_different_chains(self):
        log_t = make_normal_mean_target(10, 1.0)
        r1 = metropolis_hastings(log_t, init=1.0, n_iter=200, cand_std=1.0,
                                  n_chains=1, seed=0)
        r2 = metropolis_hastings(log_t, init=1.0, n_iter=200, cand_std=1.0,
                                  n_chains=1, seed=99)
        assert not np.allclose(r1["samples"], r2["samples"])

    def test_same_seed_reproducible(self):
        log_t = make_normal_mean_target(10, 1.0)
        r1 = metropolis_hastings(log_t, init=1.0, n_iter=200, cand_std=1.0,
                                  n_chains=1, seed=42)
        r2 = metropolis_hastings(log_t, init=1.0, n_iter=200, cand_std=1.0,
                                  n_chains=1, seed=42)
        np.testing.assert_array_equal(r1["samples"], r2["samples"])


# ---------------------------------------------------------------------------
# Sampler — statistical correctness (large-n checks)
# ---------------------------------------------------------------------------

class TestSamplerConvergence:
    def test_normal_mean_posterior_mean(self):
        """Posterior mean should be close to y_bar for a concentrated likelihood."""
        y = np.array([1.2, 1.4, -0.5, 0.3, 0.9, 2.3, 1.0, 0.1, 1.3, 1.9])
        y_bar = y.mean()
        log_t = make_normal_mean_target(len(y), y_bar)
        result = metropolis_hastings(log_t, init=0.0, n_iter=10_000, cand_std=0.5,
                                     burn_in=1_000, thin=1, n_chains=1, seed=42)
        post_mean = result["samples"].mean()
        assert abs(post_mean - y_bar) < 0.15, f"posterior mean {post_mean:.3f} far from y_bar {y_bar:.3f}"

    def test_bimodal_visits_both_modes(self):
        """Wide proposal should visit both modes of the bimodal target."""
        result = metropolis_hastings(bimodal_log_target, init=0.0,
                                     n_iter=20_000, cand_std=3.0,
                                     burn_in=2_000, thin=1, n_chains=1, seed=0)
        s = result["samples"].flatten()
        frac_left  = (s < 0).mean()
        frac_right = (s > 0).mean()
        # Both modes should get at least 30 % of visits with a wide proposal
        assert frac_left  > 0.30, f"left mode undersampled: {frac_left:.2f}"
        assert frac_right > 0.30, f"right mode undersampled: {frac_right:.2f}"


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

class TestDiagnostics:
    def _iid_chain(self, n=2000, seed=0):
        rng = np.random.default_rng(seed)
        return rng.standard_normal(n)

    def test_autocorrelation_lag0_is_one(self):
        acf = autocorrelation(self._iid_chain(), max_lag=10)
        assert abs(acf[0] - 1.0) < 1e-10

    def test_iid_acf_small_lags(self):
        """ACF of i.i.d. samples should be near 0 beyond lag 0."""
        acf = autocorrelation(self._iid_chain(5000), max_lag=20)
        assert np.all(np.abs(acf[1:]) < 0.1)

    def test_ess_leq_n(self):
        chain = self._iid_chain()
        ess = effective_sample_size(chain)
        assert ess <= len(chain) * 1.05  # small numerical slack

    def test_ess_positive(self):
        log_t = make_normal_mean_target(10, 1.0)
        result = metropolis_hastings(log_t, init=1.0, n_iter=2000, cand_std=0.5,
                                     burn_in=200, n_chains=1, seed=0)
        ess = effective_sample_size(result["samples"][0])
        assert ess > 0

    def test_gelman_rubin_converged(self):
        """Well-mixed chains should give R-hat < 1.05."""
        log_t = make_normal_mean_target(10, 1.0)
        result = metropolis_hastings(log_t, init=1.0, n_iter=5_000, cand_std=0.5,
                                     burn_in=500, n_chains=4, seed=42)
        r_hat = gelman_rubin(result["samples"])
        assert r_hat < 1.05, f"R-hat = {r_hat:.4f} suggests poor mixing"

    def test_gelman_rubin_requires_two_chains(self):
        with pytest.raises(ValueError):
            gelman_rubin(np.ones((1, 100)))

    def test_gelman_rubin_requires_2d(self):
        with pytest.raises(ValueError):
            gelman_rubin(np.ones(100))
