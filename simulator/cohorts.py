"""Cohort generation for the DECAT simulator.

Four disjoint cohorts are generated per run:

- Cohort A  : representation training set       (b ~ N(0, I))
- Cohort A' : within-cohort validation set       (b ~ N(0, I))
- Cohort B  : external cohort with batch shift   (b ~ N(mu_b, I))
- Cohort C  : external cohort with scaled shift  (b ~ N((1+epsilon)*mu_b, I))
"""

import numpy as np

from simulator.latents import sample_latents


def sample_batch_shift(k: int, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a fixed cohort-shift vector mu_b ~ N(0, I).

    Parameters
    ----------
    k : int
        Latent dimensionality.
    rng : np.random.Generator or None

    Returns
    -------
    mu_b : ndarray of shape (k,)
    """
    if rng is None:
        rng = np.random.default_rng()
    return rng.standard_normal(k)


def generate_cohorts(
    n_a: int,
    n_a_prime: int,
    n_b: int,
    n_c: int,
    k: int,
    mu_b: np.ndarray,
    epsilon: float = 0.2,
    rng: np.random.Generator | None = None,
) -> dict:
    """Generate latent variables for all four cohorts.

    Parameters
    ----------
    n_a, n_a_prime, n_b, n_c : int
        Number of patients in each cohort.
    k : int
        Latent dimensionality.
    mu_b : ndarray of shape (k,)
        Batch shift vector for Cohort B.
    epsilon : float
        Relative shift magnitude between B and C.
        Cohort C uses ``(1 + epsilon) * mu_b``.
    rng : np.random.Generator or None

    Returns
    -------
    dict mapping cohort name (``"A"``, ``"A_prime"``, ``"B"``, ``"C"``)
    to a latent dict as returned by :func:`~simulator.latents.sample_latents`.
    """
    if rng is None:
        rng = np.random.default_rng()

    rng_a       = np.random.default_rng(rng.integers(1 << 63))
    rng_a_prime = np.random.default_rng(rng.integers(1 << 63))
    rng_b       = np.random.default_rng(rng.integers(1 << 63))
    rng_c       = np.random.default_rng(rng.integers(1 << 63))

    mu_b_C = (1.0 + epsilon) * mu_b

    cohorts = {
        "A":       sample_latents(n_a,       k, batch_mean=None,   rng=rng_a),
        "A_prime": sample_latents(n_a_prime, k, batch_mean=None,   rng=rng_a_prime),
        "B":       sample_latents(n_b,       k, batch_mean=mu_b,   rng=rng_b),
        "C":       sample_latents(n_c,       k, batch_mean=mu_b_C, rng=rng_c),
    }
    return cohorts
