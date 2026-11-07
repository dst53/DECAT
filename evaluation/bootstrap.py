"""Bootstrap resampling utilities for confidence intervals.

Provides a generic bootstrap wrapper and a permutation-test helper
used across multiple DECAT metrics.
"""

from typing import Callable

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits


def bootstrap_ci(
    statistic_fn: Callable,
    n: int,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
) -> dict:
    """Compute a bootstrap confidence interval by resampling indices.

    Parameters
    ----------
    statistic_fn : callable
        ``statistic_fn(idx)`` returns a scalar statistic computed on the
        subset of data selected by index array *idx*.
    n : int
        Total number of observations to resample from.
    n_bootstrap : int
        Number of bootstrap replicates.
    alpha : float
        Significance level (default 0.05 for 95% CI).
    rng : np.random.Generator or None
    n_jobs : int
        Number of parallel workers (1 = sequential).

    Returns
    -------
    dict with keys:
        ``ci_lower``  – lower CI bound
        ``ci_upper``  – upper CI bound
        ``bootstrap`` – array of bootstrap statistic values
    """
    if rng is None:
        rng = np.random.default_rng()

    # Pre-generate all index arrays so the parent RNG stays deterministic
    indices = [rng.choice(n, size=n, replace=True) for _ in range(n_bootstrap)]

    if n_jobs == 1:
        boot_values = np.array([statistic_fn(idx) for idx in indices])
    else:
        boot_values = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(statistic_fn)(idx) for idx in indices
            )
        )

    ci_lower = np.percentile(boot_values, 100 * alpha / 2)
    ci_upper = np.percentile(boot_values, 100 * (1 - alpha / 2))

    return {
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "bootstrap": boot_values,
    }


def permutation_null(
    statistic_fn: Callable,
    n: int,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
) -> dict:
    """Estimate a null distribution via index permutation.

    Parameters
    ----------
    statistic_fn : callable
        ``statistic_fn(perm, local_rng)`` returns a scalar statistic
        computed after applying permutation array *perm*.  The second
        argument is the per-permutation RNG so the callee can draw
        additional random variates (e.g. projection directions) that
        are reproducible and independent across permutations.
    n : int
        Number of observations to permute.
    n_permutations : int
        Number of permutation resamples.
    rng : np.random.Generator or None
    n_jobs : int
        Number of parallel workers (1 = sequential).

    Returns
    -------
    dict with keys:
        ``null``       – array of null statistic values
        ``null_lower`` – 2.5th percentile
        ``null_upper`` – 97.5th percentile
    """
    if rng is None:
        rng = np.random.default_rng()

    rng_perm = np.random.default_rng(rng.integers(1 << 63))

    # Pre-generate seeds so results are reproducible and parallel-safe
    seeds = rng_perm.integers(1 << 63, size=n_permutations)

    def _one_permutation(seed):
        with threadpool_limits(limits=1):
            local_rng = np.random.default_rng(seed)
            perm = local_rng.permutation(n)
            return statistic_fn(perm, local_rng)

    if n_jobs == 1:
        null_values = np.array([_one_permutation(s) for s in seeds])
    else:
        null_values = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_one_permutation)(s) for s in seeds
            )
        )

    return {
        "null": null_values,
        "null_lower": float(np.percentile(null_values, 2.5)),
        "null_upper": float(np.percentile(null_values, 97.5)),
    }
