"""Generate ground-truth latent variables for the DECAT simulator.

Latent variables (z_s, z_h, z_r, b) are sampled independently from
multivariate normal distributions.  They are never observed by models
and are used only for simulation and evaluation.
"""

import numpy as np


def sample_latents(
    n: int,
    k: int,
    batch_mean: np.ndarray | None = None,
    rng: np.random.Generator | None = None,
) -> dict:
    """Sample all latent variables for *n* patients.

    Parameters
    ----------
    n : int
        Number of patients.
    k : int
        Latent dimensionality.
    batch_mean : array-like of shape (k,) or None
        Mean of the batch latent.  ``None`` is equivalent to the zero
        vector (Cohort A).  Use a non-zero vector for shifted cohorts
        (B, C).
    rng : np.random.Generator or None
        Random number generator.  A new default generator is created
        when ``None``.

    Returns
    -------
    dict with keys ``z_s``, ``z_h``, ``z_r``, ``b``, each of shape
    (n, k).
    """
    if rng is None:
        rng = np.random.default_rng()

    z_s = rng.standard_normal((n, k))
    z_h = rng.standard_normal((n, k))
    z_r = rng.standard_normal((n, k))

    if batch_mean is None:
        b = rng.standard_normal((n, k))
    else:
        batch_mean = np.asarray(batch_mean, dtype=np.float64)
        b = rng.standard_normal((n, k)) + batch_mean[np.newaxis, :]

    return {"z_s": z_s, "z_h": z_h, "z_r": z_r, "b": b}
