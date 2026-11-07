"""Metric A — Cross-Modal Shared Latent Agreement (A_norm).

Measures whether shared representations inferred from different
modalities agree for the *same patient* more than expected by chance.

    A_norm = Agreement_obs - Agreement_null

where Agreement_obs is the mean cosine similarity of matched patient
pairs and Agreement_null is the mean cosine similarity of mismatched
pairs.

Applicable to **all models** (entangled and factorized).
"""

import numpy as np

from evaluation.bootstrap import permutation_null


def _cosine_similarity_rows(A, B):
    """Row-wise cosine similarity between two (n, d) matrices."""
    A_hat = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-12)
    B_hat = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-12)
    return np.sum(A_hat * B_hat, axis=1)


def _mean_cross_cosine(A, B):
    """Mean cosine similarity between all pairs (i, j) where i != j."""
    A_hat = A / np.linalg.norm(A, axis=1, keepdims=True).clip(min=1e-12)
    B_hat = B / np.linalg.norm(B, axis=1, keepdims=True).clip(min=1e-12)
    sim_matrix = A_hat @ B_hat.T
    n = sim_matrix.shape[0]
    # Exclude diagonal (matched pairs)
    mask = ~np.eye(n, dtype=bool)
    return sim_matrix[mask].mean()


def compute_anorm(z_m1: np.ndarray, z_m2: np.ndarray) -> float:
    """Compute A_norm (point estimate).

    Parameters
    ----------
    z_m1 : ndarray of shape (n, d)
        Shared representations inferred from modality 1.
    z_m2 : ndarray of shape (n, d)
        Shared representations inferred from modality 2.
        Rows are matched by patient.

    Returns
    -------
    float
        A_norm = Agreement_obs - Agreement_null.
    """
    agreement_obs = _cosine_similarity_rows(z_m1, z_m2).mean()
    agreement_null = _mean_cross_cosine(z_m1, z_m2)
    return float(agreement_obs - agreement_null)


def compute_anorm_null(
    z_m1: np.ndarray,
    z_m2: np.ndarray,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute A_norm with a permutation-based null interval.

    The null distribution is formed by randomly permuting patient
    correspondence across modalities and recomputing A_norm.

    Parameters
    ----------
    z_m1, z_m2 : ndarray of shape (n, d)
        Matched shared representations from two modalities.
    n_permutations : int
        Number of permutation resamples.
    alpha : float
        Significance level (default 0.05 for 95% null interval).
    rng : np.random.Generator or None

    Returns
    -------
    dict with keys:
        ``anorm``      – observed A_norm
        ``null_lower`` – lower bound of null interval
        ``null_upper`` – upper bound of null interval
        ``null``       – array of null A_norm values
    """
    if rng is None:
        rng = np.random.default_rng()

    observed = compute_anorm(z_m1, z_m2)
    n = z_m1.shape[0]

    # Vectorized null: pre-normalize once, then batch all permutations as a
    # single (n_permutations, n) index array and compute all cosine similarities
    # in two matrix multiplies. ~40x faster than sequential permutation_null.
    z_m1_hat = z_m1 / np.linalg.norm(z_m1, axis=1, keepdims=True).clip(min=1e-12)
    z_m2_hat = z_m2 / np.linalg.norm(z_m2, axis=1, keepdims=True).clip(min=1e-12)

    # Matched (diagonal) cosine similarity for observed
    matched = (z_m1_hat * z_m2_hat).sum(axis=1).mean()
    # Cross (off-diagonal mean) — same for any permutation of z_m2
    cross = (z_m1_hat @ z_m2_hat.T)
    cross_mean = (cross.sum() - np.trace(cross)) / (n * (n - 1))

    # For each permutation, matched mean changes; cross mean is invariant to row perm
    perm_idx = np.stack([rng.permutation(n) for _ in range(n_permutations)])  # (P, n)
    # Gather permuted z_m2_hat rows: (P, n, d) — use advanced indexing
    # matched_perm[p] = mean_i( z_m1_hat[i] · z_m2_hat[perm[p,i]] )
    #                 = mean_i( (z_m1_hat @ z_m2_hat.T)[i, perm[p,i]] )
    scores = cross[np.arange(n)[None, :], perm_idx]  # (P, n)
    matched_perm = scores.mean(axis=1)                 # (P,)
    null_values = matched_perm - cross_mean

    null_lower = np.percentile(null_values, 100 * alpha / 2)
    null_upper = np.percentile(null_values, 100 * (1 - alpha / 2))

    return {
        "anorm": observed,
        "null_lower": float(null_lower),
        "null_upper": float(null_upper),
        "null": null_values,
    }
