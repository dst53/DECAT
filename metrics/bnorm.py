"""Metric B — Prototype Consistency per Modality (B_norm).

Measures whether biological interpretation is invariant to which
modality was used to infer the shared representation.

    B_norm = E_i[Delta_p(i)] - Delta_p_null

where Delta_p(i) = |cos(z_c^{i,m1}, p) - cos(z_c^{i,m2}, p)|
is the per-patient prompt discrepancy for a prototype p.

Applicable to all models.
"""

import numpy as np

from evaluation.bootstrap import permutation_null
from evaluation.prototypes import compute_prototypes, cosine_to_prototype


def _compute_bnorm_single_prototype(z_m1, z_m2, p):
    """Compute B_norm for a single prototype.

    Parameters
    ----------
    z_m1, z_m2 : ndarray of shape (n, d)
        Matched shared representations from two modalities.
    p : ndarray of shape (d,)
        Prototype vector.

    Returns
    -------
    float
        B_norm for this prototype.
    """
    sim_m1 = cosine_to_prototype(z_m1, p)
    sim_m2 = cosine_to_prototype(z_m2, p)

    # Observed: matched pairs
    delta_obs = np.abs(sim_m1 - sim_m2).mean()

    # Null: mismatched pairs
    n = len(sim_m1)
    delta_matrix = np.abs(sim_m1[:, None] - sim_m2[None, :])
    mask = ~np.eye(n, dtype=bool)
    delta_null = delta_matrix[mask].mean()

    return float(delta_obs - delta_null)


def compute_bnorm(z_m1: np.ndarray, z_m2: np.ndarray, prototypes: np.ndarray) -> float:
    """Compute B_norm averaged over all prototypes.

    Parameters
    ----------
    z_m1, z_m2 : ndarray of shape (n, d)
        Matched shared representations from two modalities.
    prototypes : ndarray of shape (k, d)
        Prototype vectors.

    Returns
    -------
    float
        B_norm averaged across prototypes.
    """
    values = [_compute_bnorm_single_prototype(z_m1, z_m2, p)
              for p in prototypes]
    return float(np.mean(values))


def compute_bnorm_null(
    z_m1: np.ndarray,
    z_m2: np.ndarray,
    prototypes: np.ndarray,
    n_permutations: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """Compute B_norm with permutation-based null interval.

    Parameters
    ----------
    z_m1, z_m2 : ndarray of shape (n, d)
        Matched shared representations from two modalities.
    prototypes : ndarray of shape (k, d)
    n_permutations : int
    alpha : float
    rng : np.random.Generator or None

    Returns
    -------
    dict with keys:
        ``bnorm``      – observed B_norm
        ``null_lower`` – lower bound of null interval
        ``null_upper`` – upper bound of null interval
        ``null``       – array of null B_norm values

    Implementation note
    -------------------
    The original per-permutation call to ``compute_bnorm`` built an O(n²)
    all-pairs matrix for each of the n_permutations × k prototypes, giving
    O(n² × k × n_permutations) total work.

    Key identity: for any permutation π of z_m2 rows,

        S_full = Σ_{i,j} |sim_m1[i] - sim_m2[j]|

    is invariant (j → π(j) is a bijection).  Therefore the off-diagonal
    null term can be expressed as a function of the diagonal delta alone:

        B_norm_perm_p = delta_obs_perm_p × n/(n-1) - S_full_p / (n(n-1))

    so only O(n) work is required per permutation per prototype after
    computing S_full once, reducing total work to O(n² × k + n × k × P).
    """
    if rng is None:
        rng = np.random.default_rng()

    n, d = z_m1.shape
    k = len(prototypes)

    # Precompute cosine similarities: (n, k) then transpose to (k, n)
    def _cos_sim_batch(Z, P):
        Z_n = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-12)
        P_n = P / np.linalg.norm(P, axis=1, keepdims=True).clip(min=1e-12)
        return (Z_n @ P_n.T).T  # (k, n)

    sim1 = _cos_sim_batch(z_m1, prototypes)  # (k, n)
    sim2 = _cos_sim_batch(z_m2, prototypes)  # (k, n)

    # S_full per prototype: sum_{i,j} |sim1[p,i] - sim2[p,j]|  (computed once)
    # Shape: (k, n, n) — for k=5, n=1000 this is 5M floats = 40 MB
    delta_all = np.abs(sim1[:, :, None] - sim2[:, None, :])  # (k, n, n)
    S_full = delta_all.sum(axis=(1, 2))  # (k,)

    # Observed: diagonal delta per prototype → (k,)
    delta_obs = np.abs(sim1 - sim2).mean(axis=1)  # (k,)
    # Using the identity: B_norm_p = delta_obs_p * n/(n-1) - S_full_p/(n*(n-1))
    denom = n * (n - 1)
    observed_per_proto = delta_obs * (n / (n - 1)) - S_full / denom
    observed = float(observed_per_proto.mean())

    # Null: batch all permutations — only need diagonal delta per perm
    perms = np.vstack([rng.permutation(n) for _ in range(n_permutations)])  # (P, n)
    # sim2 permuted: (k, P, n) via advanced indexing
    sim2_perms = sim2[:, perms]                                               # (k, P, n)
    diag_delta_perms = np.abs(sim1[:, None, :] - sim2_perms)                 # (k, P, n)
    delta_obs_perms = diag_delta_perms.mean(axis=2)                          # (k, P)
    bnorm_perms = delta_obs_perms * (n / (n - 1)) - S_full[:, None] / denom  # (k, P)
    null_values = bnorm_perms.mean(axis=0)                                    # (P,)

    return {
        "bnorm": observed,
        "null_lower": float(np.percentile(null_values, 100 * alpha / 2)),
        "null_upper": float(np.percentile(null_values, 100 * (1 - alpha / 2))),
        "null": null_values,
    }
