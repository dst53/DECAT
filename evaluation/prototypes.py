"""Prototype generation and similarity scoring for DECAT evaluation.

Prototypes are reference query vectors in shared latent space, used by
B_norm (prompt consistency) and D_quantile (biological transfer).

Prototypes are defined via k-means on training-cohort shared
representations inferred from a single fixed modality (HE).
"""

import numpy as np
from sklearn.cluster import KMeans


def compute_prototypes(
    z_train: np.ndarray,
    k: int = 2,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Compute prototypes via k-means on training shared representations.

    Parameters
    ----------
    z_train : ndarray of shape (n, d)
        Shared representations from training cohort (single modality).
    k : int
        Number of clusters (default 2).
    rng : np.random.Generator or None

    Returns
    -------
    prototypes : ndarray of shape (k, d)
        Cluster centroids.
    """
    seed = None if rng is None else int(rng.integers(0, 2**31))
    km = KMeans(n_clusters=k, n_init=10, random_state=seed)
    km.fit(z_train)
    return km.cluster_centers_


def compute_prototypes_multiseed(
    z_train: np.ndarray,
    k: int = 2,
    n_seeds: int = 5,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Compute multiple prototype sets using different k-means seeds.

    Parameters
    ----------
    z_train : ndarray of shape (n, d)
        Shared representations from training cohort.
    k : int
        Number of clusters per seed.
    n_seeds : int
        Number of independent k-means runs.
    rng : np.random.Generator or None

    Returns
    -------
    list of ndarray, each of shape (k, d)
        One prototype set per seed.
    """
    if rng is None:
        rng = np.random.default_rng()

    prototypes_list = []
    for _ in range(n_seeds):
        seed_rng = np.random.default_rng(rng.integers(1 << 63))
        prototypes = compute_prototypes(z_train, k=k, rng=seed_rng)
        prototypes_list.append(prototypes)

    return prototypes_list


def cosine_to_prototype(Z: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Cosine similarity of each row in Z to a prototype vector p.

    Parameters
    ----------
    Z : ndarray of shape (n, d)
        Representation matrix.
    p : ndarray of shape (d,)
        Prototype vector.

    Returns
    -------
    ndarray of shape (n,)
        Cosine similarities.
    """
    Z_norm = Z / np.linalg.norm(Z, axis=1, keepdims=True).clip(min=1e-12)
    p_norm = p / np.linalg.norm(p).clip(min=1e-12)
    return Z_norm @ p_norm
