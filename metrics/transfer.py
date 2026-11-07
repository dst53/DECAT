"""Metric D1 — Prediction Transfer / Functional Stability (P_transfer).

Tests whether independently trained outcome predictors induce a
consistent patient ordering in an external cohort.

    P_transfer^(s) = Spearman(f_A'(z_sig), f_B(z_sig)) | cohort C

Probes are trained independently on Cohort A' and Cohort B, then
both evaluated on Cohort C.
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.linalg import cho_factor, cho_solve
from scipy.stats import spearmanr
from sklearn.preprocessing import StandardScaler

from evaluation.bootstrap import bootstrap_ci
from evaluation.probes import fit_probe, predict_scores, DEFAULT_PROBE_BACKEND


def compute_transfer(
    z_a_prime: np.ndarray,
    y_a_prime: np.ndarray,
    z_b: np.ndarray,
    y_b: np.ndarray,
    z_c: np.ndarray,
) -> float:
    """Compute prediction transfer P_transfer.

    Parameters
    ----------
    z_a_prime : ndarray of shape (n_a', d)
        Signal-carrying representation for Cohort A'.
    y_a_prime : ndarray of shape (n_a',)
        Outcome labels for Cohort A'.
    z_b : ndarray of shape (n_b, d)
        Signal-carrying representation for Cohort B.
    y_b : ndarray of shape (n_b,)
        Outcome labels for Cohort B.
    z_c : ndarray of shape (n_c, d)
        Signal-carrying representation for Cohort C.

    Returns
    -------
    float
        Spearman correlation of predicted scores on C from the two
        independently trained probes.
    """
    scaler_a, clf_a = fit_probe(z_a_prime, y_a_prime)
    scaler_b, clf_b = fit_probe(z_b, y_b)

    scores_a = predict_scores(scaler_a, clf_a, z_c)
    scores_b = predict_scores(scaler_b, clf_b, z_c)

    rho, _ = spearmanr(scores_a, scores_b)
    return float(rho)


def compute_transfer_ci(
    z_a_prime: np.ndarray,
    y_a_prime: np.ndarray,
    z_b: np.ndarray,
    y_b: np.ndarray,
    z_c: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
) -> dict:
    """P_transfer with bootstrap CI on Cohort C.

    Parameters
    ----------
    z_a_prime, z_b, z_c : ndarray
    y_a_prime, y_b : ndarray
    n_bootstrap : int
    alpha : float
    rng : np.random.Generator or None

    Returns
    -------
    dict with keys:
        ``p_transfer`` – observed Spearman rho
        ``ci_lower``   – lower CI bound
        ``ci_upper``   – upper CI bound
        ``bootstrap``  – array of bootstrap values
    """
    if rng is None:
        rng = np.random.default_rng()

    rng_boot = np.random.default_rng(rng.integers(1 << 63))

    # Fit probes once
    scaler_a, clf_a = fit_probe(z_a_prime, y_a_prime)
    scaler_b, clf_b = fit_probe(z_b, y_b)

    scores_a = predict_scores(scaler_a, clf_a, z_c)
    scores_b = predict_scores(scaler_b, clf_b, z_c)

    observed, _ = spearmanr(scores_a, scores_b)
    n_c = z_c.shape[0]

    def _boot_rho(idx):
        rho, _ = spearmanr(scores_a[idx], scores_b[idx])
        return rho

    ci_result = bootstrap_ci(_boot_rho, n_c, n_bootstrap, alpha, rng_boot)

    return {
        "p_transfer": float(observed),
        "ci_lower": ci_result["ci_lower"],
        "ci_upper": ci_result["ci_upper"],
        "bootstrap": ci_result["bootstrap"],
    }


def _spearman_batch(a_mat: np.ndarray, b_mat: np.ndarray) -> np.ndarray:
    """Spearman correlation for each column pair of (n, P) matrices.

    Returns (P,) array of Spearman correlations.
    """
    def rank_cols(X):
        order = np.argsort(X, axis=0)
        ranks = np.empty_like(order, dtype=np.float64)
        col_idx = np.arange(X.shape[1])[None, :]
        ranks[order, col_idx] = np.arange(X.shape[0], dtype=np.float64)[:, None]
        return ranks

    ra = rank_cols(a_mat)
    rb = rank_cols(b_mat)
    ra -= ra.mean(axis=0)
    rb -= rb.mean(axis=0)
    num   = (ra * rb).sum(axis=0)
    denom = np.sqrt((ra ** 2).sum(axis=0) * (rb ** 2).sum(axis=0))
    return num / denom.clip(min=1e-12)


def compute_transfer_null(
    z_a_prime: np.ndarray,
    y_a_prime: np.ndarray,
    z_b: np.ndarray,
    y_b: np.ndarray,
    z_c: np.ndarray,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
) -> dict:
    """Estimate null distribution for P_transfer by label permutation.

    Outcome labels are shuffled within Cohorts A' and B before
    retraining probes.

    Parameters
    ----------
    z_a_prime, z_b, z_c : ndarray
    y_a_prime, y_b : ndarray
    n_permutations : int
    rng : np.random.Generator or None
    n_jobs : int
        Number of parallel workers (1 = sequential, ignored for ridge
        backend which is fully vectorized).

    Returns
    -------
    dict with keys:
        ``null``       – array of null P_transfer values
        ``null_lower`` – 2.5th percentile of null distribution
        ``null_upper`` – 97.5th percentile of null distribution

    Implementation note
    -------------------
    For the ridge backend the representations Z are fixed; only labels
    change across permutations.  We precompute the Cholesky factor of
    (Z.T Z + αI) once per cohort and solve for all P permuted label
    vectors in a single batched cho_solve, then apply a vectorized
    Spearman on the resulting (n_c, P) score matrices.
    """
    if rng is None:
        rng = np.random.default_rng()

    if DEFAULT_PROBE_BACKEND == "ridge":
        # --- vectorized ridge path ---
        alpha = 1.0

        def _prep(Z, y):
            sc = StandardScaler()
            X  = sc.fit_transform(Z)
            chol = cho_factor(X.T @ X + alpha * np.eye(X.shape[1]))
            return sc, X, chol

        sc_a, X_a, chol_a = _prep(z_a_prime, y_a_prime)
        sc_b, X_b, chol_b = _prep(z_b,       y_b)
        X_c_a = sc_a.transform(z_c)
        X_c_b = sc_b.transform(z_c)

        rng_perm = np.random.default_rng(rng.integers(1 << 63))
        Y_a = np.vstack([rng_perm.permutation(y_a_prime)
                         for _ in range(n_permutations)]).T.astype(np.float64)  # (n_a, P)
        Y_b = np.vstack([rng_perm.permutation(y_b)
                         for _ in range(n_permutations)]).T.astype(np.float64)  # (n_b, P)

        W_a = cho_solve(chol_a, X_a.T @ Y_a)   # (d, P)
        W_b = cho_solve(chol_b, X_b.T @ Y_b)   # (d, P)

        scores_a = X_c_a @ W_a   # (n_c, P)
        scores_b = X_c_b @ W_b   # (n_c, P)

        null_values = _spearman_batch(scores_a, scores_b)  # (P,)

    else:
        # --- fallback serial path for logistic backend ---
        rng_perm = np.random.default_rng(rng.integers(1 << 63))
        seeds = rng_perm.integers(1 << 63, size=n_permutations)

        def _one_permutation(seed):
            local_rng = np.random.default_rng(seed)
            y_a_perm = local_rng.permutation(y_a_prime)
            y_b_perm = local_rng.permutation(y_b)
            return compute_transfer(z_a_prime, y_a_perm, z_b, y_b_perm, z_c)

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
