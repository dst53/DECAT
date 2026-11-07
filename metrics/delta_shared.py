"""Metric C — Predictive Signal Localization (Delta_shared).

Measures where outcome-relevant signal resides: shared biology or
modality-specific effects.

    Delta_shared^(s) = AUROC(f_c^(s)) - AUROC(f_m^(s))

Includes the continuous variant for models with graded sharedness
(e.g., JIVE, DisentangledSSL) where sharedness scores gamma_k weight latent dimensions.

Only computable for **factorized models** (JIVE, DisentangledSSL).
"""

import numpy as np
from joblib import Parallel, delayed
from threadpoolctl import threadpool_limits

from evaluation.bootstrap import bootstrap_ci
from evaluation.probes import (
    cv_auroc, make_cv_splits,
    precompute_ridge_folds, cv_auroc_precomputed, cv_auroc_precomputed_batch,
    DEFAULT_PROBE_BACKEND,
)


def compute_delta_shared(
    z_shared: np.ndarray,
    z_specific: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_permutations: int = 100,
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
    splits: list | None = None,
    standardize: bool = True,
) -> dict:
    """Compute Delta_shared (hard factorization variant).

    Parameters
    ----------
    z_shared : ndarray of shape (n, d)
        Shared representation for a given modality.
    z_specific : ndarray of shape (n, d)
        Modality-specific representation.
    y : ndarray of shape (n,)
        Binary outcome labels.
    n_folds : int
    n_permutations : int
        Number of label permutations for the signal-presence gate.
        Set to 0 to skip the permutation-based signal-presence test
        (uses observed_max > 0.55 fallback).
        Permutation testing is recommended for all final analyses.
    rng : np.random.Generator or None
    n_jobs : int
        Number of parallel workers for the permutation loop (1 = sequential).
    splits : list of (train_idx, test_idx) or None
        Precomputed CV splits.  If *None*, splits are generated internally.
    standardize : bool
        Passed to cv_auroc.  Set to False for pre-standardized
        soft-masked representations (continuous gamma variant;
        DisentangledSSL and JIVE-style models).

    Returns
    -------
    dict with keys:
        ``delta_shared`` – AUROC(shared) - AUROC(specific)
        ``auroc_shared`` – AUROC on shared component
        ``auroc_specific`` – AUROC on modality-specific component
        ``auroc_both`` – AUROC on concatenated representation
        ``signal_present`` – bool, whether the label-permutation p-value
                             is below 0.05 (signal-presence gate)
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(y)
    if splits is None:
        splits = make_cv_splits(n, y, n_folds, rng)
    z_both = np.concatenate([z_shared, z_specific], axis=1)

    # Precompute fold factors for Ridge backend — reused across all permutations.
    _use_precomputed = (DEFAULT_PROBE_BACKEND == "ridge")
    if _use_precomputed:
        _folds_sh = precompute_ridge_folds(z_shared,   splits, standardize=standardize)
        _folds_sp = precompute_ridge_folds(z_specific, splits, standardize=standardize)
        _folds_bo = precompute_ridge_folds(z_both,     splits, standardize=standardize)

        def _auroc_sh(y_): return cv_auroc_precomputed(y_, _folds_sh)
        def _auroc_sp(y_): return cv_auroc_precomputed(y_, _folds_sp)
        def _auroc_bo(y_): return cv_auroc_precomputed(y_, _folds_bo)
    else:
        def _auroc_sh(y_): return cv_auroc(z_shared,   y_, splits=splits, standardize=standardize)
        def _auroc_sp(y_): return cv_auroc(z_specific, y_, splits=splits, standardize=standardize)
        def _auroc_bo(y_): return cv_auroc(z_both,     y_, splits=splits, standardize=standardize)

    auroc_shared   = _auroc_sh(y)
    auroc_specific = _auroc_sp(y)
    auroc_both     = _auroc_bo(y)

    observed_max = max(auroc_shared, auroc_specific, auroc_both)

    # Signal-presence gate via label-permutation null
    if n_permutations > 0:
        seeds = rng.integers(1 << 63, size=n_permutations)

        if _use_precomputed:
            # Batch all permutations into a single (n, n_perm) matrix and
            # compute all AUROCs in one vectorised pass — ~100x faster than
            # a Python loop over individual roc_auc_score calls.
            y_perms = np.column_stack(
                [np.random.default_rng(s).permutation(y) for s in seeds]
            )  # (n, n_perm)
            null_sh = cv_auroc_precomputed_batch(y_perms, _folds_sh)
            null_sp = cv_auroc_precomputed_batch(y_perms, _folds_sp)
            null_bo = cv_auroc_precomputed_batch(y_perms, _folds_bo)
            null_max_aurocs = np.maximum(np.maximum(null_sh, null_sp), null_bo)
        else:
            def _one_permutation(seed):
                local_rng = np.random.default_rng(seed)
                y_perm = local_rng.permutation(y)
                return max(_auroc_sh(y_perm), _auroc_sp(y_perm), _auroc_bo(y_perm))

            if n_jobs == 1:
                null_max_aurocs = np.array([_one_permutation(s) for s in seeds])
            else:
                null_max_aurocs = np.array(
                    Parallel(n_jobs=n_jobs)(delayed(_one_permutation)(s) for s in seeds)
                )

        p_value = (1 + np.sum(null_max_aurocs >= observed_max)) / (n_permutations + 1)
        signal_present = p_value < 0.05
    else:
        signal_present = observed_max > 0.55

    return {
        "delta_shared": float(auroc_shared - auroc_specific),
        "auroc_shared": auroc_shared,
        "auroc_specific": auroc_specific,
        "auroc_both": auroc_both,
        "signal_present": signal_present,
    }


def compute_delta_shared_null(
    z_shared: np.ndarray,
    z_specific: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_permutations: int = 1000,
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
    splits: list | None = None,
    standardize: bool = True,
) -> dict:
    """Estimate a permutation null distribution for Delta_shared.

    For each permutation, outcome labels are shuffled and
    Delta_shared = AUROC(shared) - AUROC(specific) is recomputed,
    producing a null distribution for the localization contrast
    independent of the signal-presence gate.

    Parameters
    ----------
    z_shared : ndarray of shape (n, d)
        Shared representation for a given modality.
    z_specific : ndarray of shape (n, d)
        Modality-specific representation.
    y : ndarray of shape (n,)
        Binary outcome labels.
    n_folds : int
    n_permutations : int
        Number of label permutations.
    rng : np.random.Generator or None
    n_jobs : int
        Number of parallel workers (1 = sequential).
    splits : list of (train_idx, test_idx) or None
        Precomputed CV splits.  If *None*, splits are generated internally.

    Returns
    -------
    dict with keys:
        ``null``       – array of permuted Delta_shared values
        ``null_lower`` – 2.5th percentile
        ``null_upper`` – 97.5th percentile
    """
    if rng is None:
        rng = np.random.default_rng()

    n = len(y)
    if splits is None:
        splits = make_cv_splits(n, y, n_folds, rng)

    _use_precomputed = (DEFAULT_PROBE_BACKEND == "ridge")
    if _use_precomputed:
        _folds_sh = precompute_ridge_folds(z_shared,   splits, standardize=standardize)
        _folds_sp = precompute_ridge_folds(z_specific, splits, standardize=standardize)

        def _auroc_sh(y_): return cv_auroc_precomputed(y_, _folds_sh)
        def _auroc_sp(y_): return cv_auroc_precomputed(y_, _folds_sp)
    else:
        def _auroc_sh(y_): return cv_auroc(z_shared,   y_, splits=splits, standardize=standardize)
        def _auroc_sp(y_): return cv_auroc(z_specific, y_, splits=splits, standardize=standardize)

    seeds = rng.integers(1 << 63, size=n_permutations)

    if _use_precomputed:
        y_perms = np.column_stack(
            [np.random.default_rng(s).permutation(y) for s in seeds]
        )  # (n, n_perm)
        null_values = (cv_auroc_precomputed_batch(y_perms, _folds_sh)
                       - cv_auroc_precomputed_batch(y_perms, _folds_sp))
    else:
        def _one_permutation(seed):
            local_rng = np.random.default_rng(seed)
            y_perm = local_rng.permutation(y)
            return _auroc_sh(y_perm) - _auroc_sp(y_perm)

        if n_jobs == 1:
            null_values = np.array([_one_permutation(s) for s in seeds])
        else:
            null_values = np.array(
                Parallel(n_jobs=n_jobs)(delayed(_one_permutation)(s) for s in seeds)
            )

    return {
        "null": null_values,
        "null_lower": float(np.percentile(null_values, 2.5)),
        "null_upper": float(np.percentile(null_values, 97.5)),
    }


def compute_delta_shared_bootstrap(
    z_shared: np.ndarray,
    z_specific: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    n_bootstrap: int = 1000,
    n_permutations: int = 100,
    alpha: float = 0.05,
    rng: np.random.Generator | None = None,
    n_jobs: int = 1,
    standardize: bool = True,
) -> dict:
    """Compute Delta_shared with bootstrap CI and permutation null.

    The point estimate and signal-presence gate are computed without
    the Delta_shared permutation null (to keep pipelines separate).
    The permutation null for Delta_shared is computed independently
    via ``compute_delta_shared_null``.

    Parameters
    ----------
    z_shared, z_specific : ndarray of shape (n, d)
    y : ndarray of shape (n,)
    n_folds : int
    n_bootstrap : int
    n_permutations : int
        Number of label permutations used for:
        (1) the signal-presence gate (max AUROC null)
        (2) the Delta_shared permutation null (computed separately)
        Set to 0 to skip both.
    alpha : float
    rng : np.random.Generator or None
    n_jobs : int
        Number of parallel workers (1 = sequential).

    Returns
    -------
    dict with keys:
        ``delta_shared``  – point estimate
        ``ci_lower``      – lower CI bound
        ``ci_upper``      – upper CI bound
        ``auroc_shared``  – point estimate
        ``auroc_specific``– point estimate
        ``auroc_both``    – point estimate
        ``signal_present``– whether the label-permutation p-value
                            is below 0.05 (signal-presence gate)
        ``bootstrap``     – array of bootstrap delta values
        ``delta_null``       – (if n_permutations > 0) array of null values
        ``delta_null_lower`` – (if n_permutations > 0) 2.5th percentile
        ``delta_null_upper`` – (if n_permutations > 0) 97.5th percentile
    """
    if rng is None:
        rng = np.random.default_rng()

    # Independent child RNGs for statistical independence between
    # CV geometry, permutation null, and bootstrap.
    rng_splits = np.random.default_rng(rng.integers(1 << 63))
    rng_perm = np.random.default_rng(rng.integers(1 << 63))
    rng_boot = np.random.default_rng(rng.integers(1 << 63))

    n = len(y)
    splits = make_cv_splits(n, y, n_folds, rng_splits)

    # Point estimate + signal-presence gate
    point = compute_delta_shared(z_shared, z_specific, y, n_folds,
                                 n_permutations=n_permutations, rng=rng_perm,
                                 n_jobs=n_jobs, splits=splits,
                                 standardize=standardize)

    # Separate permutation null for Delta_shared
    if n_permutations > 0:
        null_result = compute_delta_shared_null(
            z_shared, z_specific, y, n_folds,
            n_permutations=n_permutations, rng=rng_perm, n_jobs=n_jobs,
            splits=splits, standardize=standardize,
        )
        point["delta_null"] = null_result["null"]
        point["delta_null_lower"] = null_result["null_lower"]
        point["delta_null_upper"] = null_result["null_upper"]

    # Pre-generate seeds so the closure captures a deterministic seed
    # rather than sharing the parent rng across workers.
    boot_seeds = rng_boot.integers(1 << 63, size=n_bootstrap)

    _use_precomputed = (DEFAULT_PROBE_BACKEND == "ridge")

    def _boot_delta(idx, seed):
        # threadpool_limits removed: loky workers set OMP/MKL/OPENBLAS=2 at
        # process startup; calling it per-iteration causes lock contention
        # across the 16 threading workers sharing the same process.
        local_rng = np.random.default_rng(seed)
        z_sh_b = z_shared[idx]; z_sp_b = z_specific[idx]; y_b = y[idx]
        z_bo_b = np.concatenate([z_sh_b, z_sp_b], axis=1)
        boot_splits = make_cv_splits(len(y_b), y_b, n_folds, local_rng)
        if _use_precomputed:
            fsh = precompute_ridge_folds(z_sh_b, boot_splits, standardize=standardize)
            fsp = precompute_ridge_folds(z_sp_b, boot_splits, standardize=standardize)
            fbo = precompute_ridge_folds(z_bo_b, boot_splits, standardize=standardize)
            return (cv_auroc_precomputed(y_b, fsh)
                    - cv_auroc_precomputed(y_b, fsp))
        else:
            r = compute_delta_shared(
                z_sh_b, z_sp_b, y_b, n_folds,
                n_permutations=0, rng=local_rng,
                standardize=standardize,
            )
            return r["delta_shared"]

    # Pre-generate bootstrap index arrays from parent rng
    indices = [rng_boot.choice(n, size=n, replace=True) for _ in range(n_bootstrap)]

    if n_jobs == 1:
        boot_values = np.array([
            _boot_delta(idx, seed)
            for idx, seed in zip(indices, boot_seeds)
        ])
    else:
        boot_values = np.array(
            Parallel(n_jobs=n_jobs)(
                delayed(_boot_delta)(idx, seed)
                for idx, seed in zip(indices, boot_seeds)
            )
        )

    ci_lower = np.percentile(boot_values, 100 * alpha / 2)
    ci_upper = np.percentile(boot_values, 100 * (1 - alpha / 2))

    point["ci_lower"] = float(ci_lower)
    point["ci_upper"] = float(ci_upper)
    point["bootstrap"] = boot_values
    return point


def compute_sharedness_scores(loadings_dict: dict) -> np.ndarray:
    """Compute per-factor sharedness scores from loading matrices.

    Uses the normalized entropy of squared variance-explained across
    modalities.

    Parameters
    ----------
    loadings_dict : dict mapping modality name to ndarray of shape (d_m, K)
        Loading matrices for each modality (e.g., from JIVE).

    Returns
    -------
    gamma : ndarray of shape (K,)
        Sharedness scores in [0, 1].  1 = evenly shared, 0 = single-modality.
    """
    modalities = list(loadings_dict.keys())
    M = len(modalities)
    K = loadings_dict[modalities[0]].shape[1]

    # R^2_{m,k} proportional to column norm squared
    r2 = np.zeros((M, K))
    for i, mod in enumerate(modalities):
        L = loadings_dict[mod]
        r2[i] = np.sum(L ** 2, axis=0)

    # Normalize per factor across modalities
    r2_sq = r2 ** 2
    denom = r2_sq.sum(axis=0, keepdims=True).clip(min=1e-12)
    p = r2_sq / denom  # (M, K)

    # Entropy per factor
    H = -np.sum(p * np.log(p + 1e-12), axis=0)
    gamma = H / np.log(M)
    return gamma.clip(0, 1)


def soft_mask_representations(
    Z: np.ndarray,
    gamma: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply soft masking to split into shared and modality-specific.

    Parameters
    ----------
    Z : ndarray of shape (n, K)
        Latent representation.
    gamma : ndarray of shape (K,)
        Sharedness scores.

    Returns
    -------
    Z_shared : ndarray of shape (n, K)
    Z_specific : ndarray of shape (n, K)
    """
    Z_shared = Z * gamma[np.newaxis, :]
    Z_specific = Z * (1 - gamma)[np.newaxis, :]
    return Z_shared, Z_specific