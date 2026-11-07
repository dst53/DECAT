"""Linear probe training utilities for DECAT evaluation.

Provides logistic regression probes with per-fold standardization,
cross-validated AUROC computation, and probe fitting/scoring helpers
used by multiple metrics (Delta_shared, P_transfer).

Two probe backends are supported via the ``probe_backend`` parameter:

- ``"logistic"`` (default): scikit-learn LogisticRegression with LBFGS.
  Accurate and well-tested; slower for large permutation loops.

- ``"ridge"``: scikit-learn Ridge regression. Closed-form solution via
  Cholesky (no iterative solver), ~3x faster for permutation/bootstrap
  loops. AUROC is computed from continuous regression scores rather than
  probabilities. When ``probe_backend="ridge"`` both the observed metric
  and its permutation null use Ridge, so the relative comparison remains
  valid. Absolute AUROC values differ slightly from the logistic backend.

The backend can be set per-call or globally via the module-level constant
``DEFAULT_PROBE_BACKEND``.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import cho_factor, cho_solve
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Fast AUROC helpers — avoid sklearn overhead in tight permutation/bootstrap
# loops.  Uses the Wilcoxon rank-sum formula: equivalent to sklearn's
# roc_auc_score but ~100x faster for small arrays (n ≤ 1000).
# ---------------------------------------------------------------------------

def _auroc_fast(scores: np.ndarray, y: np.ndarray) -> float:
    """AUROC via rank-sum statistic.  No sklearn, O(n log n)."""
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    order = np.argsort(scores)
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1, dtype=np.float64)
    return (float(np.sum(ranks[y.astype(bool)])) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def _auroc_batch_2d(scores_mat: np.ndarray, y_mat: np.ndarray) -> np.ndarray:
    """Vectorized AUROC for (n, n_perm) scores and (n, n_perm) labels.

    Both scores and labels may differ across columns (permutation loop where
    both z_train labels and z_val labels change each iteration).

    Returns (n_perm,) AUROCs.
    """
    n, n_perm = scores_mat.shape
    order = np.argsort(-scores_mat, axis=0)                                   # (n, n_perm)
    sorted_y = np.take_along_axis(y_mat.astype(np.float64), order, axis=0)   # (n, n_perm)
    ranks = np.arange(1, n + 1, dtype=np.float64)[:, None]                   # (n, 1)
    n_pos = y_mat.sum(axis=0).astype(np.float64)                             # (n_perm,)
    n_neg = n - n_pos
    # D = sum of descending-order positions of positives; convert to
    # ascending rank sum via R+ = n_pos*(n+1) - D, then apply standard formula.
    D = (sorted_y * ranks).sum(axis=0)                                        # (n_perm,)
    R_plus = n_pos * (n + 1) - D
    denom = n_pos * n_neg
    return np.where(denom > 0, (R_plus - n_pos * (n_pos + 1) / 2.0) / denom, 0.5)

DEFAULT_PROBE_BACKEND: str = "ridge"


def make_cv_splits(
    n: int,
    y: np.ndarray,
    n_folds: int = 5,
    rng: np.random.Generator | None = None,
) -> list:
    """Precompute stratified CV split indices.

    Parameters
    ----------
    n : int
        Number of samples (unused, kept for API clarity).
    y : ndarray of shape (n,)
        Labels for stratification.
    n_folds : int
    rng : np.random.Generator or None

    Returns
    -------
    list of (train_idx, val_idx) tuples
    """
    seed = None if rng is None else int(rng.integers(0, 2**31))
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(skf.split(np.arange(len(y)), y))


def precompute_ridge_folds(
    Z: np.ndarray,
    splits: list,
    standardize: bool = True,
    ridge_alpha: float = 1.0,
) -> list[dict]:
    """Precompute per-fold Cholesky factors for repeated Ridge cv_auroc calls.

    Call once before a permutation/bootstrap loop, then pass the result to
    ``cv_auroc_precomputed`` on each iteration.  Amortizes the O(n·d²) XtX
    computation and O(d³) Cholesky factorization across all loop iterations.

    Parameters
    ----------
    Z : ndarray of shape (n, d)
        Representation matrix (same Z used in the loop).
    splits : list of (train_idx, val_idx)
        Precomputed CV splits.
    standardize : bool
        Match the standardize setting used in the loop.
    ridge_alpha : float
        L2 regularization strength.

    Returns
    -------
    list of dict, one per fold, with keys:
        ``chol``   – Cholesky factor of (XtX + alpha·I)
        ``X_tr``   – (possibly standardized) training features
        ``X_va``   – (possibly standardized) validation features
        ``scaler`` – fitted StandardScaler or None
        ``tr_idx`` – training indices (for y slicing)
        ``va_idx`` – validation indices
    """
    folds = []
    for train_idx, val_idx in splits:
        if standardize:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(Z[train_idx])
            X_va = scaler.transform(Z[val_idx])
        else:
            scaler = None
            X_tr = Z[train_idx]
            X_va = Z[val_idx]
        XtX = X_tr.T @ X_tr
        chol = cho_factor(XtX + ridge_alpha * np.eye(XtX.shape[0]))
        folds.append({
            "chol": chol, "X_tr": X_tr, "X_va": X_va,
            "scaler": scaler, "tr_idx": train_idx, "va_idx": val_idx,
        })
    return folds


def cv_auroc_precomputed(y: np.ndarray, folds: list[dict]) -> float:
    """Compute cv_auroc using precomputed Ridge fold factors.

    Parameters
    ----------
    y : ndarray of shape (n,)
        Label vector (may be permuted/resampled relative to the original).
    folds : list of dict
        Output of ``precompute_ridge_folds``.

    Returns
    -------
    float
        Mean AUROC across folds.
    """
    aurocs = []
    for fold in folds:
        y_tr = y[fold["tr_idx"]].astype(float)
        y_va = y[fold["va_idx"]]
        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue
        alpha_vec = cho_solve(fold["chol"], fold["X_tr"].T @ y_tr)
        scores = fold["X_va"] @ alpha_vec
        aurocs.append(_auroc_fast(scores, y_va))
    return float(np.mean(aurocs)) if aurocs else 0.5


def cv_auroc_precomputed_batch(
    y_matrix: np.ndarray,
    folds: list[dict],
) -> np.ndarray:
    """Compute mean AUROC for n_perm label vectors simultaneously.

    Batches all permutation/bootstrap label vectors into a single set of
    matrix operations per fold, avoiding per-permutation Python overhead.

    Parameters
    ----------
    y_matrix : ndarray of shape (n, n_perm)
        Label matrix — each column is one permuted/resampled label vector.
    folds : list of dict
        Output of ``precompute_ridge_folds``.

    Returns
    -------
    ndarray of shape (n_perm,)
        Mean AUROC across folds for each label vector.
    """
    n_perm = y_matrix.shape[1]
    auroc_sum = np.zeros(n_perm, dtype=np.float64)
    n_valid_folds = 0
    for fold in folds:
        Y_tr = y_matrix[fold["tr_idx"], :].astype(np.float64)  # (n_tr, n_perm)
        Y_va = y_matrix[fold["va_idx"], :]                      # (n_va, n_perm)
        rhs = fold["X_tr"].T @ Y_tr                             # (d, n_perm)
        alpha_mat = cho_solve(fold["chol"], rhs)                # (d, n_perm)
        scores_mat = fold["X_va"] @ alpha_mat                   # (n_va, n_perm)
        auroc_sum += _auroc_batch_2d(scores_mat, Y_va)
        n_valid_folds += 1
    return auroc_sum / n_valid_folds if n_valid_folds > 0 else np.full(n_perm, 0.5)


def cv_auroc(
    Z: np.ndarray,
    y: np.ndarray,
    n_folds: int = 5,
    rng: np.random.Generator | None = None,
    splits: list | None = None,
    standardize: bool = True,
    probe_backend: str | None = None,
) -> float:
    """Cross-validated AUROC with optional per-fold standardization.

    Parameters
    ----------
    Z : ndarray of shape (n, d)
        Representation matrix.
    y : ndarray of shape (n,)
        Binary outcome labels.
    n_folds : int
        Ignored when *splits* is provided.
    rng : np.random.Generator or None
        Used to generate splits when *splits* is None.
    splits : list of (train_idx, val_idx) or None
        Precomputed CV splits.  When provided, the same folds are
        reused so that observed vs. null comparisons differ only
        by labels.
    standardize : bool
        If True (default), apply per-fold StandardScaler.  Set to
        False when inputs are already standardized and re-scaling
        would destroy meaningful magnitude differences (e.g.,
        soft-masked representations from continuously factorized
        models (DisentangledSSL, JIVE).
    probe_backend : str or None
        ``"logistic"`` (default) or ``"ridge"``.  If None, uses
        ``DEFAULT_PROBE_BACKEND``.  Both observed and null should use
        the same backend so relative comparisons remain valid.

    Returns
    -------
    float
        Mean AUROC across folds.
    """
    backend = probe_backend if probe_backend is not None else DEFAULT_PROBE_BACKEND

    if splits is None:
        seed = None if rng is None else int(rng.integers(0, 2**31))
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        splits = list(skf.split(Z, y))

    if backend == "ridge":
        return _cv_auroc_ridge(Z, y, splits, standardize)

    aurocs = []
    for train_idx, val_idx in splits:
        if standardize:
            scaler = StandardScaler()
            Z_train = scaler.fit_transform(Z[train_idx])
            Z_val = scaler.transform(Z[val_idx])
        else:
            Z_train = Z[train_idx]
            Z_val = Z[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
            continue

        clf = LogisticRegression(max_iter=1000, solver="lbfgs")
        clf.fit(Z_train, y_train)
        scores = clf.predict_proba(Z_val)[:, 1]
        aurocs.append(roc_auc_score(y_val, scores))

    return float(np.mean(aurocs)) if aurocs else 0.5


def _cv_auroc_ridge(
    Z: np.ndarray,
    y: np.ndarray,
    splits: list,
    standardize: bool = True,
    ridge_alpha: float = 1.0,
) -> float:
    """Ridge cv_auroc with precomputed Cholesky factors.

    For each fold, XtX + alpha*I is factorized once.  The same factors
    are reused for the current label vector y.  When called repeatedly
    with the same splits but different y (permutation/bootstrap loops),
    the caller should use ``precompute_ridge_folds`` to share fold factors
    across calls.  This function handles the single-call case.
    """
    from scipy.linalg import cho_factor, cho_solve

    aurocs = []
    for train_idx, val_idx in splits:
        if standardize:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(Z[train_idx])
            X_va = scaler.transform(Z[val_idx])
        else:
            X_tr = Z[train_idx]
            X_va = Z[val_idx]
        y_tr, y_va = y[train_idx].astype(float), y[val_idx]

        if len(np.unique(y_tr)) < 2 or len(np.unique(y_va)) < 2:
            continue

        XtX = X_tr.T @ X_tr
        c = cho_factor(XtX + ridge_alpha * np.eye(XtX.shape[0]))
        alpha_vec = cho_solve(c, X_tr.T @ y_tr)
        scores = X_va @ alpha_vec
        aurocs.append(roc_auc_score(y_va, scores))

    return float(np.mean(aurocs)) if aurocs else 0.5


def fit_probe(
    Z_train: np.ndarray,
    y_train: np.ndarray,
    probe_backend: str | None = None,
    C: float = 1.0,
    solver: str = "lbfgs",
    max_iter: int = 1000,
    random_state: int | None = None,
    standardize: bool = True,
) -> tuple[StandardScaler | None, LogisticRegression | Ridge]:
    """Fit a linear probe with optional per-call standardization.

    Parameters
    ----------
    Z_train : ndarray of shape (n, d)
    y_train : ndarray of shape (n,)
    probe_backend : str or None
        ``"logistic"`` or ``"ridge"``.  If None, uses ``DEFAULT_PROBE_BACKEND``.
    C : float
        Inverse regularization strength for logistic backend (default 1.0).
        Set to 0.1 for small held-out cohorts (e.g. TCGA per-split probes).
    solver : str
        Solver for logistic backend (default ``"lbfgs"``).
        Use ``"liblinear"`` for low-dimensional embeddings / small N.
    max_iter : int
        Max iterations for logistic backend (default 1000).
    random_state : int or None
        Random state for logistic backend (default None).
    standardize : bool
        If True (default), fit a StandardScaler on Z_train and apply it.
        Set False when Z_train is already standardized upstream.

    Returns
    -------
    scaler : StandardScaler or None
        Fitted scaler, or None when ``standardize=False``.
    clf : LogisticRegression or Ridge
        Fitted probe.  Use ``predict_scores()`` to get positive-class scores.
    """
    backend = probe_backend if probe_backend is not None else DEFAULT_PROBE_BACKEND

    if standardize:
        scaler = StandardScaler()
        Z_scaled = scaler.fit_transform(Z_train)
    else:
        scaler = None
        Z_scaled = Z_train

    if len(np.unique(y_train)) < 2:
        clf = LogisticRegression(max_iter=max_iter, solver=solver, C=C,
                                 random_state=random_state)
        clf.classes_ = np.array([0, 1])
        clf.coef_ = np.zeros((1, Z_scaled.shape[1]))
        clf.intercept_ = np.array([0.0])
        return scaler, clf

    if backend == "ridge":
        clf = Ridge(alpha=1.0)
        clf.fit(Z_scaled, y_train.astype(float))
    else:
        clf = LogisticRegression(max_iter=max_iter, solver=solver, C=C,
                                 random_state=random_state)
        clf.fit(Z_scaled, y_train)

    return scaler, clf


def predict_scores(
    scaler: StandardScaler | None,
    clf: LogisticRegression | Ridge,
    Z: np.ndarray,
) -> np.ndarray:
    """Return scores for the positive class (probability or regression score).

    Works with both ``"logistic"`` and ``"ridge"`` backends.

    Parameters
    ----------
    scaler : StandardScaler
    clf : LogisticRegression or Ridge
    Z : ndarray of shape (n, d)

    Returns
    -------
    ndarray of shape (n,)
    """
    Z_scaled = scaler.transform(Z) if scaler is not None else Z
    if isinstance(clf, Ridge):
        return clf.predict(Z_scaled)
    return clf.predict_proba(Z_scaled)[:, 1]
