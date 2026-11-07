"""Metric D_quantile^task — Outcome-aligned quantile stability.

Primary Scenario 2 decision metric (Stage IV). Projects representations
onto the outcome-predictive probe direction learned from A', then measures
W_1 between A\'\'-referenced quantile distributions of B and C.
"""

import numpy as np
from scipy.stats import wasserstein_distance

from evaluation.probes import fit_probe

def _quantiles_relative_to_reference(scores, reference_scores):
    """Map scores to quantiles using the empirical CDF of a reference cohort.

    For each score s, computes F_A(s) = (number of reference scores <= s) / n_ref.

    Parameters
    ----------
    scores : ndarray of shape (n,)
        Scores to convert.
    reference_scores : ndarray of shape (n_ref,)
        Reference distribution (from Cohort A).

    Returns
    -------
    ndarray of shape (n,)
        Quantiles in [0, 1].
    """
    sorted_ref = np.sort(reference_scores)
    return np.searchsorted(sorted_ref, scores, side="right") / len(sorted_ref)


def _project_scores(Z, v):
    """Project representations onto direction v.

    Parameters
    ----------
    Z : ndarray of shape (n, d)
    v : ndarray of shape (d,)

    Returns
    -------
    ndarray of shape (n,)
    """
    v_norm = v / np.linalg.norm(v).clip(min=1e-12)
    return Z @ v_norm


def compute_dquantile_projection(
    z_a: np.ndarray,
    z_b: np.ndarray,
    z_c: np.ndarray,
    v: np.ndarray,
) -> float:
    """Compute D_quantile along a single projection direction.

    Parameters
    ----------
    z_a : ndarray of shape (n_a, d)
        Reference cohort representations.
    z_b, z_c : ndarray of shape (n, d)
    v : ndarray of shape (d,)
        Projection direction.

    Returns
    -------
    float
    """
    s_a = _project_scores(z_a, v)
    s_b = _project_scores(z_b, v)
    s_c = _project_scores(z_c, v)

    q_b = _quantiles_relative_to_reference(s_b, s_a)
    q_c = _quantiles_relative_to_reference(s_c, s_a)

    return float(wasserstein_distance(q_b, q_c))


def _alignment_score(
    z_b: np.ndarray,
    z_c: np.ndarray,
    w: np.ndarray,
) -> float:
    """Measure alignment between the B→C mean shift and the probe direction w.

    Returns |cos(v_shift, w)| where v_shift = (C̄ − B̄) / ‖C̄ − B̄‖.
    A value near 1 means the cohort shift is almost entirely along the
    outcome-predictive direction; near 0 means orthogonal.

    Parameters
    ----------
    z_b : (n_b, d) — raw Cohort B representations
    z_c : (n_c, d) — raw Cohort C representations
    w   : (d,)     — unit-norm outcome-predictive probe direction

    Returns
    -------
    float in [0, 1]
    """
    shift = z_c.mean(axis=0) - z_b.mean(axis=0)
    norm = np.linalg.norm(shift)
    if norm < 1e-12:
        return 0.0
    v_shift = shift / norm
    w_unit = w / (np.linalg.norm(w) + 1e-12)
    return float(abs(np.dot(v_shift, w_unit)))


def _mlp_scores(
    z_ap: np.ndarray,
    y_ap: np.ndarray,
    z_eval: np.ndarray,
    scaler,
    hidden_dim: int = 64,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Train a shallow MLP on A' and return P(y=1) scores for z_eval.

    The MLP is trained on standardized A' features using the provided scaler.
    z_eval is standardized with the same A' scaler before scoring.

    Parameters
    ----------
    z_ap : (n_ap, d) — raw A' representations
    y_ap : (n_ap,) — binary labels for A'
    z_eval : (n_eval, d) — raw representations to score
    scaler : fitted StandardScaler from A'
    hidden_dim : int — hidden layer width
    rng : np.random.Generator or None — for reproducible MLP init

    Returns
    -------
    scores : (n_eval,) — P(y=1) from MLP
    """
    from sklearn.neural_network import MLPClassifier

    seed = int(rng.integers(1 << 31)) if rng is not None else 0
    mlp = MLPClassifier(
        hidden_layer_sizes=(hidden_dim,),
        activation='relu',
        max_iter=500,
        random_state=seed,
        early_stopping=False,
    )
    mlp.fit(scaler.transform(z_ap), y_ap)
    return mlp.predict_proba(scaler.transform(z_eval))[:, 1]


def compute_dquantile_task(
    z_ap: np.ndarray,
    z_b: np.ndarray,
    z_c: np.ndarray,
    y_ap: np.ndarray,
    n_permutations: int = 200,
    rng: np.random.Generator | None = None,
    probe_backend: str = 'logistic',
    survival_data: dict | None = None,
) -> dict:
    """Metric D3: outcome-aligned quantile stability (D_quantile^task).

    Primary Scenario 2 decision metric.  Projects representations onto
    the outcome-predictive probe direction learned from A', then measures
    W_1 between A'-referenced quantile distributions of B and C.

    **No per-cohort standardisation is applied** to the input representations.
    The caller is responsible for passing raw (non-z-scored) representations
    so that the mean-shift signal between cohorts is preserved.

    The probe direction is fixed for all permutations; only B/C cohort labels
    are shuffled; probe direction w is held fixed.

    Parameters
    ----------
    z_ap : ndarray of shape (n_ap, d)
        A'-cohort representations — raw, no per-cohort standardisation.
        Used both to fit the probe and as the quantile reference CDF.
    z_b : ndarray of shape (n_b, d)
        Cohort B representations — raw, no per-cohort standardisation.
    z_c : ndarray of shape (n_c, d)
        Cohort C representations — raw, no per-cohort standardisation.
    y_ap : ndarray of shape (n_ap,)
        Binary outcome labels for A'.
    n_permutations : int
        Number of B/C label-shuffle permutations for the null.
    rng : np.random.Generator or None
    probe_backend : str
        ``'logistic'`` (default) uses a linear logistic regression probe.
        ``'mlp'`` uses a shallow 2-layer MLP (Linear → ReLU → Linear) to
        capture nonlinear confound directions. The MLP is trained once on A'
        and its P(y=1) outputs replace the linear projection scores.
        ``'alignment'`` discovers the cohort shift direction empirically
        (v_shift = (C̄ − B̄) / ‖C̄ − B̄‖) and measures its alignment with the
        outcome-predictive probe direction: stat = |cos(v_shift, ŵ)|.
        The null shuffles B/C labels and recomputes the alignment on random
        v_shift directions. This variant is robust to nonlinear confounders
        because the mean shift is detectable even when the full distribution
        shift is distributed across many directions.

    Returns
    -------
    dict with keys:
        ``dquantile_task``           – observed stat (W_1 or alignment score)
        ``dquantile_task_null_upper`` – 97.5th percentile of null
        ``probe_direction``          – unit-norm probe direction (logistic/alignment;
                                       None for mlp backend)
    """
    if rng is None:
        rng = np.random.default_rng()

    if survival_data is not None:
        # Cox probe: fit PCA-compressed Cox model, extract hazard direction.
        # Handles censored survival data better than logistic on binarized OS.
        from sklearn.decomposition import PCA as _PCA
        from lifelines import CoxPHFitter as _CoxPHFitter
        import pandas as _pd

        _n_pca = min(50, z_ap.shape[1] - 1)
        _scaler = __import__('sklearn.preprocessing', fromlist=['StandardScaler']).StandardScaler().fit(z_ap)
        _Z_s = _scaler.transform(z_ap).astype(np.float32)
        _pca = _PCA(n_components=_n_pca, random_state=42).fit(_Z_s)
        _Z_pca = _pca.transform(_Z_s)

        _os_t = survival_data['times']
        _os_e = survival_data['events']
        _valid = (_os_t > 0) & ~np.isnan(_os_t)
        _df = _pd.DataFrame(_Z_pca[_valid], columns=[f'pc{i}' for i in range(_n_pca)])
        _df['T'] = _os_t[_valid]
        _df['E'] = _os_e[_valid].astype(bool)

        try:
            _cph = _CoxPHFitter(penalizer=0.1).fit(_df, duration_col='T', event_col='E')
            # Hazard direction in PCA space → back-project to original space
            _w_pca = _cph.params_.values  # (n_pca,)
            # w in original (unscaled) space via chain rule: PCA components @ scaler
            _w_scaled = _pca.components_.T @ _w_pca  # (d,)
            w = _w_scaled / (_scaler.scale_ + 1e-8)
            w = w / (np.linalg.norm(w) + 1e-12)
        except Exception:
            # Fall back to logistic if Cox fails
            _scaler2, _clf2 = fit_probe(z_ap, y_ap, probe_backend="logistic")
            w = _clf2.coef_[0] / (_scaler2.scale_ + 1e-8)
            w = w / (np.linalg.norm(w) + 1e-12)

        observed = compute_dquantile_projection(z_ap, z_b, z_c, w)
        n_b = len(z_b)
        z_pool = np.concatenate([z_b, z_c], axis=0)
        rng_perm = np.random.default_rng(rng.integers(1 << 63))
        null_values = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng_perm.permutation(len(z_pool))
            null_values[i] = compute_dquantile_projection(
                z_ap, z_pool[perm[:n_b]], z_pool[perm[n_b:]], w
            )
        return {
            "dquantile_task":            float(observed),
            "dquantile_task_null_upper": float(np.percentile(null_values, 97.5)),
            "probe_direction":           w,
        }

    # Fit linear probe on A' and extract direction in original (unscaled) space.
    # Logistic is required: Ridge collapses on weak batch signals.
    # w is used by all backends (logistic for projection, alignment for cosine).
    scaler, clf = fit_probe(z_ap, y_ap, probe_backend="logistic")
    w = clf.coef_[0] / (scaler.scale_ + 1e-8)
    w = w / (np.linalg.norm(w) + 1e-12)

    if probe_backend == 'logistic':

        def _score(z):
            return _project_scores(z, w)

        probe_dir = w

    elif probe_backend == 'alignment':
        # Discover cohort shift direction from B and C empirically.
        # Stat = |cos(v_shift, ŵ)| — no Wasserstein needed.
        observed = _alignment_score(z_b, z_c, w)

        n_b = len(z_b)
        z_pool = np.concatenate([z_b, z_c], axis=0)
        rng_perm = np.random.default_rng(rng.integers(1 << 63))
        null_values = np.empty(n_permutations)
        for i in range(n_permutations):
            perm = rng_perm.permutation(len(z_pool))
            null_values[i] = _alignment_score(
                z_pool[perm[:n_b]], z_pool[perm[n_b:]], w
            )

        return {
            "dquantile_task":            float(observed),
            "dquantile_task_null_upper": float(np.percentile(null_values, 97.5)),
            "probe_direction":           w,
        }

    elif probe_backend == 'mlp':
        # Nonlinear probe: shallow MLP trained on standardized A'.
        # Scores are P(y=1) — used directly as input to quantile CDF.
        _mlp_rng = np.random.default_rng(rng.integers(1 << 63))

        def _score(z):
            return _mlp_scores(z_ap, y_ap, z, scaler, hidden_dim=64, rng=_mlp_rng)

        probe_dir = None

    else:
        raise ValueError(f"Unknown probe_backend: {probe_backend!r}. "
                         f"Expected 'logistic' or 'mlp'.")

    s_ap = _score(z_ap)
    s_b  = _score(z_b)
    s_c  = _score(z_c)

    q_b = _quantiles_relative_to_reference(s_b, s_ap)
    q_c = _quantiles_relative_to_reference(s_c, s_ap)
    observed = float(wasserstein_distance(q_b, q_c))

    # Null: permute B/C labels; probe scores are held fixed.
    n_b = len(z_b)
    s_pool = np.concatenate([s_b, s_c])
    rng_perm = np.random.default_rng(rng.integers(1 << 63))
    null_values = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng_perm.permutation(len(s_pool))
        s_b_null = s_pool[perm[:n_b]]
        s_c_null = s_pool[perm[n_b:]]
        q_b_null = _quantiles_relative_to_reference(s_b_null, s_ap)
        q_c_null = _quantiles_relative_to_reference(s_c_null, s_ap)
        null_values[i] = float(wasserstein_distance(q_b_null, q_c_null))

    return {
        "dquantile_task":            float(observed),
        "dquantile_task_null_upper": float(np.percentile(null_values, 97.5)),
        "probe_direction":           probe_dir,
    }
