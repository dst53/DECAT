"""Outcome generation for the DECAT simulator.

Risk scores are computed as:

    r_i = alpha_s <w_s, z_s> + alpha_h <w_h, z'_h> + alpha_r <w_r, z'_r>
          + alpha_b <w_b, b> + eps_y

where:

    z'_m = z_m + gamma * ((1 - eta) b + eta b_tilde) / sqrt((1 - eta)^2 + eta^2)

* **gamma** controls proxy strength (degree of entanglement).
* **eta** controls proxy alignment / identifiability across cohorts.
  When eta = 0, proxy signal is aligned with the shared batch axis b.
  When eta > 0, proxy signal is partially driven by an independent
  nuisance component b_tilde ~ N(0, I), introducing incoherent
  variation across patients and cohorts.
  The denominator normalizes the mixture so that proxy variance is
  constant regardless of eta.

The outcome model has two independent confounding knobs:

* **gamma** — feature-level proxy entanglement: contaminates modality-
  specific latents *before* they enter the measurement model.
* **alpha_b** — outcome-level confounding: the shared nuisance latent
  *b* directly contributes to the risk score, independently of gamma.

Binary labels: y_i = 1(r_i > 0).

Weight vectors w are sampled once per run and normalized to unit norm so
that Var(<w_bar, z>) = 1 when z ~ N(0, I).
"""

import numpy as np


def sample_outcome_weights(k: int, rng: np.random.Generator | None = None) -> dict:
    """Sample and normalize outcome weight vectors (one per latent block).

    Parameters
    ----------
    k : int
        Latent dimensionality.
    rng : np.random.Generator or None

    Returns
    -------
    dict with keys ``w_s``, ``w_h``, ``w_r``, ``w_b``, each of shape (k,),
    unit-normalized.
    """
    if rng is None:
        rng = np.random.default_rng()

    def _unit(v):
        return v / np.linalg.norm(v)

    return {
        "w_s": _unit(rng.standard_normal(k)),
        "w_h": _unit(rng.standard_normal(k)),
        "w_r": _unit(rng.standard_normal(k)),
        "w_b": _unit(rng.standard_normal(k)),
    }


def generate_outcomes(
    latents: dict,
    weights: dict,
    alpha_s: float = 1.0,
    alpha_h: float = 0.0,
    alpha_r: float = 0.0,
    alpha_b: float = 0.0,
    gamma_h: float = 0.0,
    gamma_r: float = 0.0,
    eta: float = 0.0,
    sigma_y: float = 0.0,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate binary outcome labels from latent risk scores.

    Parameters
    ----------
    latents : dict
        Must contain ``z_s``, ``z_h``, ``z_r``, ``b`` arrays of shape
        (n, k).
    weights : dict
        Must contain ``w_s``, ``w_h``, ``w_r``, ``w_b`` arrays of
        shape (k,), as returned by :func:`sample_outcome_weights`.
    alpha_s, alpha_h, alpha_r : float
        Coefficients controlling where predictive signal resides.
    alpha_b : float
        Outcome-level confounding coefficient.  When > 0, the shared
        nuisance latent *b* directly influences the risk score,
        independently of gamma_h/gamma_r.
    gamma_h : float
        Feature-level proxy contamination for H&E.  When > 0,
        z_h_prime = z_h + gamma_h * proxy.
    gamma_r : float
        Feature-level proxy contamination for RNA.  When > 0,
        z_r_prime = z_r + gamma_r * proxy.  Set equal to gamma_h for
        dual-modality proxy (2D); set to 0 for single-modality (2A).
    eta : float
        Proxy misalignment parameter.  When 0, proxy is aligned with
        the shared batch axis (proxy = b exactly; no b_tilde is sampled).
        When > 0, proxy is a normalized mixture of b and independent
        noise b_tilde, keeping variance constant across eta.
    sigma_y : float
        Standard deviation of outcome noise.
    rng : np.random.Generator or None

    Returns
    -------
    y : ndarray of shape (n,), int
        Binary outcome labels.
    risk : ndarray of shape (n,)
        Continuous risk scores (before thresholding).
    proxy : ndarray of shape (n, k)
        Proxy variable used for entanglement (zeros when gamma=0).
    """
    if rng is None:
        rng = np.random.default_rng()

    z_s = latents["z_s"]
    z_h = latents["z_h"]
    z_r = latents["z_r"]
    b = latents["b"]
    n = z_s.shape[0]

    # Feature-level proxy entanglement — shared proxy vector, modality-specific scaling
    if gamma_h > 0 or gamma_r > 0:
        if np.isclose(eta, 0.0):
            proxy = b
        else:
            rng_proxy = np.random.default_rng(rng.integers(1 << 63))
            b_tilde = rng_proxy.standard_normal(b.shape)
            norm = np.sqrt((1 - eta) ** 2 + eta ** 2)
            proxy = ((1 - eta) * b + eta * b_tilde) / norm
        z_h_prime = z_h + gamma_h * proxy
        z_r_prime = z_r + gamma_r * proxy
    else:
        proxy = np.zeros_like(b)
        z_h_prime = z_h
        z_r_prime = z_r

    # Outcome-level confounding (independent of gamma)
    risk = (alpha_s * (z_s @ weights["w_s"])
            + alpha_h * (z_h_prime @ weights["w_h"])
            + alpha_r * (z_r_prime @ weights["w_r"])
            + alpha_b * (b @ weights["w_b"]))

    if sigma_y > 0:
        risk = risk + sigma_y * rng.standard_normal(n)

    y = (risk > 0).astype(np.int32)
    return y, risk, proxy
