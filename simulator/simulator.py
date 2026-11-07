"""Main simulator entry point for the DECAT framework.

``simulate_run(config)`` generates a complete synthetic dataset
containing observed modalities, latent variables, outcomes, and
cohort labels for all four cohorts (A, A', B, C).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

from simulator.cohorts import generate_cohorts, sample_batch_shift
from simulator.measurement_model import (
    MeasurementModel,
    NonlinearBatchMeasurementModel,
    NonlinearBothMeasurementModel,
    NonlinearMeasurementModel,
)
from simulator.outcome_model import generate_outcomes, sample_outcome_weights


@dataclass
class SimulatorConfig:
    """Configuration for a single synthetic run.

    Attributes
    ----------
    latent_dim : int
        Dimensionality *k* of each latent block.
    d_h : int
        Observation dimensionality for H&E.
    d_r : int
        Observation dimensionality for RNA.
    n_a : int
        Patients in Cohort A (training).
    n_a_prime : int
        Patients in Cohort A' (within-cohort validation).
    n_b : int
        Patients in Cohort B (external, shifted).
    n_c : int
        Patients in Cohort C (external, scaled shift relative to B).
    beta_s : float
        Shared-biology mixing coefficient.
    beta_h : float
        H&E-specific mixing coefficient.
    beta_r : float
        RNA-specific mixing coefficient.
    beta_b : float
        Batch mixing coefficient.
    sigma_h : float
        H&E observation noise std.
    sigma_r : float
        RNA observation noise std.
    alpha_s : float
        Outcome coefficient for shared biology.
    alpha_h : float
        Outcome coefficient for H&E-specific biology.
    alpha_r : float
        Outcome coefficient for RNA-specific biology.
    gamma_h : float
        Feature-level proxy contamination for H&E observations.
        z_h_prime = z_h + gamma_h * proxy.  Use gamma_h > 0, gamma_r = 0
        for single-modality proxy (scanner artifact); gamma_h = gamma_r > 0
        for dual-modality proxy (site batch effect hitting both modalities).
    gamma_r : float
        Feature-level proxy contamination for RNA observations.
        z_r_prime = z_r + gamma_r * proxy.
    eta : float
        Proxy misalignment parameter.
        eta = 0 → aligned proxy (fully shared batch axis).
        eta > 0 → misaligned proxy (cohort-specific variation).
    alpha_b : float
        Outcome-level confounding coefficient (independent of gamma_h/gamma_r).
    sigma_y : float
        Outcome noise std.
    epsilon : float
        Relative shift magnitude between Cohort B and C
        (C uses ``(1 + epsilon) * mu_b``).
    batch_align : float
        Alignment between the batch shift direction ``mu_b`` and the
        outcome-predictive batch weight ``w_b``.  0.0 (default) draws
        ``mu_b`` independently of ``w_b`` — the two are unrelated random
        directions in b-space.  1.0 sets ``mu_b ∝ w_b``, so the cohort
        that shifts most relative to A is the one whose batch dimension
        predicts the outcome.  Intermediate values interpolate linearly
        before re-normalising and rescaling to ``batch_shift_magnitude``.
        Can be combined with ``proxy_shift_align`` for mixed scenarios
        (e.g. Transition B); typical use keeps their sum ≤ 1.
    proxy_shift_align : float
        Alignment between the batch shift direction ``mu_b`` and the
        modality-specific outcome weight selected by
        ``proxy_shift_target``.  0.0 (default) draws ``mu_b``
        independently.  1.0 sets ``mu_b ∝ w_target``, so the cohort
        shift is maximally aligned with the direction that creates proxy
        outcome signal via feature-level entanglement (gamma > 0).
        Combined with ``batch_align`` via three-way interpolation:
        ``raw = (1 - batch_align - proxy_shift_align) * n_hat
                + batch_align * w_b + proxy_shift_align * w_target``.
    proxy_shift_target : str
        Which modality-specific weight to align ``mu_b`` with when
        ``proxy_shift_align > 0``.  ``'h'`` (default) targets the H&E
        weight ``w_h``; ``'r'`` targets the RNA weight ``w_r``.
    batch_shift_magnitude : float or None
        Fixed L2 norm for ``mu_b``.  When ``None`` (default) the norm
        is ``sqrt(latent_dim)``, which scales with dimensionality and
        causes outcome saturation at high k.  Setting this to a constant
        (e.g. 2.0) decouples the batch effect size from k, matching the
        biological intuition that a scanner or staining batch effect has
        a fixed physical magnitude regardless of the latent space
        dimension.
    seed : int
        Random seed for reproducibility.
    """
    latent_dim: int = 50
    d_h: int = 1024
    d_r: int = 1024
    n_a: int = 2500
    n_a_prime: int = 500
    n_b: int = 500
    n_c: int = 500
    beta_s: float = 1.0
    beta_h: float = 1.0
    beta_r: float = 1.0
    beta_b: float = 0.75
    # sigma_h / sigma_r: if None, auto-computed from betas to fix SNR=3
    # (spec §01 §6).  Pass an explicit value to override.
    sigma_h: float | None = None
    sigma_r: float | None = None

    def __post_init__(self):
        if self.sigma_h is None:
            self.sigma_h = math.sqrt((self.beta_s**2 + self.beta_h**2 + self.beta_b**2) / 3.0)
        if self.sigma_r is None:
            self.sigma_r = math.sqrt((self.beta_s**2 + self.beta_r**2 + self.beta_b**2) / 3.0)
    alpha_s: float = 1.0
    alpha_h: float = 0.0
    alpha_r: float = 0.0
    gamma_h: float = 0.0
    gamma_r: float = 0.0
    eta: float = 0.0
    alpha_b: float = 0.0
    sigma_y: float = 0.0
    epsilon: float = 0.2
    batch_align: float = 0.0
    proxy_shift_align: float = 0.0
    proxy_shift_target: str = 'h'
    batch_shift_magnitude: float | None = None
    measurement_variant: str = 'linear'
    seed: int = 0


def simulate_run(config: SimulatorConfig) -> Dict[str, np.ndarray]:
    """Generate a complete synthetic dataset for one run.

    Parameters
    ----------
    config : SimulatorConfig

    Returns
    -------
    dict with keys:
        ``xh``      – (N, d_h) observed H&E vectors
        ``xr``      – (N, d_r) observed RNA vectors
        ``y``       – (N,) binary outcome labels
        ``risk``    – (N,) continuous risk scores
        ``proxy``   – (N, k) proxy variable used for entanglement
        ``z_s``     – (N, k) shared latent
        ``z_h``     – (N, k) H&E-specific latent
        ``z_r``     – (N, k) RNA-specific latent
        ``b``       – (N, k) batch latent
        ``cohort``  – (N,) string cohort labels
        ``weights`` – outcome weight vectors (per-run, fixed)
        ``measurement_model`` – :class:`MeasurementModel` instance
        ``mu_b``    – (k,) batch shift vector
    where N = n_a + n_a_prime + n_b + n_c.
    """
    rng_root = np.random.default_rng(config.seed)

    rng_batch       = np.random.default_rng(rng_root.integers(1 << 63))
    rng_measurement = np.random.default_rng(rng_root.integers(1 << 63))
    rng_weights     = np.random.default_rng(rng_root.integers(1 << 63))
    rng_latents     = np.random.default_rng(rng_root.integers(1 << 63))
    rng_obs_noise   = np.random.default_rng(rng_root.integers(1 << 63))
    rng_outcome     = np.random.default_rng(rng_root.integers(1 << 63))

    k = config.latent_dim

    # Fixed per-run components
    # Weights are sampled before mu_b so that batch_align can reference w_b.
    # RNG streams are derived independently above, so computation order does
    # not affect reproducibility for existing seeds when batch_align=0.
    _measurement_classes = {
        'linear':           MeasurementModel,
        'nonlinear_batch':  NonlinearBatchMeasurementModel,
        'nonlinear_obs':    NonlinearMeasurementModel,
        'nonlinear_both':   NonlinearBothMeasurementModel,
    }
    if config.measurement_variant not in _measurement_classes:
        raise ValueError(f"Unknown measurement_variant: {config.measurement_variant!r}")
    measurement = _measurement_classes[config.measurement_variant](
        config.d_h, config.d_r, k, rng=rng_measurement
    )
    weights = sample_outcome_weights(k, rng=rng_weights)

    _mag = (config.batch_shift_magnitude if config.batch_shift_magnitude is not None
            else np.sqrt(k))

    if config.batch_align == 0.0 and config.proxy_shift_align == 0.0:
        raw = sample_batch_shift(k, rng=rng_batch)
        mu_b = raw / (np.linalg.norm(raw) + 1e-12) * _mag
    else:
        # Three-way interpolation between an independent noise direction,
        # the confounding outcome weight w_b (batch_align), and the
        # modality-specific proxy weight w_h or w_r (proxy_shift_align).
        # When proxy_shift_align=0 this reduces to the original batch_align
        # formula.  ||mu_b|| = _mag is preserved regardless of the
        # alignment coefficients.
        noise = rng_batch.standard_normal(k)
        noise = noise / (np.linalg.norm(noise) + 1e-12)  # unit direction
        w_proxy = (weights["w_h"] if config.proxy_shift_target == 'h'
                   else weights["w_r"])
        raw = ((1.0 - config.batch_align - config.proxy_shift_align) * noise
               + config.batch_align * weights["w_b"]
               + config.proxy_shift_align * w_proxy)
        mu_b = raw / (np.linalg.norm(raw) + 1e-12) * _mag

    # Latents per cohort
    cohort_latents = generate_cohorts(
        config.n_a, config.n_a_prime, config.n_b, config.n_c,
        k, mu_b, epsilon=config.epsilon, rng=rng_latents,
    )

    # Concatenate across cohorts
    cohort_names = ["A", "A_prime", "B", "C"]
    cohort_sizes = [config.n_a, config.n_a_prime, config.n_b, config.n_c]

    z_s_all = np.concatenate([cohort_latents[c]["z_s"] for c in cohort_names])
    z_h_all = np.concatenate([cohort_latents[c]["z_h"] for c in cohort_names])
    z_r_all = np.concatenate([cohort_latents[c]["z_r"] for c in cohort_names])
    b_all = np.concatenate([cohort_latents[c]["b"] for c in cohort_names])

    latents_all = {"z_s": z_s_all, "z_h": z_h_all, "z_r": z_r_all, "b": b_all}

    cohort_labels = np.concatenate([
        np.full(n, name, dtype=object)
        for name, n in zip(cohort_names, cohort_sizes)
    ])

    # Observations
    xh, xr = measurement.generate(
        latents_all,
        beta_s=config.beta_s, beta_h=config.beta_h,
        beta_r=config.beta_r, beta_b=config.beta_b,
        sigma_h=config.sigma_h, sigma_r=config.sigma_r,
        rng=rng_obs_noise,
    )

    # Outcomes
    y, risk, proxy = generate_outcomes(
        latents_all, weights,
        alpha_s=config.alpha_s, alpha_h=config.alpha_h,
        alpha_r=config.alpha_r, alpha_b=config.alpha_b,
        gamma_h=config.gamma_h, gamma_r=config.gamma_r, eta=config.eta,
        sigma_y=config.sigma_y, rng=rng_outcome,
    )

    return {
        "xh": xh,
        "xr": xr,
        "y": y,
        "risk": risk,
        "proxy": proxy,
        "z_s": z_s_all,
        "z_h": z_h_all,
        "z_r": z_r_all,
        "b": b_all,
        "cohort": cohort_labels,
        "weights": weights,
        "measurement_model": measurement,
        "mu_b": mu_b,
    }
