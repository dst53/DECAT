"""Observation (measurement) model for the DECAT simulator.

Each modality is generated as a linear mixture of latent variables:

    x_h = beta_s * A_s @ z_s + beta_h * A_h @ z_h + beta_b * A_b @ b + eps_h
    x_r = beta_s * B_s @ z_s + beta_r * B_r @ z_r + beta_b * B_b @ b + eps_r

Projection matrices are block-normalized so that each latent block
contributes unit expected variance per observed dimension *before*
beta scaling.

Nonlinear variants (for the nonlinear experiment):

  NonlinearBatchMeasurementModel  — Condition 3: biology enters linearly,
      batch term enters through a fixed random 2-layer MLP:
          batch_h = MLP_h(beta_b * b),  MLP_h: k → 128 → d_h
      Output is variance-normalized so Var ≈ beta_b² per dimension.

  NonlinearMeasurementModel       — Condition 2: entire linear mixture
      passes through a fixed random 2-layer MLP before noise:
          x_h = MLP_h(linear_mixture_h) + noise,  MLP_h: d_h → 512 → d_h
      Output is variance-normalized to match the linear baseline.

  NonlinearBothMeasurementModel   — Condition 4: nonlinear measurement +
      nonlinear batch (combination of Conditions 2 and 3).
"""

import numpy as np


def _block_normalize(M):
    """Rescale M so that tr(M M^T) / d == 1.

    Parameters
    ----------
    M : ndarray of shape (d, k)

    Returns
    -------
    M_norm : ndarray of shape (d, k)
    """
    d = M.shape[0]
    trace = np.trace(M @ M.T)
    return M * np.sqrt(d / trace)


def sample_projection_matrices(
    d: int,
    k: int,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample and block-normalize a set of projection matrices for one modality.

    Returns three block-normalized matrices of shape (d, k) corresponding to
    the shared, modality-specific, and batch blocks.

    Parameters
    ----------
    d : int
        Observation dimensionality.
    k : int
        Latent dimensionality.
    rng : np.random.Generator or None

    Returns
    -------
    tuple of three ndarray, each (d, k).
    """
    if rng is None:
        rng = np.random.default_rng()

    M_shared = _block_normalize(rng.standard_normal((d, k)))
    M_specific = _block_normalize(rng.standard_normal((d, k)))
    M_batch = _block_normalize(rng.standard_normal((d, k)))
    return M_shared, M_specific, M_batch


class MeasurementModel:
    """Stores projection matrices and generates observed modality vectors.

    Parameters
    ----------
    d_h : int
        H&E observation dimensionality.
    d_r : int
        RNA observation dimensionality.
    k : int
        Latent dimensionality.
    rng : np.random.Generator or None
    """

    def __init__(
        self,
        d_h: int,
        d_r: int,
        k: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        self.d_h = d_h
        self.d_r = d_r
        self.k = k
        # H&E projection matrices
        self.A_s, self.A_h, self.A_b = sample_projection_matrices(d_h, k, rng)
        # RNA projection matrices
        self.B_s, self.B_r, self.B_b = sample_projection_matrices(d_r, k, rng)

    def generate(
        self,
        latents: dict,
        beta_s: float = 1.0,
        beta_h: float = 1.0,
        beta_r: float = 1.0,
        beta_b: float = 0.5,
        sigma_h: float = float(np.sqrt(0.75)),
        sigma_r: float = float(np.sqrt(0.75)),
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate observed modality vectors from latent variables.

        Parameters
        ----------
        latents : dict
            Must contain ``z_s``, ``z_h``, ``z_r``, ``b`` arrays of
            shape (n, k).
        beta_s, beta_h, beta_r, beta_b : float
            Mixing coefficients for shared, modality-specific, and batch
            blocks.
        sigma_h, sigma_r : float
            Standard deviation of observation noise for each modality.
        rng : np.random.Generator or None

        Returns
        -------
        x_h : ndarray of shape (n, d_h)
        x_r : ndarray of shape (n, d_r)
        """
        if rng is None:
            rng = np.random.default_rng()

        z_s = latents["z_s"]
        z_h = latents["z_h"]
        z_r = latents["z_r"]
        b = latents["b"]
        n = z_s.shape[0]

        x_h = (beta_s * (z_s @ self.A_s.T)
               + beta_h * (z_h @ self.A_h.T)
               + beta_b * (b @ self.A_b.T)
               + sigma_h * rng.standard_normal((n, self.d_h)))

        x_r = (beta_s * (z_s @ self.B_s.T)
               + beta_r * (z_r @ self.B_r.T)
               + beta_b * (b @ self.B_b.T)
               + sigma_r * rng.standard_normal((n, self.d_r)))

        return x_h, x_r


# MLP helpers for nonlinear variants


def _sample_mlp_weights(
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sample fixed random 2-layer MLP weights (He initialization).

    Parameters
    ----------
    input_dim, hidden_dim, output_dim : int
    rng : np.random.Generator

    Returns
    -------
    W1 : (input_dim, hidden_dim)
    b1 : (hidden_dim,)
    W2 : (hidden_dim, output_dim)
    b2 : (output_dim,)
    """
    W1 = rng.standard_normal((input_dim, hidden_dim)) * np.sqrt(2.0 / input_dim)
    b1 = np.zeros(hidden_dim)
    W2 = rng.standard_normal((hidden_dim, output_dim)) * np.sqrt(2.0 / hidden_dim)
    b2 = np.zeros(output_dim)
    return W1, b1, W2, b2


def _apply_mlp(
    x: np.ndarray,
    W1: np.ndarray,
    b1: np.ndarray,
    W2: np.ndarray,
    b2: np.ndarray,
) -> np.ndarray:
    """Apply fixed 2-layer ReLU MLP.

    Parameters
    ----------
    x : (n, input_dim)

    Returns
    -------
    out : (n, output_dim)
    """
    h = np.maximum(0.0, x @ W1 + b1)
    return h @ W2 + b2


def _variance_scale(out: np.ndarray) -> float:
    """Return mean per-dimension std of out, used for variance normalization."""
    return float(np.sqrt(np.mean(out.var(axis=0)) + 1e-12))


class NonlinearBatchMeasurementModel(MeasurementModel):
    """Condition 3: biology enters linearly, batch enters through a fixed MLP.

    x_h = beta_s*(z_s@A_s.T) + beta_h*(z_h@A_h.T) + MLP_h(beta_b*b) + noise
    x_r = beta_s*(z_s@B_s.T) + beta_r*(z_r@B_r.T) + MLP_r(beta_b*b) + noise

    MLP_h: k → 128 → d_h (separate MLP per modality, different random weights)
    Output variance-normalized so per-dim Var ≈ beta_b².

    Parameters
    ----------
    d_h, d_r, k : int
    hidden_dim : int
        Hidden layer width (default 128 — expansion from k=50 input).
    rng : np.random.Generator or None
    """

    def __init__(
        self,
        d_h: int,
        d_r: int,
        k: int,
        hidden_dim: int = 128,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        super().__init__(d_h, d_r, k, rng=rng)
        rng_mlp_h = np.random.default_rng(rng.integers(1 << 63))
        rng_mlp_r = np.random.default_rng(rng.integers(1 << 63))
        self._mlp_h = _sample_mlp_weights(k, hidden_dim, d_h, rng_mlp_h)
        self._mlp_r = _sample_mlp_weights(k, hidden_dim, d_r, rng_mlp_r)

    def generate(
        self,
        latents: dict,
        beta_s: float = 1.0,
        beta_h: float = 1.0,
        beta_r: float = 1.0,
        beta_b: float = 0.5,
        sigma_h: float = float(np.sqrt(0.75)),
        sigma_r: float = float(np.sqrt(0.75)),
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate observations with nonlinear batch term.

        Parameters
        ----------
        latents : dict
        beta_s, beta_h, beta_r, beta_b : float
        sigma_h, sigma_r : float
        rng : np.random.Generator or None

        Returns
        -------
        x_h : (n, d_h)
        x_r : (n, d_r)
        """
        if rng is None:
            rng = np.random.default_rng()

        z_s = latents["z_s"]
        z_h = latents["z_h"]
        z_r = latents["z_r"]
        b = latents["b"]
        n = z_s.shape[0]

        # Nonlinear batch term — normalize to match linear baseline variance
        raw_h = _apply_mlp(beta_b * b, *self._mlp_h)
        raw_r = _apply_mlp(beta_b * b, *self._mlp_r)
        batch_h = raw_h * (beta_b / _variance_scale(raw_h))
        batch_r = raw_r * (beta_b / _variance_scale(raw_r))

        x_h = (beta_s * (z_s @ self.A_s.T)
               + beta_h * (z_h @ self.A_h.T)
               + batch_h
               + sigma_h * rng.standard_normal((n, self.d_h)))

        x_r = (beta_s * (z_s @ self.B_s.T)
               + beta_r * (z_r @ self.B_r.T)
               + batch_r
               + sigma_r * rng.standard_normal((n, self.d_r)))

        return x_h, x_r


class NonlinearMeasurementModel(MeasurementModel):
    """Condition 2: entire linear mixture passed through a fixed MLP before noise.

    x_h = MLP_h(beta_s*(z_s@A_s.T) + beta_h*(z_h@A_h.T) + beta_b*(b@A_b.T)) + noise
    x_r = MLP_r(beta_s*(z_s@B_s.T) + beta_r*(z_r@B_r.T) + beta_b*(b@B_b.T)) + noise

    MLP_h: d_h → 512 → d_h (non-bottleneck; separate per modality)
    Output variance-normalized to match expected linear-baseline variance.

    Parameters
    ----------
    d_h, d_r, k : int
    hidden_dim : int
        Hidden layer width (default 512 — preserves most information).
    rng : np.random.Generator or None
    """

    def __init__(
        self,
        d_h: int,
        d_r: int,
        k: int,
        hidden_dim: int = 512,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        super().__init__(d_h, d_r, k, rng=rng)
        rng_mlp_h = np.random.default_rng(rng.integers(1 << 63))
        rng_mlp_r = np.random.default_rng(rng.integers(1 << 63))
        self._mlp_h = _sample_mlp_weights(d_h, hidden_dim, d_h, rng_mlp_h)
        self._mlp_r = _sample_mlp_weights(d_r, hidden_dim, d_r, rng_mlp_r)

    def generate(
        self,
        latents: dict,
        beta_s: float = 1.0,
        beta_h: float = 1.0,
        beta_r: float = 1.0,
        beta_b: float = 0.5,
        sigma_h: float = float(np.sqrt(0.75)),
        sigma_r: float = float(np.sqrt(0.75)),
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate observations with nonlinear measurement function.

        Parameters
        ----------
        latents : dict
        beta_s, beta_h, beta_r, beta_b : float
        sigma_h, sigma_r : float
        rng : np.random.Generator or None

        Returns
        -------
        x_h : (n, d_h)
        x_r : (n, d_r)
        """
        if rng is None:
            rng = np.random.default_rng()

        z_s = latents["z_s"]
        z_h = latents["z_h"]
        z_r = latents["z_r"]
        b = latents["b"]
        n = z_s.shape[0]

        # Compute linear mixture then pass through MLP
        linear_h = (beta_s * (z_s @ self.A_s.T)
                    + beta_h * (z_h @ self.A_h.T)
                    + beta_b * (b @ self.A_b.T))
        linear_r = (beta_s * (z_s @ self.B_s.T)
                    + beta_r * (z_r @ self.B_r.T)
                    + beta_b * (b @ self.B_b.T))

        raw_h = _apply_mlp(linear_h, *self._mlp_h)
        raw_r = _apply_mlp(linear_r, *self._mlp_r)

        # Target std: sqrt(beta_s² + beta_h² + beta_b²) — matches linear baseline
        target_std_h = np.sqrt(beta_s**2 + beta_h**2 + beta_b**2)
        target_std_r = np.sqrt(beta_s**2 + beta_r**2 + beta_b**2)
        obs_h = raw_h * (target_std_h / _variance_scale(raw_h))
        obs_r = raw_r * (target_std_r / _variance_scale(raw_r))

        x_h = obs_h + sigma_h * rng.standard_normal((n, self.d_h))
        x_r = obs_r + sigma_r * rng.standard_normal((n, self.d_r))

        return x_h, x_r


class NonlinearBothMeasurementModel(MeasurementModel):
    """Condition 4: nonlinear measurement + nonlinear batch.

    x_h = MLP_obs_h(beta_s*(z_s@A_s.T) + beta_h*(z_h@A_h.T)) + MLP_batch_h(beta_b*b) + noise

    Biology passes through measurement MLP (d_h → 512 → d_h).
    Batch passes through separate batch MLP (k → 128 → d_h).
    Both outputs are variance-normalized independently.

    Parameters
    ----------
    d_h, d_r, k : int
    obs_hidden_dim : int
        Hidden dim for measurement MLP (default 512).
    batch_hidden_dim : int
        Hidden dim for batch MLP (default 128).
    rng : np.random.Generator or None
    """

    def __init__(
        self,
        d_h: int,
        d_r: int,
        k: int,
        obs_hidden_dim: int = 512,
        batch_hidden_dim: int = 128,
        rng: np.random.Generator | None = None,
    ) -> None:
        if rng is None:
            rng = np.random.default_rng()
        super().__init__(d_h, d_r, k, rng=rng)
        rng_obs_h   = np.random.default_rng(rng.integers(1 << 63))
        rng_obs_r   = np.random.default_rng(rng.integers(1 << 63))
        rng_batch_h = np.random.default_rng(rng.integers(1 << 63))
        rng_batch_r = np.random.default_rng(rng.integers(1 << 63))
        self._mlp_obs_h   = _sample_mlp_weights(d_h, obs_hidden_dim, d_h, rng_obs_h)
        self._mlp_obs_r   = _sample_mlp_weights(d_r, obs_hidden_dim, d_r, rng_obs_r)
        self._mlp_batch_h = _sample_mlp_weights(k, batch_hidden_dim, d_h, rng_batch_h)
        self._mlp_batch_r = _sample_mlp_weights(k, batch_hidden_dim, d_r, rng_batch_r)

    def generate(
        self,
        latents: dict,
        beta_s: float = 1.0,
        beta_h: float = 1.0,
        beta_r: float = 1.0,
        beta_b: float = 0.5,
        sigma_h: float = float(np.sqrt(0.75)),
        sigma_r: float = float(np.sqrt(0.75)),
        rng: np.random.Generator | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate observations with nonlinear measurement and nonlinear batch.

        Parameters
        ----------
        latents : dict
        beta_s, beta_h, beta_r, beta_b : float
        sigma_h, sigma_r : float
        rng : np.random.Generator or None

        Returns
        -------
        x_h : (n, d_h)
        x_r : (n, d_r)
        """
        if rng is None:
            rng = np.random.default_rng()

        z_s = latents["z_s"]
        z_h = latents["z_h"]
        z_r = latents["z_r"]
        b = latents["b"]
        n = z_s.shape[0]

        # Nonlinear measurement (biology only, no batch in this MLP)
        bio_h = beta_s * (z_s @ self.A_s.T) + beta_h * (z_h @ self.A_h.T)
        bio_r = beta_s * (z_s @ self.B_s.T) + beta_r * (z_r @ self.B_r.T)
        raw_obs_h = _apply_mlp(bio_h, *self._mlp_obs_h)
        raw_obs_r = _apply_mlp(bio_r, *self._mlp_obs_r)
        target_bio_std_h = np.sqrt(beta_s**2 + beta_h**2)
        target_bio_std_r = np.sqrt(beta_s**2 + beta_r**2)
        obs_h = raw_obs_h * (target_bio_std_h / _variance_scale(raw_obs_h))
        obs_r = raw_obs_r * (target_bio_std_r / _variance_scale(raw_obs_r))

        # Nonlinear batch
        raw_batch_h = _apply_mlp(beta_b * b, *self._mlp_batch_h)
        raw_batch_r = _apply_mlp(beta_b * b, *self._mlp_batch_r)
        batch_h = raw_batch_h * (beta_b / _variance_scale(raw_batch_h))
        batch_r = raw_batch_r * (beta_b / _variance_scale(raw_batch_r))

        x_h = obs_h + batch_h + sigma_h * rng.standard_normal((n, self.d_h))
        x_r = obs_r + batch_r + sigma_r * rng.standard_normal((n, self.d_r))

        return x_h, x_r
