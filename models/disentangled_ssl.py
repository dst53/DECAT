"""DisentangledSSL — Information-theoretic disentangled multimodal representation model.

Implements the two-step self-supervised learning procedure from:
  Wang et al. (2025) "An Information Criterion for Controlled Disentanglement
  of Multimodal Data", ICLR 2025.  arXiv:2410.23996

Architecture
------------
Step 1 — Shared representations:
  Encoder f_c^m maps each modality x_m to a shared latent z_c^m (L2-normalized).
  Objective: maximize cross-modal InfoNCE, penalized by the conditional entropy
  bottleneck (CEB) term that minimizes I(z_c; x_m | x_{m'}).

  Tractable objective (eq. 3 in paper):
    L_c = −InfoNCE(z_c^h, z_c^r) + β · E[z_c^h ⊤ z_c^r]

  The second term is a tractable upper bound on the CEB penalty under vMF
  distributions; it penalizes modality-specific information leaking into z_c
  by pulling the encodings of paired samples closer.

Step 2 — Modality-specific representations:
  Encoder f_s^m maps (x_m, z_c^m*) → z_s^m conditioned on the frozen shared
  representation, ensuring z_s^m captures only residual modality-specific information.
  Objective: maximize InfoNCE on augmented views, penalized by an orthogonality
  loss that enforces independence between z_s^m and z_c^m.

  L_s = −InfoNCE_aug(z_s^m, z_c^{m'}) + λ · ||Z_s ⊤ Z_c||_F²

Pipeline interface
------------------
To integrate cleanly with the existing soft-masking pipeline infrastructure,
encode_h and encode_r return the CONCATENATION [z_c | z_s] (shape n × 2K).
A binary sharedness vector gamma = [1,...,1, 0,...,0] stored in gamma.npy
causes soft masking to act as a hard split: the shared component recovers z_c
and the specific component recovers z_s, with no soft masking needed.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class _SharedEncoder(nn.Module):
    """MLP encoder for shared representations with L2-normalized output."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


class _SpecificEncoder(nn.Module):
    """MLP encoder for modality-specific representations.

    Takes both the observation x and the shared representation z_c as input,
    ensuring the specific encoder captures only residual information not
    already encoded in z_c.
    """

    def __init__(self, input_dim: int, shared_dim: int, hidden_dim: int,
                 output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim + shared_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, z_c.detach()], dim=-1))


def _infonce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    """Symmetric NT-Xent (InfoNCE) loss for matched pairs."""
    sim = z1 @ z2.T / temperature
    labels = torch.arange(len(z1), device=z1.device)
    return (F.cross_entropy(sim, labels) + F.cross_entropy(sim.T, labels)) / 2


def _orthogonal_loss(z_s: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
    """Frobenius norm of pairwise cosine similarities between z_s and z_c.

    Encourages marginal independence between shared and specific representations
    by penalizing cross-correlation across latent dimensions (eq. 11 in paper).
    """
    z_s_n = F.normalize(z_s, dim=0)
    z_c_n = F.normalize(z_c, dim=0)
    return (z_s_n.T @ z_c_n).pow(2).sum()


class DisentangledSSL:
    """Two-step self-supervised disentangled representation model.

    Parameters
    ----------
    latent_dim : int
        Output dimensionality K for both shared and modality-specific components.
        encode_h / encode_r return (n, 2K) arrays: first K dims are shared z_c,
        last K dims are modality-specific z_s.
    hidden_multiplier : int
        Hidden layer width relative to input_dim.
    beta : float
        Weight of the CEB penalty in Step 1. Higher β pushes z_c to contain
        less modality-specific information (more compressed shared representation).
    lambda_orth : float
        Weight of the orthogonality loss in Step 2. Higher λ enforces stronger
        independence between z_s and z_c.
    temperature : float
        InfoNCE softmax temperature.
    aug_noise_std : float
        Standard deviation of Gaussian noise used for data augmentation in Step 2.
    lr : float
        Adam learning rate.
    n_epochs_shared : int
        Training epochs for Step 1.
    n_epochs_specific : int
        Training epochs for Step 2.
    batch_size : int
    seed : int
    device : str or None
        Torch device. Defaults to CUDA if available else CPU.
    """

    def __init__(
        self,
        latent_dim: int = 50,
        hidden_multiplier: int = 2,
        beta: float = 0.1,
        lambda_orth: float = 0.1,
        temperature: float = 0.07,
        aug_noise_std: float = 0.1,
        lr: float = 1e-3,
        n_epochs_shared: int = 100,
        n_epochs_specific: int = 100,
        batch_size: int = 256,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.hidden_multiplier = hidden_multiplier
        self.beta = beta
        self.lambda_orth = lambda_orth
        self.temperature = temperature
        self.aug_noise_std = aug_noise_std
        self.lr = lr
        self.n_epochs_shared = n_epochs_shared
        self.n_epochs_specific = n_epochs_specific
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self._enc_h_c: _SharedEncoder | None = None
        self._enc_r_c: _SharedEncoder | None = None
        self._enc_h_s: _SpecificEncoder | None = None
        self._enc_r_s: _SpecificEncoder | None = None

    def fit(
        self,
        xh: np.ndarray,
        xr: np.ndarray,
        checkpoint_fn: callable | None = None,
        checkpoint_interval: int = 25,
    ) -> "DisentangledSSL":
        """Train shared then modality-specific encoders on Cohort A data.

        Parameters
        ----------
        xh : ndarray of shape (n, d_h)
            H&E observations.
        xr : ndarray of shape (n, d_r)
            RNA observations.
        checkpoint_fn : callable or None
            If provided, called as ``checkpoint_fn(phase, epoch, model, loss)``
            every ``checkpoint_interval`` epochs within each phase ("shared"
            or "specific"), and at epoch 0 before any training.
        checkpoint_interval : int
            Epoch interval between checkpoint_fn calls.

        Returns
        -------
        self
        """
        torch.manual_seed(self.seed)
        device = torch.device(self.device)
        d_h, d_r = xh.shape[1], xr.shape[1]
        K = self.latent_dim
        h_h = self.hidden_multiplier * d_h
        h_r = self.hidden_multiplier * d_r

        Xh = torch.tensor(xh, dtype=torch.float32, device=device)
        Xr = torch.tensor(xr, dtype=torch.float32, device=device)
        n = len(Xh)
        gen = torch.Generator(device='cpu')
        gen.manual_seed(self.seed)

        # Step 1: Shared representations
        self._enc_h_c = _SharedEncoder(d_h, h_h, K).to(device)
        self._enc_r_c = _SharedEncoder(d_r, h_r, K).to(device)

        opt1 = torch.optim.Adam(
            list(self._enc_h_c.parameters()) + list(self._enc_r_c.parameters()),
            lr=self.lr,
        )

        if checkpoint_fn is not None:
            self._enc_h_c.eval(); self._enc_r_c.eval()
            checkpoint_fn("shared", 0, self, float('nan'))
            self._enc_h_c.train(); self._enc_r_c.train()

        epoch_loss = float('nan')
        for epoch in range(self.n_epochs_shared):
            idx = torch.randperm(n, generator=gen)
            batch_losses = []
            for start in range(0, n, self.batch_size):
                b = idx[start: start + self.batch_size]
                zh_c = self._enc_h_c(Xh[b])
                zr_c = self._enc_r_c(Xr[b])

                loss_nce = _infonce_loss(zh_c, zr_c, self.temperature)
                loss_ceb = -(zh_c * zr_c).sum(dim=-1).mean()
                loss = loss_nce + self.beta * loss_ceb

                opt1.zero_grad()
                loss.backward()
                opt1.step()
                batch_losses.append(loss.item())

            epoch_loss = float(np.mean(batch_losses))
            if checkpoint_fn is not None and (epoch + 1) % checkpoint_interval == 0:
                self._enc_h_c.eval(); self._enc_r_c.eval()
                checkpoint_fn("shared", epoch + 1, self, epoch_loss)
                self._enc_h_c.train(); self._enc_r_c.train()

        self._enc_h_c.eval()
        self._enc_r_c.eval()

        # Step 2: Modality-specific representations
        self._enc_h_s = _SpecificEncoder(d_h, K, h_h, K).to(device)
        self._enc_r_s = _SpecificEncoder(d_r, K, h_r, K).to(device)

        opt2 = torch.optim.Adam(
            list(self._enc_h_s.parameters()) + list(self._enc_r_s.parameters()),
            lr=self.lr,
        )

        rng_aug = torch.Generator(device='cpu')
        rng_aug.manual_seed(self.seed + 1)

        if checkpoint_fn is not None:
            self._enc_h_s.eval(); self._enc_r_s.eval()
            checkpoint_fn("specific", 0, self, float('nan'))
            self._enc_h_s.train(); self._enc_r_s.train()

        for epoch in range(self.n_epochs_specific):
            idx = torch.randperm(n, generator=gen)
            batch_losses = []
            for start in range(0, n, self.batch_size):
                b = idx[start: start + self.batch_size]
                xh_b, xr_b = Xh[b], Xr[b]

                with torch.no_grad():
                    zh_c = self._enc_h_c(xh_b)
                    zr_c = self._enc_r_c(xr_b)

                noise_h = self.aug_noise_std * torch.randn_like(xh_b)
                noise_r = self.aug_noise_std * torch.randn_like(xr_b)
                xh_aug = xh_b + noise_h
                xr_aug = xr_b + noise_r

                with torch.no_grad():
                    zh_c_aug = self._enc_h_c(xh_aug)
                    zr_c_aug = self._enc_r_c(xr_aug)

                zh_s     = self._enc_h_s(xh_b,   zh_c)
                zh_s_aug = self._enc_h_s(xh_aug,  zh_c_aug)
                zr_s     = self._enc_r_s(xr_b,   zr_c)
                zr_s_aug = self._enc_r_s(xr_aug,  zr_c_aug)

                zh_cat     = F.normalize(torch.cat([zh_s,     zr_c],     dim=-1), dim=-1)
                zh_cat_aug = F.normalize(torch.cat([zh_s_aug, zr_c_aug], dim=-1), dim=-1)
                zr_cat     = F.normalize(torch.cat([zr_s,     zh_c],     dim=-1), dim=-1)
                zr_cat_aug = F.normalize(torch.cat([zr_s_aug, zh_c_aug], dim=-1), dim=-1)

                loss_nce = (_infonce_loss(zh_cat, zh_cat_aug, self.temperature) +
                            _infonce_loss(zr_cat, zr_cat_aug, self.temperature))
                loss_orth = (_orthogonal_loss(zh_s, zh_c) +
                             _orthogonal_loss(zr_s, zr_c))
                loss = loss_nce + self.lambda_orth * loss_orth

                opt2.zero_grad()
                loss.backward()
                opt2.step()
                batch_losses.append(loss.item())

            epoch_loss = float(np.mean(batch_losses))
            if checkpoint_fn is not None and (epoch + 1) % checkpoint_interval == 0:
                self._enc_h_s.eval(); self._enc_r_s.eval()
                checkpoint_fn("specific", epoch + 1, self, epoch_loss)
                self._enc_h_s.train(); self._enc_r_s.train()

        self._enc_h_s.eval()
        self._enc_r_s.eval()
        return self

    def _encode(
        self,
        x: np.ndarray,
        enc_c: _SharedEncoder,
        enc_s: _SpecificEncoder,
    ) -> np.ndarray:
        """Encode one modality, returning [z_c | z_s] concatenation."""
        self._check_fitted()
        device = torch.device(self.device)
        with torch.no_grad():
            X = torch.tensor(x, dtype=torch.float32, device=device)
            z_c = enc_c(X)
            z_s = enc_s(X, z_c)
        return torch.cat([z_c, z_s], dim=-1).cpu().numpy()

    def encode_h(self, xh: np.ndarray) -> np.ndarray:
        """Encode H&E observations.

        Parameters
        ----------
        xh : ndarray of shape (n, d_h)

        Returns
        -------
        ndarray of shape (n, 2 * latent_dim)
            First latent_dim columns: shared z_c^h.
            Last latent_dim columns: H&E-specific z_s^h.
        """
        return self._encode(xh, self._enc_h_c, self._enc_h_s)

    def encode_r(self, xr: np.ndarray) -> np.ndarray:
        """Encode RNA observations.

        Parameters
        ----------
        xr : ndarray of shape (n, d_r)

        Returns
        -------
        ndarray of shape (n, 2 * latent_dim)
            First latent_dim columns: shared z_c^r.
            Last latent_dim columns: RNA-specific z_s^r.
        """
        return self._encode(xr, self._enc_r_c, self._enc_r_s)

    def get_shared(self) -> None:
        """Not applicable — sharedness is expressed via binary gamma in pipeline."""
        return None

    def get_private(self) -> None:
        """Not applicable — sharedness is expressed via binary gamma in pipeline."""
        return None

    def _check_fitted(self) -> None:
        if self._enc_h_c is None:
            raise RuntimeError(
                "DisentangledSSL has not been fitted. Call fit() first."
            )
