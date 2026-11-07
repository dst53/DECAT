"""CLIP — Contrastive Language-Image Pretraining baseline (multimodal adaptation).

Nonlinear entangled representation model trained with symmetric NT-Xent
contrastive loss on matched cross-modal patient pairs. Each modality is
encoded by an independent shallow MLP; representations are L2-normalized
and aligned by maximizing similarity of matched pairs.

Architecture boundary
---------------------
This module uses PyTorch for training. All post-training operations
(encode_h, encode_r) work on numpy arrays via torch inference.

Reference: Chen et al., "A Simple Framework for Contrastive Learning
of Visual Representations", ICML 2020 (NT-Xent loss formulation).
"""

from __future__ import annotations

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


class _MLP(nn.Module):
    """Two-layer MLP encoder with ReLU activation and L2-normalized output."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class CLIP:
    """Contrastive multimodal representation model.

    Parameters
    ----------
    latent_dim : int
        Output representation dimensionality K.
    hidden_multiplier : int
        Hidden layer width as a multiple of input_dim. Default 2.
    temperature : float
        NT-Xent softmax temperature τ. Default 0.07.
    lr : float
        Adam learning rate.
    n_epochs : int
        Training epochs.
    batch_size : int
        Minibatch size.
    seed : int
        Random seed for weight initialization and batch shuffling.
    device : str or None
        Torch device. If None, uses CUDA when available else CPU.
    """

    def __init__(
        self,
        latent_dim: int = 50,
        hidden_multiplier: int = 2,
        temperature: float = 0.07,
        lr: float = 1e-3,
        n_epochs: int = 100,
        batch_size: int = 256,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        self.latent_dim = latent_dim
        self.hidden_multiplier = hidden_multiplier
        self.temperature = temperature
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.seed = seed
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

        self._enc_h: _MLP | None = None
        self._enc_r: _MLP | None = None

    def fit(
        self,
        xh: np.ndarray,
        xr: np.ndarray,
        checkpoint_fn: callable | None = None,
        checkpoint_interval: int = 25,
    ) -> "CLIP":
        """Train contrastive encoders on paired modality data from Cohort A.

        Parameters
        ----------
        xh : ndarray of shape (n, d_h)
            H&E observations.
        xr : ndarray of shape (n, d_r)
            RNA observations.
        checkpoint_fn : callable or None
            If provided, called as ``checkpoint_fn(epoch, model, loss)`` every
            ``checkpoint_interval`` epochs (and at epoch 0 before any training).
            Intended for convergence analysis; no-op in normal use.
        checkpoint_interval : int
            Epoch interval between checkpoint_fn calls.

        Returns
        -------
        self
        """
        torch.manual_seed(self.seed)
        device = torch.device(self.device)

        d_h, d_r = xh.shape[1], xr.shape[1]
        h_h = self.hidden_multiplier * d_h
        h_r = self.hidden_multiplier * d_r

        self._enc_h = _MLP(d_h, h_h, self.latent_dim).to(device)
        self._enc_r = _MLP(d_r, h_r, self.latent_dim).to(device)

        optimizer = torch.optim.Adam(
            list(self._enc_h.parameters()) + list(self._enc_r.parameters()),
            lr=self.lr,
        )

        Xh = torch.tensor(xh, dtype=torch.float32, device=device)
        Xr = torch.tensor(xr, dtype=torch.float32, device=device)
        n = len(Xh)
        rng = torch.Generator(device='cpu')
        rng.manual_seed(self.seed)

        if checkpoint_fn is not None:
            self._enc_h.eval(); self._enc_r.eval()
            checkpoint_fn(0, self, float('nan'))
            self._enc_h.train(); self._enc_r.train()

        epoch_loss = float('nan')
        for epoch in range(self.n_epochs):
            idx = torch.randperm(n, generator=rng)
            batch_losses = []
            for start in range(0, n, self.batch_size):
                batch = idx[start : start + self.batch_size]
                zh = self._enc_h(Xh[batch])
                zr = self._enc_r(Xr[batch])

                sim = zh @ zr.T / self.temperature
                labels = torch.arange(len(zh), device=device)
                loss = (F.cross_entropy(sim, labels) +
                        F.cross_entropy(sim.T, labels)) / 2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                batch_losses.append(loss.item())

            epoch_loss = float(np.mean(batch_losses))

            if checkpoint_fn is not None and (epoch + 1) % checkpoint_interval == 0:
                self._enc_h.eval(); self._enc_r.eval()
                checkpoint_fn(epoch + 1, self, epoch_loss)
                self._enc_h.train(); self._enc_r.train()

        self._enc_h.eval()
        self._enc_r.eval()
        return self

    def encode_h(self, xh: np.ndarray) -> np.ndarray:
        """Encode H&E observations into the shared embedding space.

        Parameters
        ----------
        xh : ndarray of shape (n, d_h)

        Returns
        -------
        ndarray of shape (n, latent_dim)
        """
        self._check_fitted()
        with torch.no_grad():
            z = self._enc_h(
                torch.tensor(xh, dtype=torch.float32, device=self.device)
            )
        return z.cpu().numpy()

    def encode_r(self, xr: np.ndarray) -> np.ndarray:
        """Encode RNA observations into the shared embedding space.

        Parameters
        ----------
        xr : ndarray of shape (n, d_r)

        Returns
        -------
        ndarray of shape (n, latent_dim)
        """
        self._check_fitted()
        with torch.no_grad():
            z = self._enc_r(
                torch.tensor(xr, dtype=torch.float32, device=self.device)
            )
        return z.cpu().numpy()

    def get_shared(self) -> None:
        """Not applicable — CLIP produces entangled representations."""
        return None

    def get_private(self) -> None:
        """Not applicable — CLIP produces entangled representations."""
        return None

    def _check_fitted(self) -> None:
        if self._enc_h is None:
            raise RuntimeError("CLIP has not been fitted. Call fit() first.")
