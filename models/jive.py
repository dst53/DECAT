"""JIVE — Joint and Individual Variation Explained (linear factorized model).

Implements a simplified JIVE (Lock et al., 2013) using CCA for the joint
subspace and residual PCA for the individual (modality-specific) subspaces.

Architecture
------------
Step 1 — Joint subspace (shared):
    Fit CCA on (x_h, x_r) to obtain k canonical directions W_h and W_r.
    The joint scores are the canonical variates: Z_joint = x_h @ W_h = x_r @ W_r.

Step 2 — Individual subspaces (modality-specific):
    Project out the joint subspace from each modality:
        x_h_residual = x_h - x_h @ W_h @ W_h^T
        x_r_residual = x_r - x_r @ W_r @ W_r^T
    Fit PCA on each residual to obtain k individual components per modality.

Representation
--------------
encode_h returns [Z_joint | Z_h_individual] of shape (n, 2k):
    - First k dimensions: shared canonical variates (gamma = 1)
    - Last k dimensions: H&E-specific residual PCA scores (gamma = 0)
encode_r returns [Z_joint | Z_r_individual] of shape (n, 2k).

The sharedness vector gamma = [1, ..., 1, 0, ..., 0] (k ones, k zeros)
is stored in gamma.npy and triggers hard soft-masking in the evaluation
pipeline, cleanly separating shared from specific without soft thresholding.

Relation to CCA
---------------
The joint component is identical to the CCA entangled model, making JIVE
directly comparable to CCA: the only difference is the addition of the
individual component. This isolates the effect of factorization.

Reference: Lock et al. (2013), "Joint and Individual Variation Explained".
"""

import numpy as np


class JIVE:
    """Linear factorized representation model via CCA + residual PCA.

    Parameters
    ----------
    n_components : int
        Number of shared (canonical) components per modality.
    n_individual : int or None
        Number of individual (residual PCA) components per modality.
        If None, defaults to n_components.  Using a larger value (e.g.,
        3 * n_components) increases the probability of capturing
        outcome-relevant modality-specific variance that may be spread
        across many residual dimensions.  The output representation has
        (n_components + n_individual) dimensions per modality, with the
        first n_components dimensions shared (gamma=1) and the last
        n_individual dimensions specific (gamma=0).
    """

    def __init__(
        self,
        n_components: int = 50,
        n_individual: int | None = None,
    ) -> None:
        self.n_components = n_components
        self.n_individual = n_individual if n_individual is not None else n_components

        self._Wh_joint: np.ndarray | None = None
        self._Wr_joint: np.ndarray | None = None
        self._Vh_indiv: np.ndarray | None = None
        self._Vr_indiv: np.ndarray | None = None
        self._xh_mean: np.ndarray | None = None
        self._xr_mean: np.ndarray | None = None

    def fit(self, xh: np.ndarray, xr: np.ndarray) -> "JIVE":
        """Fit JIVE on paired modality data from Cohort A.

        Parameters
        ----------
        xh : ndarray of shape (n, d_h)
            H&E observations.
        xr : ndarray of shape (n, d_r)
            RNA observations.

        Returns
        -------
        self
        """
        n = xh.shape[0]
        k = self.n_components
        ki = self.n_individual
        max_k = min(n, xh.shape[1], xr.shape[1])
        if k > max_k:
            raise ValueError(
                f"n_components ({k}) exceeds maximum feasible value "
                f"({max_k}) for inputs of shape {xh.shape} and {xr.shape}."
            )
        if ki > max_k:
            raise ValueError(
                f"n_individual ({ki}) exceeds maximum feasible value ({max_k})."
            )

        self._xh_mean = xh.mean(axis=0)
        self._xr_mean = xr.mean(axis=0)
        xh_c = xh - self._xh_mean
        xr_c = xr - self._xr_mean

        # Step 1: Joint subspace via SVD of cross-covariance (same as CCA)
        C = (xh_c.T @ xr_c) / (n - 1)
        U, _s, Vt = np.linalg.svd(C, full_matrices=False)
        self._Wh_joint = U[:, :k]    # (d_h, k)
        self._Wr_joint = Vt[:k].T    # (d_r, k)

        # Step 2: Individual subspaces via residual PCA
        xh_res = xh_c - xh_c @ self._Wh_joint @ self._Wh_joint.T
        xr_res = xr_c - xr_c @ self._Wr_joint @ self._Wr_joint.T

        # PCA via SVD of residuals (take top n_individual components)
        Uh, _sh, _ = np.linalg.svd(xh_res, full_matrices=False)
        self._Vh_indiv = (xh_res.T @ Uh[:, :ki])
        self._Vh_indiv /= np.linalg.norm(self._Vh_indiv, axis=0, keepdims=True).clip(min=1e-12)

        Ur, _sr, _ = np.linalg.svd(xr_res, full_matrices=False)
        self._Vr_indiv = (xr_res.T @ Ur[:, :ki])
        self._Vr_indiv /= np.linalg.norm(self._Vr_indiv, axis=0, keepdims=True).clip(min=1e-12)

        return self

    def encode_h(self, xh: np.ndarray) -> np.ndarray:
        """Encode H&E observations.

        Parameters
        ----------
        xh : ndarray of shape (n, d_h)

        Returns
        -------
        ndarray of shape (n, 2 * n_components)
            First n_components columns: joint (shared) canonical scores.
            Last n_components columns: H&E individual (specific) PCA scores.
        """
        self._check_fitted()
        xh_c = xh - self._xh_mean
        z_joint = xh_c @ self._Wh_joint
        z_indiv = xh_c @ self._Vh_indiv
        return np.concatenate([z_joint, z_indiv], axis=1)

    def encode_r(self, xr: np.ndarray) -> np.ndarray:
        """Encode RNA observations.

        Parameters
        ----------
        xr : ndarray of shape (n, d_r)

        Returns
        -------
        ndarray of shape (n, 2 * n_components)
            First n_components columns: joint (shared) canonical scores.
            Last n_components columns: RNA individual (specific) PCA scores.
        """
        self._check_fitted()
        xr_c = xr - self._xr_mean
        z_joint = xr_c @ self._Wr_joint
        z_indiv = xr_c @ self._Vr_indiv
        return np.concatenate([z_joint, z_indiv], axis=1)

    @property
    def loadings(self) -> dict[str, np.ndarray]:
        """Return loading matrices for sharedness score computation.

        Returns binary sharedness: joint dimensions = 1 (shared),
        individual dimensions = 0 (specific). This produces gamma =
        [1, ..., 1, 0, ..., 0] after compute_sharedness_scores().
        """
        self._check_fitted()
        ki = self.n_individual
        d_r = self._Wr_joint.shape[0]

        # Joint block (first k cols): equal loadings across both modalities → gamma ≈ 1
        # Individual block (last ki cols): zero in the other modality → gamma ≈ 0
        zeros_r = np.zeros((d_r, ki))

        W_h_full = np.concatenate([self._Wh_joint, self._Vh_indiv], axis=1)  # (d_h, k+ki)
        W_r_full = np.concatenate([self._Wr_joint, zeros_r], axis=1)         # (d_r, k+ki)

        return {"H&E": W_h_full, "RNA": W_r_full}

    def get_shared(self) -> dict[str, np.ndarray]:
        """Return loading matrices for the evaluation pipeline.

        Returns
        -------
        dict mapping modality name to loading matrix of shape (d_m, 2k).
        """
        return self.loadings

    def get_private(self) -> None:
        """Sharedness is expressed via the gamma vector from get_shared()."""
        return None

    def _check_fitted(self) -> None:
        if self._Wh_joint is None:
            raise RuntimeError("JIVE has not been fitted. Call fit() first.")
