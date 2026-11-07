"""CCA — Linear Entangled Representation Model.

Canonical Correlation Analysis baseline.  Learns paired linear projections
that maximize cross-modal correlation.

Uses SVD of the cross-covariance matrix rather than iterative PLS
(NIPALS), giving exact solutions in O(n d k) time without convergence
issues.

Representations are entangled: shared and modality-specific structure are
not explicitly separated.  As a result, get_private() returns None and
Delta_shared is not computed for this model by the evaluation pipeline.
"""

import numpy as np


class CCA:
    """Linear cross-modal alignment via Canonical Correlation Analysis.

    Learns projection matrices W_h in R^{d_h x K} and W_r in R^{d_r x K}
    that map each modality into a K-dimensional shared canonical space:

        z_h = (x_h - mean_h) @ W_h
        z_r = (x_r - mean_r) @ W_r

    where mean_h and mean_r are the column means computed on the training
    set (Cohort A).  Representations are entangled: CCA enforces
    cross-modal correlation but does not separate shared from
    modality-specific structure.

    Parameters
    ----------
    n_components : int
        Number of canonical components to retain (default 50).
    """

    def __init__(self, n_components: int = 50) -> None:
        self.n_components = n_components
        self._Wh: np.ndarray | None = None
        self._Wr: np.ndarray | None = None
        self._xh_mean: np.ndarray | None = None
        self._xr_mean: np.ndarray | None = None

    def fit(self, xh: np.ndarray, xr: np.ndarray) -> "CCA":
        """Fit CCA on paired modality data from Cohort A.

        Uses SVD of the cross-covariance matrix C = Xh^T Xr / (n-1)
        to obtain canonical directions. The top-k left and right
        singular vectors are the canonical weight matrices.

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
        max_components = min(n, xh.shape[1], xr.shape[1])
        if self.n_components > max_components:
            raise ValueError(
                f"n_components ({self.n_components}) exceeds the maximum feasible "
                f"value ({max_components}) for inputs of shape "
                f"{xh.shape} and {xr.shape}."
            )

        self._xh_mean = xh.mean(axis=0)
        self._xr_mean = xr.mean(axis=0)

        xh_c = xh - self._xh_mean
        xr_c = xr - self._xr_mean

        C = (xh_c.T @ xr_c) / (n - 1)
        U, _s, Vt = np.linalg.svd(C, full_matrices=False)

        self._Wh = U[:, :self.n_components]
        self._Wr = Vt[:self.n_components, :].T

        return self

    def encode_h(self, xh: np.ndarray) -> np.ndarray:
        """Project H&E observations into the canonical space.

        Parameters
        ----------
        xh : ndarray of shape (n, d_h)

        Returns
        -------
        ndarray of shape (n, n_components)
        """
        self._check_fitted()
        return (xh - self._xh_mean) @ self._Wh

    def encode_r(self, xr: np.ndarray) -> np.ndarray:
        """Project RNA observations into the canonical space.

        Parameters
        ----------
        xr : ndarray of shape (n, d_r)

        Returns
        -------
        ndarray of shape (n, n_components)
        """
        self._check_fitted()
        return (xr - self._xr_mean) @ self._Wr

    def get_shared(self) -> None:
        """Not applicable for entangled models.

        encode_h and encode_r already return shared-aligned representations.
        Returns None to signal that no further shared/private decomposition
        is available.
        """
        return None

    def get_private(self) -> None:
        """Not applicable for entangled models.

        A return value of None signals to the evaluation pipeline that this
        model does not expose a modality-specific component and Delta_shared
        should not be computed.
        """
        return None

    def _check_fitted(self) -> None:
        if self._Wh is None:
            raise RuntimeError("CCA has not been fitted. Call fit() first.")
