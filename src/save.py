"""
JAX implementation of **Sliced Average Variance Estimation (SAVE)**.

References
----------
- Cook, R. D., & Weisberg, S. (1991). *Discussion of “Sliced Inverse
  Regression”*.
- Li, K.-C. (1991). *Sliced Inverse Regression for Dimension Reduction*.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from typing import Literal, Tuple
from utils import _to_array, _covariance, _ensure_2d, _standardise, _slice_categorical, _slice_continuous

# ----------------------------------------------------------------------
# Core SAVE implementation (JIT‑compiled)
# ----------------------------------------------------------------------
@jax.jit
def _save_core(
        X: jnp.ndarray,
        Y: jnp.ndarray,
        Ytype: Literal["continuous", "categorical"] = "continuous",
        standardiseX: bool = True,
        standardiseY: bool = True,
        num_slices: int = 20,
        k: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute SAVE and return the projected data, eigenvectors and eigenvalues.

    Parameters
    ----------
    X, Y : array‑like
        Predictor matrix (n × p) and response vector/matrix (n × 1 or n).
    Ytype : {"continuous", "categorical"}
        Determines whether ``Y`` is sliced by quantiles or by unique categories.
    standardiseX, standardiseY : bool
        Whether to centre/scale the inputs (standardiseY only for continuous Y).
    num_slices : int
        Desired number of slices for continuous responses.
    k : int
        Number of leading eigenvectors to retain.

    Returns
    -------
    X_save : jnp.ndarray, shape (n, k)
        Data projected onto the SAVE subspace.
    eigvecs : jnp.ndarray, shape (p, k)
        Eigenvectors (columns correspond to directions).
    eigvals : jnp.ndarray, shape (k,)
        Associated eigenvalues (descending order).
    """
    # --------------------------------------------------------------
    # Input conversion & optional standardisation
    # --------------------------------------------------------------
    X = _to_array(X)
    Y = _to_array(Y)
    Y = _ensure_2d(Y)

    if standardiseX:
        X, _, _ = _standardise(X)

    if Ytype == "continuous" and standardiseY:
        Y, _, _ = _standardise(Y)

    # --------------------------------------------------------------
    # 2️⃣  Slice the response
    # --------------------------------------------------------------
    if Ytype == "continuous":
        slice_ids, n_slices_eff = _slice_continuous(Y, num_slices)
    else:  # categorical
        slice_ids, n_slices_eff = _slice_categorical(Y)

    # --------------------------------------------------------------
    # Slice statistics (means, covariances, probabilities)
    # --------------------------------------------------------------
    # One‑hot encoding of slice membership (n × g)
    one_hot = jax.nn.one_hot(slice_ids, n_slices_eff)

    # Slice probabilities p_g
    probs = jnp.mean(one_hot, axis=0)                     # (g,)

    # Slice means μ_g  (g × p)
    slice_means = (one_hot.T @ X) / (probs * X.shape[0])  # broadcasting safe

    # Slice covariances Σ_g  (g × p × p)
    #   Compute centered X for each slice via broadcasting:
    X_centered = X[:, None, :] - slice_means[None, :, :]   # (n, g, p)
    #   Weight by one‑hot and average:
    weighted = one_hot[..., None] * X_centered            # (n, g, p)
    cov_slices = jnp.einsum('ngp,ngq->gpq', weighted, X_centered) / (
            (probs * X.shape[0])[:, None, None] + 1e-12
    )                                                    # (g, p, p)

    # --------------------------------------------------------------
    # Build the SAVE kernel matrix
    # --------------------------------------------------------------
    #   Σ̂ = Cov(X)   (p × p)
    sigma_x = _covariance(X)

    #   For each slice compute Δ_g = Σ_g - Σ̂
    delta = cov_slices - sigma_x[None, :, :]               # (g, p, p)

    #   SAVE matrix:  Σ̂^{-1} ( Σ_g - Σ̂ ) Σ̂^{-1} weighted by p_g
    #   Equivalent formulation from Cook & Weisberg (1991):
    #       M = Σ̂^{-1} ( Σ_g p_g Δ_g Δ_g^T ) Σ̂^{-1}
    #   We implement the symmetric version:
    weighted_delta = (probs[:, None, None] *
                      jnp.einsum('gij,gkl->gijkl', delta, delta))
    #   Collapse over slices:
    M = jnp.sum(weighted_delta, axis=0)                   # (p, p, p, p)

    #   Contract with Σ̂^{-1} twice (matrix‑matrix multiplication):
    sigma_x_inv = jnp.linalg.inv(sigma_x)
    save_matrix = sigma_x_inv @ M @ sigma_x_inv
    #   The result is symmetric; enforce symmetry numerically:
    save_matrix = 0.5 * (save_matrix + save_matrix.T)

    # --------------------------------------------------------------
    # Generalised eigen‑problem Σ̂^{-1} M v = λ v
    # --------------------------------------------------------------
    eigvals_all, eigvecs_all = jax.scipy.linalg.eigh(
        a=save_matrix, b=sigma_x, lower=False
    )  # eigenvalues sorted ascending

    # Keep the largest `k` eigenpairs
    eigvals = eigvals_all[-k:][::-1]          # descending
    eigvecs = eigvecs_all[:, -k:][:, ::-1]    # matching columns

    # --------------------------------------------------------------
    # Project the data
    # --------------------------------------------------------------
    X_save = X @ eigvecs

    return X_save, eigvecs, eigvals


# ----------------------------------------------------------------------
# Public wrapper
# ----------------------------------------------------------------------
def SAVE(
        X,
        Y,
        Ytype: Literal["continuous", "categorical"] = "continuous",
        standardiseX: bool = True,
        standardiseY: bool = True,
        num_slices: int = 20,
        k: int = 4,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    User‑facing function that delegates to the JIT‑compiled core.

    Returns
    -------
    X_save : jnp.ndarray, shape (n_samples, k)
        Projected data.
    eigvecs : jnp.ndarray, shape (n_features, k)
        Projection matrix (eigenvectors).
    eigvals : jnp.ndarray, shape (k,)
        Corresponding eigenvalues.
    """
    return _save_core(
        X,
        Y,
        Ytype=Ytype,
        standardiseX=standardiseX,
        standardiseY=standardiseY,
        num_slices=num_slices,
        k=k,
    )
