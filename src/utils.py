from __future__ import annotations
from typing import Any, List
import jax
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing import Tuple, Literal, Optional

# todo: enforce no nan values in y vector
# todo: enforce no inf values in y vector
# todo: clip num_slices to {1, ..., len(y)}
# todo: flake8 & sphinx tools

class SlicerConfig(BaseModel):
    """
    A class to validate the inputs to the slicer function

    Parameters
    ----------
    y : Any
        The response variable – anything that ``jnp.asarray`` can turn into a JAX array
        (list, NumPy array, JAX array, etc.).
    num_slices : int
        Desired number of slices (must be at least 1 and no more than len(y)).
    """

    # ------------------------------------------------------------------
    # Pydantic settings – we want strict type checking and forbid extra fields
    # ------------------------------------------------------------------
    model_config = ConfigDict(strict=True, extra="forbid")

    y: Any = Field(..., description="Response variable data")
    num_slices: int = Field(
        10,
        ge=1,
        description="Number of slices (must be a positive integer)",
    )

    # ------------------------------------------------------------------
    # Coerce ``y`` to a JAX array (executed before any other validation)
    # ------------------------------------------------------------------
    @field_validator("y", mode="before")
    @classmethod
    def _coerce_to_jax_array(cls, v: Any) -> jnp.ndarray:
        """Convert the incoming value to a JAX array."""
        try:
            return jnp.asarray(v, dtype=v.dtype)
        except Exception as exc:
            raise TypeError(f"Unable to convert {v} to a JAX array: {exc}") from exc

    # ------------------------------------------------------------------
    # Validate the relationship between ``num_slices`` and the length of ``y``.
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _check_slice_feasibility(self) -> "SlicerConfig":
        """Ensure we can actually create the requested number of slices."""
        n_obs = self.y.size
        if self.num_slices > n_obs:
            raise ValueError(
                f"`num_slices` ({self.num_slices}) cannot exceed the number of observations "
                f"({n_obs})."
            )
        return self

def slicer(y: Any, num_slices: int = 10) -> List[jnp.ndarray]:
    """
    Split ``y`` into ``num_slices`` contiguous bins based on the sorted order
    of the values.

    Parameters
    ----------
    y : Any
        Input data (list, NumPy array, JAX array, …). It will be flattened.
    num_slices : int, default 10
        Desired number of slices. Must satisfy be at least 1 and no more than len(y).

    Returns
    -------
    List[jnp.ndarray]
        A list of JAX arrays, each containing the values belonging to one slice.
        The slices are ordered from smallest to largest values.

    Raises
    ------
    pydantic.ValidationError / ValueError
        If ``y`` or ``num_slices`` cannot be converted to a JAX array or if ``num_slices`` is out of
        bounds.
    """
    # --------------------------------------------------------------
    # Validate + coerce inputs with Pydantic
    # --------------------------------------------------------------
    cfg = SlicerConfig(y=y, num_slices=num_slices)

    # --------------------------------------------------------------
    # Core JAX work – now we know the inputs are sane
    # --------------------------------------------------------------
    flat_y = jnp.ravel(cfg.y)          # fast flatten, no copy if possible
    n = flat_y.shape[0]

    # Sort once
    sorted_y = jnp.sort(flat_y)

    # Compute the size of each slice (first `remainder` slices get +1 element)
    base = flat_y.size // cfg.num_slices
    rem = flat_y.size % cfg.num_slices
    sizes = jnp.full(cfg.num_slices, base)
    sizes = sizes.at[:rem].add(1)  # vectorised addition

    # `jnp.split` expects the *indices* where the array should be broken.
    # Those are the cumulative sums of the slice sizes, except the final one.
    split_points = jnp.cumsum(sizes)[:-1]  # shape (num_slices‑1,)
    slices = jnp.split(sorted_y, split_points)  # returns a list of JAX arrays
    return slices

#####################

Array = jnp.ndarray
def _to_array(x: Array | jnp.typing.ArrayLike) -> Array:
    """Convert input to a JAX array."""
    return jnp.asarray(x)


def _ensure_2d(y: Array) -> Array:
    """Guarantee a column‑vector shape (n, 1)."""
    return y[:, None] if y.ndim == 1 else y


def _standardise(x: Array) -> Tuple[Array, Array, Array]:
    """Zero‑mean, unit‑variance scaling."""
    mean = jnp.mean(x, axis=0, keepdims=True)
    std = jnp.std(x, axis=0, keepdims=True)
    # Avoid division by zero for constant columns
    std = jnp.where(std == 0, 1.0, std)
    return (x - mean) / std, mean.squeeze(), std.squeeze()


def _slice_continuous(y: Array, n_slices: int) -> Tuple[Array, int]:
    """
    Discretise a continuous response into `n_slices` quantile bins.
    Returns the slice label for each observation and the effective number of slices.
    """
    # Compute quantile edges (excluding the last edge which is +inf)
    quantiles = jnp.linspace(0.0, 1.0, n_slices + 1)[1:-1]
    edges = jnp.quantile(y, quantiles, axis=0)
    # Assign each observation to a slice (0 … n_slices‑1)
    slice_ids = jnp.sum(y > edges, axis=1)
    return slice_ids, n_slices


def _slice_categorical(y: Array) -> Tuple[Array, int]:
    """Map each distinct category to an integer label."""
    uniq, inv = jnp.unique(y, return_inverse=True)
    return inv, uniq.shape[0]


def _slice_means(
        X: Array, slice_ids: Array, n_slices: int
) -> Tuple[Array, Array]:
    """
    Compute, for each slice, the mean of X and the proportion of samples.
    Returns:
        slice_means  – (n_slices, n_features)
        probs        – (n_slices,)
    """
    n, p = X.shape

    # One‑hot encoding of slice membership
    one_hot = jax.nn.one_hot(slice_ids, n_slices)          # (n, n_slices)

    # Slice probabilities = fraction of points per slice
    probs = jnp.mean(one_hot, axis=0)                      # (n_slices,)

    # Weighted sum of X within each slice
    #   (n, p)^T @ (n, n_slices) -> (p, n_slices)
    slice_sum = X.T @ one_hot                               # (p, n_slices)

    # Avoid division by zero for empty slices
    probs_safe = jnp.where(probs == 0, 1.0, probs)

    slice_means = (slice_sum / (probs_safe * n)).T           # (n_slices, p)

    return slice_means, probs


def _covariance(X: Array) -> Array:
    """Sample covariance matrix (unbiased estimator)."""
    n = X.shape[0]
    # Centered X is already passed in scaled form, but we recompute the mean for safety
    Xc = X - jnp.mean(X, axis=0, keepdims=True)
    cov = (Xc.T @ Xc) / (n - 1)
    return cov


