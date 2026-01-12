from __future__ import annotations
from typing import Any, List
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

class SlicerConfig(BaseModel):
    """
    A class to validate the inputs to slicer

    Parameters
    ----------
    y : Any
        The response variable – anything that ``jnp.asarray`` can turn into a JAX array
        (list, NumPy array, JAX array, etc.).
    n_slices : int
        Desired number of slices (must be ≥ 1 and ≤ len(y)).
    """

    # ------------------------------------------------------------------
    # Pydantic settings – we want strict type checking and forbid extra fields
    # ------------------------------------------------------------------
    model_config = ConfigDict(strict=True, extra="forbid")

    y: Any = Field(..., description="Response variable data")
    n_slices: int = Field(
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
            return jnp.asarray(v)
        except Exception as exc:
            raise TypeError(f"Unable to convert `y` to a JAX array: {exc}") from exc

    # ------------------------------------------------------------------
    # Validate the relationship between ``n_slices`` and the length of ``y``.
    # ------------------------------------------------------------------
    @model_validator(mode="after")
    def _check_slice_feasibility(self) -> "SlicerConfig":
        """Ensure we can actually create the requested number of slices."""
        n_obs = self.y.size
        if self.n_slices > n_obs:
            raise ValueError(
                f"`n_slices` ({self.n_slices}) cannot exceed the number of observations "
                f"({n_obs})."
            )
        return self


def _slice_bounds(num_items: int, n_slices: int) -> jnp.ndarray:
    """
    A helper function for slicer.
    Compute start‑ and end‑indices for each slice.

    The first ``remainder`` slices receive one extra element so that the total
    number of elements is preserved.
    """
    base_size = num_items // n_slices
    remainder = num_items % n_slices

    sizes = jnp.full(n_slices, base_size)
    sizes = sizes.at[:remainder].add(1)

    ends = jnp.cumsum(sizes)
    starts = jnp.concatenate([jnp.array([0]), ends[:-1]])
    return jnp.stack([starts, ends], axis=1)   # shape (n_slices, 2)

def slicer(y: Any, n_slices: int = 10) -> List[jnp.ndarray]:
    """
    Split ``y`` into ``n_slices`` contiguous bins based on the sorted order
    of the values.

    Parameters
    ----------
    y : Any
        Input data (list, NumPy array, JAX array, …). It will be flattened.
    n_slices : int, default 10
        Desired number of slices. Must satisfy be at least 1 and no more than len(y).

    Returns
    -------
    List[jnp.ndarray]
        A list of JAX arrays, each containing the values belonging to one slice.
        The slices are ordered from smallest to largest values.

    Raises
    ------
    pydantic.ValidationError / ValueError
        If ``y`` cannot be converted to a JAX array or if ``n_slices`` is out of
        bounds.
    """
    # --------------------------------------------------------------
    # Validate + coerce inputs with Pydantic
    # --------------------------------------------------------------
    cfg = SlicerConfig(y=y, n_slices=n_slices)

    # --------------------------------------------------------------
    # Core JAX work – now we know the inputs are sane
    # --------------------------------------------------------------
    flat_y = jnp.ravel(cfg.y)          # fast flatten, no copy if possible
    n = flat_y.shape[0]

    # Sort once
    sorted_y = jnp.sort(flat_y)

    # Compute slice boundaries
    bounds = _slice_bounds(n, cfg.n_slices)

    # Gather slices
    slices = [sorted_y[int(start):int(end)] for start, end in bounds]

    return slices