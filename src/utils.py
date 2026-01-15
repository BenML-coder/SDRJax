from __future__ import annotations
from typing import Any, List
import jax
import jax.numpy as jnp
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

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