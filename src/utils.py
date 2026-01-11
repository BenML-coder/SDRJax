import jax
import jax.numpy as jnp
from typing import List

def _slice_bounds(num_items: int, n_slices: int) -> jnp.ndarray:
    """
    A helper function for slicer
    Return the start indices for each slice.
    The last slice will absorb any remainder.
    """
    # Size of a regular slice (floor division)
    base_size = num_items // n_slices
    # Number of slices that will receive one extra element
    remainder = num_items % n_slices

    # Build an array of slice sizes: first `remainder` slices get +1 element
    sizes = jnp.full(n_slices, base_size)
    sizes = sizes.at[:remainder].add(1)

    # Cumulative sum gives the end index of each slice; prepend 0 for starts
    ends = jnp.cumsum(sizes)
    starts = jnp.concatenate([jnp.array([0]), ends[:-1]])
    return jnp.stack([starts, ends], axis=1)   # shape (n_slices, 2)


def slicer(y: jnp.ndarray, n_slices: int = 10) -> List[jnp.ndarray]:
    """
    Split a 1‑D array ``y`` into ``n_slices`` contiguous bins based on the
    sorted order of the values.

    Parameters
    ----------
    y : jnp.ndarray
        Input array (any shape; it will be flattened).
    n_slices : int, optional
        Desired number of slices (default = 10). Must be >= 1 and <= len(y).

    Returns
    -------
    List[jnp.ndarray]
        A list of JAX arrays, each containing the values belonging to one slice.
        The slices are ordered from smallest to largest values.
    """
    if n_slices < 1:
        raise ValueError("n_slices must be a positive integer.")
    flat_y = jnp.ravel(y)                     # fast flatten, no copy if possible
    n = flat_y.shape[0]

    if n_slices > n:
        raise ValueError("n_slices cannot exceed the number of observations.")

    # Sort once
    sorted_y = jnp.sort(flat_y)

    # Compute slice boundaries (start, end) for each bin
    bounds = _slice_bounds(n, n_slices)       # shape (n_slices, 2)

    # Gather slices without a Python loop
    #    Using `vmap` to apply the same slice operation across all rows of `bounds`.
    def _gather_one(bound):
        start, end = bound
        return sorted_y[start:end]

    slices = jnp.vstack(jnp.arange(n_slices))  # dummy to trigger vmap shape inference
    # vmap expects a static shape for the output; we therefore map manually:
    sliced_list = [_gather_one(b) for b in bounds]   # still Python loop but cheap

    return sliced_list