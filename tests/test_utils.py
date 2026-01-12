#!/usr/bin/env python3

import pytest
import jax.random as jrand
import jax.numpy as jnp
from src.utils import slicer

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------
def test_basic_split_even():
    """Even split: 100 elements → 10 slices of size 10 each."""

    key = jrand.key(0)
    data = jrand.normal(key, 100)
    slices = slicer(data, num_slices=10)

    assert len(slices) == 10
    for s in slices:
        assert s.shape[0] == 10

    # Concatenating all slices should give the fully sorted array
    reconstructed = jnp.concatenate(slices)
    assert jnp.allclose(reconstructed, jnp.sort(data))

def test_split_with_remainder():
    """Uneven split: 103 elements → 10 slices, first 3 have size 11."""

    key = jrand.key(1)
    data = jrand.normal(key, 103)
    slices = slicer(data, num_slices=10)

    assert len(slices) == 10
    # First `remainder` (=3) slices get one extra element
    expected_sizes = [11, 11, 11] + [10] * 7
    actual_sizes = [s.shape[0] for s in slices]
    assert actual_sizes == expected_sizes

    # Verify ordering across slice boundaries
    for i in range(len(slices) - 1):
        assert slices[i].max() <= slices[i + 1].min()

    # Whole reconstruction matches sorted data
    assert jnp.allclose(jnp.concatenate(slices), jnp.sort(data))

def test_input_higher_dimensional():
    """Function should flatten arbitrary shapes before slicing."""
    # 2‑D array (20 × 5 = 100 elements)

    key = jrand.key(2)
    data = jrand.normal(key, (20, 5))
    slices = slicer(data, num_slices=5)

    assert len(slices) == 5
    # 100 / 5 = 20 per slice
    assert all(s.shape[0] == 20 for s in slices)

    # Flattened + sorted reconstruction check
    assert jnp.allclose(jnp.concatenate(slices), jnp.sort(data.ravel()))

def test_invalid_num_slices_zero_or_negative():
    """num_slices must be >= 1."""

    key = jrand.key(3)
    data = jrand.normal(key, 10)

    with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
        slicer(data, num_slices=0)

    with pytest.raises(ValueError, match="Input should be greater than or equal to 1"):
        slicer(data, num_slices=-5)


def test_num_slices_greater_than_observations():
    """Should raise a clear error when requesting more slices than data points."""

    key = jrand.key(4)
    data = jrand.normal(key, 7)

    with pytest.raises(ValueError, match="cannot exceed"):
        slicer(data, num_slices=10)

def test_consistency_across_multiple_calls():
    """Repeated calls with the same seed should be deterministic."""
    key = jrand.key(6)
    data = jrand.normal(key, 42)

    first = slicer(data, num_slices=6)
    second = slicer(data, num_slices=6)

    for a, b in zip(first, second):
        assert jnp.array_equal(a, b)


if __name__ == "__main__":
    pytest.main([__file__])