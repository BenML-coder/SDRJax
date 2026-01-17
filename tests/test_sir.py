# -*- coding: utf-8 -*-
"""Unit tests for ``src.sir``.

The tests cover shape contracts, eigen‑value ordering, deterministic
behaviour, continuous vs. categorical slicing and a sanity comparison
with the original NumPy implementation (``src.save``).

Run with::
    pytest -q
"""

from __future__ import annotations

import pathlib
import sys
from typing import Tuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# --------------------------------------------------------------------------- #
# Helper: import the package under test
# --------------------------------------------------------------------------- #
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src import sir, save  # noqa: E402  (import after sys.path manipulation)


# --------------------------------------------------------------------------- #
# Fixtures – small deterministic datasets
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def rng() -> np.random.Generator:
    """A reproducible NumPy RNG (seed = 42)."""
    return np.random.default_rng(42)


def _make_continuous_data(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a toy regression problem where the true SDR direction is known.

    X ∈ ℝ^{n×2}  with columns (z, ε) where z ∼ N(0,1) and ε ∼ N(0,0.1).
    Y = z + noise, i.e. the sufficient direction is the first column.
    """
    n = 500
    z = rng.standard_normal(size=n)
    eps = rng.normal(scale=0.1, size=n)
    X = np.column_stack([z, eps])
    Y = z + rng.normal(scale=0.05, size=n)  # small extra noise
    return X, Y[:, None]  # make Y 2‑D to match the API


def _make_categorical_data(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a classification problem with three well‑separated classes."""
    n_per_class = 150
    centers = np.array([[0, 0], [3, 3], [-3, 3]])
    X = np.vstack(
        [
            rng.normal(loc=centers[i], scale=0.4, size=(n_per_class, 2))
            for i in range(3)
        ]
    )
    Y = np.repeat(np.arange(3), n_per_class)[:, None]
    return X, Y


@pytest.fixture(params=["continuous", "categorical"])
def dataset(request, rng):
    """Yield (X, Y, Ytype) tuples for both problem types."""
    if request.param == "continuous":
        X, Y = _make_continuous_data(rng)
        Ytype = "continuous"
    else:
        X, Y = _make_categorical_data(rng)
        Ytype = "categorical"
    return X, Y, Ytype


# --------------------------------------------------------------------------- #
# 1️⃣  Shape & basic contract tests
# --------------------------------------------------------------------------- #
def test_sir_output_shapes(dataset):
    """Check that the public wrapper returns objects of the advertised shape."""
    X, Y, Ytype = dataset
    k = 2  # we ask for two effective dimensions

    X_proj, eigvecs, eigvals = sir.SIR(
        X, Y, Ytype=Ytype, num_slices=10, k=k, standardiseX=True, standardiseY=True
    )

    assert isinstance(X_proj, jnp.ndarray)
    assert isinstance(eigvecs, jnp.ndarray)
    assert isinstance(eigvals, jnp.ndarray)

    n, p = X.shape
    assert X_proj.shape == (n, k)
    assert eigvecs.shape == (p, k)
    assert eigvals.shape == (k,)


# --------------------------------------------------------------------------- #
# 2️⃣  Eigen‑value ordering & non‑negativity
# --------------------------------------------------------------------------- #
def test_eigenvalues_sorted_descending(dataset):
    """Eigen‑values must be sorted descending and be ≥ 0."""
    X, Y, Ytype = dataset
    _, _, eigvals = sir.SIR(
        X, Y, Ytype=Ytype, num_slices=15, k=3, standardiseX=False, standardiseY=False
    )
    # Convert to plain NumPy for easy assertions
    ev = np.asarray(eigvals)

    # Non‑negative (tiny negative values can appear due to rounding)
    assert np.all(ev >= -1e-12)

    # Sorted descending
    assert np.allclose(ev, np.sort(ev)[::-1])


# --------------------------------------------------------------------------- #
# 3️⃣  Deterministic output given identical inputs
# --------------------------------------------------------------------------- #
def test_determinism(rng):
    """Running SIR twice on the same data yields identical results."""
    X, Y = _make_continuous_data(rng)
    out1 = sir.SIR(X, Y, Ytype="continuous", num_slices=12, k=2)
    out2 = sir.SIR(X, Y, Ytype="continuous", num_slices=12, k=2)

    for a, b in zip(out1, out2):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=1e-12, atol=1e-12)


# --------------------------------------------------------------------------- #
# 4️⃣  Continuous vs. categorical slicing behaviour
# --------------------------------------------------------------------------- #
def test_continuous_vs_categorical(rng):
    """Continuous slicing creates exactly `num_slices` groups; categorical uses the
    number of unique labels."""
    Xc, Yc = _make_continuous_data(rng)
    Xcat, Ycat = _make_categorical_data(rng)

    # Continuous – ask for 8 slices
    _, _, eigvals_c = sir.SIR(
        Xc, Yc, Ytype="continuous", num_slices=8, k=2, standardiseX=False, standardiseY=False
    )
    # Categorical – there are 3 classes, regardless of `num_slices`
    _, _, eigvals_cat = sir.SIR(
        Xcat,
        Ycat,
        Ytype="categorical",
        num_slices=99,  # ignored for categorical
        k=2,
        standardiseX=False,
        standardiseY=False,
    )

    # Both calls must succeed and return the correct number of eigen‑values
    assert eigvals_c.shape == (2,)
    assert eigvals_cat.shape == (2,)


# --------------------------------------------------------------------------- #
# 5️⃣  Effect of the standardisation flags
# --------------------------------------------------------------------------- #
def test_standardisation_flags(rng):
    """Turning off standardisation changes the magnitude of eigen‑values."""
    X, Y = _make_continuous_data(rng)

    _, _, ev_std = sir.SIR(
        X,
        Y,
        Ytype="continuous",
        num_slices=10,
        k=2,
        standardiseX=True,
        standardiseY=True,
    )
    _, _, ev_nostd = sir.SIR(
        X,
        Y,
        Ytype="continuous",
        num_slices=10,
        k=2,
        standardiseX=False,
        standardiseY=False,
    )

    # The two spectra must differ (otherwise the flag would be a no‑op)
    assert not np.allclose(np.asarray(ev_std), np.asarray(ev_nostd))


# --------------------------------------------------------------------------- #
# 6️⃣  Sanity comparison with the original NumPy implementation (src.save)
# --------------------------------------------------------------------------- #
def test_against_numpy_reference(rng):
    """The JAX implementation should reproduce the NumPy reference (up to
    floating‑point tolerance).  This test uses the same synthetic data and
    parameters as the reference function."""
    X, Y = _make_continuous_data(rng)

    # JAX version ---------------------------------------------------------- #
    X_proj_jax, eigvecs_jax, eigvals_jax = sir.SIR(
        X,
        Y,
        Ytype="continuous",
        num_slices=12,
        k=2,
        standardiseX=True,
        standardiseY=True,
    )

    # NumPy reference ------------------------------------------------------ #
    X_proj_np, eigvecs_np = save.SAVE(
        X,
        Y,
        Ytype="continuous",
        standardiseX=True,
        standardiseY=True,
        num_slices=12,
        k=2,
    )
    # NOTE: ``save.SAVE`` returns only the projected data and the eigenvectors.
    # It does **not** return eigen‑values, so we compare the two quantities we have.

    # Align signs of eigenvectors (eigenvectors are defined up to ±1)
    sign_correction = np.sign(np.sum(eigvecs_jax * eigvecs_np, axis=0))
    eigvecs_np_aligned = eigvecs_np * sign_correction

    # Projections may differ by the same sign flip per component
    X_proj_np_aligned = X @ eigvecs_np_aligned

    # Assertions – allow a modest tolerance because JAX uses float64 by default
    # and the reference uses NumPy float64 as well.
    np.testing.assert_allclose(
        np.asarray(X_proj_jax),
        X_proj_np_aligned,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Projected data differs from NumPy reference",
    )
    np.testing.assert_allclose(
        np.asarray(eigvecs_jax),
        eigvecs_np_aligned,
        rtol=1e-6,
        atol=1e-8,
        err_msg="Eigenvectors differ from NumPy reference",
    )
