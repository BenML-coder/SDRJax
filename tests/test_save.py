import numpy as np
import jax.numpy as jnp
import pytest

from src.save import SAVE

@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    # 200 samples, 10 features
    X = rng.normal(size=(200, 10))
    # Continuous response generated from a low‑dimensional linear combination
    beta = rng.normal(size=(10, 1))
    noise = rng.normal(scale=0.1, size=(200, 1))
    Y = X @ beta + noise
    return X, Y.squeeze()

def test_save_basic_projection(synthetic_data):
    X, Y = synthetic_data
    X_proj, eigvecs, eigvals = SAVE(X, Y, Ytype="continuous", k=3)

    # Shapes
    assert X_proj.shape == (200, 3)
    assert eigvecs.shape == (10, 3)
    assert eigvals.shape == (3,)

    # Eigenvalues should be non‑negative (SAVE matrix is PSD)
    assert jnp.all(eigvals >= 0)

def test_save_categorical():
    # Simple categorical example with two classes
    X = np.vstack([np.random.randn(50, 5) + 2,
                   np.random.randn(50, 5) - 2])
    Y = np.array([0]*50 + [1]*50)

    X_proj, eigvecs, eigvals = SAVE(X, Y, Ytype="categorical", k=2)

    assert X_proj.shape == (100, 2)
    assert eigvecs.shape == (5, 2)
    assert eigvals.shape == (2,)
    # With two well‑separated clusters, the leading eigenvalue should dominate
    assert eigvals[0] > eigvals[1]
