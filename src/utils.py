import jax
import jax.numpy as jnp
import jax.random as jrand
import pydantic
from pydantic import validate_call

######################################################
@validate_call(config={'strict': True}, validate_return=True)
def slicer(y, n_slices = 10):
    """
    Slice the range of a univariate response variable
    :param y: the response variable data
    :param n_slices: the number of slices
    :return y_discretised: a list containing the ordered slices
    """
    y = jnp.reshape(y, y.shape[0])
    y_sorted = jnp.sort(y) # sort y ascending
    n_obs_in_slice = int(jnp.floor(y.shape[0] / n_slices))
    y_discretised = []
    start_idx = 0
    end_idx = n_obs_in_slice
    num_so_far = 0

    for slice_num in range(n_slices):
        if slice_num < n_slices - 1:
            y_batch = y_sorted[start_idx:end_idx]
            num_so_far = num_so_far + n_obs_in_slice

            start_idx = num_so_far
            end_idx = num_so_far + n_obs_in_slice
        else:
            y_batch = y_sorted[start_idx:]
        y_discretised.append(y_batch)

    return y_discretised