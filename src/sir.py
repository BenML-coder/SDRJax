import jax
import jax.numpy as jnp

# class SIR:
#     def __init__(self, num_slices:int, num_components:int):
#         pass

def SIR(X, Y, Ytype="continuous", standardiseX=True, standardiseY=True, num_slices=20, k=4):
    X = np.array(X, dtype="double")
    Y = np.array(Y)
    num_row = X.shape[0]
    num_col = X.shape[1]

    if len(Y.shape) == 1:
        Y = Y.reshape(-1, 1)

    if standardiseX:
        X_scaled, x_meanvec, x_stdvec = custom_standardise(X)
    else:
        X_scaled = X

    if Ytype == "continuous" and standardiseY:
        Y_scaled, y_meanvec, y_stdvec = custom_standardise(Y)
    else:
        Y_scaled = Y

    Z = X_scaled  # use generalized eigenvalue solver, so no need to standardize here

    if Ytype == "continuous":
        Y_discretized = slicer(Y_scaled, n_slices=num_slices)
    elif Ytype == "categorical":
        Y_discretized = np.unique(Y_scaled)
        num_slices = len(Y_discretized)  # number of slices if Y categorical is number of categories
    else:
        raise ValueError

    exy = np.zeros((num_slices, num_col))
    probabilities = np.zeros(num_slices)

    for slice_num in range(num_slices):
        idxs_in_slice = np.isin(Y_scaled, Y_discretized[slice_num])
        probabilities[slice_num] = np.mean(idxs_in_slice)
        Z_slice, slice_meanvec = custom_centre(Z[idxs_in_slice.squeeze(), :])
        exy[slice_num, :] = slice_meanvec

    sir_lambda = np.transpose(exy) @ np.diagflat(probabilities) @ exy
    sir_lambda = (sir_lambda + np.transpose(sir_lambda)) / 2
    sir_vals, sir_vecs = eigh(sir_lambda, b=np.cov(X_scaled, rowvar=False).real,
                              subset_by_index=[num_col - k, num_col - 1], driver="gvx")

    # handles complex outputs with negligible imaginary part
    sir_vals = sir_vals.real
    sir_vecs = sir_vecs.real

    ind = np.flip(np.arange(len(sir_vals)))
    sir_vals = sir_vals[ind]
    sir_vecs = sir_vecs[:, ind]
    sir_result = X_scaled @ sir_vecs
    return sir_result, sir_vecs