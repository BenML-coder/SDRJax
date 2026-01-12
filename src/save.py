import jax
import jax.numpy as jnp

class SAVE:
    def __init__(self, num_slices:int, num_components:int):
        pass

def SAVE(X, Y, Ytype="continuous", standardiseX=True, standardiseY=True, num_slices=20, k=4):
    X = np.array(X,dtype="double")
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

    vxy = np.zeros((num_slices, num_col, num_col))  # one matrix per slice stored
    probabilities = np.zeros(num_slices)
    identity = np.eye(num_col)

    save_lambda = np.zeros((num_col, num_col))

    for slice_num in range(num_slices):
        idxs_in_slice = np.isin(Y_scaled, Y_discretized[slice_num])
        probabilities[slice_num] = np.mean(idxs_in_slice)
        vxy_slice = np.cov(Z[idxs_in_slice.squeeze(), :], rowvar=False).real
        save_lambda = save_lambda + probabilities[slice_num] * ((vxy_slice - identity) @ (vxy_slice - identity))

    save_lambda = (save_lambda + np.transpose(save_lambda)) / 2
    covx = np.cov(X_scaled, rowvar=False).real
    save_vals, save_vecs = eigh(covx - save_lambda, b=covx, subset_by_index=[num_col - k, num_col - 1], driver="gvx")
    save_vals = save_vals.real
    save_vecs = save_vecs.real
    ind = np.flip(np.arange(len(save_vals)))
    save_vals = save_vals[ind]
    save_vecs = save_vecs[:, ind]
    save_result = X_scaled @ save_vecs
    return save_result, save_vecs