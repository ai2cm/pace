import sys
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser
import numpy as np
import gt4py as gt
from copy import deepcopy

# Transform a dictionary of numpy arrays into a dictionary of gt4py
# storages of shape (iie-iis+1, jje-jjs+1, kke-kks+1)
def numpy_dict_to_gt4py_dict(np_dict):

    shape = np_dict["qi"].shape
    gt4py_dict = {}

    for var in np_dict:

        data = np_dict[var]
        if isinstance(data, np.ndarray):
            ndim = data.ndim
        else:
            ndim = 0

        if (ndim > 0) and (ndim <= 3) and (data.size >= 2):

            reshaped_data = np.empty(shape)

            if ndim == 1:  # 1D array (i-dimension)
                reshaped_data[...] = data[:, np.newaxis, np.newaxis]
            elif ndim == 2:  # 2D array (i-dimension, j-dimension)
                reshaped_data[...] = data[:, :, np.newaxis]
            elif ndim == 3:  # 3D array (i-dimension, j-dimension, k-dimension)
                reshaped_data[...] = data[...]

            dtype = DTYPE_INT if var in INT_VARS else DTYPE_FLT
            gt4py_dict[var] = gt.storage.from_array(
                reshaped_data, BACKEND, DEFAULT_ORIGIN, dtype=dtype
            )

        else:  # Scalars

            gt4py_dict[var] = deepcopy(data)

    return gt4py_dict


# Cast a dictionary of gt4py storages into dictionary of numpy arrays
def view_gt4py_storage(gt4py_dict):

    np_dict = {}

    for var in gt4py_dict:

        data = gt4py_dict[var]

        # ~ if not isinstance(data, np.ndarray): data.synchronize()
        if BACKEND == "gtcuda":
            data.synchronize()

        np_dict[var] = data.view(np.ndarray)

    return np_dict

# Scale the input dataset for benchmarks
def scale_dataset(data, factor):

    divider = factor[0]
    multiplier = factor[1]

    do_divide = divider < 1.0

    scaled_data = {}

    for var in data:

        data_var = data[var]
        ndim = data_var.ndim

        if ndim == 3:

            if do_divide:
                data_var = data_var[: DTYPE_INT(len(data_var) * divider), :, :]

            scaled_data[var] = np.tile(data_var, (multiplier, 1, 1))

        elif ndim == 2:

            if do_divide:
                data_var = data_var[: DTYPE_INT(len(data_var) * divider), :]

            scaled_data[var] = np.tile(data_var, (multiplier, 1))

        elif ndim == 1:

            if do_divide:
                data_var = data_var[: DTYPE_INT(len(data_var) * divider)]

            scaled_data[var] = np.tile(data_var, multiplier)

        elif ndim == 0:

            if var == "iie":
                scaled_data[var] = DTYPE_INT(data[var] * multiplier * divider)
            else:
                scaled_data[var] = data[var]

    return scaled_data