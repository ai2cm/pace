import logging
from functools import wraps
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import gt4py.storage as gt_storage
import numpy as np

from pace.dsl.typing import DTypes, Field, Float, FloatField


try:
    import cupy as cp
except ImportError:
    cp = None

# If True, automatically transfers memory between CPU and GPU (see gt4py.storage)
managed_memory = True

# Number of halo lines for each field and default origin
halo = 3
origin = (halo, halo, 0)

# TODO get from field_table
tracer_variables = [
    "qvapor",
    "qliquid",
    "qrain",
    "qice",
    "qsnow",
    "qgraupel",
    "qo3mr",
    "qsgs_tke",
    "qcld",
]

# Logger instance
logger = logging.getLogger("fv3core")


def mark_untested(msg="This is not tested"):
    def inner(func) -> Callable[..., Any]:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            print(f"{func.__name__}: {msg}")
            func(*args, **kwargs)

        return wrapper

    return inner


def make_storage_data(
    data: Field,
    shape: Optional[Tuple[int, ...]] = None,
    origin: Tuple[int, int, int] = origin,
    *,
    backend: str,
    dtype: DTypes = np.float64,
    mask: Optional[Tuple[bool, bool, bool]] = None,
    start: Tuple[int, int, int] = (0, 0, 0),
    dummy: Optional[Tuple[int, int, int]] = None,
    axis: int = 2,
    max_dim: int = 3,
    read_only: bool = True,
) -> Field:
    """Create a new gt4py storage from the given data.

    Args:
        data: Data array for new storage
        shape: Shape of the new storage. Number of indices should be equal
            to number of unmasked axes
        origin: Default origin for gt4py stencil calls
        dtype: Data type
        mask: Tuple indicating the axes used when initializing the storage.
            True indicates a masked axis, False is a used axis.
        start: Starting points for slices in data copies
        dummy: Dummy axes
        axis: Axis for 2D to 3D arrays
        backend: gt4py backend to use

    Returns:
        Field[..., dtype]: New storage

    Examples:
        1) ptop = utils.make_storage_data(top_p, q4_1.shape)
        2) ws3 = utils.make_storage_data(ws3[:, :, -1], shape, origin=(0, 0, 0))
        3) data_dict[names[i]] = make_storage_data(
               data[:, :, :, i],
               shape,
               origin=origin,
               start=start,
               dummy=dummy,
               axis=axis,
           )

    """
    # NOTE (jdahm): Temporary until Jenkins is updated
    backend = backend.replace("gtc:", "")
    n_dims = len(data.shape)
    if shape is None:
        shape = data.shape

    if not mask:
        if not read_only:
            mask = (True, True, True)
        else:
            if n_dims == 1:
                if axis == 1:
                    # Convert J-fields to IJ-fields
                    mask = (True, True, False)
                    shape = (1, shape[axis])
                else:
                    mask = tuple([i == axis for i in range(max_dim)])
            elif dummy or axis != 2:
                mask = (True, True, True)
            else:
                mask = (n_dims * (True,)) + ((max_dim - n_dims) * (False,))

    if n_dims == 1:
        data = _make_storage_data_1d(
            data, shape, start, dummy, axis, read_only, backend=backend
        )
    elif n_dims == 2:
        data = _make_storage_data_2d(
            data, shape, start, dummy, axis, read_only, backend=backend
        )
    else:
        data = _make_storage_data_3d(data, shape, start, backend=backend)

    storage = gt_storage.from_array(
        data=data,
        backend=backend,
        default_origin=origin,
        shape=shape,
        dtype=dtype,
        mask=mask,
        managed_memory=managed_memory,
    )
    return storage


def _make_storage_data_1d(
    data: Field,
    shape: Tuple[int, int, int],
    start: Tuple[int, int, int] = (0, 0, 0),
    dummy: Optional[Tuple[int, int, int]] = None,
    axis: int = 2,
    read_only: bool = True,
    *,
    backend: str,
) -> Field:
    # axis refers to a repeated axis, dummy refers to a singleton axis
    axis = min(axis, len(shape) - 1)
    buffer = zeros(shape[axis], backend=backend)
    if dummy:
        axis = list(set((0, 1, 2)).difference(dummy))[0]

    kstart = start[2]
    buffer[kstart : kstart + len(data)] = asarray(data, type(buffer))

    if not read_only:
        tile_spec = list(shape)
        tile_spec[axis] = 1
        if axis == 2:
            buffer = tile(buffer, tuple(tile_spec))
        elif axis == 1:
            x = repeat(buffer[np.newaxis, :], shape[0], axis=0)
            buffer = repeat(x[:, :, np.newaxis], shape[2], axis=2)
        else:
            y = repeat(buffer[:, np.newaxis], shape[1], axis=1)
            buffer = repeat(y[:, :, np.newaxis], shape[2], axis=2)
    elif axis == 1:
        buffer = buffer.reshape((1, buffer.shape[0]))

    return buffer


def _make_storage_data_2d(
    data: Field,
    shape: Tuple[int, int, int],
    start: Tuple[int, int, int] = (0, 0, 0),
    dummy: Optional[Tuple[int, int, int]] = None,
    axis: int = 2,
    read_only: bool = True,
    *,
    backend: str,
) -> Field:
    # axis refers to which axis should be repeated (when making a full 3d data),
    # dummy refers to a singleton axis
    do_reshape = dummy or axis != 2
    if do_reshape:
        d_axis = dummy[0] if dummy else axis
        shape2d = shape[:d_axis] + shape[d_axis + 1 :]
    else:
        shape2d = shape[0:2]

    start1, start2 = start[0:2]
    size1, size2 = data.shape
    buffer = zeros(shape2d, backend=backend)
    buffer[start1 : start1 + size1, start2 : start2 + size2] = asarray(
        data, type(buffer)
    )

    if not read_only:
        buffer = repeat(buffer[:, :, np.newaxis], shape[axis], axis=2)
        if axis != 2:
            buffer = moveaxis(buffer, 2, axis)
    elif do_reshape:
        buffer = buffer.reshape(shape)

    return buffer


def _make_storage_data_3d(
    data: Field,
    shape: Tuple[int, int, int],
    start: Tuple[int, int, int] = (0, 0, 0),
    *,
    backend: str,
) -> Field:
    istart, jstart, kstart = start
    isize, jsize, ksize = data.shape
    buffer = zeros(shape, backend=backend)
    buffer[
        istart : istart + isize,
        jstart : jstart + jsize,
        kstart : kstart + ksize,
    ] = asarray(data, type(buffer))
    return buffer


def make_storage_from_shape(
    shape: Tuple[int, int, int],
    origin: Tuple[int, int, int] = origin,
    *,
    backend: str,
    dtype: DTypes = np.float64,
    init: bool = False,
    mask: Optional[Tuple[bool, bool, bool]] = None,
) -> Field:
    """Create a new gt4py storage of a given shape. Do not memoize outputs.

    Args:
        shape: Shape of the new storage
        origin: Default origin for gt4py stencil calls
        dtype: Data type
        init: If True, initializes the storage to zero
        mask: Tuple indicating the axes used when initializing the storage
        backend: gt4py backend to use when making the storage

    Returns:
        Field[..., dtype]: New storage

    Examples:
        1) utmp = utils.make_storage_from_shape(ua.shape)
        2) qx = utils.make_storage_from_shape(
               qin.shape, origin=(grid().is_, grid().jsd, kstart)
           )
        3) q_out = utils.make_storage_from_shape(q_in.shape, origin, init=True)
    """
    if not mask:
        n_dims = len(shape)
        if n_dims == 1:
            mask = (False, False, True)  # Assume 1D is a k-field
        else:
            mask = (n_dims * (True,)) + ((3 - n_dims) * (False,))
    # NOTE (jdahm): Temporary until Jenkins is updated
    backend = backend.replace("gtc:", "")
    storage_func = gt_storage.zeros if init else gt_storage.empty
    storage = storage_func(
        backend=backend,
        default_origin=origin,
        shape=shape,
        dtype=dtype,
        mask=mask,
        managed_memory=managed_memory,
    )
    return storage


def make_storage_dict(
    data: Field,
    shape: Optional[Tuple[int, int, int]] = None,
    origin: Tuple[int, int, int] = origin,
    start: Tuple[int, int, int] = (0, 0, 0),
    dummy: Optional[Tuple[int, int, int]] = None,
    names: Optional[List[str]] = None,
    axis: int = 2,
    *,
    backend: str,
) -> Dict[str, "Field"]:
    assert names is not None, "for 4d variable storages, specify a list of names"
    if shape is None:
        shape = data.shape
    data_dict: Dict[str, Field] = dict()
    for i in range(data.shape[3]):
        data_dict[names[i]] = make_storage_data(
            squeeze(data[:, :, :, i]),
            shape,
            origin=origin,
            start=start,
            dummy=dummy,
            axis=axis,
            backend=backend,
        )
    return data_dict


# def k_slice_operation(key, value, ki, dictionary):
#     if isinstance(value, gt_storage.storage.Storage):
#         shape = value.shape
#         mask = dictionary[key].mask if key in dictionary else (True, True, True)
#         if len(shape) == 1:  # K-field
#             if mask[2]:
#                 shape = (1, 1, len(ki))
#                 dictionary[key] = make_storage_data(value[ki], shape, read_only=True)
#         elif len(shape) == 2:  # IK-field
#             if not mask[1]:
#                 dictionary[key] = make_storage_data(
#                     value[:, ki], (shape[0], 1, len(ki)), read_only=True
#                 )
#         else:  # IJK-field
#             dictionary[key] = make_storage_data(
#                 value[:, :, ki], (shape[0], shape[1], len(ki)), read_only=True
#             )
#     else:
#         dictionary[key] = value


def storage_dict(st_dict, names, shape, origin, *, backend: str):
    for name in names:
        st_dict[name] = make_storage_from_shape(
            shape, origin, init=True, backend=backend
        )


def get_kstarts(column_info, npz):
    compare = None
    kstarts = []
    for k in range(npz):
        column_vals = {}
        for q, v in column_info.items():
            if k < len(v):
                column_vals[q] = v[k]
        if column_vals != compare:
            kstarts.append(k)
            compare = column_vals
    for i in range(len(kstarts) - 1):
        kstarts[i] = (kstarts[i], kstarts[i + 1] - kstarts[i])
    kstarts[-1] = (kstarts[-1], npz - kstarts[-1])
    return kstarts


def k_split_run(func, data, k_indices, splitvars_values):
    for ki, nk in k_indices:
        splitvars = {}
        for name, value_array in splitvars_values.items():
            splitvars[name] = value_array[ki]
        data.update(splitvars)
        data["kstart"] = ki
        data["nk"] = nk
        logger.debug(
            "Running kstart: {}, num k:{}, variables:{}".format(ki, nk, splitvars)
        )
        func(**data)


def asarray(array, to_type=np.ndarray, dtype=None, order=None):
    if isinstance(array, gt_storage.storage.Storage):
        array = array.data
    if cp and (isinstance(array, list)):
        if to_type is np.ndarray:
            order = "F" if order is None else order
            return cp.asnumpy(array, order=order)
        else:
            return cp.asarray(array, dtype, order)
    elif isinstance(array, list):
        if to_type is np.ndarray:
            return np.asarray(array, dtype, order)
        else:
            return cp.asarray(array, dtype, order)
    if cp and (
        isinstance(array, memoryview)
        or isinstance(array.data, (cp.ndarray, cp.cuda.memory.MemoryPointer))
    ):
        if to_type is np.ndarray:
            order = "F" if order is None else order
            return cp.asnumpy(array, order=order)
        else:
            return cp.asarray(array, dtype, order)
    else:
        if to_type is np.ndarray:
            return np.asarray(array, dtype, order)
        else:
            return cp.asarray(array, dtype, order)


def is_gpu_backend(backend: str) -> bool:
    return backend.endswith("cuda") or backend.endswith("gpu")


def zeros(shape, dtype=Float, *, backend: str):
    storage_type = cp.ndarray if is_gpu_backend(backend) else np.ndarray
    xp = cp if cp and storage_type is cp.ndarray else np
    return xp.zeros(shape)


def sum(array, axis=None, dtype=Float, out=None, keepdims=False):
    if isinstance(array, gt_storage.storage.Storage):
        array = array.data
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.sum(array, axis, dtype, out, keepdims)


def repeat(array, repeats, axis=None):
    if isinstance(array, gt_storage.storage.Storage):
        array = array.data
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.repeat(array, repeats, axis)


def index(array, key):
    return asarray(array, type(key))[key]


def moveaxis(array, source: int, destination: int):
    if isinstance(array, gt_storage.storage.Storage):
        array = array.data
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.moveaxis(array, source, destination)


def tile(array, reps: Union[int, Tuple[int]]):
    if isinstance(array, gt_storage.storage.Storage):
        array = array.data
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.tile(array, reps)


def squeeze(array, axis: Union[int, Tuple[int]] = None):
    if isinstance(array, gt_storage.storage.Storage):
        array = array.data
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.squeeze(array, axis)


def reshape(array, new_shape: Tuple[int]):
    if array.shape != new_shape:
        old_dims = len(array.shape)
        new_dims = len(new_shape)
        if old_dims < new_dims:
            # Upcast using repeat...
            if old_dims == 2:  # IJ -> IJK
                return repeat(array[:, :, np.newaxis], new_shape[2], axis=2)
            else:  # K -> IJK
                arr_2d = repeat(array[:, np.newaxis], new_shape[1], axis=1)
                return repeat(arr_2d[:, :, np.newaxis], new_shape[2], axis=2)
        else:
            return array.reshape(new_shape)
    return array


def unique(
    array,
    return_index: bool = False,
    return_inverse: bool = False,
    return_counts: bool = False,
    axis: Union[int, Tuple[int]] = None,
):
    if isinstance(array, gt_storage.storage.Storage):
        array = array.data
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.unique(array, return_index, return_inverse, return_counts, axis)


def stack(tup, axis: int = 0, out=None):
    array_tup = []
    for array in tup:
        if isinstance(array, gt_storage.storage.Storage):
            array = array.data
        array_tup.append(array)
    xp = cp if cp and type(array_tup[0]) is cp.ndarray else np
    return xp.stack(array_tup, axis, out)


def device_sync(backend: str) -> None:
    if cp and is_gpu_backend(backend):
        cp.cuda.Device(0).synchronize()


def split_cartesian_into_storages(var: FloatField):
    """
    Provided a storage of dims [X_DIM, Y_DIM, CARTESIAN_DIM]
         or [X_INTERFACE_DIM, Y_INTERFACE_DIM, CARTESIAN_DIM]
    Split it into separate 2D storages for each cartesian
    dimension, and return these in a list.
    """
    var_data = []
    for cart in range(3):
        var_data.append(
            make_storage_data(
                asarray(var.data, type(var.data))[:, :, cart],
                var.data.shape[0:2],
                backend=var.backend,
            )
        )
    return var_data
