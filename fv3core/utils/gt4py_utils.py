import copy
import logging
import math
from typing import Dict, List, Optional, Tuple, Union

import gt4py as gt
import gt4py.storage as gt_storage
import numpy as np
from gt4py import gtscript

from fv3core.utils.mpi import MPI
from fv3core.utils.typing import DTypes, Field, float_type, int_type


try:
    import cupy as cp
except ImportError:
    cp = None

# Options: numpy, gtmc, gtx86, gtcuda, debug
backend = None

# Whether to rebuild stencils on every call
rebuild = False

# Set to "False" to skip validating gt4py stencil arguments
validate_args = True

# If True, automatically transfers memory between CPU and GPU (see gt4py.storage)
managed_memory = True

# [DEPRECATED] field types
sd = gtscript.Field[float_type]
si = gtscript.Field[int_type]

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
logger = logging.getLogger("fv3ser")


# 1 indexing to 0 and halos: -2, -1, 0 --> 0, 1,2
if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
    gt.config.cache_settings["dir_name"] = ".gt_cache_{:0>6d}".format(
        MPI.COMM_WORLD.Get_rank()
    )


# TODO remove when using quantities throughout model
def quantity_name(name):
    return name + "_quantity"


def make_storage_data(
    data: Field,
    shape: Optional[Tuple[int, int, int]] = None,
    *,
    origin: Tuple[int, int, int] = origin,
    dtype: DTypes = np.float64,
    mask: Tuple[bool, bool, bool] = (True, True, True),
    start: Tuple[int, int, int] = (0, 0, 0),
    dummy: Optional[Tuple[int, int, int]] = None,
    axis: int = 2,
) -> Field:
    """Create a new gt4py storage from the given data.

    Args:
        data: Data array for new storage
        shape: Shape of the new storage
        origin: Default origin for gt4py stencil calls
        dtype: Data type
        mask: Tuple indicating the axes used when initializing the storage
        start: Starting points for slices in data copies
        dummy: Dummy axes
        axis: Axis for 2D to 3D arrays

    Returns:
        Field[dtype]: New storage

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
    n_dims = len(data.shape)
    if shape is None:
        shape = data.shape

    if n_dims == 1:
        data = _make_storage_data_1d(data, shape, start, dummy, axis)
    elif n_dims == 2:
        data = _make_storage_data_2d(data, shape, start, dummy, axis)
    else:
        data = _make_storage_data_3d(data, shape, start)

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
) -> Field:
    # axis refers to a repeated axis, dummy refers to a singleton axis
    kstart = start[2]
    if dummy:
        axis = list(set((0, 1, 2)).difference(dummy))[0]
    buffer = zeros(shape[axis])
    buffer[kstart : kstart + len(data)] = asarray(data, type(buffer))
    tile_spec = list(shape)
    tile_spec[axis] = 1

    if dummy:
        if len(dummy) == len(tile_spec) - 1:
            data = buffer.reshape((shape))
    else:
        if axis == 2:
            data = tile(buffer, tuple(tile_spec))
        elif axis == 1:
            x = repeat(buffer[np.newaxis, :], shape[0], axis=0)
            data = repeat(x[:, :, np.newaxis], shape[2], axis=2)
        else:
            y = repeat(buffer[:, np.newaxis], shape[1], axis=1)
            data = repeat(y[:, :, np.newaxis], shape[2], axis=2)
    return data


def _make_storage_data_2d(
    data: Field,
    shape: Tuple[int, int, int],
    start: Tuple[int, int, int] = (0, 0, 0),
    dummy: Optional[Tuple[int, int, int]] = None,
    axis: int = 2,
) -> Field:
    # axis refers to which axis should be repeated (when making a full 3d data),
    # dummy refers to a singleton axis
    isize, jsize = data.shape
    istart, jstart = start[0:2]
    if dummy or axis != 2:
        d_axis = dummy[0] if dummy else axis
        shape2d = shape[:d_axis] + shape[d_axis + 1 :]
    else:
        shape2d = shape[0:2]
    buffer = zeros(shape2d)
    buffer[istart : istart + isize, jstart : jstart + jsize] = asarray(
        data, type(buffer)
    )
    if dummy:
        data = buffer.reshape(shape)
    else:
        data = repeat(buffer[:, :, np.newaxis], shape[axis], axis=2)
        if axis != 2:
            data = moveaxis(data, 2, axis)
    return data


def _make_storage_data_3d(
    data: Field,
    shape: Tuple[int, int, int],
    start: Tuple[int, int, int] = (0, 0, 0),
) -> Field:
    istart, jstart, kstart = start
    isize, jsize, ksize = data.shape
    buffer = zeros(shape)
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
    dtype: DTypes = np.float64,
    init: bool = True,
    mask: Tuple[bool, bool, bool] = (True, True, True),
) -> Field:
    """Create a new gt4py storage of a given shape.

    Args:
        shape: Shape of the new storage
        origin: Default origin for gt4py stencil calls
        dtype: Data type
        init: If True, initializes the storage to the default value for the type
        mask: Tuple indicating the axes used when initializing the storage

    Returns:
        Field[dtype]: New storage

    Examples:
        1) utmp = utils.make_storage_from_shape(ua.shape)
        2) qx = utils.make_storage_from_shape(
               qin.shape, origin=(grid().is_, grid().jsd, kstart)
           )
        3) q_out = utils.make_storage_from_shape(q_in.shape, origin, init=True)
    """
    storage = gt_storage.empty(
        backend=backend,
        default_origin=origin,
        shape=shape,
        dtype=dtype,
        mask=mask,
        managed_memory=managed_memory,
    )
    if init:
        storage[:] = dtype()
    return storage


def make_storage_dict(
    data: Field,
    shape: Optional[Tuple[int, int, int]] = None,
    origin: Tuple[int, int, int] = origin,
    start: Tuple[int, int, int] = (0, 0, 0),
    dummy: Optional[Tuple[int, int, int]] = None,
    names: Optional[List[str]] = None,
    axis: int = 2,
) -> Dict[str, type(Field)]:
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
        )
    return data_dict


def storage_dict(st_dict, names, shape, origin):
    for name in names:
        st_dict[name] = make_storage_from_shape(shape, origin)


def k_slice_operation(key, value, ki, dictionary):
    if isinstance(value, gt_storage.storage.Storage):
        dictionary[key] = make_storage_data(
            value[:, :, ki], (value.shape[0], value.shape[1], len(ki))
        )
    else:
        dictionary[key] = value


def k_slice_inplace(data_dict, ki):
    for k, v in data_dict.items():
        k_slice_operation(k, v, ki, data_dict)


def k_slice(data_dict, ki):
    new_dict = {}
    for k, v in data_dict.items():
        k_slice_operation(k, v, ki, new_dict)
    return new_dict


def k_subset_run(func, data, splitvars, ki, outputs, grid_data, grid, allz=False):
    grid.npz = len(ki)
    grid.slice_data_k(ki)
    d = k_slice(data, ki)
    d.update(splitvars)
    results = func(**d)
    collect_results(d, results, outputs, ki, allz)
    grid.add_data(grid_data)


def collect_results(data, results, outputs, ki, allz=False):
    outnames = list(outputs.keys())
    endz = None if allz else -1
    logger.debug("Computing results for k indices: {}".format(ki[:-1]))
    for k in outnames:
        if k in data:
            # passing fields with single item in 3rd dimension leads to errors
            outputs[k][:, :, ki[:endz]] = data[k][:, :, :endz]
    if results is not None:
        for ri in range(len(results)):
            outputs[outnames[ri]][:, :, ki[:endz]] = results[ri][:, :, :endz]


def k_split_run_dataslice(
    func, data, k_indices_array, splitvars_values, outputs, grid, allz=False
):
    num_k = grid.npz
    grid_data = copy.deepcopy(grid.data_fields)
    for ki in k_indices_array:
        splitvars = {}
        for name, value_array in splitvars_values.items():
            splitvars[name] = value_array[ki[0]]
        k_subset_run(func, data, splitvars, ki, outputs, grid_data, grid, allz)
    grid.npz = num_k


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


def kslice_from_inputs(kstart, nk, grid):
    if nk is None:
        nk = grid.npz - kstart
    kslice = slice(kstart, kstart + nk)
    return [kslice, nk]


def krange_from_slice(kslice, grid):
    kstart = kslice.start
    kend = kslice.stop
    nk = grid.npz - kstart if kend is None else kend - kstart
    return kstart, nk


def great_circle_dist(p1, p2, radius=None):
    beta = (
        math.asin(
            math.sqrt(
                math.sin((p1[1] - p2[1]) / 2.0) ** 2
                + math.cos(p1[1])
                * math.cos(p2[1])
                * math.sin((p1[0] - p2[0]) / 2.0) ** 2
            )
        )
        * 2.0
    )
    if radius is not None:
        great_circle_dist = radius * beta
    else:
        great_circle_dist = beta
    return great_circle_dist


def extrap_corner(p0, p1, p2, q1, q2):
    x1 = great_circle_dist(p1, p0)
    x2 = great_circle_dist(p2, p0)
    return q1 + x1 / (x2 - x1) * (q1 - q2)


def asarray(array, to_type=np.ndarray, dtype=None, order=None):
    if isinstance(array, gt_storage.storage.Storage):
        array = array.data
    if cp and (
        isinstance(array.data, cp.ndarray)
        or isinstance(array.data, cp.cuda.memory.MemoryPointer)
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


def zeros(shape, dtype=float_type):
    storage_type = cp.ndarray if "cuda" in backend else np.ndarray
    xp = cp if cp and storage_type is cp.ndarray else np
    return xp.zeros(shape)


def sum(array, axis=None, dtype=float_type, out=None, keepdims=False):
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
