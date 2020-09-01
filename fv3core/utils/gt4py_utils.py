#!/usr/bin/env python3

import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript
import copy
import math
import logging
import functools
from fv3core.utils.mpi import MPI

try:
    import cupy as cp
except ImportError:
    cp = None

logger = logging.getLogger("fv3ser")
backend = None  # Options: numpy, gtmc, gtx86, gtcuda, debug, dawn:gtmc
rebuild = True
_dtype = np.float_
sd = gtscript.Field[_dtype]
si = gtscript.Field[np.int_]
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
# 1 indexing to 0 and halos: -2, -1, 0 --> 0, 1,2
if MPI is not None and MPI.COMM_WORLD.Get_size() > 1:
    gt.config.cache_settings["dir_name"] = ".gt_cache_{:0>6d}".format(
        MPI.COMM_WORLD.Get_rank()
    )
# TODO remove when using quantities throughout model
def quantity_name(name):
    return name + "_quantity"


def stencil(**stencil_kwargs):
    def decorator(func):
        stencils = {}

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            key = (backend, rebuild)
            if key not in stencils:
                stencils[key] = gtscript.stencil(
                    backend=backend, rebuild=rebuild, **stencil_kwargs
                )(func)
            return stencils[key](*args, **kwargs)

        return wrapped

    return decorator


def make_storage_data(
    array,
    full_shape,
    istart=0,
    jstart=0,
    kstart=0,
    origin=origin,
    dummy=None,
    axis=2,
    names_4d=None,
):
    if len(array.shape) == 2:
        return make_storage_data_from_2d(
            array,
            full_shape,
            istart=istart,
            jstart=jstart,
            origin=origin,
            dummy=dummy,
            axis=axis,
        )
    elif len(array.shape) == 1:
        if dummy:
            axes = [0, 1, 2]
            axis = list(set(axes).difference(dummy))[0]
            return make_storage_data_from_1d(
                array, full_shape, kstart=kstart, origin=origin, axis=axis, dummy=dummy
            )
        else:
            return make_storage_data_from_1d(
                array, full_shape, kstart=kstart, origin=origin, axis=axis
            )
    elif len(array.shape) == 4:
        if names_4d is None:
            raise Exception("for 4d variable storages, specify a list of names")
        data_dict = {}
        for i in range(array.shape[3]):
            data_dict[names_4d[i]] = make_storage_data(
                np.squeeze(array[:, :, :, i]),
                full_shape,
                istart=istart,
                jstart=jstart,
                kstart=kstart,
                origin=origin,
                dummy=dummy,
                axis=axis,
            )
        return data_dict

    else:
        full_np_arr = np.zeros(full_shape)
        isize, jsize, ksize = array.shape
        full_np_arr[
            istart : istart + isize, jstart : jstart + jsize, kstart : kstart + ksize
        ] = asarray(array, type(full_np_arr))
        return gt.storage.from_array(
            data=full_np_arr, backend=backend, default_origin=origin, shape=full_shape,
        )


# axis refers to which axis should be repeated (when making a full 3d data), dummy refers to a singleton axis
def make_storage_data_from_2d(
    array2d, full_shape, istart=0, jstart=0, origin=origin, dummy=None, axis=2
):
    if dummy or axis != 2:
        d_axis = dummy[0] if dummy else axis
        shape2d = full_shape[:d_axis] + full_shape[d_axis + 1 :]
    else:
        shape2d = full_shape[0:2]
    isize, jsize = array2d.shape
    full_np_arr_2d = np.zeros(shape2d)
    full_np_arr_2d[istart : istart + isize, jstart : jstart + jsize] = asarray(
        array2d, type(full_np_arr_2d)
    )
    # full_np_arr_3d = np.lib.stride_tricks.as_strided(full_np_arr_2d, shape=full_shape, strides=(*full_np_arr_2d.strides, 0))
    if dummy:
        full_np_arr_3d = full_np_arr_2d.reshape(full_shape)
    else:
        full_np_arr_3d = np.repeat(
            full_np_arr_2d[:, :, np.newaxis], full_shape[axis], axis=2
        )
        if axis != 2:
            full_np_arr_3d = np.moveaxis(full_np_arr_3d, 2, axis)

    return gt.storage.from_array(
        data=full_np_arr_3d, backend=backend, default_origin=origin, shape=full_shape
    )


def make_2d_storage_data(array2d, shape2d, istart=0, jstart=0, origin=origin):
    # might not be i and j, could be i and k, j and k
    isize, jsize = array2d.shape
    full_np_arr_2d = np.zeros(shape2d)
    full_np_arr_2d[istart : istart + isize, jstart : jstart + jsize, 0] = array2d
    return gt.storage.from_array(
        data=full_np_arr_2d, backend=backend, default_origin=origin, shape=shape2d
    )


# axis refers to a repeated axis, dummy refers to a singleton axis
def make_storage_data_from_1d(
    array1d, full_shape, kstart=0, origin=origin, axis=2, dummy=None
):
    # r = np.zeros(full_shape)
    tilespec = list(full_shape)
    full_1d = np.zeros(full_shape[axis])
    full_1d[kstart : kstart + len(array1d)] = array1d
    tilespec[axis] = 1
    if dummy:
        if len(dummy) == len(tilespec) - 1:
            r = full_1d.reshape((full_shape))
        else:
            # TODO maybe, this is a little silly (repeat the array, then squash the dim), though eventually we shouldn't need this general capability if we refactor stencils to operate on 3d
            full_1d = make_storage_data_from_1d(
                array1d, full_shape, kstart=kstart, origin=origin, axis=axis, dummy=None
            )
            dimslice = [slice(None)] * len(tilespec)
            for dummy_axis in dummy:
                dimslice[dummy_axis] = slice(0, 1)
            r = full_1d[tuple(dimslice)]
    else:
        if axis == 2:
            r = np.tile(full_1d, tuple(tilespec))
            # r[:, :, kstart:kstart+len(array1d)] = np.tile(array1d, tuple(tilespec))
        elif axis == 1:
            x = np.repeat(full_1d[np.newaxis, :], full_shape[0], axis=0)
            r = np.repeat(x[:, :, np.newaxis], full_shape[2], axis=2)
        else:
            y = np.repeat(full_1d[:, np.newaxis], full_shape[1], axis=1)
            r = np.repeat(y[:, :, np.newaxis], full_shape[2], axis=2)
    return gt.storage.from_array(
        data=r, backend=backend, default_origin=origin, shape=full_shape
    )


def make_storage_from_shape(shape, origin, dtype=np.float64):
    return gt.storage.from_array(
        data=np.zeros(shape, dtype=dtype),
        backend=backend,
        default_origin=origin,
        shape=shape,
    )


def storage_dict(st_dict, names, shape, origin):
    for name in names:
        st_dict[name] = make_storage_from_shape(shape, origin)


def k_slice_operation(key, value, ki, dictionary):
    if isinstance(value, gt.storage.storage.Storage):
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


def zeros(shape, storage_type=np.ndarray, dtype=_dtype, order="F"):
    xp = cp if cp and storage_type is cp.ndarray else np
    return xp.zeros(shape)


def sum(array, axis=None, dtype=None, out=None, keepdims=False):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.sum(array, axis, dtype, out, keepdims)


def repeat(array, repeats, axis=None):
    xp = cp if cp and type(array) is cp.ndarray else np
    return xp.repeat(array.data, repeats, axis)


def index(array, key):
    return asarray(array, type(key))[key]
