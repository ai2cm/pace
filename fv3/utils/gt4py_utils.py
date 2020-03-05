#!/usr/bin/env python3

import numpy as np
import gt4py as gt
import gt4py.gtscript as gtscript
import copy as cp
import math

data_backend = "numpy"  # Options: numpy, gtmc, gtx86, gtcuda, debug
exec_backend = "numpy"  # Options: numpy, gtmc, gtx86, gtcuda, debug, and dawn:gtmc
rebuild = True
_dtype = np.float_
sd = gtscript.Field[_dtype]
halo = 3
origin = (halo, halo, 0)
# 1 indexing to 0 and halos: -2, -1, 0 --> 0, 1,2


def make_storage_data(
    array, full_shape, istart=0, jstart=0, kstart=0, origin=origin, backend=data_backend
):
    full_np_arr = np.zeros(full_shape)
    if len(array.shape) == 2:
        return make_storage_data_from_2d(
            array,
            full_shape,
            istart=istart,
            jstart=jstart,
            origin=origin,
            backend=backend,
        )
    elif len(array.shape) == 1:
        return make_storage_data_from_1d(
            array, full_shape, kstart=kstart, origin=origin, backend=backend
        )
    else:
        isize, jsize, ksize = array.shape
        full_np_arr[
            istart : istart + isize, jstart : jstart + jsize, kstart : kstart + ksize
        ] = array
        return gt.storage.from_array(
            data=full_np_arr, backend=backend, default_origin=origin, shape=full_shape
        )


def make_storage_data_from_2d(
    array2d, full_shape, istart=0, jstart=0, origin=origin, backend=data_backend
):
    shape2d = full_shape[0:2]
    isize, jsize = array2d.shape
    full_np_arr_2d = np.zeros(shape2d)
    full_np_arr_2d[istart : istart + isize, jstart : jstart + jsize] = array2d
    # full_np_arr_3d = np.lib.stride_tricks.as_strided(full_np_arr_2d, shape=full_shape, strides=(*full_np_arr_2d.strides, 0))
    full_np_arr_3d = np.repeat(full_np_arr_2d[:, :, np.newaxis], full_shape[2], axis=2)
    return gt.storage.from_array(
        data=full_np_arr_3d, backend=backend, default_origin=origin, shape=full_shape
    )


# TODO: surely there's a shorter, more generic way to do this.
def make_storage_data_from_1d(
    array1d, full_shape, kstart=0, origin=origin, backend=data_backend, axis=2
):
    # r = np.zeros(full_shape)
    tilespec = list(full_shape)
    full_1d = np.zeros(full_shape[axis])
    full_1d[kstart : kstart + len(array1d)] = array1d
    tilespec[axis] = 1
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


def make_storage_from_shape(shape, origin, backend=data_backend):
    return gt.storage.from_array(
        data=np.zeros(shape), backend=backend, default_origin=origin, shape=shape
    )


def k_slice_operation(key, value, ki, dictionary):
    if isinstance(value, gt.storage.storage.Storage):
        dictionary[key] = make_storage_data(
            value.data[:, :, ki], (value.data.shape[0], value.data.shape[1], len(ki))
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


def compute_column_split(func, data, column_split, split_varname, outputs, grid):
    num_k = grid.npz
    grid_data = cp.deepcopy(grid.data_fields)
    for kval in np.unique(column_split):
        ki = [i for i in range(num_k) if column_split[i] == kval]
        k_subset_run(func, data, {split_varname: kval}, ki, outputs, grid_data, grid)
    grid.npz = num_k


def k_subset_run(func, data, splitvars, ki, outputs, grid_data, grid):
    grid.npz = len(ki)
    grid.slice_data_k(ki)
    d = k_slice(data, ki)
    d.update(splitvars)
    results = func(**d)
    collect_results(d, results, outputs, ki)
    grid.add_data(grid_data)


def collect_results(data, results, outputs, ki):
    outnames = list(outputs.keys())
    print("Computing results for k indices:", ki[:-1])
    for k in outnames:
        if k in data:
            # passing fields with single item in 3rd dimension leads to errors
            outputs[k][:, :, ki[:-1]] = data[k][:, :, :-1]
            # outnames.remove(k)
        # else:
        #    print(k, 'not in data')
    if results is not None:
        for ri in range(len(results)):
            outputs[outnames[ri]][:, :, ki[:-1]] = results[ri][:, :, :-1]


def k_split_run(func, data, k_indices_array, splitvars_values, outputs, grid):
    num_k = grid.npz
    grid_data = cp.deepcopy(grid.data_fields)
    for ki in k_indices_array:
        splitvars = {}
        for name, value_array in splitvars_values.items():
            splitvars[name] = value_array[ki[0]]
        k_subset_run(func, data, splitvars, ki, outputs, grid_data, grid)
    grid.npz = num_k


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
