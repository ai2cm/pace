import sys
import numpy as np
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    interval,
)

def numpy_to_gt4py_storage_2D(arr, backend, k_depth):
    """convert numpy storage to gt4py storage"""
    data = np.reshape(arr, (arr.shape[0], 1, arr.shape[1]))
    if data.dtype == "bool":
        data = data.astype(np.int32)
    # Enforce that arrays are at least of length k_depth in the "k" direction
    if arr.shape[1] < k_depth:
        Z = np.zeros((arr.shape[0], 1, k_depth - arr.shape[1]))
        data = np.dstack((Z, data))
    return gt_storage.from_array(data, backend=BACKEND, default_origin=(0, 0, 0))


def storage_to_numpy(gt_storage, array_dim, has_zero_padding):
    if isinstance(array_dim, tuple):
        np_tmp = np.zeros(array_dim)
        if len(array_dim) == 2:
            if has_zero_padding:
                np_tmp[:, :] = gt_storage[0 : array_dim[0], 0, 1 : array_dim[1] + 1]
            else:
                np_tmp[:, :] = gt_storage[0 : array_dim[0], 0, 0 : array_dim[1]]
        elif len(array_dim) == 3:
            if has_zero_padding:
                np_tmp[:, :, :] = gt_storage[
                    0 : array_dim[0], 1 : array_dim[1] + 1, 0 : array_dim[2]
                ]
            else:
                np_tmp[:, :, :] = gt_storage[
                    0 : array_dim[0], 0 : array_dim[1], 0 : array_dim[2]
                ]
    else:
        np_tmp = np.zeros(array_dim)
        np_tmp[:] = gt_storage[0:array_dim, 0, 0]

    if gt_storage.dtype == "int32":
        np_tmp.astype(int)

    return np_tmp

def run(in_dict):
    area = in_dict["IPD_area"]
    area = area[:, np.newaxis]
    shape = (144, 1, 80)  # hard coded for now
    out_dict = update_atmos_model_state(
        in_dict["IPD_gq0"],
        in_dict["IPD_gt0"],
        in_dict["IPD_gu0"],
        in_dict["IPD_gv0"],
        shape,
    )

    return out_dict

def fill_gfs(im, km, pe2, q, q_min):
    dp = np.zeros((im,km))

    for k in range(km):
        for i in range(im):
            dp[i,k] = pe2[i,k] - pe2[i,k+1]

    for k in range(km-1):
        k1 = k+1
        for i in range(im):
            if q[i,k] < q_min:
                q[i,k1] = q[i,k1] + (q[i,k] - q_min) * dp[i,k]/dp[i,k1]
                q[i,k] = q_min

    for k in range(km, 1, -1):
        k1 = k-1
        for i in range(im):
            if q[i,k] < 0.0:
                q[i,k1] = q[i,k1] + q[i,k] * dp[i,k]/dp[i,k1]
                q[i,k] = 0.0

    return q

def atmosphere_state_update(im,
                            km,
                            npz,
                            gq0,
                            gt0,
                            gu0,
                            gv0,
                            prsi,
                            u_dt,
                            v_dt,
                            t_dt,
                            ugrs,
                            vgrs,
                            tgrs,     
                            ii,
                            jj,
                            dt_atmos,
                            flip_vc,
                            shape):

    # Need to figure out how to get the value of "nq"
    # qwat = np.zeros(nq)

    # nq_adv = nq - dnats

    rdt = 1.0e0 / dt_atmos

    gq0 = fill_gfs(im, km, prsi, gq0, 1.0e-9)

    for k in range(npz):
        if flip_vc:
            k1 = npz+1-k
        else:
            k1 = k
        for ix in range(im):
            i = ii[ix]
            j = jj[ix]
            u_dt[i,j,k1] = u_dt[i,j,k1] + (gu0[ix,k] - ugrs[ix,k]) * rdt
            v_dt[i,j,k1] = v_dt[i,j,k1] + (gv0[ix,k] - vgrs[ix,k]) * rdt
            t_dt[i,j,k1] = t_dt[i,j,k1] + (gt0[ix,k] - tgrs[ix,k]) * rdt

            if flip_vc:
                q0 = prsi[ix,k] - prsi[ix,k+1]
            else:
                q0 = prsi[ix,k+1] - prsi[ix,k]
            #qwat[0:nq_adv] = q0 * gq0[ix,k,0:nq_adv]


    return gq0, gt0, gu0, gv0

def update_atmos_model_state(im,
                             km,
                             npz,
                             gq0,
                             gt0,
                             gu0,
                             gv0,
                             prsi,
                             u_dt,
                             v_dt,
                             t_dt,
                             ugrs,
                             vgrs,
                             tgrs,
                             ii, jj,
                             dt_atmos,
                             flip_vc,
                             shape):

    atmosphere_state_update(im,km, npz, gq0,gt0, gu0, gv0, prsi, u_dt, v_dt, t_dt, ugrs, vgrs, tgrs, ii, jj, dt_atmos,flip_vc,shape)