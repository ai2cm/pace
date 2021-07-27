import sys
import numpy as np
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

sys.path.append("../")
from fv3gfsphysics.utils.global_config import *
from fv3gfsphysics.utils.global_constants import *

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

    # Note: Value of dt_atmos is in the namelist
    dt_atmos = 255

    # Value of dnats from fv_arrays.F90
    dnats = 1  # namelist.dnats

    u_dt = np.zeros(in_dict["gu0"].shape)
    v_dt = np.zeros(in_dict["gv0"].shape)
    t_dt = np.zeros(in_dict["gt0"].shape)

    q = np.zeros(
        (in_dict["qvapor"].shape[0], in_dict["qvapor"].shape[1], 8)
    )  # Assumption that there are 8 layers to q

    q[:, :, 0] = in_dict["qvapor"]
    q[:, :, 1] = in_dict["qliquid"]
    q[:, :, 2] = in_dict["qrain"]
    q[:, :, 3] = in_dict["qsnow"]
    q[:, :, 4] = in_dict["qice"]
    q[:, :, 5] = in_dict["qgraupel"]
    q[:, :, 6] = in_dict["qo3mr"]
    q[:, :, 7] = in_dict["qsgs_tke"]

    out_dict = update_atmos_model_state(
        in_dict["gq0_check_in"],
        in_dict["gq0_check_out"],
        in_dict["gq0"],
        in_dict["gt0"],
        in_dict["gu0"],
        in_dict["gv0"],
        in_dict["tgrs"],
        in_dict["ugrs"],
        in_dict["vgrs"],
        in_dict["prsi"],
        in_dict["delp"],
        q,
        t_dt,
        u_dt,
        v_dt,
        in_dict["nq"],
        dnats,
        in_dict["nwat"],
        dt_atmos,
        shape,
    )

    return out_dict


def fill_gfs(pe2, q, q_min):

    im = q.shape[0]
    km = q.shape[1]

    dp = np.zeros((im, km))

    # for k in range(km):
    #     for i in range(im):
    #         dp[i,k] = pe2[i,k] - pe2[i,k+1]

    for k in range(km - 1, -1, -1):
        for i in range(im):
            dp[i, k] = pe2[i, k + 1] - pe2[i, k]

    # for k in range(km-1):
    #     k1 = k+1
    #     for i in range(im):
    #         if q[i,k] < q_min:
    #             q[i,k1] = q[i,k1] + (q[i,k] - q_min) * dp[i,k]/dp[i,k1]
    #             q[i,k] = q_min

    for k in range(km - 1, 0, -1):
        k1 = k - 1
        for i in range(im):
            if q[i, k] < q_min:
                q[i, k1] = q[i, k1] + (q[i, k] - q_min) * dp[i, k] / dp[i, k1]
                q[i, k] = q_min

    # for k in range(km-1, 1, -1):
    #     k1 = k-1
    #     for i in range(im):
    #         if q[i,k] < 0.0:
    #             q[i,k1] = q[i,k1] + q[i,k] * dp[i,k]/dp[i,k1]
    #             q[i,k] = 0.0

    for k in range(km - 1):
        k1 = k + 1
        for i in range(im):
            if q[i, k] < 0.0:
                q[i, k1] = q[i, k1] + q[i, k] * dp[i, k] / dp[i, k1]
                q[i, k] = 0.0

    return q


def atmosphere_state_update(
    gq0_check_in,
    gq0_check_out,
    gq0,
    gt0,
    gu0,
    gv0,
    tgrs,
    ugrs,
    vgrs,
    prsi,
    delp,
    q,
    t_dt,
    u_dt,
    v_dt,
    nq,
    dnats,
    nwat,
    dt_atmos,
    shape,
):

    nq_adv = nq - dnats

    rdt = 1.0e0 / dt_atmos

    np.testing.assert_allclose(gq0, gq0_check_in)

    gq0[:, :, 0] = fill_gfs(prsi, gq0[:, :, 0], 1.0e-9)

    np.testing.assert_allclose(gq0, gq0_check_out)

    im = gu0.shape[0]
    npz = gu0.shape[1]
    qwat = np.zeros(nq)
    # Does blen value matter for ix?
    blen = 1
    for k in range(npz):
        for ix in range(im):
            u_dt[ix, k] = u_dt[ix, k] + (gu0[ix, k] - ugrs[ix, k]) * rdt
            v_dt[ix, k] = v_dt[ix, k] + (gv0[ix, k] - vgrs[ix, k]) * rdt
            t_dt[ix, k] = t_dt[ix, k] + (gt0[ix, k] - tgrs[ix, k]) * rdt

            q0 = prsi[ix, k + 1] - prsi[ix, k]
            qwat[0:nq_adv] = q0 * gq0[ix, k, 0:nq_adv]
            qt = np.sum(qwat[0:nwat])
            q_sum = np.sum(q[ix, k, 0:nwat])
            q0 = delp[ix, k] * (1.0 - q_sum) + qt

            delp[ix, k] = q0
            q[ix, k, :nq_adv] = qwat[:nq_adv] / q0

    return u_dt, v_dt, t_dt, delp, q


def update_atmos_model_state(
    gq0_check_in,
    gq0_check_out,
    gq0,
    gt0,
    gu0,
    gv0,
    tgrs,
    ugrs,
    vgrs,
    prsi,
    delp,
    q,
    t_dt,
    u_dt,
    v_dt,
    nq,
    dnats,
    nwat,
    dt_atmos,
    shape,
):

    (u_dt, v_dt, t_dt, delp, q) = atmosphere_state_update(
        gq0_check_in,
        gq0_check_out,
        gq0,
        gt0,
        gu0,
        gv0,
        tgrs,
        ugrs,
        vgrs,
        prsi,
        delp,
        q,
        t_dt,
        u_dt,
        v_dt,
        nq,
        dnats,
        nwat,
        dt_atmos,
        shape,
    )

    out_dict = {}

    out_dict["u_dt"] = u_dt
    out_dict["v_dt"] = v_dt
    out_dict["t_dt"] = t_dt
    out_dict["delp"] = delp
    out_dict["q"] = q
    return out_dict
