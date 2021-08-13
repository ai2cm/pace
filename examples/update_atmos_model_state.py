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

from mpi4py import MPI
# Access fv3core
sys.path.append("../../fv3core/")
# import fv3core
# import fv3core._config as spec
# import fv3core.testing
# import fv3core.utils.global_config as global_config
# import fv3gfs.util as util


def numpy_to_gt4py_storage_2D(arr, backend, k_depth):
    """convert numpy storage to gt4py storage"""
    data = np.reshape(arr, (arr.shape[0], 1, arr.shape[1]))
    if data.dtype == "bool":
        data = data.astype(np.int32)
    # Enforce that arrays are at least of length k_depth in the "k" direction
    if arr.shape[1] < k_depth:
        Z = np.zeros((arr.shape[0], 1, k_depth - arr.shape[1]))
        #data = np.dstack((Z, data))
        data = np.dstack((data,Z))
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
    # area = in_dict["IPD_area"]
    # area = area[:, np.newaxis]
    shape = (144, 1, 80)  # hard coded for now

    # Note: Value of dt_atmos is in the namelist
    dt_atmos = 255

    # Value of dnats from fv_arrays.F90
    dnats = 1  # namelist.dnats

    # *** Code used between physics_driver and fv_update_phys ***

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
        in_dict["gq0"],
        in_dict["gt0"],
        in_dict["gu0"],
        in_dict["gv0"],
        in_dict["tgrs"],
        in_dict["ugrs"],
        in_dict["vgrs"],
        in_dict["prsi"],
        in_dict["delp"],
        in_dict["u"],
        in_dict["v"],
        in_dict["w"],
        in_dict["pt"],
        in_dict["ua"],
        in_dict["va"],
        in_dict["ps"],
        in_dict["pe"],
        in_dict["peln"],
        in_dict["pk"],
        in_dict["pkz"],
        in_dict["phis"],
        in_dict["u_srf"],
        in_dict["v_srf"],
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

    for k in range(km - 1, -1, -1):
        for i in range(im):
            dp[i, k] = pe2[i, k + 1] - pe2[i, k]

    for k in range(km - 1, 0, -1):
        k1 = k - 1
        for i in range(im):
            if q[i, k] < q_min:
                q[i, k1] = q[i, k1] + (q[i, k] - q_min) * dp[i, k] / dp[i, k1]
                q[i, k] = q_min

    for k in range(km - 1):
        k1 = k + 1
        for i in range(im):
            if q[i, k] < 0.0:
                q[i, k1] = q[i, k1] + q[i, k] * dp[i, k] / dp[i, k1]
                q[i, k] = 0.0

    return q


def atmosphere_state_update(
    gq0, gt0, gu0, gv0,
    tgrs, ugrs, vgrs, prsi,
    delp,
    u, v, w, 
    pt, 
    ua, va, ps, 
    pe, peln, pk, pkz, phis, 
    u_srf, v_srf,
    q,
    t_dt, u_dt, v_dt,
    nq, dnats, nwat, dt_atmos,
    shape,
):

    nq_adv = nq - dnats

    rdt = 1.0e0 / dt_atmos

    gq0[:, :, 0] = fill_gfs(prsi, gq0[:, :, 0], 1.0e-9)

    im = gu0.shape[0]
    npz = gu0.shape[1]
    qwat = np.zeros(nq)
    
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

    # Note : pe has shape (14,80,14)
    #        peln has shape(12,80,12)

    pe, peln, pk, ps, pt, u_srf, v_srf = fv_update_phys(dt_atmos, u, v, w, delp, pt, ua, va, ps, pe, peln, pk, pkz, 
                   phis, u_srf, v_srf, False, u_dt, v_dt, t_dt, False, 0,0,0,False,
                   q[:,:,0], q[:,:,1], q[:,:,2], q[:,:,3], q[:,:,4], q[:,:,5], nwat)

    return u_dt, v_dt, t_dt, delp, q, pe, peln, pk, ps, pt, u_srf, v_srf

# *** Version of update_atmos_model_state to test code between physics_driver and fv_update_phys ***
def update_atmos_model_state(    
    gq0, gt0, gu0, gv0,
    tgrs, ugrs, vgrs, prsi,
    delp,
    u, v, w,
    pt, ua, va,
    ps, pe, peln, pk, pkz, phis,
    u_srf, v_srf,
    q,
    t_dt, u_dt, v_dt,
    nq, dnats, nwat, dt_atmos,
    shape,
):

    (u_dt, v_dt, t_dt, delp, q,
    pe, peln, pk, ps, pt, u_srf, v_srf) = atmosphere_state_update(
        gq0, gt0, gu0, gv0,
        tgrs, ugrs, vgrs, prsi,
        delp,
        u, v, w,
        pt, ua, va,
        ps, pe, peln, pk, pkz, phis,
        u_srf, v_srf,
        q,
        t_dt, u_dt, v_dt,
        nq, dnats, nwat, dt_atmos,
        shape,
    )

    out_dict = {}

    out_dict["u_dt"] = u_dt
    out_dict["v_dt"] = v_dt
    out_dict["t_dt"] = t_dt
    out_dict["delp"] = delp
    out_dict["q"] = q

    out_dict["pe"] = pe
    out_dict["ps"] = ps
    out_dict["peln"] = peln
    out_dict["pk"] = pk
    out_dict["pt"] = pt
    out_dict["u_srf"] = u_srf
    out_dict["v_srf"] = v_srf 

    return out_dict

def fv_update_phys(dt, #is_, ie, js, je, isd, ied, jsd, jed,
                   u, v, w, delp, pt, ua, va, ps, pe, peln, pk, pkz,
                   phis, u_srf, v_srf, hydrostatic,
                   u_dt, v_dt, t_dt,
                   gridstruct, npx, npy, npz, 
                   domain,
                   qvapor, qliquid, qrain, qsnow, qice, qgraupel, nwat):

    # Parameters from Fortran that're currently not implemented in Python
    # ng, q_dt, q, qdiag, nq, ak, bk, ts, delz, moist_phys, Time, nudge
    # lona, lata, flagstruct, neststruct, bd, ptop, physics_tendency_diag
    # column_moistening_implied_by_nudging, q_dt
    hydrostatic = False
    con_cp = cp_air
    # cvm = np.zeros(pk.shape[0])
    # qc  = np.zeros(pk.shape[0])
    cvm = np.zeros(12)
    qc  = np.zeros(12)

    # Hard coded tracer indexes : Maybe dynamically set these later?
    sphum   = 0
    liq_wat = 1
    rainwat = 2
    snowwat = 3
    ice_wat = 4
    graupel = 5
    cld_amt = 8

    # is_ = int((ua.shape[0]-u_srf.shape[0]) / 2)
    # ie = u_srf.shape[0]+is_

    # js = int((ua.shape[1]-u_srf.shape[1]) / 2)
    # je = u_srf.shape[1]+js
    npz = u.shape[1]

    for k in range(npz):
        # For testing, hard code the range of j
        #for j in range(js,je):
        for j in range(12):
            # Note : There's already a GT4Py-ported version of moist_cv
            qc, cvm = moist_cv(j, k, nwat, qvapor, qliquid, qrain, qsnow, qice, qgraupel, 
                               qc, cvm)
            
            for i in range(12):
                # pt[i,j,k] = pt[i,j,k] + t_dt[i-3,j-3,k] * dt * con_cp/cvm[i-3]
                pt[j*12 + i ,k] = pt[j*12 + i,k] + t_dt[j*12 + i,k] * dt * con_cp/cvm[i-3]


    # HALO EXCHANGES

    for j in range(12):
        for k in range(1,npz+1):
            for i in range(12):
                pe[i+1,k,j+1] = pe[i+1,k-1,j+1] + delp[12*j + i,k-1]
                peln[i,k,j] = np.log(pe[i+1,k,j+1])
                pk[12*j + i,k] = np.exp(KAPPA*peln[i,k,j])

        for i in range(12):
            ps[12*j + i] = pe[i+1,npz,j+1]
            u_srf[12*j + i] = ua[12*j + i,npz-1]
            v_srf[12*j + i] = va[12*j + i,npz-1]


    # COMPLETE_GROUP_HALO_UPDATE

    # UPDATE_DWINDS_PHYS

    #CUBED_TO_LATLON
    return pe, peln, pk, ps, pt, u_srf, v_srf

# Note : There already exists a moist_cv stencil within fv3core
def moist_cv(j, k, nwat, qvapor, qliquid, qrain, qsnow, qice, qgraupel, qd, cvm):


    # Note: Fortran code has a select case contruct to select how to 
    #       update qv, ql, qs, and qd.
    #       Currently we're only implementing case(6)

    # $$$ Previous moist_cv implementation $$$

    # qvapor, qliquid, qrain, qsnow, qice, and qgraupel have halo regions in i and j,
    # so their index in i have to be incremented by 3 to account for the halo.  The j
    # index as a input parameter already takes takes the halo into consideration

    # for i in range(qd.shape[0]):
    #     qv = qvapor[i+3, j, k]
    #     ql = qliquid[i+3, j, k] + qrain[i+3, j, k]
    #     qs = qice[i+3, j, k] + qsnow[i+3, j, k] + qgraupel[i+3, j, k]
    #     qd[i] = ql + qs
    #     cvm[i] = (1.0 - (qv + qd[i])) * cv_air + qv * cv_vap + ql * c_liq + qs * c_ice

    # $$$$$$

    for i in range(12):
        qv = qvapor[j*12 + i, k]
        ql = qliquid[j*12 + i, k] + qrain[j*12 + i, k]
        qs = qice[j*12+i, k] + qsnow[j*12 + i, k] + qgraupel[j*12 + i, k]
        qd[i] = ql + qs
        cvm[i] = (1.0 - (qv + qd[i])) * cv_air + qv * cv_vap + ql * c_liq + qs * c_ice

    return qd, cvm