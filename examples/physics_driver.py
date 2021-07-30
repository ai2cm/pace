import sys
import numpy as np
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage

from fv3gfs.physics.global_config import *
from fv3gfs.physics.global_constants import *
import fv3gfs.physics.stencils.get_prs_fv3 as get_prs_fv3
import fv3gfs.physics.stencils.get_phi_fv3 as get_phi_fv3
import fv3gfs.physics.stencils.microphysics_standalone as microphysics

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
    out_dict = physics_driver(
        in_dict["IPD_dtp"],
        in_dict["IPD_levs"],
        in_dict["IPD_phii"],
        in_dict["IPD_prsi"],
        in_dict["IPD_qgrs"],
        in_dict["IPD_tgrs"],
        in_dict["IPD_ntrac"],
        in_dict["IPD_gt0"],
        in_dict["IPD_gq0"],
        area,
        in_dict["IPD_vvl"],
        in_dict["IPD_prsl"],
        in_dict["IPD_gu0"],
        in_dict["IPD_gv0"],
        in_dict["IPD_ugrs"],
        in_dict["IPD_vgrs"],
        in_dict["IPD_refl_10cm"],
        shape,
    )

    return out_dict


def physics_driver(
    dtp,
    levs,
    phii,
    prsi,
    qgrs,
    tgrs,
    ntrac,
    gt0,
    gq0,
    area,
    vvl,
    prsl,
    gu0,
    gv0,
    ugrs,
    vgrs,
    refl_10cm,
    full_shape,
):
    # dtp = spec.namelist.dt_atmos
    # ntrac = state.nq_tot
    # levs = spec.namelist.npz
    ix = full_shape[0]
    im = ix

    phii = numpy_to_gt4py_storage_2D(phii, BACKEND, levs + 1)
    prsi = numpy_to_gt4py_storage_2D(prsi, BACKEND, levs + 1)
    tgrs = numpy_to_gt4py_storage_2D(tgrs, BACKEND, levs + 1)
    qgrs_0 = numpy_to_gt4py_storage_2D(qgrs[:, :, 0], BACKEND, levs + 1)
    gt0 = numpy_to_gt4py_storage_2D(gt0, BACKEND, levs + 1)
    vvl = numpy_to_gt4py_storage_2D(vvl, BACKEND, levs + 1)
    prsl = numpy_to_gt4py_storage_2D(prsl, BACKEND, levs + 1)
    gu0 = numpy_to_gt4py_storage_2D(gu0, BACKEND, levs + 1)
    gv0 = numpy_to_gt4py_storage_2D(gv0, BACKEND, levs + 1)
    refl_10cm = numpy_to_gt4py_storage_2D(refl_10cm, BACKEND, levs + 1)
    ugrs = numpy_to_gt4py_storage_2D(ugrs, BACKEND, levs + 1)
    vgrs = numpy_to_gt4py_storage_2D(vgrs, BACKEND, levs + 1)

    gq0 = gt_storage.from_array(gq0, backend=BACKEND, default_origin=(0, 0, 0))
    qgrs = gt_storage.from_array(qgrs, backend=BACKEND, default_origin=(0, 0, 0))

    dtdt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    del_ = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    del_gz = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )

    phil = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )

    gq_0 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )

    get_prs_fv3.get_prs_fv3_stencil(
        phii, prsi, tgrs, qgrs_0, del_, del_gz, domain=full_shape
    )

    debug = {}
    debug["phii"] = phii
    debug["prsi"] = prsi
    debug["pt"] = tgrs
    debug["qvapor"] = qgrs_0
    debug["del"] = del_
    debug["del_gz"] = del_gz
    np.save("standalone_after_prsfv3.npy", debug)

    # These copies can be done within a stencil
    gt0 = tgrs  # + dtdt * dtp (tendencies from PBL and others)
    gq0 = qgrs
    gu0 = ugrs
    gv0 = vgrs

    gq_0[:, 0, 1:] = gq0[:, :, 0]

    get_phi_fv3.get_phi_fv3_stencil(gt0, gq_0, del_gz, phii, phil, domain=full_shape)

    land = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=(full_shape[0], full_shape[1]),
        default_origin=(0, 0, 0),
    )

    area = gt_storage.from_array(area, backend=BACKEND, default_origin=(0, 0, 0))
    rain0 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=(full_shape[0], full_shape[1]),
        default_origin=(0, 0, 0),
    )
    snow0 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=(full_shape[0], full_shape[1]),
        default_origin=(0, 0, 0),
    )
    ice0 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=(full_shape[0], full_shape[1]),
        default_origin=(0, 0, 0),
    )
    graupel0 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=(full_shape[0], full_shape[1]),
        default_origin=(0, 0, 0),
    )
    qn1 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qv_dt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    ql_dt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qr_dt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qi_dt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qs_dt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qg_dt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qa_dt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    pt_dt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    udt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    vdt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qv1 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    ql1 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qr1 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qi1 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qs1 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qg1 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qa1 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    pt = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    w = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    uin = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    vin = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    delp = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    dz = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    p123 = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    refl = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    Z = np.zeros((gq0.shape[0], 1, gq0.shape[2]))
    gq0 = np.concatenate((Z, gq0), axis=1)
    qv1[:, :, :] = gq0[:, :, 0][:, np.newaxis, :]  # using newaxis for now
    ql1[:, :, :] = gq0[:, :, 1][:, np.newaxis, :]
    qr1[:, :, :] = gq0[:, :, 2][:, np.newaxis, :]
    qi1[:, :, :] = gq0[:, :, 3][:, np.newaxis, :]
    qs1[:, :, :] = gq0[:, :, 4][:, np.newaxis, :]
    qg1[:, :, :] = gq0[:, :, 5][:, np.newaxis, :]
    qa1[:, :, :] = gq0[:, :, 8][:, np.newaxis, :]
    pt[:, :, :] = gt0
    w[:, :, 1:] = (
        -vvl[:, :, 1:]
        * (1.0 + con_fvirt * qv1[:, :, 1:])
        * gt0[:, :, 1:]
        / prsl[:, :, 1:]
        * (rdgas * rgrav)
    )
    uin[:, :, :] = gu0
    vin[:, :, :] = gv0
    delp[:, :, :] = del_
    for k in range(1, levs + 1):
        dz[:, :, k] = (phii[:, :, k] - phii[:, :, k - 1]) * rgrav
    p123[:, :, :] = prsl
    refl[:, :, :] = refl_10cm
    mph_input = {}
    mph_input["area"] = area
    mph_input["delp"] = delp
    mph_input["dtp_in"] = dtp
    mph_input["dz"] = dz
    mph_input["graupel0"] = graupel0
    mph_input["ice0"] = ice0
    mph_input["im"] = im
    mph_input["land"] = land
    mph_input["levs"] = levs
    mph_input["lradar"] = False  # lradar
    mph_input["p123"] = p123
    mph_input["pt"] = pt
    mph_input["pt_dt"] = pt_dt
    mph_input["qa1"] = qa1
    mph_input["qa_dt"] = qa_dt
    mph_input["qg1"] = qg1
    mph_input["qg_dt"] = qg_dt
    mph_input["qi1"] = qi1
    mph_input["qi_dt"] = qi_dt
    mph_input["ql1"] = ql1
    mph_input["ql_dt"] = ql_dt
    mph_input["qn1"] = qn1
    mph_input["qr1"] = qr1
    mph_input["qr_dt"] = qr_dt
    mph_input["qs1"] = qs1
    mph_input["qs_dt"] = qs_dt
    mph_input["qv1"] = qv1
    mph_input["qv_dt"] = qv_dt
    mph_input["rain0"] = rain0
    mph_input["refl"] = refl
    mph_input["reset"] = True  # reset
    mph_input["seconds"] = 0.0  # seconds
    mph_input["snow0"] = snow0
    mph_input["udt"] = udt
    mph_input["uin"] = uin
    mph_input["vdt"] = vdt
    mph_input["vin"] = vin
    mph_input["w"] = w
    mph_output = microphysics.run(mph_input)
    output = {}

    gq0[:, :, 0] = qv1[:, 0, :] + qv_dt[:, 0, :] * dtp
    gq0[:, :, 1] = ql1[:, 0, :] + ql_dt[:, 0, :] * dtp
    gq0[:, :, 2] = qr1[:, 0, :] + qr_dt[:, 0, :] * dtp
    gq0[:, :, 3] = qi1[:, 0, :] + qi_dt[:, 0, :] * dtp
    gq0[:, :, 4] = qs1[:, 0, :] + qs_dt[:, 0, :] * dtp
    gq0[:, :, 5] = qg1[:, 0, :] + qg_dt[:, 0, :] * dtp
    gq0[:, :, 8] = qa1[:, 0, :] + qa_dt[:, 0, :] * dtp

    # These computations could be put into a stencil
    gt0[:, 0, :] = gt0[:, 0, :] + pt_dt[:, 0, :] * dtp
    gu0[:, 0, :] = gu0[:, 0, :] + udt[:, 0, :] * dtp
    gv0[:, 0, :] = gv0[:, 0, :] + vdt[:, 0, :] * dtp

    output["IPD_gq0"] = storage_to_numpy(gq0, (144, 79, 9), True)
    output["IPD_gt0"] = storage_to_numpy(gt0, (144, 79), True)
    output["IPD_gu0"] = storage_to_numpy(gu0, (144, 79), True)
    output["IPD_gv0"] = storage_to_numpy(gv0, (144, 79), True)
    return output
