import sys
import numpy as np
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
from config import *

import get_prs_fv3
import get_phi_fv3

from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    interval,
)

backend = "numpy"

def numpy_to_gt4py_storage_2D(arr, backend, k_depth):
    """convert numpy storage to gt4py storage"""
    data = np.reshape(arr, (arr.shape[0], 1, arr.shape[1]))
    if data.dtype == "bool":
        data = data.astype(np.int32)
    # Enforce that arrays are at least of length k_depth in the "k" direction
    if arr.shape[1] < k_depth:
        Z = np.zeros((arr.shape[0], 1, k_depth - arr.shape[1]))
        data = np.dstack((data, Z))
    return gt_storage.from_array(data, backend=backend, default_origin=(0, 0, 0))

def storage_to_numpy(gt_storage, array_dim):
    if isinstance(array_dim, tuple):
        np_tmp = np.zeros(array_dim)
        if(len(array_dim) == 2):
            np_tmp[:, :] = gt_storage[0 : array_dim[0], 0, 0 : array_dim[1]]
        elif(len(array_dim) == 3):
            np_tmp[:, :, :] = gt_storage[:,:,:]
    else:
        np_tmp = np.zeros(array_dim)
        np_tmp[:] = gt_storage[0:array_dim, 0, 0]

    if gt_storage.dtype == "int32":
        np_tmp.astype(int)

    return np_tmp

def run(in_dict):
    del_gz, phii, phil = physics_driver(
                            in_dict["IPD_levs"],
                            in_dict["IPD_phii"],
                            in_dict["IPD_prsi"],
                            in_dict["IPD_qgrs"],
                            in_dict["IPD_tgrs"],
                            in_dict["IPD_xlon"],
                            in_dict["IPD_ntrac"],
                            in_dict["IPD_gt0"],
                            in_dict["IPD_gq0"],
                        )
    # setup output
    out_dict = {}
    for key in ["phi_del_gz", "phi_phii", "phi_phil"]:
        out_dict[key] = np.zeros(1, dtype=np.float64)

    out_dict["phi_del_gz"] = del_gz
    out_dict["phi_phii"]   = phii
    out_dict["phi_phil"]   = phil

    return out_dict

def physics_driver(levs, phii, prsi, qgrs, tgrs, xlon, ntrac, gt0, gq0):

    ix = xlon.shape[0]
    im = ix
    dtp = 0.0 # CK: This is a guess right now

    phii   = numpy_to_gt4py_storage_2D(phii,        backend, levs+1)
    prsi   = numpy_to_gt4py_storage_2D(prsi,        backend, levs+1)
    tgrs   = numpy_to_gt4py_storage_2D(tgrs,        backend, levs+1)
    qgrs_0 = numpy_to_gt4py_storage_2D(qgrs[:,:,0], backend, levs+1)
    gt0    = numpy_to_gt4py_storage_2D(gt0,         backend, levs+1)

    gq0    = gt_storage.from_array(gq0,  backend=backend, default_origin=(0, 0, 0))
    qgrs   = gt_storage.from_array(qgrs, backend=backend, default_origin=(0, 0, 0))

    dtdt   = gt_storage.zeros(backend=backend, dtype=FIELD_FLT, shape=(ix, 1, levs+1),
                default_origin=(0, 0, 0))
    del_   = gt_storage.zeros(backend=backend, dtype=FIELD_FLT, shape=(ix, 1, levs+1),
                default_origin=(0, 0, 0))
    del_gz = gt_storage.zeros(backend=backend, dtype=FIELD_FLT, shape=(ix, 1, levs+1),
                default_origin=(0, 0, 0))

    phil   = gt_storage.zeros(backend=backend, dtype=FIELD_FLT, shape=(ix, 1, levs+1),
                default_origin=(0, 0, 0))

    gq_0   = gt_storage.zeros(backend=backend, dtype=FIELD_FLT, shape=(ix, 1, levs+1),
                default_origin=(0, 0, 0))

    get_prs_fv3.get_prs_fv3_stencil(phii,
                        prsi,
                        tgrs,
                        qgrs_0,
                        del_,
                        del_gz,
                        domain=(ix,1,levs+1))

    # These copies can be done within a stencil
    gt0 = tgrs
    gq0 = qgrs

    gq_0[:,0,:-1] = gq0[:,:,0]

    get_phi_fv3.get_phi_fv3_stencil(gt0,
                                    gq_0,
                                    del_gz,
                                    phii,
                                    phil,
                                    domain=(ix,1,levs+1))

    del_gz = storage_to_numpy(del_gz, (ix, levs+1))
    phii   = storage_to_numpy(phii,   (ix, levs+1))
    phil   = storage_to_numpy(phil,   (ix, levs))

    return del_gz, phii, phil