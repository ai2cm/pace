import numpy as np
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
import timeit
from config import *

from gt4py.gtscript import (
    __INLINED,
    PARALLEL,
    computation,
    interval,
)

backend = BACKEND

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
        np_tmp[:, :] = gt_storage[0 : array_dim[0], 0, 0 : array_dim[1]]
    else:
        np_tmp = np.zeros(array_dim)
        np_tmp[:] = gt_storage[0:array_dim, 0, 0]

    if gt_storage.dtype == "int32":
        np_tmp.astype(int)

    return np_tmp

def run(in_dict, timing):
    del_, del_gz = get_prs_fv3(
                    in_dict["ix"],
                    in_dict["levs"],
                    in_dict["ntrac"],
                    in_dict["phii"],
                    in_dict["prsi"],
                    in_dict["tgrs"],
                    in_dict["qgrs"],
                    in_dict["del"],
                    in_dict["del_gz"]
                )
    # setup output
    out_dict = {}
    for key in OUT_VARS:
        out_dict[key] = np.zeros(1, dtype=np.float64)

    out_dict["del"]    = del_
    out_dict["del_gz"] = del_gz

    return out_dict

def get_prs_fv3(ix, 
                levs, 
                ntrac, 
                phii, 
                prsi, 
                tgrs,
                qgrs, 
                del_, 
                del_gz):
    
    phii   = numpy_to_gt4py_storage_2D(phii,        backend, levs+1)
    prsi   = numpy_to_gt4py_storage_2D(prsi,        backend, levs+1)
    tgrs   = numpy_to_gt4py_storage_2D(tgrs,        backend, levs+1)
    qgrs_0 = numpy_to_gt4py_storage_2D(qgrs[:,:,0], backend, levs+1)
    del_   = numpy_to_gt4py_storage_2D(del_,        backend, levs+1)
    del_gz = numpy_to_gt4py_storage_2D(del_gz,      backend, levs+1)

    get_prs_fv3_stencil(phii,
                        prsi,
                        tgrs,
                        qgrs_0,
                        del_,
                        del_gz,
                        domain=(ix,1,levs+1))

    del_   = storage_to_numpy(del_,   (ix, levs))
    del_gz = storage_to_numpy(del_gz, (ix, levs+1))

    return del_, del_gz

@gtscript.stencil(backend=backend)
def get_prs_fv3_stencil(phii: FIELD_FLT,
                        prsi: FIELD_FLT,
                        tgrs: FIELD_FLT,
                        qgrs: FIELD_FLT,
                        del_: FIELD_FLT,
                        del_gz: FIELD_FLT):
    with computation(PARALLEL), interval(0,-1):
        del_ = prsi[0,0,0] - prsi[0,0,1]
        del_gz = (phii[0,0,1] - phii[0,0,0]) / (tgrs[0,0,0] * (1.0 + con_fvirt * max(0.0, qgrs[0,0,0])))