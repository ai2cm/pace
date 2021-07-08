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

def run(in_dict, timings):
    del_gz, phii, phil = get_phi_fv3(
                    in_dict["ix"],
                    in_dict["levs"],
                    in_dict["ntrac"],
                    in_dict["gt0"],
                    in_dict["gq0"],
                    in_dict["del_gz"],
                    in_dict["phii"],
                    in_dict["phil"]
                )
    # setup output
    out_dict = {}
    for key in OUT_VARS:
        out_dict[key] = np.zeros(1, dtype=np.float64)

    out_dict["del_gz"] = del_gz
    out_dict["phii"]   = phii
    out_dict["phil"]   = phil

    return out_dict

def get_phi_fv3(ix, 
                levs, 
                ntrac, 
                gt0, 
                gq0, 
                del_gz,
                phii, 
                phil):
    
    gt0    = numpy_to_gt4py_storage_2D(gt0,        backend, levs+1)
    gq_0   = numpy_to_gt4py_storage_2D(gq0[:,:,0], backend, levs+1)    
    del_gz = numpy_to_gt4py_storage_2D(del_gz,     backend, levs+1)
    phii   = numpy_to_gt4py_storage_2D(phii,       backend, levs+1)
    phil   = numpy_to_gt4py_storage_2D(phil,       backend, levs+1)

    get_phi_fv3_stencil(gt0,
                        gq_0,
                        del_gz,
                        phii,
                        phil,
                        domain=(ix,1,levs+1))

    del_gz = storage_to_numpy(del_gz, (ix, levs+1))
    phii   = storage_to_numpy(phii,   (ix, levs+1))
    phil   = storage_to_numpy(phil,   (ix, levs))

    return del_gz, phii, phil

@gtscript.stencil(backend=backend)
def get_phi_fv3_stencil(gt0: FIELD_FLT,
                        gq0: FIELD_FLT,
                        del_gz: FIELD_FLT,
                        phii: FIELD_FLT,
                        phil: FIELD_FLT):
    with computation(FORWARD), interval(0,1):
        phii = 0.0

    with computation(FORWARD):
        with interval(0,1):
            del_gz = del_gz[0,0,0] * gt0[0,0,0] * (1.0 + con_fvirt * max(0.0, gq0[0,0,0]))
            phil   = 0.5 * (phii[0,0,0] + phii[0,0,0] + del_gz[0,0,0])
        with interval(1,-1):
            phii = phii[0,0,-1] + del_gz[0,0,-1]
            del_gz = del_gz[0,0,0] * gt0[0,0,0] * (1.0 + con_fvirt * max(0.0, gq0[0,0,0]))
            phil   = 0.5 * (phii[0,0,0] + phii[0,0,0] + del_gz[0,0,0])
        with interval(-1,None):
            phii = phii[0,0,-1] + del_gz[0,0,-1]