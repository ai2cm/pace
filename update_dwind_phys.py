#!/usr/bin/env python3

import os
import sys
import numpy as np
import gt4py.storage as gt_storage
from gt4py.gtscript import (
    PARALLEL,
    FORWARD,
    BACKWARD,
    computation,
    horizontal,
    interval,
)
from fv3gfsphysics.utils.global_config import *
SERIALBOX_DIR = "/usr/local/serialbox2.6.1"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser
SELECT_SP = None

def add_composite_vvar_storage(d, varname, data3d, max_shape, start_indices):
    shape = data3d.shape
    start1, start2 = start_indices.get(varname, (0,0))
    size1, size2 = data3d.shape[0:2]
    for s in range(data3d.shape[2]):
        buffer =  np.zeros(max_shape[0:2])
        buffer[start1 : start1 + size1, start2 : start2 + size2] = np.squeeze(data3d[:, :, s])
        d[varname + str(s + 1)] = gt_storage.from_array(
            data=buffer, backend=BACKEND, default_origin=(start1, start2), shape=max_shape[0:2],  dtype=DTYPE_FLT
        )

def add_composite_evar_storage(d, varname, data4d, max_shape, start_indices):
    shape = data4d.shape
    start1, start2 = start_indices.get(varname, (0,0))
    size1, size2 = data4d.shape[1:3]
    for s in range(data4d.shape[0]):
        for t in range(data4d.shape[3]):
            buffer =  np.zeros(max_shape[0:2])
            buffer[start1 : start1 + size1, start2 : start2 + size2] = np.squeeze(data4d[s, :, :, t])
            d[varname + str(s + 1) + "_"+str(t+1)] = gt_storage.from_array(
                data=buffer, backend=BACKEND, default_origin=(start1, start2), shape=max_shape[0:2],  dtype=DTYPE_FLT
            )

def storage_dict_from_var_list(var_list, serializer, savepoint, max_shape, start_indices):
    d = {}
    for var in var_list:
        data = serializer.read(var, savepoint)
        if var in ["vlat", "vlon"]:
            add_composite_vvar_storage(d, var, data, max_shape, start_indices)
            continue
        if var in ["es", "ew"]:
            add_composite_evar_storage(d, var, data, max_shape, start_indices)
            continue
        # convert single element numpy arrays to scalars
        elif data.size == 1:
            data = data.item()
            d[var] = data
            continue
        elif len(data.shape) < 2:
            d[var] = data
        elif len(data.shape) == 2:
            d[var] = np.zeros(max_shape[0:2])
            start1, start2 = start_indices.get(var, (0,0))
            size1, size2 = data.shape
            d[var][start1 : start1 + size1, start2 : start2 + size2] = data
        else:
            istart, jstart, kstart = start_indices.get(var, (0, 0,0))
            isize, jsize, ksize = data.shape
            print(max_shape)
            d[var] = np.zeros(max_shape)
            d[var][
                istart : istart + isize,
                jstart : jstart + jsize,
                kstart : kstart + ksize,
            ] = data
        d[var] = gt_storage.from_array(data=d[var], backend=BACKEND, default_origin=DEFAULT_ORIGIN, shape=d[var].shape, dtype=DTYPE_FLT)
    return d


def compare_data(exp_data, ref_data):
    assert set(exp_data.keys()) == set(
        ref_data.keys()
    ), "Entries of exp and ref dictionaries don't match"
    for key in ref_data:
        print('comparing', key)
        atol=1e-16
        rtol=1e-16
        ind = np.array(
            np.nonzero(~np.isclose(exp_data[key].data, ref_data[key].data, equal_nan=True, atol=atol, rtol=rtol))
        )
        if ind.size > 0:
            i = tuple(ind[:, 0])
            print("FAIL at ", key, i, exp_data[key][i], ref_data[key][i],  exp_data[key][i] - ref_data[key][i])

        #for i in range(ref_data[key].shape[0]):
        #    for j in range(ref_data[key].shape[1]):
        #        for k in range(ref_data[key].shape[2]):
        #            ref = ref_data[key][i,j,k]
        #            val = exp_data[key][i,j,k]
        #            if ref != val:
        #                print(i,j,k, val, ref, val - ref)
        assert np.allclose(exp_data[key].data, ref_data[key].data, equal_nan=False, atol=atol, rtol=rtol), (
            "Data does not match for field " + key
        )
        
@gtscript.stencil(backend=BACKEND)
def update_dwind_stencil(
    u: FIELD_FLT,
    v: FIELD_FLT,
    u_dt: FIELD_FLT,
    v_dt: FIELD_FLT,
    vlon1: FIELD_FLTIJ,
    vlon2: FIELD_FLTIJ,
    vlon3: FIELD_FLTIJ,
    vlat1: FIELD_FLTIJ,
    vlat2: FIELD_FLTIJ,
    vlat3: FIELD_FLTIJ,
    es1_1: FIELD_FLTIJ,
    es2_1: FIELD_FLTIJ,
    es3_1: FIELD_FLTIJ,
    ew1_2: FIELD_FLTIJ,
    ew2_2: FIELD_FLTIJ,
    ew3_2: FIELD_FLTIJ,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        #is -1 : ie + 1; js - 1: je + 1
        v3_1 = u_dt * vlon1 + v_dt * vlat1
        v3_2 = u_dt * vlon2 + v_dt * vlat2
        v3_3 = u_dt * vlon3 + v_dt * vlat3
        # is - 1: ie + 1 ; js: je + 1
        ue_1 = v3_1[0, -1, 0]+ v3_1
        ue_2 = v3_2[0, -1, 0]+ v3_2
        ue_3 = v3_3[0, -1, 0]+ v3_3
        # is: ie + 1 ; js - 1: je + 1
        ve_1 = v3_1[-1, 0, 0] + v3_1
        ve_2 = v3_2[-1, 0, 0] + v3_2
        ve_3 = v3_3[-1, 0, 0] + v3_3
        # edges
        # is: ie; js:je+1
        u = u + dt5 * (ue_1 * es1_1 + ue_2 * es2_1 + ue_3 * es3_1)
        # is: ie+1; js:je
        v = v + dt5 * (ve_1 * ew1_2 + ve_2 * ew2_2 + ve_3 * ew3_2)
def update_dwind_phys(data):
    print(data.keys())
    dt5 = 0.5 * data["dt"]
    im2 = (data["npx"] - 1) / 2
    jm2 = (data["npy"] - 1) / 2
    # assert grid_type < 3
    print(data["vlon1"].shape)
    print(type(data["vlon1"]))
    print(data["es1_1"].shape)
    print(data["ew2_2"].shape)
    update_dwind_stencil(data["u"], data["v"], data["u_dt"], data["v_dt"], data["vlon1"], data["vlon2"], data["vlon3"], data["vlat1"], data["vlat2"], data["vlat3"], data["es1_1"], data["es2_1"], data["es3_1"], data["ew1_2"],  data["ew2_2"],  data["ew3_2"], dt5, origin=(3, 3, 0))
# "UpdateDWindsPhys-IN"
# edge_vect_e, edge_vect_<n,s,w>
# es, ew, ied, is, isd, je, jed, js, jsd, npx, npy, npz, regional,nested
# vlat, vlon
# u, v, u_dt, v_dt

# UpdateDWindsPhys-OUT
# u, v, u_dt, v_dt
IN_VARS = ["u", "v", "u_dt", "v_dt", "npx", "npy", "vlat", "vlon", "es","ew"]
# vlon(is_2d-2:ie_2d+2,js_2d-2:je_2d+2,3)
# %ew(3,isd_2d:ied_2d+1,jsd_2d:jed_2d,  2)
# %es(3,isd_2d:ied_2d  ,jsd_2d:jed_2d+1,2)

OUT_VARS = ["u", "v", "u_dt", "v_dt"]
for tile in range(6):

    if SELECT_SP is not None:
        if tile != SELECT_SP["tile"]:
            continue

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        "c12_6ranks_baroclinic_dycore_microphysics",
        "Generator_rank" + str(tile),
    )
    in_savepoint = serializer.get_savepoint("UpdateDWindsPhys-IN")[0]
    out_savepoint = serializer.get_savepoint("UpdateDWindsPhys-OUT")[0]

    print("> running ", f"tile-{tile}", in_savepoint)
    fortran2py_index_offset = 2
    jsd =  serializer.read("jsd", in_savepoint)[0] + fortran2py_index_offset
    jed =  serializer.read("jed", in_savepoint)[0] + fortran2py_index_offset
    isd =  serializer.read("isd", in_savepoint)[0] + fortran2py_index_offset
    ied =  serializer.read("ied", in_savepoint)[0] + fortran2py_index_offset
    npz =  serializer.read("npz", in_savepoint)[0] + fortran2py_index_offset
    max_shape = (ied - isd + 2, jed - jsd + 2, npz+1)
    start_indices = {"vlon": (isd + 1, jsd + 1), "vlat": (isd+1, jsd+1)}
    # read serialized input data
    in_data = storage_dict_from_var_list(IN_VARS, serializer, in_savepoint, max_shape, start_indices)
    in_data["dt"] = 225.0 
    # run Python version
    update_dwind_phys(in_data)
    out_data = {key: value for key, value in in_data.items() if key in OUT_VARS}
    # read serialized output data
    ref_data = storage_dict_from_var_list(OUT_VARS, serializer, out_savepoint, max_shape, start_indices)
    compare_data(out_data, ref_data)
