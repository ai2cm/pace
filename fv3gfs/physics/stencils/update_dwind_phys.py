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
    region,
)
from fv3gfs.physics.global_config import *

sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser

SELECT_SP = None


def add_composite_vvar_storage(d, var, data3d, max_shape, start_indices):
    shape = data3d.shape
    start1, start2 = start_indices.get(var, (0, 0))
    size1, size2 = data3d.shape[0:2]
    for s in range(data3d.shape[2]):
        buffer = np.zeros(max_shape[0:2])
        buffer[start1 : start1 + size1, start2 : start2 + size2] = np.squeeze(
            data3d[:, :, s]
        )
        d[var + str(s + 1)] = gt_storage.from_array(
            data=buffer,
            backend=BACKEND,
            default_origin=(start1, start2),
            shape=max_shape[0:2],
            dtype=DTYPE_FLT,
        )


def add_composite_evar_storage(d, var, data4d, max_shape, start_indices):
    shape = data4d.shape
    start1, start2 = start_indices.get(var, (0, 0))
    size1, size2 = data4d.shape[1:3]
    for s in range(data4d.shape[0]):
        for t in range(data4d.shape[3]):
            buffer = np.zeros(max_shape[0:2])
            buffer[start1 : start1 + size1, start2 : start2 + size2] = np.squeeze(
                data4d[s, :, :, t]
            )
            d[var + str(s + 1) + "_" + str(t + 1)] = gt_storage.from_array(
                data=buffer,
                backend=BACKEND,
                default_origin=(start1, start2),
                shape=max_shape[0:2],
                dtype=DTYPE_FLT,
            )


def edge_vector_storage(d, var, axis):
    if axis == 1:
        default_origin = (0, 0)
        d[var] = d[var][np.newaxis, ...]
        d[var] = np.repeat(d[var], max_shape[0], axis=0)
    if axis == 0:
        default_origin = (0,)
    d[var] = gt_storage.from_array(
        data=d[var],
        backend=BACKEND,
        default_origin=default_origin,
        shape=d[var].shape,
        dtype=DTYPE_FLT,
    )


def storage_dict_from_var_list(
    var_list, serializer, savepoint, max_shape, start_indices, axes={}
):
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
            start1 = start_indices.get(var, 0)
            size1 = data.shape[0]
            axis = axes.get(var, 2)
            d[var] = np.zeros(max_shape[axis])
            d[var][start1 : start1 + size1] = data
            if "edge_vect" in var:
                edge_vector_storage(d, var, axis)
                continue
        elif len(data.shape) == 2:
            d[var] = np.zeros(max_shape[0:2])
            start1, start2 = start_indices.get(var, (0, 0))
            size1, size2 = data.shape
            d[var][start1 : start1 + size1, start2 : start2 + size2] = data
        else:
            start1, start2, start3 = start_indices.get(var, (0, 0, 0))
            size1, size2, size3 = data.shape
            d[var] = np.zeros(max_shape)
            d[var][
                start1 : start1 + size1,
                start2 : start2 + size2,
                start3 : start3 + size3,
            ] = data
        d[var] = gt_storage.from_array(
            data=d[var],
            backend=BACKEND,
            default_origin=DEFAULT_ORIGIN,
            shape=d[var].shape,
            dtype=DTYPE_FLT,
        )
    return d


def compare_data(exp_data, ref_data):
    assert set(exp_data.keys()) == set(
        ref_data.keys()
    ), "Entries of exp and ref dictionaries don't match"
    for key in ref_data:
        print("comparing", key)
        atol = 1e-20
        rtol = 1e-20
        ind = np.array(
            np.nonzero(
                ~np.isclose(
                    exp_data[key].data,
                    ref_data[key].data,
                    equal_nan=True,
                    atol=atol,
                    rtol=rtol,
                )
            )
        )
        if ind.size > 0:
            i = tuple(ind[:, 0])
            print(
                "FAIL at ",
                key,
                i,
                exp_data[key][i],
                ref_data[key][i],
                exp_data[key][i] - ref_data[key][i],
            )
        failcount = 0
        for i in range(ref_data[key].shape[0]):
            for j in range(ref_data[key].shape[1]):
                for k in range(ref_data[key].shape[2]):
                    ref = ref_data[key][i, j, k]
                    val = exp_data[key][i, j, k]
                    if ref != val:
                        print(i, j, k, val, ref, val - ref)
                        failcount += 1
        if failcount > 0:
            print("FAILED count", failcount)
        assert np.allclose(
            exp_data[key].data,
            ref_data[key].data,
            equal_nan=False,
            atol=atol,
            rtol=rtol,
        ), (
            "Data does not match for field " + key
        )


# TODO use regions to merge into one stencil
@gtscript.stencil(backend=BACKEND)
def update_dwind_prep_stencil(
    u_dt: FIELD_FLT,
    v_dt: FIELD_FLT,
    vlon1: FIELD_FLTIJ,
    vlon2: FIELD_FLTIJ,
    vlon3: FIELD_FLTIJ,
    vlat1: FIELD_FLTIJ,
    vlat2: FIELD_FLTIJ,
    vlat3: FIELD_FLTIJ,
    ue_1: FIELD_FLT,
    ue_2: FIELD_FLT,
    ue_3: FIELD_FLT,
    ve_1: FIELD_FLT,
    ve_2: FIELD_FLT,
    ve_3: FIELD_FLT,
):
    with computation(PARALLEL), interval(...):
        # is -1 : ie + 1; js - 1: je + 1
        v3_1 = u_dt * vlon1 + v_dt * vlat1
        v3_2 = u_dt * vlon2 + v_dt * vlat2
        v3_3 = u_dt * vlon3 + v_dt * vlat3
        # is - 1: ie + 1 ; js: je + 1
        ue_1 = v3_1[0, -1, 0] + v3_1
        ue_2 = v3_2[0, -1, 0] + v3_2
        ue_3 = v3_3[0, -1, 0] + v3_3
        # is: ie + 1 ; js - 1: je + 1
        ve_1 = v3_1[-1, 0, 0] + v3_1
        ve_2 = v3_2[-1, 0, 0] + v3_2
        ve_3 = v3_3[-1, 0, 0] + v3_3


# edges
@gtscript.stencil(backend=BACKEND)
def update_dwind_y_edge_south_stencil(
    ve_1: FIELD_FLT,
    ve_2: FIELD_FLT,
    ve_3: FIELD_FLT,
    vt_1: FIELD_FLT,
    vt_2: FIELD_FLT,
    vt_3: FIELD_FLT,
    edge_vect: FIELD_FLTIJ,
):
    with computation(PARALLEL), interval(...):
        vt_1 = edge_vect * ve_1[0, 1, 0] + (1.0 - edge_vect) * ve_1
        vt_2 = edge_vect * ve_2[0, 1, 0] + (1.0 - edge_vect) * ve_2
        vt_3 = edge_vect * ve_3[0, 1, 0] + (1.0 - edge_vect) * ve_3


@gtscript.stencil(backend=BACKEND)
def update_dwind_y_edge_north_stencil(
    ve_1: FIELD_FLT,
    ve_2: FIELD_FLT,
    ve_3: FIELD_FLT,
    vt_1: FIELD_FLT,
    vt_2: FIELD_FLT,
    vt_3: FIELD_FLT,
    edge_vect: FIELD_FLTIJ,
):
    with computation(PARALLEL), interval(...):
        vt_1 = edge_vect * ve_1[0, -1, 0] + (1.0 - edge_vect) * ve_1
        vt_2 = edge_vect * ve_2[0, -1, 0] + (1.0 - edge_vect) * ve_2
        vt_3 = edge_vect * ve_3[0, -1, 0] + (1.0 - edge_vect) * ve_3


@gtscript.stencil(backend=BACKEND)
def update_dwind_x_edge_west_stencil(
    ue_1: FIELD_FLT,
    ue_2: FIELD_FLT,
    ue_3: FIELD_FLT,
    ut_1: FIELD_FLT,
    ut_2: FIELD_FLT,
    ut_3: FIELD_FLT,
    edge_vect: FIELD_FLTI,
):
    with computation(PARALLEL), interval(...):
        ut_1 = edge_vect * ue_1[1, 0, 0] + (1.0 - edge_vect) * ue_1
        ut_2 = edge_vect * ue_2[1, 0, 0] + (1.0 - edge_vect) * ue_2
        ut_3 = edge_vect * ue_3[1, 0, 0] + (1.0 - edge_vect) * ue_3


@gtscript.stencil(backend=BACKEND)
def update_dwind_x_edge_east_stencil(
    ue_1: FIELD_FLT,
    ue_2: FIELD_FLT,
    ue_3: FIELD_FLT,
    ut_1: FIELD_FLT,
    ut_2: FIELD_FLT,
    ut_3: FIELD_FLT,
    edge_vect: FIELD_FLTI,
):
    with computation(PARALLEL), interval(...):
        ut_1 = edge_vect * ue_1[-1, 0, 0] + (1.0 - edge_vect) * ue_1
        ut_2 = edge_vect * ue_2[-1, 0, 0] + (1.0 - edge_vect) * ue_2
        ut_3 = edge_vect * ue_3[-1, 0, 0] + (1.0 - edge_vect) * ue_3


@gtscript.stencil(backend=BACKEND)
def copy3_stencil(
    in_field1: FIELD_FLT,
    in_field2: FIELD_FLT,
    in_field3: FIELD_FLT,
    out_field1: FIELD_FLT,
    out_field2: FIELD_FLT,
    out_field3: FIELD_FLT,
):
    with computation(PARALLEL), interval(...):
        out_field1 = in_field1
        out_field2 = in_field2
        out_field3 = in_field3


@gtscript.stencil(backend=BACKEND)
def update_uwind_stencil(
    u: FIELD_FLT,
    es1_1: FIELD_FLTIJ,
    es2_1: FIELD_FLTIJ,
    es3_1: FIELD_FLTIJ,
    ue_1: FIELD_FLT,
    ue_2: FIELD_FLT,
    ue_3: FIELD_FLT,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        # is: ie; js:je+1
        u = u + dt5 * (ue_1 * es1_1 + ue_2 * es2_1 + ue_3 * es3_1)


@gtscript.stencil(backend=BACKEND)
def update_vwind_stencil(
    v: FIELD_FLT,
    ew1_2: FIELD_FLTIJ,
    ew2_2: FIELD_FLTIJ,
    ew3_2: FIELD_FLTIJ,
    ve_1: FIELD_FLT,
    ve_2: FIELD_FLT,
    ve_3: FIELD_FLT,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        # is: ie+1; js:je
        v = v + dt5 * (ve_1 * ew1_2 + ve_2 * ew2_2 + ve_3 * ew3_2)


def make_storage_from_shape(shape):
    return gt_storage.zeros(
        backend=BACKEND,
        default_origin=DEFAULT_ORIGIN,
        shape=shape,
        dtype=DTYPE_FLT,
    )


def read_index_var(index_var, savepoint):
    fortran2py_index_offset = 2
    return int(serializer.read(index_var, savepoint)[0] + fortran2py_index_offset)


def update_dwind_phys(data):
    dt5 = 0.5 * data["dt"]
    im2 = int((data["npx"] - 1) / 2) + 2
    jm2 = int((data["npy"] - 1) / 2) + 2
    # assert grid_type < 3
    shape = data["u"].shape
    ue_1 = make_storage_from_shape(shape)
    ue_2 = make_storage_from_shape(shape)
    ue_3 = make_storage_from_shape(shape)
    ut_1 = make_storage_from_shape(shape)
    ut_2 = make_storage_from_shape(shape)
    ut_3 = make_storage_from_shape(shape)
    ve_1 = make_storage_from_shape(shape)
    ve_2 = make_storage_from_shape(shape)
    ve_3 = make_storage_from_shape(shape)
    vt_1 = make_storage_from_shape(shape)
    vt_2 = make_storage_from_shape(shape)
    vt_3 = make_storage_from_shape(shape)
    update_dwind_prep_stencil(
        data["u_dt"],
        data["v_dt"],
        data["vlon1"],
        data["vlon2"],
        data["vlon3"],
        data["vlat1"],
        data["vlat2"],
        data["vlat3"],
        ue_1,
        ue_2,
        ue_3,
        ve_1,
        ve_2,
        ve_3,
        origin=(HALO - 1, HALO - 1, 0),
        domain=(data["nic"] + 2, data["njc"] + 2, data["npz"]),
    )

    # if grid.west_edge
    if data["is"] == HALO:
        if data["js"] <= jm2:
            je_lower = min(jm2, data["je"])
            origin_lower = (HALO, HALO, 0)
            domain_lower = (1, je_lower - data["js"] + 1, data["npz"])
            if domain_lower[1] > 0:
                update_dwind_y_edge_south_stencil(
                    ve_1,
                    ve_2,
                    ve_3,
                    vt_1,
                    vt_2,
                    vt_3,
                    data["edge_vect_w"],
                    origin=origin_lower,
                    domain=domain_lower,
                )

        if data["je"] > jm2:
            js_upper = max(jm2 + 1, data["js"])
            origin_upper = (HALO, js_upper, 0)
            domain_upper = (1, data["je"] - js_upper + 1, data["npz"])

            if domain_upper[1] > 0:
                update_dwind_y_edge_north_stencil(
                    ve_1,
                    ve_2,
                    ve_3,
                    vt_1,
                    vt_2,
                    vt_3,
                    data["edge_vect_w"],
                    origin=origin_upper,
                    domain=domain_upper,
                )
                copy3_stencil(
                    vt_1,
                    vt_2,
                    vt_3,
                    ve_1,
                    ve_2,
                    ve_3,
                    origin=origin_upper,
                    domain=domain_upper,
                )
        if data["js"] <= jm2 and domain_lower[1] > 0:
            copy3_stencil(
                vt_1,
                vt_2,
                vt_3,
                ve_1,
                ve_2,
                ve_3,
                origin=origin_lower,
                domain=domain_lower,
            )

    # if grid.east_edge
    if data["ie"] - 1 == data["npx"]:
        i_origin = max_shape[0] - HALO - 1

        if data["js"] <= jm2:
            je_lower = min(jm2, data["je"])
            origin_lower = (i_origin, HALO, 0)
            domain_lower = (1, je_lower - data["js"] + 1, data["npz"])
            if domain_lower[1] > 0:
                update_dwind_y_edge_south_stencil(
                    ve_1,
                    ve_2,
                    ve_3,
                    vt_1,
                    vt_2,
                    vt_3,
                    data["edge_vect_e"],
                    origin=origin_lower,
                    domain=domain_lower,
                )
        if data["je"] > jm2:
            js_upper = max(jm2 + 1, data["js"])
            origin_upper = (i_origin, js_upper, 0)
            domain_upper = (1, data["je"] - js_upper + 1, data["npz"])
            if domain_upper[1] > 0:
                update_dwind_y_edge_north_stencil(
                    ve_1,
                    ve_2,
                    ve_3,
                    vt_1,
                    vt_2,
                    vt_3,
                    data["edge_vect_e"],
                    origin=origin_upper,
                    domain=domain_upper,
                )
                copy3_stencil(
                    vt_1,
                    vt_2,
                    vt_3,
                    ve_1,
                    ve_2,
                    ve_3,
                    origin=origin_upper,
                    domain=domain_upper,
                )
        if data["js"] <= jm2 and domain_lower[1] > 0:
            copy3_stencil(
                vt_1,
                vt_2,
                vt_3,
                ve_1,
                ve_2,
                ve_3,
                origin=origin_lower,
                domain=domain_lower,
            )

    # if grid.south_edge
    if data["js"] == HALO:
        if data["is"] <= im2:
            ie_lower = min(im2, data["ie"])
            origin_lower = (HALO, HALO, 0)
            domain_lower = (ie_lower - data["is"] + 1, 1, data["npz"])
            if domain_lower[0] > 0:
                update_dwind_x_edge_west_stencil(
                    ue_1,
                    ue_2,
                    ue_3,
                    ut_1,
                    ut_2,
                    ut_3,
                    data["edge_vect_s"],
                    origin=origin_lower,
                    domain=domain_lower,
                )
        if data["ie"] > im2:
            is_upper = max(im2 + 1, data["is"])
            origin_upper = (is_upper, HALO, 0)
            domain_upper = (data["ie"] - is_upper + 1, 1, data["npz"])
            if domain_upper[0] > 0:
                update_dwind_x_edge_east_stencil(
                    ue_1,
                    ue_2,
                    ue_3,
                    ut_1,
                    ut_2,
                    ut_3,
                    data["edge_vect_s"],
                    origin=origin_upper,
                    domain=domain_upper,
                )
                copy3_stencil(
                    ut_1,
                    ut_2,
                    ut_3,
                    ue_1,
                    ue_2,
                    ue_3,
                    origin=origin_upper,
                    domain=domain_upper,
                )
        if data["is"] <= im2 and domain_lower[0] > 0:
            copy3_stencil(
                ut_1,
                ut_2,
                ut_3,
                ue_1,
                ue_2,
                ue_3,
                origin=origin_lower,
                domain=domain_lower,
            )

    # if grid.north_edge
    if data["je"] - 1 == data["npy"]:
        j_origin = max_shape[1] - HALO - 1
        if data["is"] < im2:
            ie_lower = min(im2, data["ie"])
            origin_lower = (HALO, j_origin, 0)
            domain_lower = (ie_lower - data["is"] + 1, 1, data["npz"])
            if domain_lower[0] > 0:
                update_dwind_x_edge_west_stencil(
                    ue_1,
                    ue_2,
                    ue_3,
                    ut_1,
                    ut_2,
                    ut_3,
                    data["edge_vect_n"],
                    origin=origin_lower,
                    domain=domain_lower,
                )

        if data["je"] >= jm2:
            is_upper = max(im2 + 1, data["is"])
            origin_upper = (is_upper, j_origin, 0)
            domain_upper = (data["ie"] - is_upper + 1, 1, data["npz"])
            if domain_upper[0] > 0:
                update_dwind_x_edge_east_stencil(
                    ue_1,
                    ue_2,
                    ue_3,
                    ut_1,
                    ut_2,
                    ut_3,
                    data["edge_vect_n"],
                    origin=origin_upper,
                    domain=domain_upper,
                )
                copy3_stencil(
                    ut_1,
                    ut_2,
                    ut_3,
                    ue_1,
                    ue_2,
                    ue_3,
                    origin=origin_upper,
                    domain=domain_upper,
                )
        if data["is"] < im2 and domain_lower[0] > 0:
            copy3_stencil(
                ut_1,
                ut_2,
                ut_3,
                ue_1,
                ue_2,
                ue_3,
                origin=origin_lower,
                domain=domain_lower,
            )
    update_uwind_stencil(
        data["u"],
        data["es1_1"],
        data["es2_1"],
        data["es3_1"],
        ue_1,
        ue_2,
        ue_3,
        dt5,
        origin=(HALO, HALO, 0),
        domain=(data["nic"], data["njc"] + 1, data["npz"]),
    )
    update_vwind_stencil(
        data["v"],
        data["ew1_2"],
        data["ew2_2"],
        data["ew3_2"],
        ve_1,
        ve_2,
        ve_3,
        dt5,
        origin=(HALO, HALO, 0),
        domain=(data["nic"] + 1, data["njc"], data["npz"]),
    )


IN_VARS = [
    "u",
    "v",
    "u_dt",
    "v_dt",
    "npx",
    "npy",
    "vlat",
    "vlon",
    "es",
    "ew",
    "edge_vect_e",
    "edge_vect_w",
    "edge_vect_s",
    "edge_vect_n",
]

OUT_VARS = ["u", "v"]
for tile in range(6):

    if SELECT_SP is not None:
        if tile != SELECT_SP["tile"]:
            continue

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        "../../../examples/c12_6ranks_baroclinic_dycore_microphysics_day_10",
        "Generator_rank" + str(tile),
    )
    in_savepoint = serializer.get_savepoint("UpdateDWindsPhys-IN")[0]
    out_savepoint = serializer.get_savepoint("UpdateDWindsPhys-OUT")[0]

    print("> running ", f"tile-{tile}", in_savepoint)
    fortran2py_index_offset = 2
    index_data = {}
    for index_var in ["isd", "ied", "jsd", "jed", "is", "js", "je"]:
        index_data[index_var] = read_index_var(index_var, in_savepoint)
    index_data["npz"] = serializer.read("npz", in_savepoint)[0]
    max_shape = (
        index_data["ied"] - index_data["isd"] + 2,
        index_data["jed"] - index_data["jsd"] + 2,
        index_data["npz"] + 1,
    )
    start_indices = {
        "vlon": (index_data["isd"] + 1, index_data["jsd"] + 1),
        "vlat": (index_data["isd"] + 1, index_data["jsd"] + 1),
    }
    axes = {"edge_vect_s": 0, "edge_vect_n": 0, "edge_vect_w": 1, "edge_vect_e": 1}
    # read serialized input data
    in_data = storage_dict_from_var_list(
        IN_VARS, serializer, in_savepoint, max_shape, start_indices, axes
    )
    in_data.update(index_data)
    # TODO, put ie in savepoint data or move to code with a grid
    # this is only valid for 6 ranks:
    in_data["ie"] = in_data["je"]
    in_data["dt"] = 225.0
    in_data["nic"] = in_data["ie"] - in_data["is"] + 1
    in_data["njc"] = in_data["je"] - in_data["js"] + 1
    in_data["npx"] = int(in_data["npx"])
    in_data["npy"] = int(in_data["npy"])

    # run Python version
    update_dwind_phys(in_data)
    out_data = {key: value for key, value in in_data.items() if key in OUT_VARS}
    # read serialized output data
    ref_data = storage_dict_from_var_list(
        OUT_VARS, serializer, out_savepoint, max_shape, start_indices, axes
    )
    compare_data(out_data, ref_data)
    print("SUCCESS tile", tile)
