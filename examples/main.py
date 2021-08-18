#!/usr/bin/env python3

import os
import sys
import numpy as np

SERIALBOX_DIR = "/usr/local/serialbox"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser
import physics_driver
from ser_savepoint_var import *

SELECT_SP = None


def data_dict_from_var_list(var_list, serializer, savepoint):
    d = {}
    for var in var_list:
        data = serializer.read(var, savepoint)
        # convert single element numpy arrays to scalars
        if data.size == 1:
            data = data.item()
            d[var] = data
        elif len(data.shape) < 2:
            d[var] = data
        elif len(data.shape) == 2:
            d[var] = data[:, ::-1]
        else:
            d[var] = data[:, ::-1, :]
    return d


def compare_data(exp_data, ref_data):
    assert set(exp_data.keys()) == set(
        ref_data.keys()
    ), "Entries of exp and ref dictionaries don't match"
    for key in ref_data:
        print(key)
        if isinstance(exp_data[key], np.ndarray):
            if exp_data[key].shape != ref_data[key].shape:
                exp_data[key] = physics_driver.storage_to_numpy(
                    exp_data[key], (144, 79), True
                )  # hard coding for now
                exp_data[key] = exp_data[key][:, np.newaxis, :]
        ind = np.array(
            np.nonzero(~np.isclose(exp_data[key], ref_data[key], equal_nan=True))
        )
        if ind.size > 0:
            i = tuple(ind[:, 0])
            print("FAIL at ", key, i, exp_data[key][i], ref_data[key][i])
        # assert np.allclose(exp_data[key], ref_data[key], equal_nan=True), (
        #     "Data does not match for field " + key
        # )


for tile in range(6):

    if SELECT_SP is not None:
        if tile != SELECT_SP["tile"]:
            continue

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        "c12_6ranks_baroclinic_dycore_microphysics_day_10",
        "Generator_rank" + str(tile),
    )

    savepoints = serializer.savepoint_list()

    isready = False
    for sp in savepoints:

        if SELECT_SP is not None:
            if sp.name != SELECT_SP["savepoint"] and sp.name != SELECT_SP[
                "savepoint"
            ].replace("-in-", "-out-"):
                continue

        if sp.name.startswith("GFSPhysicsDriver-IN"):

            if isready:
                raise Exception("out-of-order data enountered: " + sp.name)

            print("> running ", f"tile-{tile}", sp)

            # read serialized input data
            in_data = data_dict_from_var_list(IN_VARS_GFSPD, serializer, sp)

            # run Python version
            out_data = physics_driver.run(in_data, tile)

            isready = True

        # if sp.name.startswith("PrsFV3-In"):
        #     print("> running ", f"tile-{tile}", sp)

        #     # read serialized input data
        #     ref_data = data_dict_from_var_list(IN_VARS_PRS, serializer, sp)

        #     compare_data(out_data_preprs, ref_data)

        # if sp.name.startswith("PrsFV3-Out"):
        #     print("> running ", f"tile-{tile}", sp)

        #     # read serialized input data
        #     ref_data = data_dict_from_var_list(OUT_VARS_PRS, serializer, sp)

        #     compare_data(out_data_postphi, ref_data)

        # if sp.name.startswith("PhiFV3-In"):
        #     print("> running ", f"tile-{tile}", sp)

        #     # read serialized input data
        #     ref_data = data_dict_from_var_list(IN_VARS_PHI, serializer, sp)

        #     compare_data(out_data_prephi, ref_data)

        # if sp.name.startswith("Microph-In"):
        #     print("> running ", f"tile-{tile}", sp)

        #     # read serialized input data
        #     ref_data = data_dict_from_var_list(IN_VARS_MICROPH, serializer, sp)
        #     ref_data_rename = {}
        #     for key in ref_data.keys():
        #         ref_data_rename[key.replace("mph_", "")] = ref_data[key]
        #     compare_data(out_data, ref_data_rename)

        # if sp.name.startswith("Microph-Out"):
        #     print("> running ", f"tile-{tile}", sp)

        #     # read serialized input data
        #     ref_data = data_dict_from_var_list(OUT_VARS_MICROPH, serializer, sp)
        #     ref_data_rename = {}
        #     for key in ref_data.keys():
        #         ref_data_rename[key.replace("mph_", "")] = ref_data[key]
        #     compare_data(out_data, ref_data_rename)

        if sp.name.startswith("GFSPhysicsDriver-OUT"):
            print("> running ", f"tile-{tile}", sp)

            # read serialized input data
            ref_data = data_dict_from_var_list(OUT_VARS_GFSPD, serializer, sp)

            compare_data(out_data, ref_data)
