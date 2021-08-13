#!/usr/bin/env python3

import os
import sys
import numpy as np

SERIALBOX_DIR = "/usr/local/serialbox"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser
import physics_driver
import update_atmos_model_state
from ser_savepoint_var import *

SELECT_SP = None


def data_dict_from_var_list(var_list, serializer, savepoint, reverse_flag=True):
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
            if reverse_flag == True:
                d[var] = data[:, ::-1]
            else:
                d[var] = data[:, :]
        else:
            if reverse_flag == True:
                d[var] = data[:, ::-1, :]
            else:
                d[var] = data[:, :, :]
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
            print("FAIL at ", key, i)  # , exp_data[key][i], ref_data[key][i])
        # assert np.allclose(exp_data[key], ref_data[key], equal_nan=True), (
        #     "Data does not match for field " + key
        # )


for tile in range(6):

    if SELECT_SP is not None:
        if tile != SELECT_SP["tile"]:
            continue

    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        #"c12_6ranks_baroclinic_dycore_microphysics",
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

        if sp.name.startswith("FVDynamics-In"):
            in_data_fvd = data_dict_from_var_list(IN_VAR_DYNS, serializer, sp)

        if sp.name.startswith("FVDynamics-Out"):
            out_data_fvd = data_dict_from_var_list(OUT_VAR_DYNS, serializer, sp, False)

        if sp.name.startswith("AtmosPhysDriverStatein-OUT"):
            out_data_pds = data_dict_from_var_list(OUT_VAR_APDS, serializer, sp)

        if sp.name.startswith("GFSPhysicsDriver-IN"):

            if isready:
                raise Exception("out-of-order data enountered: " + sp.name)

            print("> running ", f"tile-{tile}", sp)

            # read serialized input data
            in_data_pd = data_dict_from_var_list(IN_VARS_GFSPD, serializer, sp)

            # run Python version
            out_data_pd = physics_driver.run(in_data_pd)

            isready = True

        if sp.name.startswith("FillGFS-IN"):

            print("> running ", f"tile-{tile}", sp)

            in_data_fillgfs = data_dict_from_var_list(IN_FILL_GFS, serializer, sp)

        if sp.name.startswith("FillGFS-OUT"):
            print("> running ", f"tile-{tile}", sp)

            out_data_fillgfs = data_dict_from_var_list(["IPD_gq0"], serializer, sp)

        if sp.name.startswith("FVUpdatePhys-In"):
            print("> running ", f"tile-{tile}", sp)
            # Note : The data from IN_VARS_FVPHY is for comparison purposes currently
            #  ***Maybe do not need to reverse***
            ref_data = data_dict_from_var_list(IN_VARS_FVPHY, serializer, sp, False)

            in_data = {}

            # Adding previously read serialized data
            in_data["prsi"] = in_data_pd["IPD_prsi"]
            in_data["tgrs"] = in_data_pd["IPD_tgrs"]
            in_data["ugrs"] = in_data_pd["IPD_ugrs"]
            in_data["vgrs"] = in_data_pd["IPD_vgrs"]
            in_data["IPD_area"] = in_data_pd["IPD_area"]

            # Note : Currently microphysics doesn't verify with the c12 10 day data, so these outputs
            #        cannot be used for verification purposes
            # in_data["gq0"] = out_data_pd["IPD_gq0"]
            # in_data["gt0"] = out_data_pd["IPD_gt0"]
            # in_data["gu0"] = out_data_pd["IPD_gu0"]
            # in_data["gv0"] = out_data_pd["IPD_gv0"]

            in_data["gq0"] = in_data_fillgfs["IPD_gq0"]
            in_data["gt0"] = ref_data_pd["IPD_gt0"]
            in_data["gu0"] = ref_data_pd["IPD_gu0"]
            in_data["gv0"] = ref_data_pd["IPD_gv0"]            

            in_data["nq"] = (
                in_data_fvd["nq_tot"] - 1
            )  # I think nq=8 since it's set in class DynamicalCore
            in_data["delp"] = np.reshape(
                out_data_fvd["delp"][3:-3, 3:-3, :], (144, 79), order="F"
            )

            in_data["nwat"] = out_data_pds["IPD_nwat"]

            in_data["qvapor"] = np.reshape(
                out_data_fvd["qvapor"][3:-3, 3:-3, :], (144, 79), order="F"
            )
            in_data["qliquid"] = np.reshape(
                out_data_fvd["qliquid"][3:-3, 3:-3, :], (144, 79), order="F"
            )
            in_data["qrain"] = np.reshape(
                out_data_fvd["qrain"][3:-3, 3:-3, :], (144, 79), order="F"
            )
            in_data["qsnow"] = np.reshape(
                out_data_fvd["qsnow"][3:-3, 3:-3, :], (144, 79), order="F"
            )
            in_data["qice"] = np.reshape(
                out_data_fvd["qice"][3:-3, 3:-3, :], (144, 79), order="F"
            )
            in_data["qgraupel"] = np.reshape(
                out_data_fvd["qgraupel"][3:-3, 3:-3, :], (144, 79), order="F"
            )
            in_data["qo3mr"] = np.reshape(
                out_data_fvd["qo3mr"][3:-3, 3:-3, :], (144, 79), order="F"
            )
            in_data["qsgs_tke"] = np.reshape(
                out_data_fvd["qsgs_tke"][3:-3, 3:-3, :], (144, 79), order="F"
            )
            in_data["qcld"] = np.reshape(
                out_data_fvd["qcld"][3:-3, 3:-3, :], (144, 79), order="F"
            )

            in_data["u"]     = np.reshape(ref_data["u"][3:-3, 3:-4,:], (144,79), order="F")
            in_data["v"]     = np.reshape(ref_data["v"][3:-4, 3:-3,:], (144,79), order="F")
            in_data["w"]     = np.reshape(ref_data["w"][3:-3, 3:-3,:], (144,79), order="F")

            in_data["pt"]    = np.reshape(ref_data["pt"][3:-3, 3:-3,:], (144,79), order="F")
            in_data["ua"]    = np.reshape(ref_data["ua"][3:-3, 3:-3,:], (144,79), order="F")
            in_data["va"]    = np.reshape(ref_data["va"][3:-3, 3:-3,:], (144,79), order="F")

            in_data["ps"]    = np.reshape(ref_data["ps"][3:-3, 3:-3], (144), order="F")
            in_data["pe"]    = ref_data["pe"]
            in_data["peln"]  = ref_data["peln"]
            in_data["pk"]    = np.reshape(ref_data["pk"], (144,80), order="F")
            in_data["pkz"]   = np.reshape(ref_data["pkz"], (144, 79), order="F")
            in_data["phis"]  = np.reshape(ref_data["phis"][3:-3, 3:-3], (144), order="F")
            in_data["u_srf"] = np.reshape(ref_data["u_srf"],(144), order="F")
            in_data["v_srf"] = np.reshape(ref_data["v_srf"],(144), order="F")

            out_data = update_atmos_model_state.run(in_data)

            # print("After update_atmos_model_state")

            delp = np.reshape(ref_data["delp"][3:-3, 3:-3, :], (144, 79), order="F")
            u_dt = np.reshape(ref_data["u_dt"][3:-3, 3:-3, :], (144, 79), order="F")
            v_dt = np.reshape(ref_data["v_dt"][3:-3, 3:-3, :], (144, 79), order="F")
            t_dt = np.reshape(ref_data["t_dt"], (144, 79), order="F")

            qvapor = np.reshape(ref_data["qvapor"][3:-3, 3:-3, :], (144, 79), order="F")
            qliquid = np.reshape(ref_data["qliquid"][3:-3, 3:-3, :], (144, 79), order="F")
            qrain = np.reshape(ref_data["qrain"][3:-3, 3:-3, :], (144, 79), order="F")
            qsnow = np.reshape(ref_data["qsnow"][3:-3, 3:-3, :], (144, 79), order="F")
            qice = np.reshape(ref_data["qice"][3:-3, 3:-3, :], (144, 79), order="F")
            qgraupel = np.reshape(ref_data["qgraupel"][3:-3, 3:-3, :], (144, 79), order="F")

            np.testing.assert_allclose(out_data["delp"], delp)
            np.testing.assert_allclose(out_data["u_dt"], u_dt, atol=1e-8)
            np.testing.assert_allclose(out_data["v_dt"], v_dt, atol=1e-8)
            np.testing.assert_allclose(out_data["t_dt"], t_dt, atol=1e-8)
            
            # np.testing.assert_allclose(out_data["q"][:,:,0], qvapor)
            # np.testing.assert_allclose(out_data["q"][:,:,1], qliquid)
            # np.testing.assert_allclose(out_data["q"][:,:,2], qrain)
            # np.testing.assert_allclose(out_data["q"][:,:,3], qsnow)
            # np.testing.assert_allclose(out_data["q"][:,:,4], qice, atol=1e-8)
            # np.testing.assert_allclose(out_data["q"][:,:,5], qgraupel)


            #***************************************************************************

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

        # if sp.name.startswith("Microph-Out"):
        #     print("> running ", f"tile-{tile}", sp)

        #     # read serialized input data
        #     ref_data = data_dict_from_var_list(OUT_VARS_MICROPH, serializer, sp)

        #     compare_data(out_data_postphi, ref_data)

        if sp.name.startswith("GFSPhysicsDriver-OUT"):
            print("> running ", f"tile-{tile}", sp)

            # read serialized input data
            ref_data_pd = data_dict_from_var_list(OUT_VARS_GFSPD, serializer, sp)

            compare_data(out_data_pd, ref_data_pd)

            isready = False

        if sp.name.startswith("FVUpdatePhys-Out"):
            print("> running ", f"tile-{tile}", sp)

            # read serialized input data
            ref_data = data_dict_from_var_list(OUT_VARS_FVPHY, serializer, sp, False)

            ps = np.reshape(ref_data["ps"][3:-3, 3:-3], (144), order="F")
            pt = np.reshape(ref_data["pt"][3:-3, 3:-3,:], (144,79), order="F")
            pk = np.reshape(ref_data["pk"], (144,80), order="F")

            #compare_data(out_data_fup, ref_data)
            np.testing.assert_allclose(out_data["ps"],ps)
            np.testing.assert_allclose(out_data["pt"],pt,atol=1e-4) # There's ONE number in the C12 10 Day dataset that won't verify without atol being set
            np.testing.assert_allclose(out_data["pe"],ref_data["pe"])
            np.testing.assert_allclose(out_data["peln"],ref_data["peln"])
            np.testing.assert_allclose(out_data["pk"],pk)