import sys
import numpy as np

from fv3gfs.physics.global_config import *
from fv3gfs.physics.global_constants import *
from ser_savepoint_var import *

SERIALBOX_DIR = "/usr/local/serialbox"
sys.path.append(SERIALBOX_DIR + "/python")
import serialbox as ser
import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
import collections
import physics_driver


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


def long_to_short_name(state):
    ArgSpec = collections.namedtuple(
        "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
    )
    arg_specs = (
        ArgSpec("qvapor", "specific_humidity", "kg/kg", intent="inout"),
        ArgSpec("qliquid", "cloud_water_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qrain", "rain_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qsnow", "snow_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qice", "cloud_ice_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qgraupel", "graupel_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qo3mr", "ozone_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qsgs_tke", "turbulent_kinetic_energy", "m**2/s**2", intent="inout"),
        ArgSpec("qcld", "cloud_fraction", "", intent="inout"),
        ArgSpec("pt", "air_temperature", "degK", intent="inout"),
        ArgSpec(
            "delp", "pressure_thickness_of_atmospheric_layer", "Pa", intent="inout"
        ),
        ArgSpec("delz", "vertical_thickness_of_atmospheric_layer", "m", intent="inout"),
        ArgSpec("peln", "logarithm_of_interface_pressure", "ln(Pa)", intent="inout"),
        ArgSpec("u", "x_wind", "m/s", intent="inout"),
        ArgSpec("v", "y_wind", "m/s", intent="inout"),
        ArgSpec("w", "vertical_wind", "m/s", intent="inout"),
        ArgSpec("ua", "eastward_wind", "m/s", intent="inout"),
        ArgSpec("va", "northward_wind", "m/s", intent="inout"),
        ArgSpec("uc", "x_wind_on_c_grid", "m/s", intent="inout"),
        ArgSpec("vc", "y_wind_on_c_grid", "m/s", intent="inout"),
        ArgSpec("q_con", "total_condensate_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("pe", "interface_pressure", "Pa", intent="inout"),
        ArgSpec("phis", "surface_geopotential", "m^2 s^-2", intent="in"),
        ArgSpec(
            "pk",
            "interface_pressure_raised_to_power_of_kappa",
            "unknown",
            intent="inout",
        ),
        ArgSpec(
            "pkz",
            "layer_mean_pressure_raised_to_power_of_kappa",
            "unknown",
            intent="inout",
        ),
        ArgSpec("ps", "surface_pressure", "Pa", intent="inout"),
        ArgSpec("omga", "vertical_pressure_velocity", "Pa/s", intent="inout"),
        ArgSpec("ak", "atmosphere_hybrid_a_coordinate", "Pa", intent="in"),
        ArgSpec("bk", "atmosphere_hybrid_b_coordinate", "", intent="in"),
        ArgSpec("mfxd", "accumulated_x_mass_flux", "unknown", intent="inout"),
        ArgSpec("mfyd", "accumulated_y_mass_flux", "unknown", intent="inout"),
        ArgSpec("cxd", "accumulated_x_courant_number", "", intent="inout"),
        ArgSpec("cyd", "accumulated_y_courant_number", "", intent="inout"),
        ArgSpec(
            "diss_estd",
            "dissipation_estimate_from_heat_source",
            "unknown",
            intent="inout",
        ),
    )
    state_out = {}
    for sp in arg_specs:
        arg_name, standard_name, units, intent = sp
        state_out[arg_name] = state[standard_name]
    return state_out


def atmos_phys_driver_statein(state, tile):
    NQ = 8  # state.nq_tot - spec.namelist.dnats
    dnats = 1  # spec.namelist.dnats
    nwat = 6  # spec.namelist.nwat
    p00 = 1.0e5
    ptop = state["ak"][0]
    pktop = (ptop / p00) ** KAPPA
    pk0inv = (1.0 / p00) ** KAPPA
    full_shape = state["pk"].shape
    shape_2d = state["ps"].shape
    npz = state["omga"].shape[2]
    qmin = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=shape_2d,
        default_origin=(0, 0, 0),
    )
    qmin[:, :] = 1.0e-10
    phii = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    prsi = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    prsik = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    qgrs = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape + (9,),
        default_origin=(0, 0, 0, 0),
    )
    tgrs = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    ugrs = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    vgrs = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    vvl = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    prsl = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    diss_est = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=full_shape,
        default_origin=(0, 0, 0),
    )
    prsik[:, :, :] = 1.0e25
    tgrs[:, :, 0:-1] = state["pt"]
    ugrs[:, :, 0:-1] = state["ua"]
    vgrs[:, :, 0:-1] = state["va"]
    vvl[:, :, 0:-1] = state["omga"]
    prsl[:, :, 0:-1] = state["delp"]
    diss_est[:, :, 0:-1] = state["diss_estd"]
    for k in range(npz - 1, -1, -1):
        phii[:, :, k] = phii[:, :, k + 1] - state["delz"][:, :, k] * grav
    q_all = [
        state["qvapor"],
        state["qliquid"],
        state["qrain"],
        state["qice"],
        state["qsnow"],
        state["qgraupel"],
        state["qo3mr"],
        state["qsgs_tke"],
        state["qcld"],
    ]
    for l in range(NQ):
        qgrs[:, :, 0:-1, l] = q_all[l][:, :, :] * prsl[:, :, 0:-1]
    if dnats > 0:  # tracer includes none-mixing ratio var
        qgrs[:, :, 0:-1, NQ] = q_all[NQ]
    if nwat == 6:
        prsl[:, :, :] = (
            prsl[:, :, :]
            - qgrs[:, :, :, 1]
            - qgrs[:, :, :, 2]
            - qgrs[:, :, :, 3]
            - qgrs[:, :, :, 4]
            - qgrs[:, :, :, 5]
        )
    # debug = {"delp": prsl}
    # np.save("standalone_driver_statein_rank" + str(tile) + ".npy", debug)
    # passed
    prsi[:, :, 0] = ptop
    for k in range(npz):
        prsi[:, :, k + 1] = prsi[:, :, k] + prsl[:, :, k]
        prsik[:, :, k] = np.log(prsi[:, :, k])
        for n in range(NQ):
            qgrs[:, :, k, n] = qgrs[:, :, k, n] / prsl[:, :, k]
    prsik[:, :, -1] = np.log(prsi[:, :, -1])
    prsik[:, :, 0] = np.log(ptop)
    dm = gt_storage.zeros(
        backend=BACKEND,
        dtype=FIELD_FLT,
        shape=shape_2d,
        default_origin=(0, 0, 0),
    )
    for k in range(npz):
        qgrs_rad = np.maximum(qmin, qgrs[:, :, k, 0])
        rTv = rdgas * tgrs[:, :, k] * (1.0 + con_fvirt * qgrs_rad)
        dm[:, :] = prsl[:, :, k]
        prsl[:, :, k] = dm * rTv / (phii[:, :, k] - phii[:, :, k + 1])
        # if not hydrostatic, replaces it with hydrostatic pressure if violated
        prsl[:, :, k] = np.minimum(prsl[:, :, k], prsi[:, :, k + 1] - 0.01 * dm)
        prsl[:, :, k] = np.maximum(prsl[:, :, k], prsi[:, :, k] + 0.01 * dm)

    prsik[:, :, -1] = np.exp(KAPPA * prsik[:, :, -1]) * pk0inv
    prsik[:, :, 0] = pktop
    debug = {"delp": prsl, "prsik": prsik, "qvapor": qgrs[:, :, :, 0]}
    np.save("standalone_driver_statein_rank" + str(tile) + ".npy", debug)
    # temporary transformation to match with fortran data
    output = {}
    output["tgrs"] = np.reshape(tgrs.data, (144, 80), order="F")[:, 0:79]
    output["qgrs"] = np.reshape(qgrs.data, (144, 80, 9), order="F")[:, 0:79, :]
    output["ugrs"] = np.reshape(ugrs.data, (144, 80), order="F")[:, 0:79]
    output["vgrs"] = np.reshape(vgrs.data, (144, 80), order="F")[:, 0:79]
    output["vvl"] = np.reshape(vvl.data, (144, 80), order="F")[:, 0:79]
    output["prsl"] = np.reshape(prsl.data, (144, 80), order="F")[:, 0:79]
    output["phii"] = np.reshape(phii.data, (144, 80), order="F")
    output["prsi"] = np.reshape(prsi.data, (144, 80), order="F")
    output["prsik"] = np.reshape(prsik.data, (144, 80), order="F")
    return output


for tile in range(6):
    print("Validating tile " + str(tile))
    state = np.load("state_rank" + str(tile) + ".npy", allow_pickle=True).item()
    state = long_to_short_name(state)
    input_data = np.load(
        "input_data_rank" + str(tile) + ".npy", allow_pickle=True
    ).item()
    serializer = ser.Serializer(
        ser.OpenModeKind.Read,
        "c12_6ranks_baroclinic_dycore_microphysics",
        "Generator_rank" + str(tile),
    )

    exp_data = atmos_phys_driver_statein(state, tile)
    ref_data = data_dict_from_var_list(
        OUT_VAR_APDS,
        serializer,
        serializer.get_savepoint("AtmosPhysDriverStatein-OUT")[0],
    )
    for key in exp_data:
        print(key)
        ind = np.array(
            np.nonzero(
                ~np.isclose(exp_data[key], ref_data["IPD_" + key], equal_nan=True)
            )
        )
        if ind.size > 0:
            i = tuple(ind[:, 0])
            print("FAIL at ", key, i)
            np.testing.assert_allclose(ref_data["IPD_" + key], exp_data[key])
