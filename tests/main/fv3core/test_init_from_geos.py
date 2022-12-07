import f90nml
import numpy as np
import pytest  # noqa

from pace import fv3core
from pace.util.null_comm import NullComm


def test_geos_wrapper():

    namelist_dict = {
        "stencil_config": {
            "compilation_config": {
                "backend": "numpy",
                "rebuild": False,
                "validate_args": True,
                "format_source": False,
                "device_sync": False,
            }
        },
        "initialization": {"type": "baroclinic"},
        "nx_tile": 12,
        "nz": 91,
        "dt_atmos": 225,
        "minutes": 15,
        "layout": [1, 1],
        "dycore_config": {
            "a_imp": 1.0,
            "beta": 0.0,
            "consv_te": 0.0,
            "d2_bg": 0.0,
            "d2_bg_k1": 0.2,
            "d2_bg_k2": 0.1,
            "d4_bg": 0.15,
            "d_con": 1.0,
            "d_ext": 0.0,
            "dddmp": 0.5,
            "delt_max": 0.002,
            "do_sat_adj": True,
            "do_vort_damp": True,
            "fill": True,
            "hord_dp": 6,
            "hord_mt": 6,
            "hord_tm": 6,
            "hord_tr": 8,
            "hord_vt": 6,
            "hydrostatic": False,
            "k_split": 1,
            "ke_bg": 0.0,
            "kord_mt": 9,
            "kord_tm": -9,
            "kord_tr": 9,
            "kord_wz": 9,
            "n_split": 1,
            "nord": 3,
            "nwat": 6,
            "p_fac": 0.05,
            "rf_cutoff": 3000.0,
            "rf_fast": True,
            "tau": 10.0,
            "vtdm4": 0.06,
            "z_tracer": True,
            "do_qa": True,
            "tau_i2s": 1000.0,
            "tau_g2v": 1200.0,
            "ql_gen": 0.001,
            "ql_mlt": 0.002,
            "qs_mlt": 1e-06,
            "qi_lim": 1.0,
            "dw_ocean": 0.1,
            "dw_land": 0.15,
            "icloud_f": 0,
            "tau_l2v": 300.0,
            "tau_v2l": 90.0,
            "fv_sg_adj": 0,
            "n_sponge": 48,
        },
    }

    namelist = f90nml.namelist.Namelist(namelist_dict)

    comm = NullComm(rank=0, total_ranks=6, fill_value=0.0)
    backend = "numpy"

    wrapper = fv3core.GeosDycoreWrapper(namelist, comm, backend)
    nhalo = 3
    shape_centered = (
        namelist["nx_tile"] + 2 * nhalo,
        namelist["nx_tile"] + 2 * nhalo,
        namelist["nz"],
    )
    shape_x_interface = (
        namelist["nx_tile"] + 2 * nhalo + 1,
        namelist["nx_tile"] + 2 * nhalo,
        namelist["nz"],
    )
    shape_y_interface = (
        namelist["nx_tile"] + 2 * nhalo,
        namelist["nx_tile"] + 2 * nhalo + 1,
        namelist["nz"],
    )
    shape_z_interface = (
        namelist["nx_tile"] + 2 * nhalo,
        namelist["nx_tile"] + 2 * nhalo,
        namelist["nz"] + 1,
    )
    shape2d = (
        namelist["nx_tile"] + 2 * nhalo,
        namelist["nx_tile"] + 2 * nhalo,
    )
    shape_tracers = (
        namelist["nx_tile"] + 2 * nhalo,
        namelist["nx_tile"] + 2 * nhalo,
        namelist["nz"],
        7,
    )

    assert isinstance(wrapper, fv3core.GeosDycoreWrapper)
    assert isinstance(wrapper.dynamical_core, fv3core.DynamicalCore)

    u = np.ones(shape_y_interface)
    v = np.ones(shape_x_interface)
    w = np.ones(shape_centered)
    delz = np.ones(shape_centered)
    pt = np.ones(shape_centered)
    delp = np.ones(shape_centered)
    q = np.ones(shape_tracers)
    ps = np.ones(shape2d)
    pe = np.ones(
        (
            namelist["nx_tile"] + 2,
            namelist["nx_tile"] + 2,
            namelist["nz"] + 1,
        )
    )
    pk = np.ones(
        (
            namelist["nx_tile"],
            namelist["nx_tile"],
            namelist["nz"] + 1,
        )
    )
    peln = np.ones(
        (
            namelist["nx_tile"],
            namelist["nx_tile"],
            namelist["nz"] + 1,
        )
    )
    pkz = np.ones(
        (
            namelist["nx_tile"],
            namelist["nx_tile"],
            namelist["nz"],
        )
    )
    phis = np.ones(shape2d)
    q_con = np.ones(shape_centered)
    omga = np.ones(shape_centered)
    ua = np.ones(shape_centered)
    va = np.ones(shape_centered)
    uc = np.ones(shape_x_interface)
    vc = np.ones(shape_y_interface)
    mfxd = np.ones(
        (
            namelist["nx_tile"] + 1,
            namelist["nx_tile"],
            namelist["nz"],
        )
    )
    mfyd = np.ones(
        (
            namelist["nx_tile"],
            namelist["nx_tile"] + 1,
            namelist["nz"],
        )
    )
    cxd = np.ones(
        (
            namelist["nx_tile"] + 1,
            namelist["nx_tile"] + 2 * nhalo,
            namelist["nz"],
        )
    )
    cyd = np.ones(
        (
            namelist["nx_tile"] + 2 * nhalo,
            namelist["nx_tile"] + 1,
            namelist["nz"],
        )
    )
    diss_estd = np.ones(shape_centered)

    output_dict = wrapper(
        u,
        v,
        w,
        delz,
        pt,
        delp,
        q,
        ps,
        pe,
        pk,
        peln,
        pkz,
        phis,
        q_con,
        omga,
        ua,
        va,
        uc,
        vc,
        mfxd,
        mfyd,
        cxd,
        cyd,
        diss_estd,
    )

    assert isinstance(output_dict["u"], np.ndarray)
