import numpy as np

import pace.util as fv3util
import pace.util.constants as constants
from pace.fv3core.initialization.dycore_state import DycoreState
from pace.util.grid import GridData, MetricTerms, great_circle_distance_lon_lat
from pace.util.grid.gnomonic import (
    get_lonlat_vect,
    get_unit_vector_direction,
    lon_lat_midpoint,
)

from .baroclinic import (
    empty_numpy_dycore_state,
    initialize_delp,
    initialize_edge_pressure,
    local_compute_size,
)


nhalo = fv3util.N_HALO_DEFAULT


def init_tc_state(
    metric_terms: MetricTerms,
    hydrostatic: bool,
    comm: fv3util.CubedSphereCommunicator,
) -> DycoreState:
    """
    Create a DycoreState object with quantities initialized to the
    FV3 tropical cyclone test case (test_case 55).

    This case involves a grid_transformation (done on metric terms)
    to locally increase resolution.
    """

    sample_quantity = metric_terms.lat
    shape = (*sample_quantity.data.shape[:2], metric_terms.ak.data.shape[0])
    nx, ny, nz = local_compute_size(shape)
    numpy_state = empty_numpy_dycore_state(shape)

    # Initializing to values the Fortran does for easy comparison
    numpy_state.delp[:nhalo, :nhalo] = 0.0
    numpy_state.delp[:nhalo, nhalo + ny :] = 0.0
    numpy_state.delp[nhalo + nx :, :nhalo] = 0.0
    numpy_state.delp[nhalo + nx :, nhalo + ny :] = 0.0
    #numpy_state.pe[:] = 0.0
    #numpy_state.phis[:] = 1.0e30

    tc_properties = {
        "hydrostatic": hydrostatic,
        "dp": 1115.0,
        "exppr": 1.5,
        "exppz": 2.0,
        "gamma": 0.007,
        "lat_tc": 10.0,
        "lon_tc": 180.0,
        "p_ref": 101500.0,
        "ptop": 1.0,
        "qtrop": 1e-11,
        "q00": 0.021,
        "rp": 282000.0,
        "Ts0": 302.15,
        "vort": True,
        "ztrop": 15000.0,
        "zp": 7000.0,
        "zq1": 3000.0,
        "zq2": 8000.0,
    }

    calc = _some_inital_calculations(tc_properties)

    ps_output = _initialize_vortex_ps_phis(metric_terms, shape, tc_properties, calc)
    ps, ps_u, ps_v = ps_output["ps"], ps_output["ps_uc"], ps_output["ps_vc"]

    # TODO restart file had different ak, bk. Figure out where they came from;
    # for now, take from metric terms
    ak = metric_terms.ak.data
    bk = metric_terms.bk.data
    delp = initialize_delp(ps, ak, bk)
    pe = np.zeros(shape)
    pe[:, :, :-1] = initialize_edge_pressure(delp, tc_properties["ptop"])

    pe_u = _initialize_edge_pressure_cgrid(ak, bk, ps_u, shape, tc_properties["ptop"])
    pe_v = _initialize_edge_pressure_cgrid(ak, bk, ps_v, shape, tc_properties["ptop"])

    ud, vd = _initialize_wind_dgrid(
        metric_terms, tc_properties, calc, pe_u, pe_v, ps_u, ps_v, shape
    )
    ua, va = _interpolate_winds_dgrid_agrid(metric_terms, ud, vd, tc_properties, shape)

    qvapor, pt = _initialize_qvapor_temperature(
        metric_terms, pe, ps, tc_properties, calc, shape
    )
    delz, w = _initialize_delz_w(pe, pt, qvapor, tc_properties, shape)

    # numpy_state.cxd[:] =
    # numpy_state.cyd[:] =
    numpy_state.delp[:, :, :-1] = delp
    numpy_state.delz[:] = delz
    # numpy_state.diss_estd[:] =
    # numpy_state.mfxd[:] =
    # numpy_state.mfyd[:] =
    # numpy_state.omga[:] =
    numpy_state.pe[:] = pe
    numpy_state.peln[:] = np.log(pe)
    numpy_state.phis[:] = ps_output["phis"]
    # numpy_state.pk[:] =
    # numpy_state.pkz[:] =
    numpy_state.ps[:] = pe[:, :, -2]
    #numpy_state.pt[:] = pt
    # numpy_state.qcld[:] =
    # numpy_state.qgraupel[:] =
    # numpy_state.qice[:] =
    # numpy_state.qliquid[:] =
    # numpy_state.qo3mr[:] =
    # numpy_state.qrain[:] =
    # numpy_state.qsgs_tke[:] =
    # numpy_state.qsnow[:] =
    numpy_state.qvapor[:] = qvapor
    # numpy_state.q_con[:] =
    numpy_state.u[:] = ud
    numpy_state.ua[:] = ua
    # numpy_state.uc[:] =
    numpy_state.v[:] = vd
    numpy_state.va[:] = va
    # numpy_state.uc[:] =
    numpy_state.w[:] = w

    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=metric_terms.quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    return state


def _calculate_distance_from_tc_center(pe_v, ps_v, muv, calc, tc_properties):

    d1 = (
        np.sin(calc["p0"][1]) * np.cos(muv["midpoint"][:, :, 1])
        - np.cos(calc["p0"][1])
        * np.sin(muv["midpoint"][:, :, 1])
        * np.cos(muv["midpoint"][:, :, 0] - calc["p0"][0])
    )
    d2 = np.cos(calc["p0"][1]) * np.sin(muv["midpoint"][:, :, 0] - calc["p0"][0])
    d = np.sqrt(d1 ** 2 + d2 ** 2)
    d[d < 1e-15] = 1e-15

    r = great_circle_distance_lon_lat(
        calc["p0"][0],
        muv["midpoint"][:, :, 0],
        calc["p0"][1],
        muv["midpoint"][:, :, 1],
        constants.RADIUS,
        np,
    )
    ptmp = 0.5 * (pe_v[:, :, :-1] + pe_v[:, :, 1:])
    height = (calc["t00"] / tc_properties["gamma"]) * (
        1.0 - (ptmp / ps_v[:, :, None]) ** calc["exponent"]
    )

    distance_dict = {"d": d, "d1": d1, "d2": d2, "height": height, "r": r}

    return distance_dict


def _calculate_pt_height(height, qvapor, r, tc_properties, calc):

    aa = calc["t00"] - tc_properties["gamma"] * height
    bb = (1.0 + constants.ZVIR * qvapor)[:, :, :-1]
    cc = np.exp((height / tc_properties["zp"]) ** tc_properties["exppz"]) * np.exp(
        (r[:, :, None] / tc_properties["rp"]) ** tc_properties["exppr"]
    )
    dd = 1.0 - tc_properties["p_ref"] / tc_properties["dp"] * cc
    ee = constants.GRAV * tc_properties["zp"] ** tc_properties["exppz"] * dd
    ff = 1.0 + tc_properties["exppz"] * constants.RDGAS * aa * height / ee

    pt = aa / bb / ff

    return pt


def _calculate_utmp(height, dist, calc, tc_properties):

    aa = (height / tc_properties["zp"]) # (134, 135, 79)
    bb = (dist["r"] / tc_properties["rp"]) # (134, 135)
    cc = (aa**tc_properties["exppz"]) # (134, 135, 79)
    dd = (bb**tc_properties["exppr"]) # (134, 135)
    ee = (1. - tc_properties["p_ref"]/tc_properties["dp"] * np.exp(dd[:, :, None]) * np.exp(cc)) # (134, 135, 79)
    ff = (constants.GRAV * tc_properties["zp"]**tc_properties["exppz"]) # number
    gg = (calc["t00"] - tc_properties["gamma"] * height) # (134, 135, 79)
    hh = (tc_properties["exppz"] * height * constants.RDGAS * gg / ff + ee) # (134, 135, 79)
    ii = (calc["cor"] * dist["r"] / 2.0) # (134, 135)
    jj = (2.0) # number
    kk = (ii[:, :, None]**jj - tc_properties["exppr"] * bb[:, :, None]**tc_properties["exppr"] * constants.RDGAS * gg / hh) # (134, 135, 79)
    ll = (-calc["cor"] * dist["r"][:, :, None] / 2.0 + np.sqrt(kk)) # (134, 135, 79)

    utmp = 1.0 / dist["d"][:, :, None] * ll

    return utmp


def _calculate_vortex_surface_pressure_with_radius(p0, p_grid, tc_properties):
    """
    p0 is the tc center point
    p_grid is the metric_terms.grid variable corresponding to what is needed
    for ps on A-grid, p_grid is metric_terms.agrid.data
    """

    r = great_circle_distance_lon_lat(
        p0[0], p_grid[:, :, 0], p0[1], p_grid[:, :, 1], constants.RADIUS, np
    )
    ps = tc_properties["p_ref"] - tc_properties["dp"] * np.exp(
        -((r / tc_properties["rp"]) ** 1.5)
    )

    return ps


def _find_midpoint_unit_vectors(p1, p2):

    midpoint = np.array(
        lon_lat_midpoint(p1[:, :, 0], p2[:, :, 0], p1[:, :, 1], p2[:, :, 1], np)
    ).transpose([1, 2, 0])
    unit_dir = get_unit_vector_direction(p1, p2, np)
    exv, eyv = get_lonlat_vect(midpoint, np)

    muv = {"midpoint": midpoint, "unit_dir": unit_dir, "exv": exv, "eyv": eyv}

    return muv


def _initialize_delz_w(pe, pt, qvapor, tc_properties, shape):

    delz = np.zeros(shape)
    w = np.zeros(shape)

    if tc_properties["hydrostatic"] is False:
        delz[:, :, :-1] = (
            constants.RDGAS
            * pt[:, :, :-1]
            * (1 + constants.ZVIR * qvapor[:, :, :-1])
            / constants.GRAV
            * np.log(pe[:, :, :-1] / pe[:, :, 1:])
        )
        w[:] = 0.0

    return delz, w


def _initialize_edge_pressure_cgrid(ak, bk, ps, shape, ptop):
    """
    Initialize edge pressure on c-grid for u and v points,
    depending on which ps is input (ps_uc or ps_vc)
    """
    pe_cgrid = np.zeros(shape)
    pe_cgrid[:, :, 0] = ptop

    pe_cgrid[:, :, :] = ak[None, None, :] + ps[:, :, None] * bk[None, None, :]

    return pe_cgrid


def _initialize_qvapor_temperature(metric_terms, pe, ps, tc_properties, calc, shape):

    qvapor = np.zeros(shape)
    pt = np.zeros(shape)

    ptmp = 0.5 * (pe[:, :, :-1] + pe[:, :, 1:])
    height = (calc["t00"] / tc_properties["gamma"]) * (
        1.0 - (ptmp / ps[:, :, None]) ** calc["exponent"]
    )
    qvapor[:, :, :-1] = (
        tc_properties["q00"]
        * np.exp(-height / tc_properties["zq1"])
        * np.exp(-((height / tc_properties["zq2"]) ** tc_properties["exppz"]))
    )

    p2 = metric_terms.agrid.data
    r = great_circle_distance_lon_lat(
        calc["p0"][0], p2[:, :, 0], calc["p0"][1], p2[:, :, 1], constants.RADIUS, np
    )

    pt[:, :, :-1] = _calculate_pt_height(height, qvapor, r, tc_properties, calc)

    qvapor[:, :, :-1][height > tc_properties["ztrop"]] = tc_properties["qtrop"]
    pt[:, :, :-1][height > tc_properties["ztrop"]] = calc["ttrop"]

    return qvapor, pt


def _initialize_vortex_ps_phis(metric_terms, shape, tc_properties, calc):
    p0 = [np.deg2rad(tc_properties["lon_tc"]), np.deg2rad(tc_properties["lat_tc"])]

    phis = np.zeros(shape[:2])
    ps = np.zeros(shape[:2])
    ps = _calculate_vortex_surface_pressure_with_radius(
        calc["p0"], metric_terms.agrid.data, tc_properties
    )

    ps_vc = np.zeros(shape[:2])
    p_grid = 0.5 * (
        metric_terms.grid.data[:, :-1, :] + metric_terms.grid.data[:, 1:, :]
    )
    ps_vc[:, :-1] = _calculate_vortex_surface_pressure_with_radius(
        p0, p_grid, tc_properties
    )

    ps_uc = np.zeros(shape[:2])
    p_grid = 0.5 * (
        metric_terms.grid.data[:-1, :, :] + metric_terms.grid.data[1:, :, :]
    )
    ps_uc[:-1, :] = _calculate_vortex_surface_pressure_with_radius(
        p0, p_grid, tc_properties
    )

    output_dict = {"ps": ps, "ps_uc": ps_uc, "ps_vc": ps_vc, "phis": phis}

    return output_dict


def _initialize_wind_dgrid(
    metric_terms, tc_properties, calc, pe_u, pe_v, ps_u, ps_v, shape
):
    # u-wind
    ud = np.zeros(shape)
    p1 = metric_terms.grid.data[:-1, :, :]
    p2 = metric_terms.grid.data[1:, :, :]
    muv = _find_midpoint_unit_vectors(p1, p2)
    dist = _calculate_distance_from_tc_center(pe_u, ps_u, muv, calc, tc_properties)

    utmp = _calculate_utmp(
        dist["height"][:-1, :, :], dist, calc, tc_properties
    )
    vtmp = utmp * dist["d2"][:, :, None]
    print()
    utmp = utmp * dist["d1"][:, :, None]

    ud[:-1, :, :-1] = (
        utmp * np.sum(muv["unit_dir"] * muv["exv"], 2)[:, :, None]
        + vtmp * np.sum(muv["unit_dir"] * muv["eyv"], 2)[:, :, None]
    )
    ud[:, :, :-1][dist["height"] > tc_properties["ztrop"]] = 0

    # v-wind
    vd = np.zeros(shape)
    p1 = metric_terms.grid.data[:, :-1, :]
    p2 = metric_terms.grid.data[:, 1:, :]
    muv = _find_midpoint_unit_vectors(p1, p2)
    dist = _calculate_distance_from_tc_center(pe_v, ps_v, muv, calc, tc_properties)

    utmp = _calculate_utmp(
        dist["height"][:, :-1, :], dist, calc, tc_properties
    )
    vtmp = utmp * dist["d2"][:, :, None]
    utmp *= dist["d1"][:, :, None]

    vd[:, :-1, :-1] = (
        utmp * np.sum(muv["unit_dir"] * muv["exv"], 2)[:, :, None]
        + vtmp * np.sum(muv["unit_dir"] * muv["eyv"], 2)[:, :, None]
    )
    vd[:, :, :-1][dist["height"] > tc_properties["ztrop"]] = 0

    return ud, vd


def _interpolate_winds_dgrid_agrid(metric_terms, ud, vd, tc_properties, shape):

    ua = np.zeros(shape)
    va = np.zeros(shape)

    if tc_properties["vort"] is True:
        ua[:, :-1, :] = (
            0.5
            * (
                ud[:, :-1, :] * metric_terms.dx.data[:, :-1, None]
                + ud[:, 1:, :] * metric_terms.dx.data[:, 1:, None]
            )
            / metric_terms.dxa.data[:, :-1, None]
        )
        va[:-1, :, :] = (
            0.5
            * (
                vd[:-1, :, :] * metric_terms.dy.data[:-1, :, None]
                + vd[1:, :, :] * metric_terms.dy.data[1:, :, None]
            )
            / metric_terms.dya.data[:-1, :, None]
        )
    else:
        pass

    # TODO translate the not vort case#
    #             do i=isd,ied
    #             tmp1j(:) = 0.0
    #             tmp2j(:) = uin(i,:)*dyc(i,:)
    #             tmp3j(:) = dyc(i,:)
    #             call interp_left_edge_1d(tmp1j, tmp2j, tmp3j, jsd, jed+1, interpOrder)
    #             uout(i,jsd:jed) = tmp1j(jsd+1:jed+1)/dya(i,jsd:jed)
    #          enddo
    #          do j=jsd,jed
    #             tmp1i(:) = 0.0
    #             tmp2i(:) = vin(:,j)*dxc(:,j)
    #             tmp3i(:) = dxc(:,j)
    #             call interp_left_edge_1d(tmp1i, tmp2i, tmp3i, isd, ied+1, interpOrder)
    #             vout(isd:ied,j) = tmp1i(isd+1:ied+1)/dxa(isd:ied,j)
    #          enddo #2934

    return ua, va


def _some_inital_calculations(tc_properties):
    t00 = tc_properties["Ts0"] * (1.0 + constants.ZVIR * tc_properties["q00"])  # num
    p0 = [np.deg2rad(tc_properties["lon_tc"]), np.deg2rad(tc_properties["lat_tc"])]
    exponent = constants.RDGAS * tc_properties["gamma"] / constants.GRAV  # num
    cor = 2.0 * constants.OMEGA * np.sin(np.deg2rad(tc_properties["lat_tc"]))  # num
    ttrop = t00 - tc_properties["gamma"] * tc_properties["ztrop"]

    calc = {
        "cor": cor,
        "exponent": exponent,
        "p0": p0,
        "ttrop": ttrop,
        "t00": t00,
    }

    return calc
