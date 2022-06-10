import math
from dataclasses import fields
from types import SimpleNamespace

import numpy as np

import fv3core.initialization.baroclinic_jablonowski_williamson as jablo_init
import pace.dsl.gt4py_utils as utils
import pace.util as fv3util
import pace.util.constants as constants
from fv3core.initialization.dycore_state import DycoreState
from pace.util.grid import MetricTerms, lon_lat_midpoint


nhalo = fv3util.N_HALO_DEFAULT
ptop_min = 1e-8
pcen = [math.pi / 9.0, 2.0 * math.pi / 9.0]


def initialize_delp(ps, ak, bk):
    return (
        ak[None, None, 1:]
        - ak[None, None, :-1]
        + ps[:, :, None] * (bk[None, None, 1:] - bk[None, None, :-1])
    )


def initialize_edge_pressure(delp, ptop):
    pe = np.zeros(delp.shape)
    pe[:, :, 0] = ptop
    for k in range(1, pe.shape[2]):
        pe[:, :, k] = pe[:, :, k - 1] + delp[:, :, k - 1]
    return pe


def initialize_log_pressure_interfaces(pe, ptop):
    peln = np.zeros(pe.shape)
    peln[:, :, 0] = math.log(ptop)
    peln[:, :, 1:] = np.log(pe[:, :, 1:])
    return peln


def initialize_kappa_pressures(pe, peln, ptop):
    """
    Compute the edge_pressure**kappa (pk) and the layer mean of this (pkz)
    """
    pk = np.zeros(pe.shape)
    pkz = np.zeros(pe.shape)
    pk[:, :, 0] = ptop ** constants.KAPPA
    pk[:, :, 1:] = np.exp(constants.KAPPA * np.log(pe[:, :, 1:]))
    pkz[:, :, :-1] = (pk[:, :, 1:] - pk[:, :, :-1]) / (
        constants.KAPPA * (peln[:, :, 1:] - peln[:, :, :-1])
    )
    return pk, pkz


def local_coordinate_transformation(u_component, lon, grid_vector_component):
    """
    Transform the zonal wind component to the cubed sphere grid using a grid vector
    """
    return (
        u_component
        * (
            grid_vector_component[:, :, 1] * np.cos(lon)
            - grid_vector_component[:, :, 0] * np.sin(lon)
        )[:, :, None]
    )


def wind_component_calc(
    shape,
    eta_v,
    lon,
    lat,
    grid_vector_component,
    islice,
    islice_grid,
    jslice,
    jslice_grid,
):
    slice_grid = (islice_grid, jslice_grid)
    slice_3d = (islice, jslice, slice(None))
    u_component = np.zeros(shape)
    u_component[slice_3d] = jablo_init.baroclinic_perturbed_zonal_wind(
        eta_v, lon[slice_grid], lat[slice_grid]
    )
    u_component[slice_3d] = local_coordinate_transformation(
        u_component[slice_3d],
        lon[slice_grid],
        grid_vector_component[islice_grid, jslice_grid, :],
    )
    return u_component


def initialize_zonal_wind(
    u,
    eta,
    eta_v,
    lon,
    lat,
    east_grid_vector_component,
    center_grid_vector_component,
    islice,
    islice_grid,
    jslice,
    jslice_grid,
    axis,
):
    shape = u.shape
    uu1 = wind_component_calc(
        shape,
        eta_v,
        lon,
        lat,
        east_grid_vector_component,
        islice,
        islice,
        jslice,
        jslice_grid,
    )
    uu3 = wind_component_calc(
        shape,
        eta_v,
        lon,
        lat,
        east_grid_vector_component,
        islice,
        islice_grid,
        jslice,
        jslice,
    )
    upper = (slice(None),) * axis + (slice(0, -1),)
    lower = (slice(None),) * axis + (slice(1, None),)
    pa1, pa2 = lon_lat_midpoint(lon[upper], lon[lower], lat[upper], lat[lower], np)
    uu2 = wind_component_calc(
        shape,
        eta_v,
        pa1,
        pa2,
        center_grid_vector_component,
        islice,
        islice,
        jslice,
        jslice,
    )
    u[islice, jslice, :] = 0.25 * (uu1 + 2.0 * uu2 + uu3)[islice, jslice, :]


def compute_grid_edge_midpoint_latitude_components(lon, lat):
    _, lat_avg_x_south = lon_lat_midpoint(
        lon[0:-1, :], lon[1:, :], lat[0:-1, :], lat[1:, :], np
    )
    _, lat_avg_y_east = lon_lat_midpoint(
        lon[1:, 0:-1], lon[1:, 1:], lat[1:, 0:-1], lat[1:, 1:], np
    )
    _, lat_avg_x_north = lon_lat_midpoint(
        lon[0:-1, 1:], lon[1:, 1:], lat[0:-1, 1:], lat[1:, 1:], np
    )
    _, lat_avg_y_west = lon_lat_midpoint(
        lon[:, 0:-1], lon[:, 1:], lat[:, 0:-1], lat[:, 1:], np
    )
    return lat_avg_x_south, lat_avg_y_east, lat_avg_x_north, lat_avg_y_west


def cell_average_nine_point(pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9):
    """
    9-point average: should be 2nd order accurate for a rectangular cell
    9  4  8
    5  1  3
    6  2  7
    """
    return (
        0.25 * pt1 + 0.125 * (pt2 + pt3 + pt4 + pt5) + 0.0625 * (pt6 + pt7 + pt8 + pt9)
    )


def cell_average_nine_components(
    component_function,
    component_args,
    lon,
    lat,
    lat_agrid,
):
    """
    Outputs the weighted average of a field that is a function of latitude,
    averaging over the 9 points on the corners, edges, and center of each
    gridcell.

    Args:
        component_function: callable taking in an array of latitude and
            returning an output array
        component_args: arguments to pass on to component_function,
            should not be a function of latitude
        lon: longitude array, defined on cell corners
        lat: latitude array, defined on cell corners
        lat_agrid: latitude array, defined on cell centers
    """
    # this weighting is done to reproduce the behavior of the Fortran code
    # Compute cell lats in the midpoint of each cell edge
    lat2, lat3, lat4, lat5 = compute_grid_edge_midpoint_latitude_components(lon, lat)
    pt1 = component_function(*component_args, lat=lat_agrid)
    pt2 = component_function(*component_args, lat=lat2[:, :-1])
    pt3 = component_function(*component_args, lat=lat3)
    pt4 = component_function(*component_args, lat=lat4)
    pt5 = component_function(*component_args, lat=lat5[:-1, :])
    pt6 = component_function(*component_args, lat=lat[:-1, :-1])
    pt7 = component_function(*component_args, lat=lat[1:, :-1])
    pt8 = component_function(*component_args, lat=lat[1:, 1:])
    pt9 = component_function(*component_args, lat=lat[:-1, 1:])
    return cell_average_nine_point(pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9)


def initialize_delz(pt, peln):
    return constants.RDG * pt[:, :, :-1] * (peln[:, :, 1:] - peln[:, :, :-1])


def moisture_adjusted_temperature(pt, qvapor):
    """
    Update initial temperature to include water vapor contribution
    """
    return pt / (1.0 + constants.ZVIR * qvapor)


def setup_pressure_fields(
    eta,
    eta_v,
    delp,
    ps,
    pe,
    peln,
    pk,
    pkz,
    ak,
    bk,
    ptop,
):
    ps[:] = jablo_init.surface_pressure
    delp[:, :, :-1] = initialize_delp(ps, ak, bk)
    pe[:] = initialize_edge_pressure(delp, ptop)
    peln[:] = initialize_log_pressure_interfaces(pe, ptop)
    pk[:], pkz[:] = initialize_kappa_pressures(pe, peln, ptop)
    eta[:-1], eta_v[:-1] = jablo_init.compute_eta(ak, bk)


def baroclinic_initialization(
    eta,
    eta_v,
    peln,
    qvapor,
    delp,
    u,
    v,
    pt,
    phis,
    delz,
    w,
    lon,
    lat,
    lon_agrid,
    lat_agrid,
    ee1,
    ee2,
    es1,
    ew2,
    ptop,
    adiabatic,
    hydrostatic,
    nx,
    ny,
):
    """
    Calls methods that compute initial state via the Jablonowski perturbation test case
    Transforms results to the cubed sphere grid
    Creates an initial baroclinic state for u(x-wind), v(y-wind), pt(temperature),
    phis(surface geopotential)w (vertical windspeed) and delz (vertical coordinate layer
    width)

    Inputs lon, lat, lon_agrid, lat_agrid, ee1, ee2, es1, ew2, ptop are defined by the
           grid and can be computed using an instance of the MetricTerms class.
    Inputs eta and eta_v are vertical coordinate columns derived from the ak and bk
           variables, also found in the Metric Terms class.
    """

    # Equation (2) for v
    # Although meridional wind is 0 in this scheme
    # on the cubed sphere grid, v is not 0 on every tile
    initialize_zonal_wind(
        v,
        eta,
        eta_v,
        lon,
        lat,
        east_grid_vector_component=ee2,
        center_grid_vector_component=ew2,
        islice=slice(0, nx + 1),
        islice_grid=slice(0, nx + 1),
        jslice=slice(0, ny),
        jslice_grid=slice(1, ny + 1),
        axis=1,
    )

    initialize_zonal_wind(
        u,
        eta,
        eta_v,
        lon,
        lat,
        east_grid_vector_component=ee1,
        center_grid_vector_component=es1,
        islice=slice(0, nx),
        islice_grid=slice(1, nx + 1),
        jslice=slice(0, ny + 1),
        jslice_grid=slice(0, ny + 1),
        axis=0,
    )

    slice_3d = (slice(0, nx), slice(0, ny), slice(None))
    slice_2d = (slice(0, nx), slice(0, ny))
    slice_2d_buffer = (slice(0, nx + 1), slice(0, ny + 1))
    # initialize temperature
    t_mean = jablo_init.horizontally_averaged_temperature(eta)
    pt[slice_3d] = cell_average_nine_components(
        jablo_init.temperature,
        [eta, eta_v, t_mean],
        lon[slice_2d_buffer],
        lat[slice_2d_buffer],
        lat_agrid[slice_2d],
    )

    # initialize surface geopotential
    phis[slice_2d] = cell_average_nine_components(
        jablo_init.surface_geopotential_perturbation,
        [],
        lon[slice_2d_buffer],
        lat[slice_2d_buffer],
        lat_agrid[slice_2d],
    )

    if not hydrostatic:
        # vertical velocity is set to 0 for nonhydrostatic setups
        w[slice_3d] = 0.0
        delz[:nx, :ny, :-1] = initialize_delz(pt[slice_3d], peln[slice_3d])

    if not adiabatic:
        qvapor[:nx, :ny, :-1] = jablo_init.specific_humidity(
            delp[slice_3d], peln[slice_3d], lat_agrid[slice_2d]
        )
        pt[slice_3d] = moisture_adjusted_temperature(pt[slice_3d], qvapor[slice_3d])


def initialize_pkz_moist(delp, pt, qvapor, delz):
    return np.exp(
        constants.KAPPA
        * np.log(
            constants.RDG
            * delp[:, :, :-1]
            * pt[:, :, :-1]
            * (1.0 + constants.ZVIR * qvapor[:, :, :-1])
            / delz[:, :, :-1]
        )
    )


def initialize_pkz_dry(delp, pt, delz):
    return np.exp(
        constants.KAPPA
        * np.log(constants.RDG * delp[:, :, :-1] * pt[:, :, :-1] / delz[:, :, :-1])
    )


def fix_top_log_edge_pressure(peln, ptop):
    if ptop < ptop_min:
        ak1 = (constants.KAPPA + 1.0) / constants.KAPPA
        peln[:, :, 0] = peln[:, :, 1] - ak1
    else:
        peln[:, :, 0] = np.log(ptop)


def p_var(
    delp,
    delz,
    pt,
    ps,
    qvapor,
    pe,
    peln,
    pkz,
    ptop,
    moist_phys,
    make_nh,
):
    """
    Computes auxiliary pressure variables for a hydrostatic state.

    The Fortran code also recomputes some more pressure variables,
    pe, pk, but since these are already done in setup_pressure_fields
    we don't duplicate them here
    """

    ps[:] = pe[:, :, -1]
    fix_top_log_edge_pressure(peln, ptop)

    if make_nh:
        delz[:, :, :-1] = initialize_delz(pt, peln)
    if moist_phys:
        pkz[:, :, :-1] = initialize_pkz_moist(delp, pt, qvapor, delz)
    else:
        pkz[:, :, :-1] = initialize_pkz_dry(delp, pt, delz)


# TODO: maybe extract from quantity related objects
def local_compute_size(data_array_shape):
    nx = data_array_shape[0] - 2 * nhalo - 1
    ny = data_array_shape[1] - 2 * nhalo - 1
    nz = data_array_shape[2]
    return nx, ny, nz


def compute_slices(nx, ny):
    islice = slice(nhalo, nhalo + nx)
    jslice = slice(nhalo, nhalo + ny)
    slice_3d = (islice, jslice, slice(None))
    slice_2d = (islice, jslice)
    return islice, jslice, slice_3d, slice_2d


def empty_numpy_dycore_state(shape):
    numpy_dict = {}
    for _field in fields(DycoreState):
        if "dims" in _field.metadata.keys():
            numpy_dict[_field.name] = np.zeros(shape[: len(_field.metadata["dims"])])
    numpy_state = SimpleNamespace(**numpy_dict)
    return numpy_state


def init_baroclinic_state(
    metric_terms: MetricTerms,
    adiabatic: bool,
    hydrostatic: bool,
    moist_phys: bool,
    comm: fv3util.CubedSphereCommunicator,
) -> DycoreState:
    """
    Create a DycoreState object with quantities initialized to the Jablonowski &
    Williamson baroclinic test case perturbation applied to the cubed sphere grid.
    """
    sample_quantity = metric_terms.lat
    shape = (*sample_quantity.data.shape[0:2], metric_terms.ak.data.shape[0])
    nx, ny, nz = local_compute_size(shape)
    numpy_state = empty_numpy_dycore_state(shape)
    # Initializing to values the Fortran does for easy comparison
    numpy_state.delp[:] = 1e30
    numpy_state.delp[:nhalo, :nhalo] = 0.0
    numpy_state.delp[:nhalo, nhalo + ny :] = 0.0
    numpy_state.delp[nhalo + nx :, :nhalo] = 0.0
    numpy_state.delp[nhalo + nx :, nhalo + ny :] = 0.0
    numpy_state.pe[:] = 0.0
    numpy_state.pt[:] = 1.0
    numpy_state.ua[:] = 1e35
    numpy_state.va[:] = 1e35
    numpy_state.uc[:] = 1e30
    numpy_state.vc[:] = 1e30
    numpy_state.w[:] = 1.0e30
    numpy_state.delz[:] = 1.0e25
    numpy_state.phis[:] = 1.0e25
    numpy_state.ps[:] = jablo_init.surface_pressure
    eta = np.zeros(nz)
    eta_v = np.zeros(nz)
    islice, jslice, slice_3d, slice_2d = compute_slices(nx, ny)
    # Slices with extra buffer points in the horizontal dimension
    # to accomodate averaging over shifted calculations on the grid
    _, _, slice_3d_buffer, slice_2d_buffer = compute_slices(nx + 1, ny + 1)

    setup_pressure_fields(
        eta=eta,
        eta_v=eta_v,
        delp=numpy_state.delp[slice_3d],
        ps=numpy_state.ps[slice_2d],
        pe=numpy_state.pe[slice_3d],
        peln=numpy_state.peln[slice_3d],
        pk=numpy_state.pk[slice_3d],
        pkz=numpy_state.pkz[slice_3d],
        ak=utils.asarray(metric_terms.ak.data),
        bk=utils.asarray(metric_terms.bk.data),
        ptop=metric_terms.ptop,
    )

    baroclinic_initialization(
        eta=eta,
        eta_v=eta_v,
        peln=numpy_state.peln[slice_3d_buffer],
        qvapor=numpy_state.qvapor[slice_3d_buffer],
        delp=numpy_state.delp[slice_3d_buffer],
        u=numpy_state.u[slice_3d_buffer],
        v=numpy_state.v[slice_3d_buffer],
        pt=numpy_state.pt[slice_3d_buffer],
        phis=numpy_state.phis[slice_2d_buffer],
        delz=numpy_state.delz[slice_3d_buffer],
        w=numpy_state.w[slice_3d_buffer],
        lon=utils.asarray(metric_terms.lon.data[slice_2d_buffer]),
        lat=utils.asarray(metric_terms.lat.data[slice_2d_buffer]),
        lon_agrid=utils.asarray(metric_terms.lon_agrid.data[slice_2d_buffer]),
        lat_agrid=utils.asarray(metric_terms.lat_agrid.data[slice_2d_buffer]),
        ee1=utils.asarray(metric_terms.ee1.data[slice_3d_buffer]),
        ee2=utils.asarray(metric_terms.ee2.data[slice_3d_buffer]),
        es1=utils.asarray(metric_terms.es1.data[slice_3d_buffer]),
        ew2=utils.asarray(metric_terms.ew2.data[slice_3d_buffer]),
        ptop=metric_terms.ptop,
        adiabatic=adiabatic,
        hydrostatic=hydrostatic,
        nx=nx,
        ny=ny,
    )

    p_var(
        delp=numpy_state.delp[slice_3d],
        delz=numpy_state.delz[slice_3d],
        pt=numpy_state.pt[slice_3d],
        ps=numpy_state.ps[slice_2d],
        qvapor=numpy_state.qvapor[slice_3d],
        pe=numpy_state.pe[slice_3d],
        peln=numpy_state.peln[slice_3d],
        pkz=numpy_state.pkz[slice_3d],
        ptop=metric_terms.ptop,
        moist_phys=moist_phys,
        make_nh=(not hydrostatic),
    )
    state = DycoreState.init_from_numpy_arrays(
        numpy_state.__dict__,
        sizer=metric_terms.quantity_factory.sizer,
        backend=sample_quantity.metadata.gt4py_backend,
    )

    comm.halo_update(state.phis, n_points=nhalo)

    comm.vector_halo_update(state.u, state.v, n_points=nhalo)

    return state
