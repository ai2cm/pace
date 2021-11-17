import math
import numpy as np
import fv3core.utils.global_constants as constants
from fv3core.grid import lon_lat_midpoint, great_circle_distance_lon_lat
import fv3gfs.util as fv3util
import fv3core.utils.baroclinic_initialization_jablonowski_williamson as jablonowski_init
nhalo = fv3util.N_HALO_DEFAULT
ptop_min = 1e-8
pcen = [math.pi / 9.0, 2.0 * math.pi / 9.0]


def initialize_delp(ps, ak, bk):
    return  ak[None, None, 1:] - ak[None, None, :-1] + ps[:, :, None] * (bk[None, None, 1:] - bk[None, None, :-1])

def initialize_edge_pressure(delp, ptop):
    pe = np.zeros(delp.shape)
    pe[:, :, 0] = ptop
    for k in range(1, pe.shape[2]):
        pe[:, :, k] = pe[:, :, k - 1] + delp[:, :, k - 1]
    return pe

def initialize_log_pressure_interfaces(pe, ptop):
    peln = np.zeros(pe.shape)
    peln[:, :, 0] = math.log(ptop)
    peln[:, :, 1:]  = np.log(pe[:, :, 1:])
    return peln

def initialize_kappa_pressures(pe, peln, ptop):
    """
    Compute the edge_pressure**kappa (pk) and the layer mean of this (pkz)
    """
    pk = np.zeros(pe.shape)
    pkz = np.zeros(pe.shape)
    pk[:, :, 0] = ptop**constants.KAPPA
    pk[:, :, 1:] = np.exp(constants.KAPPA * np.log(pe[:, :, 1:]))
    pkz[:, :, :-1] = (pk[:, :, 1:] - pk[:, :, :-1]) / (constants.KAPPA * (peln[:, :, 1:] - peln[:, :, :-1]))
    return pk, pkz

def setup_pressure_fields(eta, eta_v, delp, ps, pe, peln, pk, pkz, qvapor, ak, bk, ptop, lat_agrid, adiabatic):
    delp[:, :, :-1] = initialize_delp(ps, ak, bk)
    pe[:] = initialize_edge_pressure(delp, ptop)
    peln[:] = initialize_log_pressure_interfaces(pe, ptop)
    pk[:], pkz[:] = initialize_kappa_pressures(pe, peln, ptop)
    jablonowski_init.compute_eta(eta, eta_v, ak, bk)

    if not adiabatic:        
        jablonowski_init.specific_humidity(delp, peln, lat_agrid, qvapor)

  

def local_coordinate_transformation(u_component, lon, grid_vector_component):
    """
    Transform the zonal wind component to the cubed sphere grid using a grid vector
    """
    return u_component * (grid_vector_component[:, :, 1]*np.cos(lon) - grid_vector_component[:, :, 0]*np.sin(lon))[:,:,None]

def wind_component_calc(shape, eta_v, lon, lat, grid_vector_component, islice, islice_grid, jslice, jslice_grid):
    slice_grid = (islice_grid, jslice_grid)
    slice_3d = (islice, jslice, slice(None))
    u_component = np.zeros(shape)
    u_component[slice_3d] = jablonowski_init.baroclinic_perturbed_zonal_wind(eta_v, lon[slice_grid], lat[slice_grid])
    u_component[slice_3d] = local_coordinate_transformation(u_component[slice_3d], lon[slice_grid], grid_vector_component[islice_grid, jslice_grid, :])
    return u_component

def initialize_zonal_wind(u, eta, eta_v, lon, lat, east_grid_vector_component, center_grid_vector_component, islice, islice_grid, jslice, jslice_grid, axis):
    shape = u.shape
    uu1 = wind_component_calc(shape, eta_v, lon, lat,  east_grid_vector_component, islice, islice, jslice, jslice_grid)
    uu3 = wind_component_calc(shape, eta_v, lon, lat,  east_grid_vector_component, islice, islice_grid, jslice, jslice)
    upper = (slice(None),) * axis + (slice(0, -1),) 
    lower = (slice(None),) * axis + (slice(1, None),)
    pa1, pa2 = lon_lat_midpoint(lon[upper], lon[lower], lat[upper], lat[lower], np)
    uu2 = wind_component_calc(shape, eta_v, pa1, pa2, center_grid_vector_component,islice, islice, jslice, jslice)
    u[islice, jslice,:] = 0.25 * (uu1 + 2.0 * uu2 + uu3)[islice, jslice,:]


def compute_grid_edge_midpoint_latitude_components(lon, lat):
    _, lat_avg_x_south = lon_lat_midpoint(lon[0:-1, :], lon[1:, :], lat[0:-1, :], lat[1:, :], np)
    _, lat_avg_y_east = lon_lat_midpoint(lon[1:, 0:-1], lon[1:, 1:], lat[1:, 0:-1], lat[1:, 1:], np)
    _, lat_avg_x_north = lon_lat_midpoint(lon[0:-1, 1:], lon[1:, 1:], lat[0:-1, 1:], lat[1:, 1:], np)
    _, lat_avg_y_west  = lon_lat_midpoint(lon[:, 0:-1], lon[:, 1:], lat[:, 0:-1], lat[:, 1:], np)
    return  lat_avg_x_south, lat_avg_y_east, lat_avg_x_north,  lat_avg_y_west

def cell_average_nine_point(pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9):
    """
    9-point average: should be 2nd order accurate for a rectangular cell
    9  4  8
    5  1  3          
    6  2  7
    """
    return 0.25 * pt1 + 0.125 * (pt2 + pt3 + pt4 + pt5) + 0.0625 * (pt6 + pt7 + pt8 + pt9)

def cell_average_nine_components(component_function, component_args, lat, lat_agrid, lat2, lat3, lat4, lat5, grid_slice):
    """
    Componet 9 cell components of a variable, calling a component_function definiting the equation with component_args
    (arguments unique to that component_function), and precomputed lat arrays 
    """
    pt1 = component_function(*component_args, lat=lat_agrid[grid_slice])
    pt2 = component_function(*component_args, lat=lat2[grid_slice])
    pt3 = component_function(*component_args, lat=lat3[grid_slice])
    pt4 = component_function(*component_args, lat=lat4[grid_slice])
    pt5 = component_function(*component_args, lat=lat5[grid_slice])
    pt6 = component_function(*component_args, lat=lat[grid_slice])
    pt7 = component_function(*component_args, lat=lat[1:,:][grid_slice])
    pt8 = component_function(*component_args, lat=lat[1:,1:][grid_slice])
    pt9 = component_function(*component_args, lat=lat[:,1:][grid_slice])
    return cell_average_nine_point(pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8, pt9)



def initialize_delz(pt, peln):
    return constants.RDG * pt[:, :, :-1] * (peln[:, :, 1:] - peln[:, :, :-1])


def moisture_adjusted_temperature(pt, qvapor):
    """
    Update initial temperature to include water vapor contribution
    """
    return pt/(1. + constants.ZVIR * qvapor)
    
def baroclinic_initialization(peln, qvapor, delp, u, v, pt, phis, delz, w, eta, eta_v, lon, lat, lon_agrid, lat_agrid, ee1, ee2, es1, ew2, ptop, adiabatic, hydrostatic):

    nx = delp.shape[0] - 1
    ny = delp.shape[1] - 1

    # Equation (2) for v
    # Although meridional wind is 0 in this scheme
    # on the cubed sphere grid, v is not 0 on every tile
    initialize_zonal_wind(v, eta, eta_v, lon, lat,
                          east_grid_vector_component=ee2,
                          center_grid_vector_component=ew2,
                          islice=slice(0, nx + 1),
                          islice_grid=slice(0,  nx + 1),
                          jslice=slice(0, ny),
                          jslice_grid=slice(1, ny + 1),
                          axis=1)

    initialize_zonal_wind(u, eta, eta_v, lon, lat,
                          east_grid_vector_component=ee1,
                          center_grid_vector_component=es1,
                          islice=slice(0, nx),
                          islice_grid=slice(1,  nx + 1),
                          jslice=slice(0, ny + 1),
                          jslice_grid=slice(0,  ny + 1),
                          axis=0)
    
    # Compute cell lats in the midpoint of each cell edge
    lat2, lat3, lat4, lat5 =  compute_grid_edge_midpoint_latitude_components(lon, lat)
    slice_3d = (slice(0, nx), slice(0, ny), slice(None))
    slice_2d = (slice(0, nx), slice(0, ny))
    # initialize temperature
    t_mean = jablonowski_init.horizontally_averaged_temperature(eta)
    pt[slice_3d] = cell_average_nine_components(jablonowski_init.temperature, [eta, eta_v, t_mean], lat, lat_agrid, lat2, lat3, lat4, lat5, slice_2d)

    # initialize surface geopotential
    phis[slice_2d] = cell_average_nine_components(jablonowski_init.surface_geopotential_perturbation, [], lat, lat_agrid, lat2, lat3, lat4, lat5, slice_2d)
    
    if not hydrostatic:
        # vertical velocity is set to 0 for nonhydrostatic setups
        w[slice_3d] = 0.0
        delz[:nx, :ny, :-1] = initialize_delz(pt[slice_3d], peln[slice_3d])
        
    if not adiabatic:
        pt[slice_3d] = moisture_adjusted_temperature(pt[slice_3d], qvapor[slice_3d])



def initialize_pkz_moist(delp, pt, qvapor, delz):
     return np.exp(constants.KAPPA * np.log(constants.RDG * delp[:, :, :-1] * pt[:, :, :-1] * (1. + constants.ZVIR * qvapor[:, :, :-1]) / delz[:, :, :-1]))

def initialize_pkz_dry(delp, pt, delz):
    return np.exp(constants.KAPPA * np.log(constants.RDG * delp[:, :, :-1] * pt[:, :, :-1] / delz[:, :, :-1]))

def fix_top_log_edge_pressure(peln, ptop):
    if ptop < ptop_min:
        ak1 = (constants.KAPPA + 1.0) / constants.KAPPA
        peln[:, :, 0] =  peln[:, :, 1] - ak1
    else:
        peln[:, :, 0] = np.log(ptop)
        
def p_var(delp, delz, pt, ps, qvapor, pe, peln, pk, pkz, ptop, moist_phys, make_nh, hydrostatic=False,  adjust_dry_mass=False):
    """
    Computes auxiliary pressure variables for a hydrostatic state.
    The variables are: surface, interface, layer-mean pressure, exner function
    Given (ptop, delp) computes (ps, pk, pe, peln, pkz)
    """
    assert(not adjust_dry_mass)
    assert(not hydrostatic)
     
    ps[:] = pe[:, :, -1]
    fix_top_log_edge_pressure(peln, ptop)

    if not hydrostatic:
        if make_nh:
            delz[:]= 1.e25
            delz[:, :, :-1] = initialize_delz(pt, peln)
        if moist_phys:
            pkz[:, :, :-1] = initialize_pkz_moist(delp, pt, qvapor, delz)
        else:
            pkz[:, :, :-1] = initialize_pkz_dry(delp, pt, delz)



def horizontal_compute_shape(full_array_shape):
    full_nx, full_ny, _ = full_array_shape
    nx = full_nx - 2*  nhalo - 1
    ny = full_ny - 2 * nhalo - 1
    return nx, ny

def compute_slices(nx, ny):
    islice = slice(nhalo, nhalo + nx)
    jslice = slice(nhalo, nhalo + ny)
    slice_3d = (islice, jslice, slice(None))
    slice_2d = (islice, jslice)
    return islice, jslice, slice_3d, slice_2d

            
def init_case(ua, va, uc, vc, eta, eta_v, delp, ps, pe, peln, pk, pkz, qvapor, ak, bk, ptop, u, v, pt, phis, delz, w,  lon, lat, lon_agrid, lat_agrid, ee1, ee2, es1, ew2, adiabatic, hydrostatic, moist_phys): 
    nx, ny = horizontal_compute_shape(delp.shape)  
    delp[:] = 1e30
    delp[:nhalo, :nhalo] = 0.0
    delp[:nhalo, nhalo + ny:] = 0.0
    delp[nhalo + nx:, :nhalo] = 0.0
    delp[nhalo + nx:,  nhalo + ny:] = 0.0
    pe[:] = 0.0
    pt[:] = 1.0
    ua[:] = 1e35
    va[:] = 1e35
    uc[:] = 1e30
    vc[:] = 1e30
    delz[:]= 1.e25
    w[:] = 1.e30
    phis[:] =  1.e25
    ps[:] = jablonowski_init.surface_pressure
    islice, jslice, slice_3d, slice_2d = compute_slices(nx, ny)
    isliceb, jsliceb, slice_3db, slice_2db = compute_slices(nx + 1, ny+ 1)
    setup_pressure_fields(eta, eta_v, delp[slice_3d], ps[slice_2d], pe[slice_3d], peln[slice_3d], pk[slice_3d], pkz[slice_3d], qvapor[slice_3d], ak, bk, ptop, lat_agrid=lat_agrid[:-1, :-1][slice_2d], adiabatic=adiabatic)
    baroclinic_initialization(peln[slice_3db], qvapor[slice_3db], delp[slice_3db], u[slice_3db], v[slice_3db], pt[slice_3db], phis[slice_2db], delz[slice_3db], w[slice_3db], eta, eta_v,  lon[slice_2db], lat[slice_2db], lon_agrid[slice_2db], lat_agrid[slice_2db], ee1[slice_3db], ee2[slice_3db], es1[slice_3db], ew2[slice_3db], ptop, adiabatic, hydrostatic)
  
    p_var(delp[slice_3d], delz[slice_3d], pt[slice_3d], ps[slice_2d], qvapor[slice_3d], pe[slice_3d], peln[slice_3d], pk[slice_3d], pkz[slice_3d], ptop, moist_phys, make_nh=(not hydrostatic), hydrostatic=hydrostatic)
  
    # halo update phis
    # halo update u and v
