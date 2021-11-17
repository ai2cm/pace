import numpy as np
import math
from fv3core.grid import lon_lat_midpoint, great_circle_distance_lon_lat
import fv3core.utils.global_constants as constants
# maximum windspeed amplitude - close to windspeed of zonal-mean time-mean jet stream in troposphere
u0 = 35.0
# indicates perturbation location 20E, 40N
pcen = [math.pi / 9.0, 2.0 * math.pi / 9.0]
u1 = 1.0
pt0 = 0.0
eta_0 = 0.252
eta_s = 1.0 # surface level
eta_t = 0.2 # tropopause
t_0 = 288.0
delta_t = 480000.0
lapse_rate = 0.005

# RADIUS = 6.3712e6 in fv3 vs Jabowski paper  6.371229 1e6
R = constants.RADIUS / 10.0 # specifically for test case == 13, otherwise R = 1

"""
  Functions for computing components of a baroclinic perturbation test case, by 
  Jablonowski & Williamson Baroclinic test case Perturbation. JRMS2006
  and additional computations depicted in DCMIP2016 Tes Case Documentation
  JRMS2006 eqautions 3, 8, 9, 12, 13 are not represented
"""

def vertical_coordinate(eta_value):
    """
    Equation (1) JRMS2006
    computes eta_v, the auxiliary variable vertical coordinate
    """
    return (eta_value - eta_0) * math.pi * 0.5

def compute_eta(eta, eta_v, ak, bk):
    """
    Equation (1) JRMS2006
    eta is the vertical coordinate and eta_v is an auxiliary vertical coordinate
    """
    eta[:-1] = 0.5 * ((ak[:-1] + ak[1:])/1.e5 + bk[:-1] + bk[1:])
    eta_v[:-1] = vertical_coordinate(eta[:-1])


def zonal_wind(eta_v, latitude):
    """
    Equation (2) JRMS2006
    Returns the zonal wind u
    """
    return u0 * np.cos(eta_v[:])**(3.0/2.0) * np.sin(2.0 * latitude[:, :, None])**2.0


def apply_perturbation(u_component, up, longitude, latitude):
    """
    Apply a Gaussian perturbation to intiate a baroclinic wave in JRMS2006
    up is the maximum amplitude of the perturbation
    modifies u_component to include the perturbation of radius R
    """
    r = np.zeros((u_component.shape[0], u_component.shape[1], 1))
    # Equation (11), distance from perturbation at 20E, 40N in JRMS2006
    r = great_circle_distance_lon_lat(pcen[0], longitude,  pcen[1], latitude, constants.RADIUS, np)[:, :, None]
    r3d = np.repeat(r, u_component.shape[2], axis=2)
    near_perturbation = ((r3d/R)**2.0 < 40.0)
    # Equation(10) in JRMS2006 perturbation applied to u_component
    # Equivalent to Equation (14) in DCMIP 2016, where Zp = 1.0
    u_component[near_perturbation] = u_component[near_perturbation] + up * np.exp(-(r3d[near_perturbation]/R)**2.0)

    
def baroclinic_perturbed_zonal_wind(eta_v, longitude, latitude):
    u = zonal_wind(eta_v, latitude)
    apply_perturbation(u, u1, longitude, latitude)
    return u


def horizontally_averaged_temperature(eta):
    """
    Equations (4) and (5) JRMS2006 for characteristic temperature profile
    """
    # for troposphere:
    t_mean = t_0 * eta[:] ** (constants.RDGAS * lapse_rate / constants.GRAV)
    # above troposphere
    t_mean[eta_t > eta] = t_mean[eta_t > eta] + delta_t*(eta_t - eta[eta_t > eta])**5.0
    return t_mean


def temperature(eta, eta_v, t_mean, latitude):
    """
    Equation (6)JRMS2006
    The total temperature distribution from the horizontal-mean temperature and a horizontal variation at each level
    """
    lat = latitude[:, :, None]
    return t_mean + 0.75*(eta[:] * math.pi * u0 / constants. RDGAS) * np.sin(eta_v[:])*np.sqrt(np.cos(eta_v[:])) * (( -2.0 * (np.sin(lat)**6.0) *(np.cos(lat)**2.0 + 1.0/3.0) + 10.0/63.0 ) *2.0*u0*np.cos(eta_v[:])**(3.0/2.0) + ((8.0/5.0)*(np.cos(lat)**3.0)*(np.sin(lat)**2.0 + 2.0/3.0) - math.pi/4.0 ) * constants.RADIUS * constants.OMEGA )


def geopotential_perturbation(latitude, eta_value):
    """
    Equation (7) JRMS2006, just the perturbation component
    """
    u_comp = u0 * (np.cos(eta_value)**(3.0/2.0))
    return  u_comp * (( -2.0*(np.sin(latitude)**6.0) * (np.cos(latitude)**2.0 + 1.0/3.0) + 10.0/63.0 ) * u_comp + ((8.0/5.0)*(np.cos(latitude)**3.0)*(np.sin(latitude)**2.0 + 2.0/3.0) - math.pi/4.0 )*constants.RADIUS * constants.OMEGA)

def surface_geopotential_perturbation(latitude):
    """
    From JRMS2006:
    * 'In hydrostatic models with pressure-based vertical coordinates, it's only necessary to initialize 
    surface geopotential.'
    * 'balances the non-zero zonal wind at the surface with surface elevation zs' 
    """
    surface_level = vertical_coordinate(eta_s)
    return geopotential_perturbation(latitude, surface_level)
