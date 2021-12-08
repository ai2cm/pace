ROOT_RANK = 0
X_DIM = "x"
X_INTERFACE_DIM = "x_interface"
Y_DIM = "y"
Y_INTERFACE_DIM = "y_interface"
Z_DIM = "z"
Z_INTERFACE_DIM = "z_interface"
Z_SOIL_DIM = "z_soil"
TILE_DIM = "tile"
X_DIMS = (X_DIM, X_INTERFACE_DIM)
Y_DIMS = (Y_DIM, Y_INTERFACE_DIM)
Z_DIMS = (Z_DIM, Z_INTERFACE_DIM)
HORIZONTAL_DIMS = X_DIMS + Y_DIMS
INTERFACE_DIMS = (X_INTERFACE_DIM, Y_INTERFACE_DIM, Z_INTERFACE_DIM)

WEST = 0
EAST = 1
NORTH = 2
SOUTH = 3
NORTHWEST = 4
NORTHEAST = 5
SOUTHWEST = 6
SOUTHEAST = 7
INTERIOR = 8
EDGE_BOUNDARY_TYPES = (NORTH, SOUTH, WEST, EAST)
CORNER_BOUNDARY_TYPES = (NORTHWEST, NORTHEAST, SOUTHWEST, SOUTHEAST)
BOUNDARY_TYPES = EDGE_BOUNDARY_TYPES + CORNER_BOUNDARY_TYPES
N_HALO_DEFAULT = 3

#####################
# Physical constants
#####################
RADIUS = 6.3712e6  # Radius of the Earth [m] #6371.0e3
PI = 3.1415926535897931  # 3.14159265358979323846
OMEGA = 7.2921e-5  # Rotation of the earth  # 7.292e-5
GRAV = 9.80665  # Acceleration due to gravity [m/s^2]
RDGAS = 287.05  # Gas constant for dry air [J/kg/deg] # 287.04
RVGAS = 461.50  # Gas constant for water vapor [J/kg/deg]
HLV = 2.5e6  # Latent heat of evaporation [J/kg]
HLF = 3.3358e5  # Latent heat of fusion [J/kg]  # 3.34e5
RGRAV = 1.0 / GRAV  # Inverse of gravitational acceleration
# CP_AIR: Specific heat capacity of dry air at
# constant pressure [J/kg/deg] # RDGAS / KAPPA
CP_AIR = 1004.6  # Heat capacity of dry air at constant pressure
KAPPA = RDGAS / CP_AIR  # 2.0 / 7.0
DZ_MIN = 2.0
CV_AIR = CP_AIR - RDGAS  # Heat capacity of dry air at constant volume
RDG = -RDGAS / GRAV
CNST_0P20 = 0.2
TFREEZE = 273.15  # Freezing temperature of fresh water [K]
K1K = RDGAS / CV_AIR
CNST_0P20 = 0.2
CV_VAP = 3.0 * RVGAS  # Heat capacity of water vapor at constant volume
CV_AIR = CP_AIR - RDGAS  # Heat capacity of dry air at constant volume
ZVIR = RVGAS / RDGAS - 1  # con_fvirt in Fortran physics
C_ICE = 1972.0  # Heat capacity of ice at -15 degrees Celsius
C_LIQ = 4.1855e3  # Heat capacity of water at 15 degrees Celsius
CP_VAP = 4.0 * RVGAS  # Heat capacity of water vapor at constant pressure
TICE = 273.16  # Freezing temperature
TICE_MICRO = (
    TICE  # Freezing temp, set to 165. to turn off ice-phase phys (kessler emulator)
)
DC_ICE = C_LIQ - C_ICE  # Isobaric heating / cooling
DC_VAP = CP_VAP - C_LIQ  # Isobaric heating / cooling
D2ICE = DC_VAP + DC_ICE  # Isobaric heating / cooling
LI0 = HLF - DC_ICE * TICE
EPS = RDGAS / RVGAS
LV0 = (
    HLV - DC_VAP * TICE
)  # 3.13905782e6, evaporation latent heat coefficient at 0 degrees Kelvin
LI00 = (
    HLF - DC_ICE * TICE
)  # -2.7105966e5, fusion latent heat coefficient at 0 degrees Kelvin
LI2 = (
    LV0 + LI00
)  # 2.86799816e6, sublimation latent heat coefficient at 0 degrees Kelvin
E00 = 611.21  # Saturation vapor pressure at 0 degrees Celsius
T_WFR = TICE - 40.0  # homogeneous freezing temperature
TICE0 = TICE - 0.01
T_MIN = 178.0  # Minimum temperature to freeze-dry all water vapor
T_SAT_MIN = TICE - 160.0
LAT2 = (HLV + HLF) ** 2  # used in bigg mechanism
##################
# Physics only
###################
RHOS = 1.0e2
RHOG = 4.0e2
QCMIN = 1.0e-12  # Minimum value for cloud condensation
VR_MIN = 1.0e-3  # Minimum fall speed for rain
VF_MIN = 1.0e-5  # Minimum fall speed for cloud ice, snow, graupel

QRMIN = 1.0e-8  # Minimum value for rain water
QVMIN = 1.0e-20  # Minimum value for water vapor (treated as zero)

DZ_MIN_FLIP = 1.0e-2  # Use for correcting flipped height
SFCRHO = 1.2  # Surface air density
RHOR = 1.0e3  # Density of rain water, lin83
DT_FR = 8.0  # Homogeneous freezing of all cloud water at t_wfr - dt_fr
P_MIN = 100.0  # Minimum pressure (Pascal) for mp to operate

ALIN = 842.0  # "a" in lin1983
CLIN = 4.8  # "c" in lin 1983, 4.8 -- > 6. (to ehance ql -- > qs)

VI_FAC = 1.0  # if const_vi: 1/3
VS_FAC = 1.0  # if const_vs: 1.
VG_FAC = 1.0  # if const_vg: 2.
VR_FAC = 1.0  # if const_vr: 4.


# Fall velocity constants
VCONR = 2503.23638966667
NORMR = 25132741228.7183
THR = 1.0e-8
THI = 1.0e-8  # Cloud ice threshold for terminal fall
THG = 1.0e-8
THS = 1.0e-8
AA = -4.14122e-5
BB = -0.00538922
CC = -0.0516344
DD_FS = 0.00216078
EE = 1.9714

# Marshall-Palmer constants ###
VCONS = 6.6280504
VCONG = 87.2382675
NORMS = 942477796.076938
NORMG = 5026548245.74367


########################################################################
# Tunable parameters (Fortran namelist parameters)
########################################################################

c_cracw = 0.8  # Rain accretion efficiency
c_paut = 0.5  # Autoconversion cloud water to rain (use 0.5 to reduce autoconversion)
c_pgacs = 0.01  # Snow to graupel "accretion" eff. (was 0.1 in zetac)
c_psaci = 0.05  # Accretion: cloud ice to snow (was 0.1 in zetac)
ccn_l = 300.0  # CCN over land (cm^-3)
ccn_o = 100.0  # CCN over ocean (cm^-3)
const_vg = 0  # Fall velocity tuning constant of graupel
const_vi = 0  # Fall velocity tuning constant of ice
const_vr = 0  # Fall velocity tuning constant of rain water
const_vs = 0  # Fall velocity tuning constant of snow
de_ice = 0  # To prevent excessive build-up of cloud ice from external sources
do_qa = 1  # Do inline cloud fraction
do_sedi_heat = 0  # Transport of heat in sedimentation
dw_land = 0.15  # Base value for subgrid deviation / variability over land
dw_ocean = 0.1  # Base value for ocean
fast_sat_adj = 1  # Has fast saturation adjustments
fix_negative = 1  # Fix negative water species
irain_f = 0  # Cloud water to rain auto conversion scheme
mono_prof = False  # Perform terminal fall with mono ppm scheme
mp_time = 225.0  # Maximum microphysics timestep (sec)
prog_ccn = 0  # Do prognostic ccn (yi ming's method)
qi0_crt = 8e-05  # Cloud ice to snow autoconversion threshold
qi_lim = 1.0  # Cloud ice limiter to prevent large ice build up
ql_mlt = 0.002  # Maximum value of cloud water allowed from melted cloud ice
qs0_crt = 0.003  # Snow to graupel density threshold (0.6e-3 in purdue lin scheme)
qs_mlt = 1e-06  # Maximum cloud water due to snow melt
rad_rain = 1  # Consider rain in cloud fraction calculation
rad_snow = 1  # Consider snow in cloud fraction calculation
rh_inc = 0.2  # RH increment for complete evaporation of cloud water and cloud ice
rh_inr = 0.3  # RH increment for minimum evaporation of rain
rthresh = 1e-05  # Critical cloud drop radius (micrometers)
sedi_transport = 1  # Transport of momentum in sedimentation
tau_g2v = 1200.0  # Graupel sublimation
tau_i2s = 1000.0  # Cloud ice to snow autoconversion
tau_l2v = 300.0  # Cloud water to water vapor (evaporation)
tau_v2l = 90.0  # Water vapor to cloud water (condensation)
use_ppm = 0  # Use ppm fall scheme
vg_max = 16.0  # Maximum fall speed for graupel
vi_max = 1.0  # Maximum fall speed for ice
vr_max = 16.0  # Maximum fall speed for rain
vs_max = 2.0  # Maximum fall speed for snow
z_slope_ice = 1  # Use linear mono slope for autoconversions
z_slope_liq = 1  # Use linear mono slope for autoconversions
