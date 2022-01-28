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
SPATIAL_DIMS = X_DIMS + Y_DIMS + Z_DIMS

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
ZVIR = RVGAS / RDGAS - 1  # con_fvirt in Fortran physics
C_ICE = 1972.0  # Heat capacity of ice at -15 degrees Celsius
C_LIQ = 4.1855e3  # Heat capacity of water at 15 degrees Celsius
CP_VAP = 4.0 * RVGAS  # Heat capacity of water vapor at constant pressure
TICE = 273.16  # Freezing temperature
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
