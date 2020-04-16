RADIUS = 6371.0e3  # 6.3712e6  # Radius of the Earth [m]
PI = 3.14159265358979323846  # 3.141592653589793
OMEGA = 7.292e-5  # 7.2921e-5  # Rotation of the earth
GRAV = 9.8  # 9.80665  # Acceleration due to gravity [m/s^2]
RDGAS = 287.04  # 287.05  # Gas constant for dry air [J/kg/deg]
RVGAS = 461.50  # Gas constant for water vapor [J/kg/deg]
HLV = 2.5e6  # Latent heat of evaporation [J/kg]
HLF = 3.34e5  # 3.3358e5  # Latent heat of fusion [J/kg]
KAPPA = 2.0 / 7.0  # RDGAS / CP_AIR
# 1004.6  # Specific heat capacity of dry air at constant pressure [J/kg/deg]
CP_AIR = RDGAS / KAPPA
DZ_MIN = 2.0
CV_AIR = CP_AIR - RDGAS
RDG = -RDGAS / GRAV
CNST_0P20 = 0.2
