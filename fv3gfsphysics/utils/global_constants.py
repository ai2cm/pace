########################################################################
# Physical constants
########################################################################

rhos = 1.0e2
rhog = 4.0e2
grav = 9.80665  # Acceleration due to gravity
rgrav = 1.0 / grav  # Inverse of gravitational acceleration
rdgas = 287.05  # Gas constant for dry air
rvgas = 461.50  # Gas constant for water vapor

cp_air = 1004.6  # Heat capacity of dry air at constant pressure
cp_vap = 4.0 * rvgas  # Heat capacity of water vapor at constant pressure
cv_air = cp_air - rdgas  # Heat capacity of dry air at constant volume
cv_vap = 3.0 * rvgas  # Heat capacity of water vapor at constant volume
c_ice = 1972.0  # Heat capacity of ice at -15 degrees Celsius
c_liq = 4185.5  # Heat capacity of water at 15 degrees Celsius

eps = rdgas / rvgas

t_ice = 273.16  # Freezing temperature

e00 = 611.21  # Saturation vapor pressure at 0 degrees Celsius

dc_vap = cp_vap - c_liq  # Isobaric heating / cooling
dc_ice = c_liq - c_ice  # Isobaric heating / cooling

hlv = 2.5e6  # Latent heat of evaporation
hlf = 3.3358e5  # Latent heat of fusion
hlv0 = hlv  # Evaporation latent heat coefficient at 0 degrees Celsius
hlf0 = hlf  # Fusion latent heat coefficient at 0 degrees Celsius

lv0 = hlv0 - dc_vap * t_ice  # Evaporation latent heat coefficient at 0 degrees Kelvin
li00 = hlf0 - dc_ice * t_ice  # Fusion latent heat coefficient at 0 degrees Kelvin

d2ice = dc_vap + dc_ice  # Isobaric heating / cooling
li2 = lv0 + li00  # Sublimation latent heat coefficient at 0 degrees Kelvin

qcmin = 1.0e-12  # Minimum value for cloud condensation

vr_min = 1.0e-3  # Minimum fall speed for rain
vf_min = 1.0e-5  # Minimum fall speed for cloud ice, snow, graupel

qrmin = 1.0e-8  # Minimum value for rain water
qvmin = 1.0e-20  # Minimum value for water vapor (treated as zero)

dz_min = 1.0e-2  # Use for correcting flipped height

sfcrho = 1.2  # Surface air density
rhor = 1.0e3  # Density of rain water, lin83

dt_fr = 8.0  # Homogeneous freezing of all cloud water at t_wfr - dt_fr

p_min = 100.0  # Minimum pressure (Pascal) for mp to operate

qi0_max = 1.0e-4  # Max cloud ice value (by other sources)

alin = 842.0  # "a" in lin1983
clin = 4.8  # "c" in lin 1983, 4.8 -- > 6. (to ehance ql -- > qs)

vi_fac = 1.0  # if const_vi: 1/3
vs_fac = 1.0  # if const_vs: 1.
vg_fac = 1.0  # if const_vg: 2.
vr_fac = 1.0  # if const_vr: 4.

tice = 273.16  # Set tice = 165. to trun off ice-phase phys (kessler emulator)

t_min = 178.0  # Minimum temperature to freeze-dry all water vapor
t_sub = 184.0  # Minimum temperature for sublimation of cloud ice

tau_imlt = 600.0  # Cloud ice melting
tau_v2g = 21600.0  # Graupel deposition -- make it a slow process

qc_crt = 5.0e-8  # Minimum condensate mixing ratio to allow partial cloudiness

qi_gen = 1.82e-6  # Maximum cloud ice generation during remapping step

### Latent heat coefficients used in wet bulb and bigg mechanism ###
latv = hlv
lati = hlf
lats = latv + lati
lat2 = lats * lats
lcp = latv / cp_air
icp = lati / cp_air
tcp = (latv + lati) / cp_air

### Fall velocity constants ###
vconr = 2503.23638966667
normr = 25132741228.7183
thr = 1.0e-8
thi = 1.0e-8  # Cloud ice threshold for terminal fall
thg = 1.0e-8
ths = 1.0e-8
aa = -4.14122e-5
bb = -0.00538922
cc = -0.0516344
dd_fs = 0.00216078
ee = 1.9714

### Marshall-Palmer constants ###
vcons = 6.6280504
vcong = 87.2382675
norms = 942477796.076938
normg = 5026548245.74367


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
qi0_crt = 8e-05  # Cloud ice to snow autoconversion threshold (highly dependent on horizontal resolution)
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
