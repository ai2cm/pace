from typing import Iterable, Mapping, Union

from .constants import (
    X_DIM,
    X_INTERFACE_DIM,
    Y_DIM,
    Y_INTERFACE_DIM,
    Z_DIM,
    Z_INTERFACE_DIM,
    Z_SOIL_DIM,
)


RestartProperties = Mapping[str, Mapping[str, Union[str, Iterable[str]]]]
RESTART_PROPERTIES: RestartProperties = {
    "accumulated_x_courant_number": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "cx",
        "units": "",
    },
    "accumulated_x_mass_flux": {
        "dims": [Z_DIM, Y_DIM, X_INTERFACE_DIM],
        "restart_name": "mfx",
        "units": "unknown",
    },
    "accumulated_y_courant_number": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "cy",
        "units": "unknown",
    },
    "accumulated_y_mass_flux": {
        "dims": [Z_DIM, Y_INTERFACE_DIM, X_DIM],
        "restart_name": "mfy",
        "units": "unknown",
    },
    "air_temperature": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "T",
        "units": "degK",
    },
    "air_temperature_after_physics": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "gt0",
        "units": "K",
    },
    "air_temperature_at_2m": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "t2m",
        "units": "degK",
    },
    "area_of_grid_cell": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "area",
        "units": "m^2",
    },
    "atmosphere_hybrid_a_coordinate": {
        "dims": [Z_INTERFACE_DIM],
        "restart_name": "ak",
        "units": "Pa",
    },
    "atmosphere_hybrid_b_coordinate": {
        "dims": [Z_INTERFACE_DIM],
        "restart_name": "bk",
        "units": "",
    },
    "canopy_water": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "canopy",
        "units": "unknown",
    },
    "clear_sky_downward_longwave_flux_at_surface": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "dnfx0",
        "restart_name": "sfcflw",
        "units": "W/m^2",
    },
    "clear_sky_downward_shortwave_flux_at_surface": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "dnfx0",
        "restart_name": "sfcfsw",
        "units": "W/m^2",
    },
    "clear_sky_upward_longwave_flux_at_surface": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "upfx0",
        "restart_name": "sfcflw",
        "units": "W/m^2",
    },
    "clear_sky_upward_longwave_flux_at_top_of_atmosphere": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "upfx0",
        "restart_name": "topflw",
        "units": "W/m^2",
    },
    "clear_sky_upward_shortwave_flux_at_surface": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "upfx0",
        "restart_name": "sfcfsw",
        "units": "W/m^2",
    },
    "clear_sky_upward_shortwave_flux_at_top_of_atmosphere": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "upfx0",
        "restart_name": "topfsw",
        "units": "W/m^2",
    },
    "convective_cloud_bottom_pressure": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "cvb",
        "units": "Pa",
    },
    "convective_cloud_fraction": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "cv",
        "units": "",
    },
    "convective_cloud_top_pressure": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "cvt",
        "units": "Pa",
    },
    "deep_soil_temperature": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "tg3",
        "units": "degK",
    },
    "dissipation_estimate_from_heat_source": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "diss_est",
        "units": "unknown",
    },
    "eastward_wind": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "ua",
        "units": "m/s",
    },
    "eastward_wind_after_physics": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "gu0",
        "units": "m/s",
    },
    "eastward_wind_at_surface": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "u_srf",
        "units": "m/s",
    },
    "fh_parameter": {
        "description": "used in PBL scheme",
        "dims": [Y_DIM, X_DIM],
        "restart_name": "ffhh",
        "units": "unknown",
    },
    "fm_at_10m": {
        "description": "Ratio of sigma level 1 wind and 10m wind",
        "dims": [Y_DIM, X_DIM],
        "restart_name": "f10m",
        "units": "unknown",
    },
    "fm_parameter": {
        "description": "used in PBL scheme",
        "dims": [Y_DIM, X_DIM],
        "restart_name": "ffmm",
        "units": "unknown",
    },
    "fractional_coverage_with_strong_cosz_dependency": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "facsf",
        "units": "",
    },
    "fractional_coverage_with_weak_cosz_dependency": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "facwf",
        "units": "",
    },
    "friction_velocity": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "uustar",
        "units": "m/s",
    },
    "ice_fraction_over_open_water": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "fice",
        "units": "",
    },
    "interface_pressure": {
        "dims": [Y_DIM, Z_INTERFACE_DIM, X_DIM],
        "restart_name": "pe",
        "units": "Pa",
    },
    "interface_pressure_raised_to_power_of_kappa": {
        "dims": [Z_INTERFACE_DIM, Y_DIM, X_DIM],
        "restart_name": "pk",
        "units": "unknown",
    },
    "land_sea_mask": {
        "description": "sea=0, land=1, sea-ice=2",
        "dims": [Y_DIM, X_DIM],
        "restart_name": "slmsk",
        "units": "",
    },
    "latent_heat_flux": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "dqsfci",
        "units": "W/m^2",
    },
    "latitude": {"dims": [Y_DIM, X_DIM], "restart_name": "xlat", "units": "radians"},
    "layer_mean_pressure_raised_to_power_of_kappa": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "pkz",
        "units": "unknown",
    },
    "liquid_soil_moisture": {
        "dims": [Z_SOIL_DIM, Y_DIM, X_DIM],
        "restart_name": "slc",
        "units": "unknown",
    },
    "logarithm_of_interface_pressure": {
        "dims": [Y_DIM, Z_INTERFACE_DIM, X_DIM],
        "restart_name": "peln",
        "units": "ln(Pa)",
    },
    "longitude": {"dims": [Y_DIM, X_DIM], "restart_name": "xlon", "units": "radians"},
    "maximum_fractional_coverage_of_green_vegetation": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "shdmax",
        "units": "",
    },
    "maximum_snow_albedo_in_fraction": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "snoalb",
        "units": "",
    },
    "mean_cos_zenith_angle": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "coszen",
        "units": "",
    },
    "mean_near_infrared_albedo_with_strong_cosz_dependency": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "alnsf",
        "units": "",
    },
    "mean_near_infrared_albedo_with_weak_cosz_dependency": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "alnwf",
        "units": "",
    },
    "mean_visible_albedo_with_strong_cosz_dependency": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "alvsf",
        "units": "",
    },
    "mean_visible_albedo_with_weak_cosz_dependency": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "alvwf",
        "units": "",
    },
    "minimum_fractional_coverage_of_green_vegetation": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "shdmin",
        "units": "",
    },
    "northward_wind": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "va",
        "units": "m/s",
    },
    "northward_wind_after_physics": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "gv0",
        "units": "m/s",
    },
    "northward_wind_at_surface": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "v_srf",
        "units": "m/s",
    },
    "pressure_thickness_of_atmospheric_layer": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "delp",
        "units": "Pa",
    },
    "sea_ice_thickness": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "hice",
        "units": "unknown",
    },
    "sensible_heat_flux": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "dtsfci",
        "units": "W/m^2",
    },
    "snow_cover_in_fraction": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "sncovr",
        "units": "",
    },
    "snow_depth_water_equivalent": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "snwdph",
        "units": "mm",
    },
    "snow_rain_flag": {
        "description": "snow/rain flag for precipitation",
        "dims": [Y_DIM, X_DIM],
        "restart_name": "srflag",
        "units": "",
    },
    "soil_temperature": {
        "dims": [Z_SOIL_DIM, Y_DIM, X_DIM],
        "restart_name": "stc",
        "units": "degK",
    },
    "soil_type": {"dims": [Y_DIM, X_DIM], "restart_name": "stype", "units": ""},
    "specific_humidity_at_2m": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "q2m",
        "units": "kg/kg",
    },
    "surface_geopotential": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "phis",
        "units": "m^2 s^-2",
    },
    "surface_pressure": {"dims": [Y_DIM, X_DIM], "restart_name": "ps", "units": "Pa"},
    "surface_roughness": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "zorl",
        "units": "cm",
    },
    "surface_slope_type": {
        "description": "used in land surface model",
        "dims": [Y_DIM, X_DIM],
        "restart_name": "slope",
        "units": "",
    },
    "surface_temperature": {
        "description": "surface skin temperature",
        "dims": [Y_DIM, X_DIM],
        "restart_name": "tsea",
        "units": "degK",
    },
    "surface_temperature_over_ice_fraction": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "tisfc",
        "units": "degK",
    },
    "total_condensate_mixing_ratio": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "q_con",
        "units": "kg/kg",
    },
    "total_precipitation": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "tprcp",
        "units": "m",
    },
    "total_sky_downward_longwave_flux_at_surface": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "dnfxc",
        "restart_name": "sfcflw",
        "units": "W/m^2",
    },
    "total_sky_downward_shortwave_flux_at_surface": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "dnfxc",
        "restart_name": "sfcfsw",
        "units": "W/m^2",
    },
    "total_sky_downward_shortwave_flux_at_top_of_atmosphere": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "dnfxc",
        "restart_name": "topfsw",
        "units": "W/m^2",
    },
    "total_sky_upward_longwave_flux_at_surface": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "upfxc",
        "restart_name": "sfcflw",
        "units": "W/m^2",
    },
    "total_sky_upward_longwave_flux_at_top_of_atmosphere": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "upfxc",
        "restart_name": "topflw",
        "units": "W/m^2",
    },
    "total_sky_upward_shortwave_flux_at_surface": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "upfxc",
        "restart_name": "sfcfsw",
        "units": "W/m^2",
    },
    "total_sky_upward_shortwave_flux_at_top_of_atmosphere": {
        "dims": [Y_DIM, X_DIM],
        "fortran_subname": "upfxc",
        "restart_name": "topfsw",
        "units": "W/m^2",
    },
    "total_soil_moisture": {
        "dims": [Z_SOIL_DIM, Y_DIM, X_DIM],
        "restart_name": "smc",
        "units": "unknown",
    },
    "vegetation_fraction": {
        "dims": [Y_DIM, X_DIM],
        "restart_name": "vfrac",
        "units": "",
    },
    "vegetation_type": {"dims": [Y_DIM, X_DIM], "restart_name": "vtype", "units": ""},
    "vertical_pressure_velocity": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "omga",
        "units": "Pa/s",
    },
    "vertical_thickness_of_atmospheric_layer": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "DZ",
        "units": "m",
    },
    "vertical_wind": {
        "dims": [Z_DIM, Y_DIM, X_DIM],
        "restart_name": "W",
        "units": "m/s",
    },
    "water_equivalent_of_accumulated_snow_depth": {
        "description": "weasd in Fortran code, over land and sea ice only",
        "dims": [Y_DIM, X_DIM],
        "restart_name": "sheleg",
        "units": "kg/m^2",
    },
    "x_wind": {
        "dims": [Z_DIM, Y_INTERFACE_DIM, X_DIM],
        "restart_name": "u",
        "units": "m/s",
    },
    "x_wind_on_c_grid": {
        "dims": [Z_DIM, Y_DIM, X_INTERFACE_DIM],
        "restart_name": "uc",
        "units": "m/s",
    },
    "y_wind": {
        "dims": [Z_DIM, Y_DIM, X_INTERFACE_DIM],
        "restart_name": "v",
        "units": "m/s",
    },
    "y_wind_on_c_grid": {
        "dims": [Z_DIM, Y_INTERFACE_DIM, X_DIM],
        "restart_name": "vc",
        "units": "m/s",
    },
}
