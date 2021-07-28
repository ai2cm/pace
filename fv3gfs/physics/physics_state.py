from fv3gfs.physics.stencils.microphysics import MicrophysicsState
from fv3core.decorators import ArgSpec, get_namespace


class PhysicsState:
    arg_specs = (
        ArgSpec("qvapor", "specific_humidity", "kg/kg", intent="inout"),
        ArgSpec("qliquid", "cloud_water_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qrain", "rain_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qsnow", "snow_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qice", "cloud_ice_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qgraupel", "graupel_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qo3mr", "ozone_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("qsgs_tke", "turbulent_kinetic_energy", "m**2/s**2", intent="inout"),
        ArgSpec("qcld", "cloud_fraction", "", intent="inout"),
        ArgSpec("pt", "air_temperature", "degK", intent="inout"),
        ArgSpec(
            "delp", "pressure_thickness_of_atmospheric_layer", "Pa", intent="inout"
        ),
        ArgSpec("delz", "vertical_thickness_of_atmospheric_layer", "m", intent="inout"),
        ArgSpec("peln", "logarithm_of_interface_pressure", "ln(Pa)", intent="inout"),
        ArgSpec("u", "x_wind", "m/s", intent="inout"),
        ArgSpec("v", "y_wind", "m/s", intent="inout"),
        ArgSpec("w", "vertical_wind", "m/s", intent="inout"),
        ArgSpec("ua", "eastward_wind", "m/s", intent="inout"),
        ArgSpec("va", "northward_wind", "m/s", intent="inout"),
        ArgSpec("uc", "x_wind_on_c_grid", "m/s", intent="inout"),
        ArgSpec("vc", "y_wind_on_c_grid", "m/s", intent="inout"),
        ArgSpec("q_con", "total_condensate_mixing_ratio", "kg/kg", intent="inout"),
        ArgSpec("pe", "interface_pressure", "Pa", intent="inout"),
        ArgSpec("phis", "surface_geopotential", "m^2 s^-2", intent="in"),
        ArgSpec(
            "pk",
            "interface_pressure_raised_to_power_of_kappa",
            "unknown",
            intent="inout",
        ),
        ArgSpec(
            "pkz",
            "layer_mean_pressure_raised_to_power_of_kappa",
            "unknown",
            intent="inout",
        ),
        ArgSpec("ps", "surface_pressure", "Pa", intent="inout"),
        ArgSpec("omga", "vertical_pressure_velocity", "Pa/s", intent="inout"),
        ArgSpec("ak", "atmosphere_hybrid_a_coordinate", "Pa", intent="in"),
        ArgSpec("bk", "atmosphere_hybrid_b_coordinate", "", intent="in"),
        ArgSpec("mfxd", "accumulated_x_mass_flux", "unknown", intent="inout"),
        ArgSpec("mfyd", "accumulated_y_mass_flux", "unknown", intent="inout"),
        ArgSpec("cxd", "accumulated_x_courant_number", "", intent="inout"),
        ArgSpec("cyd", "accumulated_y_courant_number", "", intent="inout"),
        ArgSpec(
            "diss_estd",
            "dissipation_estimate_from_heat_source",
            "unknown",
            intent="inout",
        ),
    )

    def __init__(self, state, grid):
        self.state = get_namespace(self.arg_specs, state)
        self.grid = grid

    @property
    def microphysics(self) -> MicrophysicsState:
        microphysics_state = MicrophysicsState(
            self.grid,
            self.state.pt_quantity,
            self.state.qvapor_quantity,
            self.state.qliquid_quantity,
            self.state.qrain_quantity,
            self.state.qice_quantity,
            self.state.qsnow_quantity,
            self.state.qgraupel_quantity,
            self.state.qcld_quantity,
            self.state.ua_quantity,
            self.state.va_quantity,
            self.state.delp_quantity,
            self.state.delz_quantity,
            self.state.omga_quantity,
        )
        return microphysics_state
