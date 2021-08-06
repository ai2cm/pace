from fv3gfs.physics.stencils.microphysics import MicrophysicsState
from fv3core.decorators import ArgSpec, get_namespace
from fv3gfs.util.quantity import Quantity
import fv3core.utils.gt4py_utils as utils
import copy
import fv3gfs.util as fv3util


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
        ArgSpec("ua", "eastward_wind", "m/s", intent="inout"),
        ArgSpec("va", "northward_wind", "m/s", intent="inout"),
        ArgSpec("w", "vertical_wind", "m/s", intent="inout"),
        ArgSpec("omga", "vertical_pressure_velocity", "Pa/s", intent="inout"),
        ArgSpec("phii", "interface_geopotential_height", "m", intent="inout"),
        ArgSpec("phil", "layer_geopotential_height", "m", intent="inout"),
        ArgSpec("dz", "geopotential_height_thickness", "m", intent="inout"),
        ArgSpec("wmp", "layer_mean_vertical_velocity_microph", "m/s", intent="inout"),
    )

    def __init__(self, state, grid):
        self.grid = grid
        self.physics_state = self.make_physics_state_from_dycore(state)
        self.physics_state = self.make_physics_state_internal(self.physics_state)
        self.physics_state = get_namespace(self.arg_specs, self.physics_state)

    def make_physics_state_from_dycore(self, dycore_state):
        phy_vars = [
            "air_temperature",
            "specific_humidity",
            "cloud_water_mixing_ratio",
            "rain_mixing_ratio",
            "cloud_ice_mixing_ratio",
            "snow_mixing_ratio",
            "graupel_mixing_ratio",
            "ozone_mixing_ratio",
            "turbulent_kinetic_energy",
            "cloud_fraction",
            "eastward_wind",
            "northward_wind",
            "vertical_wind",
            "pressure_thickness_of_atmospheric_layer",
            "vertical_thickness_of_atmospheric_layer",
            "vertical_pressure_velocity",
        ]
        phy_state = {key: copy.deepcopy(dycore_state[key]) for key in phy_vars}
        return phy_state

    def make_physics_state_internal(self, phy_state):
        """
        Make internal quantities used for physics.
        These variables do not exist in the dynamical core.
        """
        internal_vars = [
            "interface_geopotential_height",
            "layer_geopotential_height",
            "geopotential_height_thickness",
            "layer_mean_vertical_velocity_microph",
        ]
        dimensions = [
            [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
            [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
        ]
        units = ["m", "m", "m", "m/s"]
        origin = self.grid.compute_origin()
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        storage = utils.make_storage_from_shape(shape, origin=origin, init=True)
        for i in range(len(internal_vars)):
            quantity = fv3util.Quantity(
                storage, dims=dimensions[i], units=units[i], origin=origin, extent=None
            )
            phy_state[internal_vars[i]] = copy.deepcopy(quantity)
        return phy_state

    @property
    def microphysics(self) -> MicrophysicsState:
        microphysics_state = MicrophysicsState(
            self.grid,
            self.physics_state.pt_quantity,
            self.physics_state.qvapor_quantity,
            self.physics_state.qliquid_quantity,
            self.physics_state.qrain_quantity,
            self.physics_state.qice_quantity,
            self.physics_state.qsnow_quantity,
            self.physics_state.qgraupel_quantity,
            self.physics_state.qcld_quantity,
            self.physics_state.ua_quantity,
            self.physics_state.va_quantity,
            self.physics_state.delz_quantity,
            self.physics_state.omga_quantity,
            self.physics_state.dz_quantity,
            self.physics_state.wmp_quantity,
            self.physics_state.delp_quantity,
        )
        return microphysics_state
