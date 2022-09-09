from types import SimpleNamespace

import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.fv3core.stencils.fv_subgridz as fv_subgridz
import pace.util
import pace.util as fv3util
from pace.stencils.testing import ParallelTranslateBaseSlicing


# NOTE, does no halo updates, does not need to be a Parallel test,
# but doing so here to make the interface match fv_dynamics.
# Could add support to the TranslateDycoreFortranData2Py class
class TranslateFVSubgridZ(ParallelTranslateBaseSlicing):
    inputs = {
        "delp": {
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa",
        },
        "delz": {
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m",
        },
        "pe": {
            "name": "interface_pressure",
            "dims": [fv3util.X_DIM, fv3util.Z_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "Pa",
            "n_halo": 1,
        },
        "pkz": {
            "name": "layer_mean_pressure_raised_to_power_of_kappa",
            "units": "unknown",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "n_halo": 0,
        },
        "peln": {
            "name": "logarithm_of_interface_pressure",
            "units": "ln(Pa)",
            "dims": [fv3util.X_DIM, fv3util.Z_INTERFACE_DIM, fv3util.Y_DIM],
            "n_halo": 0,
        },
        "pt": {
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
        },
        "ua": {
            "name": "eastward_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "va": {
            "name": "northward_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "w": {
            "name": "vertical_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "qvapor": {
            "name": "specific_humidity",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qliquid": {
            "name": "cloud_water_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qice": {
            "name": "cloud_ice_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qrain": {
            "name": "rain_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qsnow": {
            "name": "snow_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qgraupel": {
            "name": "graupel_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qo3mr": {
            "name": "ozone_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
        "qsgs_tke": {
            "name": "turbulent_kinetic_energy",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m**2/s**2",
        },
        "qcld": {
            "name": "cloud_fraction",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
        },
        "u_dt": {
            "name": "eastward_wind_tendency_due_to_physics",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s**2",
        },
        "v_dt": {
            "name": "northward_wind_tendency_due_to_physics",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s**2",
        },
        "dt": {"dims": []},
    }
    outputs = inputs.copy()

    for name in ("dt", "pe", "peln", "delp", "delz", "pkz"):
        outputs.pop(name)

    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
        *args,
        **kwargs,
    ):
        super().__init__(grid, namelist, stencil_factory, *args, **kwargs)
        self._base.in_vars["data_vars"] = {
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "delp": {},
            "delz": {},
            "pkz": grid.compute_dict(),
            "ua": {},
            "va": {},
            "w": {},
            "pt": {},
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qo3mr": {},
            "qsgs_tke": {},
            "qcld": {},
            "u_dt": {},
            "v_dt": {},
        }

        self._base.out_vars = self._base.in_vars["data_vars"].copy()
        for var in ["pe", "peln", "delp", "delz", "pkz"]:
            self._base.out_vars.pop(var)

        self.ignore_near_zero_errors = {}
        for qvar in utils.tracer_variables:
            self.ignore_near_zero_errors[qvar] = True
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    def compute_parallel(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        fvsubgridz = fv_subgridz.DryConvectiveAdjustment(
            self.stencil_factory,
            self.namelist.nwat,
            self.namelist.fv_sg_adj,
            self.namelist.n_sponge,
            self.namelist.hydrostatic,
        )
        state_namespace = SimpleNamespace(**state)
        fvsubgridz(
            state_namespace,
            state_namespace.u_dt,
            state_namespace.v_dt,
            state_namespace.dt,
        )
        return self.outputs_from_state(state)

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = self.state_list_from_inputs_list(inputs_list)
        for state in state_list:
            fvsubgridz = fv_subgridz.DryConvectiveAdjustment(
                self.stencil_factory,
                self.namelist.nwat,
                self.namelist.fv_sg_adj,
                self.namelist.n_sponge,
                self.namelist.hydrostatic,
            )
            state_namespace = SimpleNamespace(**state)
            fvsubgridz(
                state_namespace,
                state_namespace.u_dt,
                state_namespace.v_dt,
                state_namespace.dt,
            )
        return self.outputs_list_from_state_list(state_list)
