from dataclasses import fields
from datetime import timedelta
from typing import Any, Dict, Optional, Tuple

import pytest

import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.fv3core.stencils.fv_dynamics as fv_dynamics
import pace.util
from pace.fv3core._config import DynamicalCoreConfig
from pace.fv3core.initialization.dycore_state import DycoreState
from pace.stencils.testing import ParallelTranslateBaseSlicing
from pace.stencils.testing.translate import TranslateFortranData2Py
from pace.util.grid import GridData


class TranslateDycoreFortranData2Py(TranslateFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, stencil_factory)
        self.namelist = DynamicalCoreConfig.from_namelist(namelist)


class TranslateFVDynamics(ParallelTranslateBaseSlicing):
    compute_grid_option = True
    inputs: Dict[str, Any] = {
        "q_con": {
            "name": "total_condensate_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
        },
        "delp": {
            "name": "pressure_thickness_of_atmospheric_layer",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "Pa",
        },
        "delz": {
            "name": "vertical_thickness_of_atmospheric_layer",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m",
        },
        "ps": {
            "name": "surface_pressure",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
            "units": "Pa",
        },
        "pe": {
            "name": "interface_pressure",
            "dims": [pace.util.X_DIM, pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM],
            "units": "Pa",
            "n_halo": 1,
        },
        "ak": {
            "name": "atmosphere_hybrid_a_coordinate",
            "dims": [pace.util.Z_INTERFACE_DIM],
            "units": "Pa",
        },
        "bk": {
            "name": "atmosphere_hybrid_b_coordinate",
            "dims": [pace.util.Z_INTERFACE_DIM],
            "units": "",
        },
        "pk": {
            "name": "interface_pressure_raised_to_power_of_kappa",
            "units": "unknown",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_INTERFACE_DIM],
            "n_halo": 0,
        },
        "pkz": {
            "name": "layer_mean_pressure_raised_to_power_of_kappa",
            "units": "unknown",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "n_halo": 0,
        },
        "peln": {
            "name": "logarithm_of_interface_pressure",
            "units": "ln(Pa)",
            "dims": [pace.util.X_DIM, pace.util.Z_INTERFACE_DIM, pace.util.Y_DIM],
            "n_halo": 0,
        },
        "mfxd": {
            "name": "accumulated_x_mass_flux",
            "dims": [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
        },
        "mfyd": {
            "name": "accumulated_y_mass_flux",
            "dims": [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
        },
        "cxd": {
            "name": "accumulated_x_courant_number",
            "dims": [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "",
            "n_halo": (0, 3),
        },
        "cyd": {
            "name": "accumulated_y_courant_number",
            "dims": [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM],
            "units": "",
            "n_halo": (3, 0),
        },
        "diss_estd": {
            "name": "dissipation_estimate_from_heat_source",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "unknown",
        },
        "pt": {
            "name": "air_temperature",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "degK",
        },
        "u": {
            "name": "x_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "name": "y_wind",
            "dims": [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
        "ua": {
            "name": "eastward_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
        "va": {
            "name": "northward_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
        "uc": {
            "name": "x_wind_on_c_grid",
            "dims": [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
        "vc": {
            "name": "y_wind_on_c_grid",
            "dims": [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
        "w": {
            "name": "vertical_wind",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m/s",
        },
        "phis": {
            "name": "surface_geopotential",
            "units": "m^2 s^-2",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM],
        },
        "qvapor": {
            "name": "specific_humidity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
        },
        "qliquid": {
            "name": "cloud_water_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
        },
        "qice": {
            "name": "cloud_ice_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
        },
        "qrain": {
            "name": "rain_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
        },
        "qsnow": {
            "name": "snow_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
        },
        "qgraupel": {
            "name": "graupel_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
        },
        "qo3mr": {
            "name": "ozone_mixing_ratio",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "kg/kg",
        },
        "qsgs_tke": {
            "name": "turbulent_kinetic_energy",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "m**2/s**2",
        },
        "qcld": {
            "name": "cloud_fraction",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "",
        },
        "omga": {
            "name": "vertical_pressure_velocity",
            "dims": [pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            "units": "Pa/s",
        },
        "bdt": {"dims": []},
        "ptop": {"dims": []},
    }

    outputs = inputs.copy()

    for name in ("bdt", "ak", "bk", "ptop"):
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
        fv_dynamics_vars = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "delz": {},
            "qvapor": grid.compute_dict(),
            "qliquid": grid.compute_dict(),
            "qice": grid.compute_dict(),
            "qrain": grid.compute_dict(),
            "qsnow": grid.compute_dict(),
            "qgraupel": grid.compute_dict(),
            "qo3mr": grid.compute_dict(),
            "qsgs_tke": grid.compute_dict(),
            "qcld": {},
            "ps": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "pk": grid.compute_buffer_k_dict(),
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "pkz": grid.compute_dict(),
            "phis": {},
            "q_con": {},
            "delp": {},
            "pt": {},
            "omga": {},
            "ua": {},
            "va": {},
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
            "diss_estd": {},
        }
        self._base.in_vars["data_vars"].update(fv_dynamics_vars)

        self._base.out_vars.update(fv_dynamics_vars)
        self._base.out_vars["ps"] = {"kstart": grid.npz - 1, "kend": grid.npz - 1}
        self._base.out_vars["phis"] = {"kstart": grid.npz - 1, "kend": grid.npz - 1}

        self.max_error = 1e-5

        self.ignore_near_zero_errors = {}
        for qvar in utils.tracer_variables:
            self.ignore_near_zero_errors[qvar] = True
        self.ignore_near_zero_errors["q_con"] = True
        self.dycore: Optional[fv_dynamics.DynamicalCore] = None
        self.stencil_factory = stencil_factory
        self.namelist: DynamicalCoreConfig = DynamicalCoreConfig.from_namelist(namelist)

    def state_from_inputs(self, inputs):
        input_storages = super().state_from_inputs(inputs)
        # making sure we init DycoreState with the exact set of variables
        accepted_keys = [_field.name for _field in fields(DycoreState)]
        todelete = []
        for name in input_storages.keys():
            if name not in accepted_keys:
                todelete.append(name)
        for name in todelete:
            del input_storages[name]

        state = DycoreState.init_from_storages(input_storages, sizer=self.grid.sizer)
        return state

    def prepare_data(self, inputs) -> Tuple[DycoreState, GridData]:
        for name in ("ak", "bk"):
            inputs[name] = utils.make_storage_data(
                inputs[name],
                inputs[name].shape,
                len(inputs[name].shape) * (0,),
                backend=self.stencil_factory.backend,
            )
        grid_data = self.grid.grid_data
        # These aren't in the Grid-Info savepoint, but are in the generated grid
        if grid_data.ak is None or grid_data.bk is None:
            grid_data.ak = inputs["ak"]
            grid_data.bk = inputs["bk"]
            grid_data.ptop = inputs["ptop"]

        state = self.state_from_inputs(inputs)
        return state, grid_data

    def compute_parallel(self, inputs, communicator):
        state, grid_data = self.prepare_data(inputs)
        self.dycore = fv_dynamics.DynamicalCore(
            comm=communicator,
            grid_data=grid_data,
            stencil_factory=self.stencil_factory,
            damping_coefficients=self.grid.damping_coefficients,
            config=DynamicalCoreConfig.from_namelist(self.namelist),
            phis=state.phis,
            state=state,
            timestep=timedelta(seconds=inputs["bdt"]),
        )
        self.dycore.step_dynamics(state, pace.util.NullTimer())
        outputs = self.outputs_from_state(state)
        return outputs

    def outputs_from_state(self, state: dict):
        if len(self.outputs) == 0:
            return {}
        outputs = {}
        storages = {}
        for name, properties in self.outputs.items():
            if isinstance(state[name], pace.util.Quantity):
                storages[name] = state[name].storage
            elif len(self.outputs[name]["dims"]) > 0:
                storages[name] = state[name]  # assume it's a storage
            else:
                outputs[name] = state[name]  # scalar
        outputs.update(self._base.slice_output(storages))
        return outputs

    def compute_sequential(self, *args, **kwargs):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, "
            "not running in mock-parallel"
        )
