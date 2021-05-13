from typing import Optional

import pytest

import fv3core._config as spec
import fv3core.stencils.fv_dynamics as fv_dynamics
import fv3core.utils.gt4py_utils as utils
import fv3gfs.util as fv3util
from fv3core.testing import ParallelTranslateBaseSlicing


ADVECTED_TRACER_NAMES = utils.tracer_variables[: fv_dynamics.DynamicalCore.NQ]


class TranslateFVDynamics(ParallelTranslateBaseSlicing):
    python_regression = True
    inputs = {
        "q_con": {
            "name": "total_condensate_mixing_ratio",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/kg",
        },
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
        "ps": {
            "name": "surface_pressure",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "Pa",
        },
        "pe": {
            "name": "interface_pressure",
            "dims": [fv3util.X_DIM, fv3util.Z_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "Pa",
            "n_halo": 1,
        },
        "ak": {
            "name": "atmosphere_hybrid_a_coordinate",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "Pa",
        },
        "bk": {
            "name": "atmosphere_hybrid_b_coordinate",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "",
        },
        "pk": {
            "name": "interface_pressure_raised_to_power_of_kappa",
            "units": "unknown",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
            "n_halo": 0,
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
        "mfxd": {
            "name": "accumulated_x_mass_flux",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
        },
        "mfyd": {
            "name": "accumulated_y_mass_flux",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "unknown",
            "n_halo": 0,
        },
        "cxd": {
            "name": "accumulated_x_courant_number",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "",
            "n_halo": (0, 3),
        },
        "cyd": {
            "name": "accumulated_y_courant_number",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "",
            "n_halo": (3, 0),
        },
        "diss_estd": {
            "name": "dissipation_estimate_from_heat_source",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "unknown",
        },
        "pt": {
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
        },
        "u": {
            "name": "x_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "name": "y_wind",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
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
        "uc": {
            "name": "x_wind_on_c_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "vc": {
            "name": "y_wind_on_c_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "w": {
            "name": "vertical_wind",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "phis": {
            "name": "surface_geopotential",
            "units": "m^2 s^-2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
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
        "omga": {
            "name": "vertical_pressure_velocity",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa/s",
        },
        "do_adiabatic_init": {"dims": []},
        "consv_te": {"dims": []},
        "bdt": {"dims": []},
        "ptop": {"dims": []},
        "n_split": {"dims": []},
        "ks": {"dims": []},
    }

    outputs = inputs.copy()

    for name in (
        "do_adiabatic_init",
        "consv_te",
        "bdt",
        "ptop",
        "n_split",
        "ak",
        "bk",
        "ks",
    ):
        outputs.pop(name)

    def __init__(self, grids, *args, **kwargs):
        super().__init__(grids, *args, **kwargs)
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "delz": {},
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qo3mr": {},
            "qsgs_tke": {},
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
            "ak": {},
            "bk": {},
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
            "diss_estd": {},
        }

        self._base.out_vars = self._base.in_vars["data_vars"].copy()
        for var in ["ak", "bk"]:
            self._base.out_vars.pop(var)
        self._base.out_vars["ps"] = {"kstart": grid.npz - 1, "kend": grid.npz - 1}
        self._base.out_vars["phis"] = {"kstart": grid.npz - 1, "kend": grid.npz - 1}

        self.max_error = 1e-5

        self.ignore_near_zero_errors = {}
        for qvar in utils.tracer_variables:
            self.ignore_near_zero_errors[qvar] = True
        self.ignore_near_zero_errors["q_con"] = True
        self.dycore: Optional[fv_dynamics.DynamicalCore] = None

    def compute_parallel(self, inputs, communicator):
        inputs["comm"] = communicator
        state = self.state_from_inputs(inputs)
        self.dycore = fv_dynamics.DynamicalCore(
            communicator,
            spec.namelist,
            state["atmosphere_hybrid_a_coordinate"],
            state["atmosphere_hybrid_b_coordinate"],
            state["surface_geopotential"],
        )
        self.dycore.step_dynamics(
            state,
            inputs["consv_te"],
            inputs["do_adiabatic_init"],
            inputs["bdt"],
            inputs["ptop"],
            inputs["n_split"],
            inputs["ks"],
        )
        outputs = self.outputs_from_state(state)
        for name in ADVECTED_TRACER_NAMES:
            outputs[name] = self.dycore.tracer_advection.subset_output(
                "tracers", outputs[name]
            )
        return outputs

    def compute_sequential(self, *args, **kwargs):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, "
            "not running in mock-parallel"
        )

    def subset_output(self, varname: str, output):
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        if self.dycore is None:
            raise RuntimeError(
                "cannot call subset_output before calling compute_parallel "
                "to initialize dycore"
            )
        if varname in ADVECTED_TRACER_NAMES:
            return self.dycore.tracer_advection.subset_output(  # type: ignore
                "tracers", output
            )
        else:
            return output
