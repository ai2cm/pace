from typing import Any, Dict

import numpy as np
import pytest

import fv3core.initialization.baroclinic as baroclinic_init
import fv3core.initialization.baroclinic_jablonowski_williamson as jablo_init
import fv3core.stencils.fv_dynamics as fv_dynamics
import pace.dsl.gt4py_utils as utils
import pace.util as fv3util
from pace.stencils.testing import (
    ParallelTranslateBaseSlicing,
    TranslateDycoreFortranData2Py,
)
from pace.stencils.testing.grid import TRACER_DIM
from pace.util.grid import MetricTerms


class TranslateInitCase(ParallelTranslateBaseSlicing):

    outputs: Dict[str, Any] = {
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
        "pt": {
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
        },
        "q4d": {
            "name": "tracers",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM, TRACER_DIM],
            "units": "kg/kg",
        },
    }

    def __init__(self, grid_list, namelist, stencil_factory):
        super().__init__(grid_list, namelist, stencil_factory)
        grid = grid_list[0]
        self._base.in_vars["data_vars"] = {}
        self._base.in_vars["parameters"] = ["ptop"]
        self._base.out_vars = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "ua": {},
            "va": {},
            "w": {},
            "pt": {},
            "delp": {},
            "q4d": {},
            "phis": {},
            "delz": {},
            "ps": {"kstart": grid.npz, "kend": grid.npz},
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
            "pk": grid.compute_buffer_k_dict(),
            "pkz": grid.compute_dict(),
        }
        self.max_error = 6e-14
        self.ignore_near_zero_errors = {}
        for var in ["u", "v"]:
            self.ignore_near_zero_errors[var] = {"near_zero": 2e-13}
        self.namelist = namelist
        self.stencil_factory = stencil_factory

    def compute_sequential(self, *args, **kwargs):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, "
            "not running in mock-parallel"
        )

    def outputs_from_state(self, state: dict):
        outputs = {}
        arrays = {}
        for name, properties in self.outputs.items():
            if isinstance(state[name], dict):
                for tracer, quantity in state[name].items():
                    state[name][tracer] = state[name][tracer].data
                arrays[name] = state[name]
            elif len(self.outputs[name]["dims"]) > 0:

                arrays[name] = state[name].data
            else:
                outputs[name] = state[name]  # scalar
        outputs.update(self._base.slice_output(arrays))
        return outputs

    def compute_parallel(self, inputs, communicator):
        state = {}
        full_shape = (*self.grid.domain_shape_full(add=(1, 1, 1)), fv_dynamics.NQ)
        for variable, properties in self.outputs.items():
            dims = properties["dims"]
            state[variable] = fv3util.Quantity(
                np.zeros(full_shape[0 : len(dims)]),
                dims,
                properties["units"],
                origin=self.grid.sizer.get_origin(dims),
                extent=self.grid.sizer.get_extent(dims),
                gt4py_backend=self.stencil_factory.backend,
            )

        metric_terms = MetricTerms.from_tile_sizing(
            npx=self.namelist.npx,
            npy=self.namelist.npy,
            npz=self.namelist.npz,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        state = baroclinic_init.init_baroclinic_state(
            metric_terms=metric_terms,
            adiabatic=self.namelist.adiabatic,
            hydrostatic=self.namelist.hydrostatic,
            moist_phys=self.namelist.moist_phys,
            comm=communicator,
        )

        state.q4d = {}
        for tracer in utils.tracer_variables:
            state.q4d[tracer] = getattr(state, tracer)
        return self.outputs_from_state(state.__dict__)


def make_sliced_inputs_dict(inputs, slice_2d):
    sliced_inputs = {}
    for k, v in inputs.items():
        if isinstance(v, np.ndarray) and len(v.shape) > 1:
            if len(v.shape) == 3:
                slices = (*slice_2d, slice(None))
            if len(v.shape) == 2:
                slices = slice_2d
            sliced_inputs[k] = inputs[k][slices]
        else:
            sliced_inputs[k] = inputs[k]
    return sliced_inputs


class TranslateInitPreJab(TranslateDycoreFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {"ak": {}, "bk": {}, "delp": {}}
        self.in_vars["parameters"] = ["ptop"]
        self.out_vars = {
            "delp": {},
            "ps": {"kstart": grid.npz, "kend": grid.npz},
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
            "pk": grid.compute_buffer_k_dict(),
            "pkz": grid.compute_dict(),
            "eta": {"istart": 0, "iend": 0, "jstart": 0, "jend": 0},
            "eta_v": {"istart": 0, "iend": 0, "jstart": 0, "jend": 0},
        }
        self.namelist = namelist
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        for k, v in inputs.items():
            if k != "ptop":
                inputs[k] = v.data
        full_shape = self.grid.domain_shape_full(add=(1, 1, 1))
        for variable in ["pe", "peln", "pk", "pkz"]:
            inputs[variable] = np.zeros(full_shape)
        inputs["ps"] = np.zeros(full_shape[0:2])
        for zvar in ["eta", "eta_v"]:
            inputs[zvar] = np.zeros(self.grid.npz + 1)
        inputs["ps"][:] = jablo_init.surface_pressure
        sliced_inputs = make_sliced_inputs_dict(
            inputs, self.grid.compute_interface()[0:2]
        )

        baroclinic_init.setup_pressure_fields(
            **sliced_inputs,
        )
        return self.slice_output(inputs)


class TranslateJablonowskiBaroclinic(TranslateDycoreFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "delp": {},
            "eta_v": {"istart": 0, "iend": 0, "jstart": 0, "jend": 0},
            "eta": {"istart": 0, "iend": 0, "jstart": 0, "jend": 0},
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
        }

        self.in_vars["parameters"] = ["ptop"]
        self.out_vars = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "pt": {},
            "phis": {},
            "delz": {},
            "qvapor": {},
        }
        self.ignore_near_zero_errors = {}
        for var in ["u", "v"]:
            self.ignore_near_zero_errors[var] = {"near_zero": 2e-13}

        self.max_error = 1e-13
        self.namelist = namelist
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        # testing just numpy arrays for this
        for k, v in inputs.items():
            if k != "ptop":
                inputs[k] = v.data
        full_shape = self.grid.domain_shape_full(add=(1, 1, 1))
        for variable in ["u", "v", "pt", "delz", "w", "qvapor"]:
            inputs[variable] = np.zeros(full_shape)
        for var2d in ["phis"]:
            inputs[var2d] = np.zeros(full_shape[0:2])

        slice_2d = (
            slice(self.grid.is_, self.grid.ie + 2),
            slice(self.grid.js, self.grid.je + 2),
        )
        grid_vars = {
            "lon": self.grid.bgrid1.data[slice_2d],
            "lat": self.grid.bgrid2.data[slice_2d],
            "lon_agrid": self.grid.agrid1.data[slice_2d],
            "lat_agrid": self.grid.agrid2.data[slice_2d],
            "ee1": self.grid.ee1.data[slice_2d],
            "ee2": self.grid.ee2.data[slice_2d],
            "es1": self.grid.es1.data[slice_2d],
            "ew2": self.grid.ew2.data[slice_2d],
        }
        inputs["w"][:] = 1e30
        inputs["delz"][:] = 1e30
        inputs["pt"][:] = 1.0
        inputs["phis"][:] = 1.0e25
        sliced_inputs = make_sliced_inputs_dict(inputs, slice_2d)
        baroclinic_init.baroclinic_initialization(
            **sliced_inputs,
            **grid_vars,
            adiabatic=self.namelist.adiabatic,
            hydrostatic=self.namelist.hydrostatic,
            nx=self.grid.nic,
            ny=self.grid.njc,
        )
        return self.slice_output(inputs)


class TranslatePVarAuxiliaryPressureVars(TranslateDycoreFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "delp": {},
            "delz": {},
            "pt": {},
            "ps": {"kstart": grid.npz, "kend": grid.npz},
            "qvapor": {},
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
            "pkz": grid.compute_dict(),
        }

        self.in_vars["parameters"] = ["ptop"]
        self.out_vars = {}
        for var in ["delz", "delp", "ps", "peln"]:
            self.out_vars[var] = self.in_vars["data_vars"][var]
        self.namelist = namelist
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        # testing just numpy arrays for this
        for k, v in inputs.items():
            if k != "ptop":
                inputs[k] = v.data

        namelist = self.namelist
        inputs["delz"][:] = 1.0e25
        sliced_inputs = make_sliced_inputs_dict(
            inputs, self.grid.compute_interface()[0:2]
        )
        baroclinic_init.p_var(
            **sliced_inputs,
            moist_phys=namelist.moist_phys,
            make_nh=(not namelist.hydrostatic),
        )
        return self.slice_output(inputs)
