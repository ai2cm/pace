from typing import Any, Dict

import numpy as np
import pytest

import pace.dsl.gt4py_utils as utils
import pace.util as fv3util
from fv3core.stencils.a2b_ord4 import AGrid2BGridFourthOrder
from pace.stencils.testing.parallel_translate import ParallelTranslateGrid
from pace.util.grid import MetricTerms, set_hybrid_pressure_coefficients
from pace.util.grid.global_setup import global_mirror_grid, gnomonic_grid


class TranslateGnomonicGrids(ParallelTranslateGrid):

    max_error = 2e-14

    inputs = {
        "lon": {
            "name": "longitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
        "lat": {
            "name": "latitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
    }
    outputs = {
        "lon": {
            "name": "longitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
        "lat": {
            "name": "latitude_on_cell_corners",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "radians",
            "n_halo": 0,
        },
    }

    def compute_parallel(self, inputs, communicator):
        pytest.skip(f"{self.__class__} not running in parallel")

    def compute_sequential(self, inputs_list, communicator_list):
        outputs = []
        self.max_error = 2e-14
        for inputs in inputs_list:
            outputs.append(self.compute(inputs))
        return outputs

    def compute(self, inputs):
        gnomonic_grid(
            self.grid.grid_type,
            inputs["lon"],
            inputs["lat"],
            np,
        )
        return inputs


class TranslateMirrorGrid(ParallelTranslateGrid):

    inputs = {
        "master_grid_global": {
            "name": "grid_global",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
                MetricTerms.TILE_DIM,
            ],
            "units": "radians",
            "n_halo": 3,
        },
        "master_ng": {"name": "n_ghost", "dims": []},
        "master_npx": {"name": "npx", "dims": []},
        "master_npy": {"name": "npy", "dims": []},
    }
    outputs = {
        "master_grid_global": {
            "name": "grid_global",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
                MetricTerms.TILE_DIM,
            ],
            "units": "radians",
            "n_halo": 3,
        },
    }

    def compute_parallel(self, inputs, communicator):
        pytest.skip(f"{self.__class__} not running in parallel")

    def compute_sequential(self, inputs_list, communicator_list):
        outputs = []
        outputs.append(self.compute(inputs_list[0]))
        for inputs in inputs_list[1:]:
            outputs.append(inputs)
        return outputs

    def compute(self, inputs):
        global_mirror_grid(
            inputs["master_grid_global"],
            self.inputs["master_grid_global"]["n_halo"],
            inputs["master_npx"],
            inputs["master_npy"],
            np,
            MetricTerms.RIGHT_HAND_GRID,
        )
        return inputs


class TranslateGridAreas(ParallelTranslateGrid):
    def __init__(self, rank_grids, namelist, stencil_factory):
        super().__init__(rank_grids, namelist, stencil_factory)
        self.max_error = 1e-10
        self.near_zero = 3e-14
        self.ignore_near_zero_errors = {"agrid": True, "dxc": True, "dyc": True}
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    inputs = {
        "grid": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.LON_OR_LAT_DIM],
            "units": "radians",
        },
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dxa": {
            "name": "dx_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dya": {
            "name": "dy_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
    }
    outputs = {
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dxa": {
            "name": "dx_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dya": {
            "name": "dy_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._agrid.data[:] = in_state["agrid"].data[:]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateGridGrid(ParallelTranslateGrid):

    max_error = 1e-14
    inputs: Dict[str, Any] = {
        "grid_global": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
                MetricTerms.TILE_DIM,
            ],
            "units": "radians",
        }
    }
    outputs = {
        "grid": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "radians",
        },
    }

    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        self.max_error = 1.0e-13
        self.near_zero = 1e-14
        self.ignore_near_zero_errors = {"grid": True}
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateDxDy(ParallelTranslateGrid):
    def __init__(self, rank_grids, namelist, stencil_factory):
        super().__init__(rank_grids, namelist, stencil_factory)
        self.max_error = 3e-14
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    inputs = {
        "grid": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "radians",
        },
    }
    outputs = {
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateAGrid(ParallelTranslateGrid):
    def __init__(self, rank_grids, namelist, stencil_factory):
        super().__init__(rank_grids, namelist, stencil_factory)
        self.max_error = 1e-13
        self.namelist = namelist
        self.stencil_factory = stencil_factory

    inputs = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.LON_OR_LAT_DIM],
            "units": "radians",
        },
        "grid": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "radians",
        },
    }
    outputs = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.LON_OR_LAT_DIM],
            "units": "radians",
        },
        "grid": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "radians",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._init_agrid()
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateInitGrid(ParallelTranslateGrid):
    inputs = {
        "ndims": {"name": "ndims", "dims": []},
        "nregions": {
            "name": "nregions",
            "dims": [],
        },
        "grid_name": {
            "name": "grid_name",
            "dims": [],
        },
        "sw_corner": {
            "name": "sw_corner",
            "dims": [],
        },
        "se_corner": {
            "name": "se_corner",
            "dims": [],
        },
        "nw_corner": {
            "name": "nw_corner",
            "dims": [],
        },
        "ne_corner": {
            "name": "ne_corner",
            "dims": [],
        },
    }
    outputs: Dict[str, Any] = {
        "gridvar": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.LON_OR_LAT_DIM],
            "units": "radians",
        },
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dxa": {
            "name": "dx_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dya": {
            "name": "dy_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
    }

    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        self.max_error = 3e-12
        self.near_zero = 3e-14
        self.ignore_near_zero_errors = {"gridvar": True, "agrid": True}
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateSetEta(ParallelTranslateGrid):
    inputs: Dict[str, Any] = {
        "npz": {
            "name": "npz",
            "dims": [],
            "units": "",
        },
        "ks": {
            "name": "ks",
            "dims": [],
            "units": "",
        },
        "ptop": {
            "name": "ptop",
            "dims": [],
            "units": "mb",
        },
        "ak": {
            "name": "ak",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "mb",
        },
        "bk": {
            "name": "bk",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "ks": {
            "name": "ks",
            "dims": [],
            "units": "",
        },
        "ptop": {
            "name": "ptop",
            "dims": [],
            "units": "mb",
        },
        "ak": {
            "name": "ak",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "mb",
        },
        "bk": {
            "name": "bk",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "",
        },
    }

    def compute_parallel(self, inputs, communicator):
        pytest.skip(f"{self.__class__} not running in parallel")

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = []
        for inputs in inputs_list:
            state_list.append(self._compute_local(inputs))
        return self.outputs_list_from_state_list(state_list)

    def _compute_local(self, inputs):
        state = self.state_from_inputs(inputs)
        pressure_coefficients = set_hybrid_pressure_coefficients(state["npz"])
        state["ks"] = pressure_coefficients.ks
        state["ptop"] = pressure_coefficients.ptop
        array_type = type(state["ak"].data[:])
        state["ak"].data[:] = utils.asarray(
            pressure_coefficients.ak, to_type=array_type
        )
        state["bk"].data[:] = utils.asarray(
            pressure_coefficients.bk, to_type=array_type
        )
        return state


class TranslateUtilVectors(ParallelTranslateGrid):
    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        self.max_error = 3e-12
        self.near_zero = 1e-13
        self.ignore_near_zero_errors = {
            "ew1": True,
            "ew2": True,
            "es1": True,
            "es2": True,
            "ec1": True,
            "ec2": True,
        }
        self._base.in_vars["data_vars"] = {
            "ec1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2": {
                "kend": 2,
                "kaxis": 0,
            },
            "ew1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ew2": {
                "kend": 2,
                "kaxis": 0,
            },
            "es1": {
                "kend": 2,
                "kaxis": 0,
            },
            "es2": {
                "kend": 2,
                "kaxis": 0,
            },
        }
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.LON_OR_LAT_DIM],
            "units": "radians",
        },
        "ec1": {
            "name": "ec1",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name": "ec2",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew1": {
            "name": "ew1",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew2": {
            "name": "ew2",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "es1": {
            "name": "es1",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "es2": {
            "name": "es2",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "ec1": {
            "name": "ec1",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name": "ec2",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew1": {
            "name": "ew1",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew2": {
            "name": "ew2",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "es1": {
            "name": "es1",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "es2": {
            "name": "es2",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._agrid.data[:] = in_state["agrid"].data[:]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateTrigSg(ParallelTranslateGrid):
    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        self.max_error = 2.5e-10
        self.near_zero = 1e-14
        self.ignore_near_zero_errors = {
            "cos_sg5": True,
            "cos_sg6": True,
            "cos_sg7": True,
            "cos_sg8": True,
            "cos_sg9": True,
        }
        self._base.in_vars["data_vars"] = {
            "ec1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2": {
                "kend": 2,
                "kaxis": 0,
            },
            "ee1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ee2": {
                "kend": 2,
                "kaxis": 0,
            },
        }
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.LON_OR_LAT_DIM],
            "units": "radians",
        },
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec1": {
            "name": "ec1",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name": "ec2",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._agrid.data[:] = in_state["agrid"].data[:]
        grid_generator._ec1 = in_state["ec1"]
        grid_generator._ec2 = in_state["ec2"]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateAAMCorrection(ParallelTranslateGrid):
    # TODO: THIS IS DISABLED because it fails on
    # c48 and c128 with large relative errors, investigate!
    # these values are super tiny, so ignore_near_zero
    # will eliminate most points getting tested.
    def __init__(self, rank_grids, namelist, stencil_factory):
        super().__init__(rank_grids, namelist, stencil_factory)
        self.max_error = 1e-14
        self.near_zero = 1e-14
        self.ignore_near_zero_errors = {"l2c_v": True, "l2c_u": True}
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "radians",
        },
        "l2c_v": {
            "name": "l2c_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 0,
        },
        "l2c_u": {
            "name": "l2c_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
    }
    outputs: Dict[str, Any] = {
        "l2c_v": {
            "name": "l2c_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 0,
        },
        "l2c_u": {
            "name": "l2c_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateDerivedTrig(ParallelTranslateGrid):
    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        self.max_error = 8.5e-14
        self.near_zero = 3e-14
        self.ignore_near_zero_errors = {"ee1": True, "ee2": True}
        self._base.in_vars["data_vars"] = {
            "ee1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ee2": {
                "kend": 2,
                "kaxis": 0,
            },
        }
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "radians",
        },
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ee1": {
            "name": "ee1",
            "dims": [
                MetricTerms.CARTESIAN_DIM,
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
            ],
            "units": "",
        },
        "ee2": {
            "name": "ee2",
            "dims": [
                MetricTerms.CARTESIAN_DIM,
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
            ],
            "units": "",
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "rsin2": {"name": "rsin2", "dims": [fv3util.X_DIM, fv3util.Y_DIM], "units": ""},
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "ee1": {
            "name": "ee1",
            "dims": [
                MetricTerms.CARTESIAN_DIM,
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
            ],
            "units": "",
        },
        "ee2": {
            "name": "ee2",
            "dims": [
                MetricTerms.CARTESIAN_DIM,
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
            ],
            "units": "",
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "rsin2": {"name": "rsin2", "dims": [fv3util.X_DIM, fv3util.Y_DIM], "units": ""},
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._cos_sg1 = in_state["cos_sg1"]
        grid_generator._cos_sg2 = in_state["cos_sg2"]
        grid_generator._cos_sg3 = in_state["cos_sg3"]
        grid_generator._cos_sg4 = in_state["cos_sg4"]
        grid_generator._cos_sg5 = in_state["cos_sg5"]
        grid_generator._cos_sg6 = in_state["cos_sg6"]
        grid_generator._cos_sg7 = in_state["cos_sg7"]
        grid_generator._cos_sg8 = in_state["cos_sg8"]
        grid_generator._cos_sg9 = in_state["cos_sg9"]
        grid_generator._sin_sg1 = in_state["sin_sg1"]
        grid_generator._sin_sg2 = in_state["sin_sg2"]
        grid_generator._sin_sg3 = in_state["sin_sg3"]
        grid_generator._sin_sg4 = in_state["sin_sg4"]
        grid_generator._sin_sg5 = in_state["sin_sg5"]
        grid_generator._sin_sg6 = in_state["sin_sg6"]
        grid_generator._sin_sg7 = in_state["sin_sg7"]
        grid_generator._sin_sg8 = in_state["sin_sg8"]
        grid_generator._sin_sg9 = in_state["sin_sg9"]
        state = {}
        grid_generator._calculate_derived_trig_terms_for_testing()
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateDivgDel6(ParallelTranslateGrid):
    def __init__(self, rank_grids, namelist, stencil_factory):
        super().__init__(rank_grids, namelist, stencil_factory)
        self.max_error = 4e-14
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    inputs: Dict[str, Any] = {
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._sin_sg1 = in_state["sin_sg1"]
        grid_generator._sin_sg2 = in_state["sin_sg2"]
        grid_generator._sin_sg3 = in_state["sin_sg3"]
        grid_generator._sin_sg4 = in_state["sin_sg4"]
        grid_generator._sina_u = in_state["sina_u"]
        grid_generator._sina_v = in_state["sina_v"]
        grid_generator._dx = in_state["dx"]
        grid_generator._dy = in_state["dy"]
        grid_generator._dx_cgrid = in_state["dx_cgrid"]
        grid_generator._dy_cgrid = in_state["dy_cgrid"]
        grid_generator._calculate_divg_del6_nohalos_for_testing()
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateInitCubedtoLatLon(ParallelTranslateGrid):
    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        self.max_error = 3.0e-14
        self._base.in_vars["data_vars"] = {
            "ec1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2": {
                "kend": 2,
                "kaxis": 0,
            },
        }
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    inputs: Dict[str, Any] = {
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.LON_OR_LAT_DIM],
            "units": "radians",
        },
        "ec1": {
            "name": "ec1",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name": "ec2",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "vlon": {
            "name": "vlon",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "vlat": {
            "name": "vlat",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "z11": {
            "name": "z11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z12": {
            "name": "z12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z21": {
            "name": "z21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z22": {
            "name": "z22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a11": {
            "name": "a11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a12": {
            "name": "a12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a21": {
            "name": "a21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a22": {
            "name": "a22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._sin_sg5 = in_state["sin_sg5"]
        grid_generator._agrid.data[:] = in_state["agrid"].data[:]
        grid_generator._ec1 = in_state["ec1"]
        grid_generator._ec2 = in_state["ec2"]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)


class TranslateEdgeFactors(ParallelTranslateGrid):
    def __init__(self, rank_grids, namelist, stencil_factory):
        super().__init__(rank_grids, namelist, stencil_factory)
        self.max_error = 3e-13
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    inputs: Dict[str, Any] = {
        "grid": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.LON_OR_LAT_DIM],
            "units": "radians",
        },
        "edge_s": {
            "name": "edge_s",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_n": {
            "name": "edge_n",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_vect_s": {
            "name": "edge_vect_s",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_n": {
            "name": "edge_vect_n",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_e": {
            "name": "edge_vect_e",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_vect_w": {
            "name": "edge_vect_w",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "edge_s": {
            "name": "edge_s",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_n": {
            "name": "edge_n",
            "dims": [fv3util.X_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_e": {
            "name": "edge_e",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_w": {
            "name": "edge_w",
            "dims": [fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "edge_vect_s": {
            "name": "edge_vect_s",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_n": {
            "name": "edge_vect_n",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_e": {
            "name": "edge_vect_e",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_vect_w": {
            "name": "edge_vect_w",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=1,
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )

        in_state = self.state_from_inputs(inputs)
        grid_generator._grid.data[:] = in_state["grid"].data[:]
        grid_generator._agrid.data[:] = in_state["agrid"].data[:]
        a2b = AGrid2BGridFourthOrder(
            self.stencil_factory, self.grid.grid_data, namelist.grid_type
        )
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)

        return self.outputs_from_state(state)


class TranslateInitGridUtils(ParallelTranslateGrid):
    def __init__(self, grids, namelist, stencil_factory):
        super().__init__(grids, namelist, stencil_factory)
        self.max_error = 2.5e-10
        self.near_zero = 5e-14
        self.ignore_near_zero_errors = {
            "l2c_v": True,
            "l2c_u": True,
            "ee1": True,
            "ee2": True,
            "ew1": True,
            "ew2": True,
            "es1": True,
            "es2": True,
            "cos_sg5": True,
            "cos_sg6": True,
            "cos_sg7": True,
            "cos_sg8": True,
            "cos_sg9": True,
            "cosa": True,
            "cosa_u": True,
            "cosa_v": True,
        }
        self._base.in_vars["data_vars"] = {
            "ec1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ec2": {
                "kend": 2,
                "kaxis": 0,
            },
            "ew1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ew2": {
                "kend": 2,
                "kaxis": 0,
            },
            "es1": {
                "kend": 2,
                "kaxis": 0,
            },
            "es2": {
                "kend": 2,
                "kaxis": 0,
            },
            "ee1": {
                "kend": 2,
                "kaxis": 0,
            },
            "ee2": {
                "kend": 2,
                "kaxis": 0,
            },
        }
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    inputs: Dict[str, Any] = {
        "gridvar": {
            "name": "grid",
            "dims": [
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
                MetricTerms.LON_OR_LAT_DIM,
            ],
            "units": "radians",
        },
        "agrid": {
            "name": "agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.LON_OR_LAT_DIM],
            "units": "radians",
        },
        "area": {
            "name": "area",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m^2",
        },
        "area_c": {
            "name": "area_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m^2",
        },
        "dx": {
            "name": "dx",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dy": {
            "name": "dy",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dxc": {
            "name": "dx_cgrid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dyc": {
            "name": "dy_cgrid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "m",
        },
        "dxa": {
            "name": "dx_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "dya": {
            "name": "dy_agrid",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "m",
        },
        "npz": {
            "name": "npz",
            "dims": [],
            "units": "",
        },
    }
    outputs: Dict[str, Any] = {
        "ks": {
            "name": "ks",
            "dims": [],
            "units": "",
        },
        "ptop": {
            "name": "ptop",
            "dims": [],
            "units": "mb",
        },
        "ak": {
            "name": "ak",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "mb",
        },
        "bk": {
            "name": "bk",
            "dims": [fv3util.Z_INTERFACE_DIM],
            "units": "",
        },
        "ec1": {
            "name": "ec1",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ec2": {
            "name": "ec2",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew1": {
            "name": "ew1",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "ew2": {
            "name": "ew2",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "es1": {
            "name": "es1",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "es2": {
            "name": "es2",
            "dims": [MetricTerms.CARTESIAN_DIM, fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cos_sg1": {
            "name": "cos_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg1": {
            "name": "sin_sg1",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg2": {
            "name": "cos_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg2": {
            "name": "sin_sg2",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg3": {
            "name": "cos_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg3": {
            "name": "sin_sg3",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg4": {
            "name": "cos_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg4": {
            "name": "sin_sg4",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg5": {
            "name": "cos_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg5": {
            "name": "sin_sg5",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg6": {
            "name": "cos_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg6": {
            "name": "sin_sg6",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg7": {
            "name": "cos_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg7": {
            "name": "sin_sg7",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg8": {
            "name": "cos_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg8": {
            "name": "sin_sg8",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cos_sg9": {
            "name": "cos_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sin_sg9": {
            "name": "sin_sg9",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "l2c_v": {
            "name": "l2c_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 0,
        },
        "l2c_u": {
            "name": "l2c_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "ee1": {
            "name": "ee1",
            "dims": [
                MetricTerms.CARTESIAN_DIM,
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
            ],
            "units": "",
        },
        "ee2": {
            "name": "ee2",
            "dims": [
                MetricTerms.CARTESIAN_DIM,
                fv3util.X_INTERFACE_DIM,
                fv3util.Y_INTERFACE_DIM,
            ],
            "units": "",
        },
        "cosa_u": {
            "name": "cosa_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "cosa_v": {
            "name": "cosa_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "cosa_s": {
            "name": "cosa_s",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_u": {
            "name": "sina_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "sina_v": {
            "name": "sina_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsin_u": {
            "name": "rsin_u",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "rsin_v": {
            "name": "rsin_v",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "rsina": {
            "name": "rsina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
            "n_halo": 0,
        },
        "rsin2": {"name": "rsin2", "dims": [fv3util.X_DIM, fv3util.Y_DIM], "units": ""},
        "cosa": {
            "name": "cosa",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "sina": {
            "name": "sina",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_u": {
            "name": "divg_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "divg_v": {
            "name": "divg_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "del6_u": {
            "name": "del6_u",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM],
            "units": "",
        },
        "del6_v": {
            "name": "del6_v",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM],
            "units": "",
        },
        "vlon": {
            "name": "vlon",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "vlat": {
            "name": "vlat",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, MetricTerms.CARTESIAN_DIM],
            "units": "",
            "n_halo": 2,
        },
        "z11": {
            "name": "z11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z12": {
            "name": "z12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z21": {
            "name": "z21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "z22": {
            "name": "z22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a11": {
            "name": "a11",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a12": {
            "name": "a12",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a21": {
            "name": "a21",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "a22": {
            "name": "a22",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM],
            "units": "",
            "n_halo": 1,
        },
        "edge_vect_s": {
            "name": "edge_vect_s",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_n": {
            "name": "edge_vect_n",
            "dims": [fv3util.X_DIM],
            "units": "",
        },
        "edge_vect_e": {
            "name": "edge_vect_e",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "edge_vect_w": {
            "name": "edge_vect_w",
            "dims": [fv3util.Y_DIM],
            "units": "",
        },
        "da_min": {
            "name": "da_min",
            "dims": [],
            "units": "m^2",
        },
        "da_min_c": {
            "name": "da_min_c",
            "dims": [],
            "units": "m^2",
        },
        "da_max": {
            "name": "da_max",
            "dims": [],
            "units": "m^2",
        },
        "da_max_c": {
            "name": "da_max_c",
            "dims": [],
            "units": "m^2",
        },
    }

    def compute_parallel(self, inputs, communicator):
        namelist = self.namelist
        grid_generator = MetricTerms.from_tile_sizing(
            npx=namelist.npx,
            npy=namelist.npy,
            npz=inputs["npz"],
            communicator=communicator,
            backend=self.stencil_factory.backend,
        )
        input_state = self.state_from_inputs(inputs)
        grid_generator._grid = input_state["grid"]
        grid_generator._agrid = input_state["agrid"]
        state = {}
        for metric_term, metadata in self.outputs.items():
            state[metadata["name"]] = getattr(grid_generator, metric_term)
        return self.outputs_from_state(state)
