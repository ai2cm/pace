import logging

import pace.dsl
import pace.util
import pace.util as fv3util
from pace.dsl import gt4py_utils as utils
from pace.stencils.testing import ParallelTranslate


logger = logging.getLogger("fv3ser")


class TranslateHaloUpdate(ParallelTranslate):

    inputs = {
        "array": {
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
            "n_halo": utils.halo,
        }
    }

    outputs = {
        "array": {
            "name": "air_temperature",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "degK",
            "n_halo": utils.halo,
        }
    }
    halo_update_varname = "air_temperature"

    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)

    def compute_parallel(self, inputs, communicator):
        state = self.state_from_inputs(inputs)
        req = communicator.start_halo_update(
            state[self.halo_update_varname], n_points=utils.halo
        )
        req.wait()
        return self.outputs_from_state(state)

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = self.state_list_from_inputs_list(inputs_list)
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            logger.debug(f"starting on {communicator.rank}")
            req_list.append(
                communicator.start_halo_update(
                    state[self.halo_update_varname], n_points=utils.halo
                )
            )
        for communicator, req in zip(communicator_list, req_list):
            logger.debug(f"finishing on {communicator.rank}")
            req.wait()
        return self.outputs_list_from_state_list(state_list)


class TranslateHaloUpdate_2(TranslateHaloUpdate):

    inputs = {
        "array2": {
            "name": "height_on_interface_levels",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
            "units": "m",
            "n_halo": utils.halo,
        }
    }

    outputs = {
        "array2": {
            "name": "height_on_interface_levels",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_INTERFACE_DIM],
            "units": "m",
            "n_halo": utils.halo,
        }
    }

    halo_update_varname = "height_on_interface_levels"


class TranslateMPPUpdateDomains(TranslateHaloUpdate):

    inputs = {
        "update_arr": {
            "name": "z_wind_as_tendency_of_pressure",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa/s",
            "n_halo": utils.halo,
        }
    }

    outputs = {
        "update_arr": {
            "name": "z_wind_as_tendency_of_pressure",
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "Pa/s",
            "n_halo": utils.halo,
        }
    }

    halo_update_varname = "z_wind_as_tendency_of_pressure"


class TranslateHaloVectorUpdate(ParallelTranslate):

    inputs = {
        "array_u": {
            "name": "x_wind_on_c_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
        "array_v": {
            "name": "y_wind_on_c_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
    }

    outputs = {
        "array_u": {
            "name": "x_wind_on_c_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
        "array_v": {
            "name": "y_wind_on_c_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
    }

    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super(TranslateHaloVectorUpdate, self).__init__(grid, namelist, stencil_factory)

    def compute_parallel(self, inputs, communicator):
        logger.debug(f"starting on {communicator.rank}")
        state = self.state_from_inputs(inputs)
        req = communicator.start_vector_halo_update(
            state["x_wind_on_c_grid"], state["y_wind_on_c_grid"], n_points=utils.halo
        )

        logger.debug(f"finishing on {communicator.rank}")
        req.wait()
        return self.outputs_from_state(state)

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = self.state_list_from_inputs_list(inputs_list)
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            logger.debug(f"starting on {communicator.rank}")
            req_list.append(
                communicator.start_vector_halo_update(
                    state["x_wind_on_c_grid"],
                    state["y_wind_on_c_grid"],
                    n_points=utils.halo,
                )
            )
        for communicator, req in zip(communicator_list, req_list):
            logger.debug(f"finishing on {communicator.rank}")
            req.wait()
        return self.outputs_list_from_state_list(state_list)


class TranslateMPPBoundaryAdjust(ParallelTranslate):

    inputs = {
        "u": {
            "name": "x_wind_on_d_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
        "v": {
            "name": "y_wind_on_d_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
    }

    outputs = {
        "u": {
            "name": "x_wind_on_d_grid",
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
        "v": {
            "name": "y_wind_on_d_grid",
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
            "n_halo": utils.halo,
        },
    }

    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super(TranslateMPPBoundaryAdjust, self).__init__(
            grid, namelist, stencil_factory
        )

    def compute_parallel(self, inputs, communicator):
        logger.debug(f"starting on {communicator.rank}")
        state = self.state_from_inputs(inputs)
        req = communicator.start_synchronize_vector_interfaces(
            state["x_wind_on_d_grid"], state["y_wind_on_d_grid"]
        )
        logger.debug(f"finishing on {communicator.rank}")
        req.wait()
        return self.outputs_from_state(state)

    def compute_sequential(self, inputs_list, communicator_list):
        state_list = self.state_list_from_inputs_list(inputs_list)
        req_list = []
        for state, communicator in zip(state_list, communicator_list):
            req_list.append(
                communicator.start_synchronize_vector_interfaces(
                    state["x_wind_on_d_grid"], state["y_wind_on_d_grid"]
                )
            )
        for communicator, req in zip(communicator_list, req_list):
            logger.debug(f"finishing on {communicator.rank}")
            req.wait()
        return self.outputs_list_from_state_list(state_list)
