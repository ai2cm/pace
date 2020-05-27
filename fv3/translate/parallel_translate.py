from typing import List
import copy
from .translate import TranslateFortranData2Py, read_serialized_data
import fv3util
from fv3.utils import gt4py_utils as utils
import fv3
import pytest


class ParallelTranslate:

    max_error = TranslateFortranData2Py.max_error

    inputs = {}
    outputs = {}

    def __init__(self, rank_grids):
        if not hasattr(rank_grids, "__getitem__"):
            raise TypeError(
                "rank_grids should be a sequence of grids, one for each rank, "
                f"is {self.__class__} being properly called as a parallel test?"
            )
        self._base = TranslateFortranData2Py(rank_grids[0])
        self._base.in_vars = {
            "data_vars": {name: {} for name in self.inputs},
            "parameters": {},
        }
        self.max_error = self._base.max_error
        self._rank_grids = rank_grids

    def state_list_from_inputs_list(self, inputs_list: List[list]) -> list:
        state_list = []
        for inputs in inputs_list:
            state_list.append(self.state_from_inputs(inputs))
        return state_list

    def state_from_inputs(self, inputs: dict, grid=None) -> dict:
        if grid is None:
            grid = self.grid
        state = copy.copy(inputs)
        self._base.make_storage_data_input_vars(state)
        for name, properties in self.inputs.items():
            if "name" not in properties:
                properties["name"] = name
            input_data = state[name]
            if len(properties["dims"]) > 0:
                state[properties["name"]] = grid.quantity_factory.empty(
                    properties["dims"], properties["units"], dtype=inputs[name].dtype
                )
                if len(properties["dims"]) == 3:
                    state[properties["name"]].data[:] = input_data
                elif len(properties["dims"]) == 2:
                    state[properties["name"]].data[:] = input_data[:, :, 0]
                else:
                    raise NotImplementedError(
                        "only 0, 2, and 3-d variables are supported"
                    )
            else:
                state[properties["name"]] = input_data
        return state

    def outputs_list_from_state_list(self, state_list):
        outputs_list = []
        for state in state_list:
            outputs_list.append(self.outputs_from_state(state))
        return outputs_list

    def collect_input_data(self, serializer, savepoint):
        input_data = {}
        for varname in self.inputs.keys():
            input_data[varname] = read_serialized_data(serializer, savepoint, varname)
        return input_data

    def outputs_from_state(self, state: dict):
        return_dict = {}
        if len(self.outputs) == 0:
            return return_dict
        for name, properties in self.outputs.items():
            standard_name = properties["name"]
            output_slice = _serialize_slice(
                state[standard_name], properties.get("n_halo", utils.halo)
            )
            return_dict[name] = state[standard_name].data[output_slice]
        return return_dict

    @property
    def rank_grids(self):
        return self._rank_grids

    @property
    def grid(self):
        return self._rank_grids[0]

    @property
    def layout(self):
        return fv3._config.namelist["layout"]

    def compute_sequential(self, inputs_list, communicator_list):
        """Compute the outputs while iterating over a set of communicator objects sequentially"""
        raise NotImplementedError()

    def compute_parallel(self, inputs, communicator):
        """Compute the outputs using one communicator operating in parallel"""
        self.compute_sequential(self, [inputs], [communicator])


def _serialize_slice(quantity, n_halo):
    slice_list = []
    for dim, origin, extent in zip(quantity.dims, quantity.origin, quantity.extent):
        if dim in fv3util.HORIZONTAL_DIMS:
            halo = n_halo
        else:
            halo = 0
        slice_list.append(slice(origin - halo, origin + extent + halo))
    return tuple(slice_list)


class ParallelTranslate2Py(ParallelTranslate):
    def collect_input_data(self, serializer, savepoint):
        input_data = super().collect_input_data(serializer, savepoint)
        input_data.update(self._base.collect_input_data(serializer, savepoint))
        return input_data

    def compute_parallel(self, inputs, communicator):
        inputs["comm"] = communicator
        inputs = self.state_from_inputs(inputs)
        result = self._base.compute_from_storage(inputs)
        quantity_result = self.outputs_from_state(inputs)
        result.update(quantity_result)
        return result

    def compute_sequential(self, a, b):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, not running in mock-parallel"
        )
