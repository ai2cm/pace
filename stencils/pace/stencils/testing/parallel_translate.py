import copy
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np
import pytest

import pace.util as fv3util
from pace.dsl import gt4py_utils as utils

from .translate import TranslateFortranData2Py, read_serialized_data


class ParallelTranslate:

    max_error = TranslateFortranData2Py.max_error
    near_zero = TranslateFortranData2Py.near_zero
    compute_grid_option = False
    tests_grid = False
    inputs: Dict[str, Any] = {}
    outputs: Dict[str, Any] = {}

    def __init__(self, rank_grids, namelist, stencil_factory, *args, **kwargs):
        if len(args) > 0:
            raise TypeError(
                "received {} positional arguments, expected 0".format(len(args))
            )
        if len(kwargs) > 0:
            raise TypeError(
                "received {} keyword arguments, expected 0".format(len(kwargs))
            )
        if not hasattr(rank_grids, "__getitem__"):
            rank_grid = rank_grids
        else:
            rank_grid = rank_grids[0]
        self.grid = rank_grid
        self._base = TranslateFortranData2Py(rank_grid, stencil_factory)
        self._base.in_vars = {
            "data_vars": {
                name: {}
                for name, data in self.inputs.items()
                if len(data.get("dims", [])) > 0
            },
            "parameters": [
                name
                for name, data in self.inputs.items()
                if len(data.get("dims", [])) == 0
            ],
        }
        self._base.out_vars = {name: {} for name in self.outputs}
        self.max_error = self._base.max_error
        self._rank_grids = rank_grids
        self.ignore_near_zero_errors = {}
        self.namelist = namelist
        self.skip_test = False

    def state_list_from_inputs_list(self, inputs_list: List[dict]) -> list:
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
                dims = properties["dims"]
                state[properties["name"]] = fv3util.Quantity(
                    input_data,
                    dims,
                    properties["units"],
                    origin=grid.sizer.get_origin(dims),
                    extent=grid.sizer.get_extent(dims),
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
        return_dict: Dict[str, np.ndarray] = {}
        if len(self.outputs) == 0:
            return return_dict
        for name, properties in self.outputs.items():
            standard_name = properties["name"]
            if name in self._base.in_vars["data_vars"].keys():
                if "kaxis" in self._base.in_vars["data_vars"][name].keys():
                    kaxis = int(self._base.in_vars["data_vars"][name]["kaxis"])
                    dims = list(state[standard_name].dims)
                    dims.insert(kaxis, dims.pop(-1))
                    state[standard_name] = state[standard_name].transpose(dims)
            if len(properties["dims"]) > 0:
                output_slice = _serialize_slice(
                    state[standard_name], properties.get("n_halo", utils.halo)
                )
                return_dict[name] = utils.asarray(
                    state[standard_name].data[output_slice]
                )
            else:
                return_dict[name] = state[standard_name]
        return return_dict

    @property
    def rank_grids(self):
        return self._rank_grids

    @property
    def layout(self):
        return self.namelist.layout

    def compute_sequential(self, inputs_list, communicator_list):
        """Compute the outputs while iterating over a set of communicator
        objects sequentially."""
        raise NotImplementedError()

    def compute_parallel(self, inputs, communicator):
        """Compute the outputs using one communicator operating in parallel."""
        self.compute_sequential([inputs], [communicator])


class ParallelTranslateBaseSlicing(ParallelTranslate):
    def outputs_from_state(self, state: dict):
        if len(self.outputs) == 0:
            return {}
        outputs = {}
        storages = {}
        for name, properties in self.outputs.items():
            standard_name = properties.get("name", name)
            if isinstance(state[standard_name], fv3util.Quantity):
                storages[name] = state[standard_name].data
            elif len(self.outputs[name]["dims"]) > 0:
                storages[name] = state[standard_name]  # assume it's a storage
            else:
                outputs[name] = state[standard_name]  # scalar
        outputs.update(self._base.slice_output(storages))
        return outputs


def _serialize_slice(quantity, n_halo, real_dims=None):
    if real_dims is None:
        real_dims = quantity.dims
    slice_list = []
    for dim, origin, extent in zip(quantity.dims, quantity.origin, quantity.extent):
        if dim in real_dims:
            if dim in fv3util.HORIZONTAL_DIMS:
                if isinstance(n_halo, int):
                    halo = n_halo
                elif dim in fv3util.X_DIMS:
                    halo = n_halo[0]
                elif dim in fv3util.Y_DIMS:
                    halo = n_halo[1]
                else:
                    raise RuntimeError(n_halo)
            else:
                halo = 0
            slice_list.append(slice(origin - halo, origin + extent + halo))
        else:
            slice_list.append(-1)
    return tuple(slice_list)


class ParallelTranslateGrid(ParallelTranslate):
    """
    Translation class which only uses quantity factory for initialization, to
    support some non-standard array dimension layouts not supported by the
    TranslateFortranData2Py initializers.
    """

    tests_grid = True

    def state_from_inputs(self, inputs: dict, grid=None) -> dict:
        if grid is None:
            grid = self.grid
        state = {}
        for name, properties in self.inputs.items():
            standard_name = properties.get("name", name)
            if len(properties["dims"]) > 0:
                state[standard_name] = grid.quantity_factory.empty(
                    properties["dims"], properties["units"], dtype=inputs[name].dtype
                )
                input_slice = _serialize_slice(
                    state[standard_name], properties.get("n_halo", utils.halo)
                )
                if len(properties["dims"]) > 0:
                    state[standard_name].data[input_slice] = utils.asarray(
                        inputs[name], to_type=type(state[standard_name].data)
                    )
                else:
                    state[standard_name].data[:] = inputs[name]
                if name in self._base.in_vars["data_vars"].keys():
                    if "kaxis" in self._base.in_vars["data_vars"][name].keys():
                        kaxis = int(self._base.in_vars["data_vars"][name]["kaxis"])
                        dims = list(state[standard_name].dims)
                        k_dim = dims.pop(kaxis)
                        dims.insert(len(dims), k_dim)
                        state[standard_name] = state[standard_name].transpose(dims)
            else:
                state[standard_name] = inputs[name]
        return state

    def compute_sequential(self, *args, **kwargs):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, "
            "not running in mock-parallel"
        )


class ParallelTranslate2Py(ParallelTranslate):
    def collect_input_data(self, serializer, savepoint):
        input_data = super().collect_input_data(serializer, savepoint)
        input_data.update(self._base.collect_input_data(serializer, savepoint))
        return input_data

    def compute_parallel(self, inputs, communicator):
        inputs = {**inputs}
        inputs = self.state_from_inputs(inputs)
        inputs["comm"] = communicator
        result = self._base.compute_from_storage(inputs)
        quantity_result = self.outputs_from_state(result)
        result.update(quantity_result)
        for name, data in result.items():
            if isinstance(data, fv3util.Quantity):
                result[name] = data.data
        result.update(self._base.slice_output(result))
        return result

    def compute_sequential(self, a, b):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, "
            "not running in mock-parallel"
        )


class ParallelTranslate2PyState(ParallelTranslate2Py):
    def compute_parallel(self, inputs, communicator):
        self._base.make_storage_data_input_vars(inputs)
        for name, properties in self.inputs.items():
            self.grid.quantity_dict_update(
                inputs, name, dims=properties["dims"], units=properties["units"]
            )
        statevars = SimpleNamespace(**inputs)
        state = {"state": statevars}
        self._base.compute_func(**state)
        return self._base.slice_output(vars(state["state"]))
