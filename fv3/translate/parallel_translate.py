from typing import List
import copy
from .translate import TranslateFortranData2Py, read_serialized_data
import fv3util
from fv3.utils import gt4py_utils as utils
import fv3


class ParallelTranslate:

    inputs = {}

    def __init__(self, rank_grids):
        self._base = TranslateFortranData2Py(rank_grids[0])
        self._base.in_vars = {
            "data_vars": {name: {} for name in self.inputs},
            "parameters": {},
        }
        self.max_error = self._base.max_error
        self._rank_grids = rank_grids  # one for each rank

    def state_list_from_inputs_list(self, inputs_list: List[list]) -> list:
        state_list = []
        for inputs in inputs_list:
            state_list.append(self.state_from_inputs(inputs))
        return state_list

    def state_from_inputs(self, inputs: dict) -> dict:
        inputs = copy.copy(inputs)  # don't want to modify the dict we were passed
        self._base.make_storage_data_input_vars(inputs)
        state = {}
        for name, properties in self.inputs.items():
            origin, extent = get_origin_and_extent(
                properties["dims"],
                inputs[name].shape,
                self.grid.npx,
                self.grid.npy,
                self.grid.npz,
                utils.halo,
                self.layout,
            )
            state[properties["name"]] = fv3util.Quantity(
                inputs[name],
                dims=properties["dims"],
                units=properties["units"],
                origin=origin,
                extent=extent,
            )
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
        for name, properties in self.outputs.items():
            standard_name = properties["name"]
            output_slice = _serialize_slice(state[standard_name], utils.halo)
            return_dict[name] = state[standard_name].data[output_slice]
        return return_dict

    @property
    def layout(self):
        return fv3._config.namelist["layout"]

    @property
    def grid(self):
        return self._rank_grids[0]

    @property
    def rank_grids(self):
        return self._rank_grids

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


def get_origin_and_extent(dims, shape, npx, npy, npz, n_halo, layout):
    nx_rank = (npx - 1) / layout[1]
    ny_rank = (npy - 1) / layout[0]
    dim_lengths = {
        fv3util.X_DIM: nx_rank,
        fv3util.X_INTERFACE_DIM: nx_rank + 1,
        fv3util.Y_DIM: ny_rank,
        fv3util.Y_INTERFACE_DIM: ny_rank + 1,
        fv3util.Z_DIM: npz,
        fv3util.Z_INTERFACE_DIM: npz + 1,
    }
    origin = []
    extent = []
    for dim, current_length in zip(dims, shape):
        extent.append(int(dim_lengths.get(dim, current_length)))
        if dim in fv3util.HORIZONTAL_DIMS:
            halo = n_halo
        else:
            halo = 0
        origin.append(int(halo))
    return origin, extent
