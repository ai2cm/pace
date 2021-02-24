import logging
from typing import Dict, List, Optional, Tuple

import numpy as np

import fv3core._config
import fv3core.utils.gt4py_utils as utils
from fv3core.utils.grid import Grid
from fv3core.utils.typing import Field


logger = logging.getLogger("fv3ser")


def read_serialized_data(serializer, savepoint, variable):
    data = serializer.read(variable, savepoint)
    if len(data.flatten()) == 1:
        return data[0]
    return data


class TranslateFortranData2Py:
    max_error = 1e-14
    near_zero = 1e-18

    def __init__(self, grid, origin=utils.origin):
        self.origin = origin
        self.in_vars = {"data_vars": {}, "parameters": []}
        self.out_vars = {}
        self.grid = grid
        self.maxshape = grid.domain_shape_full(add=(1, 1, 1))
        self.ordered_input_vars = None
        self.ignore_near_zero_errors = {}

    def compute_func(self, **inputs):
        raise NotImplementedError("Implement a child class compute method")

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        return self.slice_output(self.compute_from_storage(inputs))

    # assume inputs already has been turned into gt4py storages (or Quantities)
    def compute_from_storage(self, inputs):
        outputs = self.compute_func(**inputs)
        if outputs is not None:
            inputs.update(outputs)
        return inputs

    def column_split_compute(self, inputs, info_mapping):
        column_info = {}
        for pyfunc_var, serialbox_var in info_mapping.items():
            column_info[pyfunc_var] = self.column_namelist_vals(serialbox_var, inputs)
        self.make_storage_data_input_vars(inputs)
        for k in info_mapping.values():
            del inputs[k]
        kstarts = utils.get_kstarts(column_info, self.grid.npz)
        utils.k_split_run(self.compute_func, inputs, kstarts, column_info)
        return self.slice_output(inputs)

    def collect_input_data(self, serializer, savepoint):
        input_data = {}
        for varname in (
            self.serialnames(self.in_vars["data_vars"]) + self.in_vars["parameters"]
        ):
            input_data[varname] = read_serialized_data(serializer, savepoint, varname)
        return input_data

    def make_storage_data(
        self,
        array: np.ndarray,
        istart: int = 0,
        jstart: int = 0,
        kstart: int = 0,
        dummy_axes: Optional[Tuple[int, int, int]] = None,
        axis: int = 2,
        names_4d: Optional[List[str]] = None,
    ) -> Dict[str, type(Field)]:
        use_shape = list(self.maxshape)
        if dummy_axes:
            for axis in dummy_axes:
                use_shape[axis] = 1
        use_shape = tuple(use_shape)
        start = (istart, jstart, kstart)
        if names_4d:
            return utils.make_storage_dict(
                array,
                use_shape,
                start=start,
                origin=start,
                dummy=dummy_axes,
                axis=axis,
                names=names_4d,
            )
        else:
            return utils.make_storage_data(
                array,
                use_shape,
                start=start,
                origin=start,
                dummy=dummy_axes,
                axis=axis,
            )

    def storage_vars(self):
        return self.in_vars["data_vars"]

    # TODO: delete this when ready to let it go
    """
    def ordered_stencil_arg_values(self, data):
        if self.ordered_input_vars is not None:
            return [data[key] for key in self.ordered_input_vars]
        data_vars = [data[key] for key in self.in_vars['data_vars'].keys()]
        parameters = [data[key] for key in self.in_vars['parameters']]
        return data_vars + parameters

    #[data[key] for parent_key in ['data_vars', 'parameters'] for key in self.in_vars[parent_key]] # noqa: E501

    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        from fv3core._config import grid
        if storage_vars is None:
            storage_vars = self.storage_vars()
        storage = {}
        for d in storage_vars:
            istart, jstart = grid.horizontal_starts_from_shape(inputs[d].shape)
            storage[d] = self.make_storage_data(np.squeeze(inputs[d]), istart=istart, jstart=jstart) # noqa: E501
        for p in self.in_vars['parameters'] + self.in_vars['grid_parameters']:
            storage[p] = inputs[p]
            if type(inputs[p]) == np.int64:
                storage[p] = int(storage[p])
        return storage

    """

    def get_index_from_info(self, varinfo, index_name, initial_index):
        index = initial_index
        if index_name in varinfo:
            index = varinfo[index_name]
        return index

    def update_info(self, info, inputs):
        for k, v in info.items():
            if k == "serialname" or isinstance(v, list):
                continue
            if v in inputs.keys():
                info[k] = inputs[v]

    def collect_start_indices(self, datashape, varinfo):
        istart, jstart = self.grid.horizontal_starts_from_shape(datashape)
        istart = self.get_index_from_info(varinfo, "istart", istart)
        jstart = self.get_index_from_info(varinfo, "jstart", jstart)
        kstart = self.get_index_from_info(varinfo, "kstart", 0)
        return istart, jstart, kstart

    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        if storage_vars is None:
            storage_vars = self.storage_vars()
        for p in self.in_vars["parameters"]:
            if type(inputs[p]) in [np.int64, np.int32]:
                inputs[p] = int(inputs[p])
        for d, info in storage_vars.items():
            serialname = info["serialname"] if "serialname" in info else d
            self.update_info(info, inputs)
            if "kaxis" in info:
                inputs[serialname] = np.moveaxis(inputs[serialname], info["kaxis"], 2)
            istart, jstart, kstart = self.collect_start_indices(
                inputs[serialname].shape, info
            )

            logger.debug(
                "Making storage for {} with istart = {}, jstart = {}".format(
                    d, istart, jstart
                )
            )
            names_4d = None
            if len(inputs[serialname].shape) == 4:
                names_4d = info.get("names_4d", utils.tracer_variables)

            dummy_axes = info.get("dummy_axes", None)
            axis = info.get("axis", 2)
            inputs[d] = self.make_storage_data(
                np.squeeze(inputs[serialname]),
                istart=istart,
                jstart=jstart,
                kstart=kstart,
                dummy_axes=dummy_axes,
                axis=axis,
                names_4d=names_4d,
            )
            if d != serialname:
                del inputs[serialname]

    def slice_output(self, inputs, out_data=None):
        if out_data is None:
            out_data = inputs
        else:
            out_data.update(inputs)
        out = {}
        for var in self.out_vars.keys():
            info = self.out_vars[var]
            self.update_info(info, inputs)
            serialname = info["serialname"] if "serialname" in info else var
            ds = self.grid.default_domain_dict()
            ds.update(info)
            data_result = out_data[var]
            if isinstance(data_result, dict):
                names_4d = info.get("names_4d", utils.tracer_variables)
                var4d = np.zeros(
                    (
                        ds["iend"] - ds["istart"] + 1,
                        ds["jend"] - ds["jstart"] + 1,
                        ds["kend"] - ds["kstart"] + 1,
                        len(data_result),
                    )
                )
                for varname, data_element in data_result.items():
                    index = names_4d.index(varname)
                    data_element.synchronize()
                    var4d[:, :, :, index] = np.squeeze(
                        np.asarray(data_element)[self.grid.slice_dict(ds)]
                    )
                out[serialname] = var4d
            else:
                data_result.synchronize()
                out[serialname] = np.squeeze(
                    np.asarray(data_result)[self.grid.slice_dict(ds)]
                )
            if "kaxis" in info:
                out[serialname] = np.moveaxis(out[serialname], 2, info["kaxis"])
        return out

    def serialnames(self, dict):
        return [
            info["serialname"] if "serialname" in info else d
            for d, info in dict.items()
        ]

    def column_namelist_vals(self, varname, inputs):
        info = self.in_vars["data_vars"][varname]
        name = info["serialname"] if "serialname" in info else varname
        if len(inputs[name].shape) == 1:
            return inputs[name]
        return [i for i in inputs[name][0, 0, :]]


class TranslateGrid:
    fpy_model_index_offset = 2
    fpy_index_offset = -1
    composite_grid_vars = ["sin_sg", "cos_sg"]
    edge_var_axis = {"edge_w": 1, "edge_e": 1, "edge_s": 0, "edge_n": 0}
    # Super (composite) grid
    #     9---4---8
    #     |       |
    #     1   5   3
    #     |       |
    #     6---2---7

    def __init__(self, inputs, rank):
        self.indices = {}
        self.shape_params = {}
        self.data = {}
        for s in Grid.shape_params:
            self.shape_params[s] = inputs[s]
            del inputs[s]
        self.rank = rank
        self.layout = fv3core._config.namelist.layout
        for i, j in Grid.index_pairs:
            for index in [i, j]:
                self.indices[index] = inputs[index] + self.fpy_model_index_offset
                del inputs[index]

        self.data = inputs

    def make_composite_var_storage(self, varname, data3d, shape):
        for s in range(9):
            self.data[varname + str(s + 1)] = utils.make_storage_data(
                np.squeeze(data3d[:, :, s]), shape, origin=(0, 0, 0)
            )

    def make_grid_storage(self, pygrid):
        shape = pygrid.domain_shape_full(add=(1, 1, 1))
        for k in TranslateGrid.composite_grid_vars:
            if k in self.data:
                self.make_composite_var_storage(k, self.data[k], shape)
                del self.data[k]
        for k, axis in TranslateGrid.edge_var_axis.items():
            if k in self.data:
                edge_offset = pygrid.local_to_global_indices(pygrid.isd, pygrid.jsd)[
                    axis
                ]
                if axis == 0:
                    edge_offset = pygrid.global_isd
                    width = pygrid.subtile_width_x
                else:
                    edge_offset = pygrid.global_jsd
                    width = pygrid.subtile_width_y
                edge_slice = slice(int(edge_offset), int(edge_offset + width + 1))
                self.data[k] = utils.make_storage_data(
                    self.data[k][edge_slice],
                    shape,
                    start=(0, 0, pygrid.halo),
                    axis=axis,
                )
        for k, v in self.data.items():
            if type(v) is np.ndarray:
                # TODO: when grid initialization model exists, may want to use
                # it to inform this
                istart, jstart = pygrid.horizontal_starts_from_shape(v.shape)
                logger.debug(
                    "Storage for Grid variable {}, {}, {}, {}".format(
                        k, istart, jstart, v.shape
                    )
                )
                origin = (istart, jstart, 0)
                self.data[k] = utils.make_storage_data(
                    v, shape, origin=origin, start=origin
                )

    def python_grid(self):
        pygrid = Grid(self.indices, self.shape_params, self.rank, self.layout)
        self.make_grid_storage(pygrid)
        pygrid.add_data(self.data)
        return pygrid
