import logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

import pace.dsl.gt4py_utils as utils
from pace.dsl.stencil import StencilFactory
from pace.dsl.typing import Field  # noqa: F401
from pace.stencils.testing.grid import Grid


logger = logging.getLogger("fv3ser")


def read_serialized_data(serializer, savepoint, variable):
    data = serializer.read(variable, savepoint)
    if len(data.flatten()) == 1:
        return data[0]
    return data


def pad_field_in_j(field, nj, backend: str):
    utils.device_sync(backend)
    outfield = utils.tile(field[:, 0, :], [nj, 1, 1]).transpose(1, 0, 2)
    return outfield


class TranslateFortranData2Py:
    max_error = 1e-14
    near_zero = 1e-18

    def __init__(self, grid, stencil_factory: StencilFactory, origin=utils.origin):
        self.origin = origin
        self.stencil_factory = stencil_factory
        self.in_vars: Dict[str, Any] = {"data_vars": {}, "parameters": []}
        self.out_vars: Dict[str, Any] = {}
        self.write_vars: List = []
        self.grid = grid
        self.maxshape = grid.domain_shape_full(add=(1, 1, 1))
        self.ordered_input_vars = None
        self.ignore_near_zero_errors: Dict[str, Any] = {}

    def setup(self, inputs):
        self.make_storage_data_input_vars(inputs)

    def compute_func(self, **inputs):
        raise NotImplementedError("Implement a child class compute method")

    def compute(self, inputs):
        self.setup(inputs)
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

    def make_storage_data(
        self,
        array: np.ndarray,
        istart: int = 0,
        jstart: int = 0,
        kstart: int = 0,
        dummy_axes: Optional[Tuple[int, int, int]] = None,
        axis: int = 2,
        names_4d: Optional[List[str]] = None,
        read_only: bool = False,
        full_shape: bool = False,
    ) -> Dict[str, "Field"]:
        use_shape = list(self.maxshape)
        if dummy_axes:
            for axis in dummy_axes:
                use_shape[axis] = 1
        elif not full_shape and len(array.shape) < 3 and axis == len(array.shape) - 1:
            use_shape[1] = 1
        start = (int(istart), int(jstart), int(kstart))
        if names_4d:
            return utils.make_storage_dict(
                array,
                use_shape,
                start=start,
                origin=start,
                dummy=dummy_axes,
                axis=axis,
                names=names_4d,
                backend=self.stencil_factory.backend,
            )
        else:
            return utils.make_storage_data(
                array,
                use_shape,
                start=start,
                origin=start,
                dummy=dummy_axes,
                axis=axis,
                read_only=read_only,
                backend=self.stencil_factory.backend,
            )

    def storage_vars(self):
        return self.in_vars["data_vars"]

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
        inputs_in = {**inputs}
        inputs_out = {}
        if storage_vars is None:
            storage_vars = self.storage_vars()
        for p in self.in_vars["parameters"]:
            if type(inputs_in[p]) in [np.int64, np.int32]:
                inputs_out[p] = int(inputs_in[p])
            else:
                inputs_out[p] = inputs_in[p]
        for d, info in storage_vars.items():
            serialname = info["serialname"] if "serialname" in info else d
            self.update_info(info, inputs_in)
            if "kaxis" in info:
                inputs_in[serialname] = np.moveaxis(
                    inputs_in[serialname], info["kaxis"], 2
                )
            else:
                inputs_in[serialname] = inputs_in[serialname]
            istart, jstart, kstart = self.collect_start_indices(
                inputs_in[serialname].shape, info
            )

            names_4d = None
            if len(inputs_in[serialname].shape) == 4:
                names_4d = info.get("names_4d", utils.tracer_variables)

            dummy_axes = info.get("dummy_axes", None)
            axis = info.get("axis", 2)
            inputs_out[d] = self.make_storage_data(
                np.squeeze(inputs_in[serialname]),
                istart=istart,
                jstart=jstart,
                kstart=kstart,
                dummy_axes=dummy_axes,
                axis=axis,
                names_4d=names_4d,
                read_only=d not in self.write_vars,
                full_shape="full_shape" in storage_vars[d],
            )
        # update the input in-place because that's how it was originally written
        # feel free to refactor
        inputs.clear()
        inputs.update(inputs_out)

    def slice_output(self, inputs, out_data=None):
        utils.device_sync(backend=self.stencil_factory.backend)
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
                    if hasattr(data_element, "synchronize"):
                        data_element.synchronize()
                    var4d[:, :, :, index] = np.squeeze(
                        np.asarray(data_element)[self.grid.slice_dict(ds)]
                    )
                out[serialname] = var4d
            else:
                if hasattr(data_result, "synchronize"):
                    data_result.synchronize()
                slice_tuple = self.grid.slice_dict(ds, len(data_result.shape))
                out[serialname] = np.squeeze(np.asarray(data_result)[slice_tuple])
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
    vvars = ["vlon", "vlat"]
    edge_var_axis = {
        "edge_w": 1,
        "edge_e": 1,
        "edge_s": 0,
        "edge_n": 0,
    }
    edge_vect_axis = {
        "edge_vect_w": 1,
        "edge_vect_e": 1,
        "edge_vect_s": 0,
        "edge_vect_n": 0,
    }
    ee_vars = ["ee1", "ee2", "ew1", "ew2", "es1", "es2"]
    edge_vect_axis = {
        "edge_vect_s": 0,
        "edge_vect_n": 0,
        "edge_vect_w": 1,
        "edge_vect_e": 1,
    }
    # Super (composite) grid
    #     9---4---8
    #     |       |
    #     1   5   3
    #     |       |
    #     6---2---7

    @classmethod
    def new_from_serialized_data(cls, serializer, rank, layout, backend):
        grid_savepoint = serializer.get_savepoint("Grid-Info")[0]
        grid_data = {}
        grid_fields = serializer.fields_at_savepoint(grid_savepoint)
        for field in grid_fields:
            grid_data[field] = read_serialized_data(serializer, grid_savepoint, field)
        return cls(grid_data, rank, layout, backend=backend)

    def __init__(self, inputs, rank, layout, *, backend: str):
        self.backend = backend
        self.indices = {}
        self.shape_params = {}
        self.data = {}
        for s in Grid.shape_params:
            self.shape_params[s] = inputs[s]
            del inputs[s]
        self.rank = rank
        self.layout = layout
        for i, j in Grid.index_pairs:
            for index in [i, j]:
                self.indices[index] = inputs[index] + self.fpy_model_index_offset
                del inputs[index]

        self.data = inputs

    def _make_composite_var_storage(self, varname, data3d, shape, count):

        for s in range(count):
            self.data[varname + str(s + 1)] = utils.make_storage_data(
                np.squeeze(data3d[:, :, s]),
                shape,
                origin=(0, 0, 0),
                backend=self.backend,
            )

    def _edge_vector_storage(self, varname, axis, max_shape):
        default_origin = (0, 0, 0)
        mask = None
        if axis == 1:
            buffer = np.zeros(max_shape[1])
            buffer[: self.data[varname].shape[0]] = self.data[varname]
            default_origin = (0, 0)
            buffer = buffer[np.newaxis, ...]
            buffer = np.repeat(buffer, max_shape[0], axis=0)
        if axis == 0:
            buffer = np.zeros(max_shape[0])
            buffer[: self.data[varname].shape[0]] = self.data[varname]
            default_origin = (0,)
            mask = (True, False, False)
        self.data[varname] = utils.make_storage_data(
            data=buffer,
            origin=default_origin,
            shape=buffer.shape,
            backend=self.backend,
            mask=mask,
        )

    def _make_composite_vvar_storage(self, varname, data3d, shape):
        """This function is needed to transform vlat, vlon"""

        size1, size2 = data3d.shape[0:2]
        buffer = np.zeros((shape[0], shape[1], 3))
        buffer[1 : 1 + size1, 1 : 1 + size2, :] = data3d
        self.data[varname] = utils.make_storage_data(
            data=buffer,
            shape=buffer.shape,
            origin=(1, 1, 0),
            backend=self.backend,
        )

    def make_grid_storage(self, pygrid):
        shape = pygrid.domain_shape_full(add=(1, 1, 1))
        for key in TranslateGrid.composite_grid_vars:
            if key in self.data:
                self._make_composite_var_storage(key, self.data[key], shape, 9)
                del self.data[key]

        for key in TranslateGrid.vvars:
            if key in self.data:
                self._make_composite_vvar_storage(key, self.data[key], shape)

        for key in TranslateGrid.ee_vars:
            if key in self.data:
                self.data[key] = np.moveaxis(self.data[key], 0, 2)
                self.data[key] = utils.make_storage_data(
                    self.data[key],
                    (shape[0], shape[1], 3),
                    origin=(0, 0, 0),
                    backend=self.backend,
                )
        for key, axis in TranslateGrid.edge_var_axis.items():
            if key in self.data:
                self.data[key] = utils.make_storage_data(
                    self.data[key],
                    shape,
                    start=(0, 0, pygrid.halo),
                    axis=axis,
                    read_only=True,
                    backend=self.backend,
                )
        for key, axis in TranslateGrid.edge_vect_axis.items():
            if key in self.data:
                self._edge_vector_storage(key, axis, shape)

        for key, value in self.data.items():
            if type(value) is np.ndarray and len(value.shape) > 0:
                # TODO: when grid initialization model exists, may want to use
                # it to inform this
                istart, jstart = pygrid.horizontal_starts_from_shape(value.shape)
                logger.debug(
                    "Storage for Grid variable {}, {}, {}, {}".format(
                        key, istart, jstart, value.shape
                    )
                )
                origin = (istart, jstart, 0)
                self.data[key] = utils.make_storage_data(
                    value,
                    shape,
                    origin=origin,
                    start=origin,
                    read_only=True,
                    backend=self.backend,
                )

    def python_grid(self):
        pygrid = Grid(
            self.indices, self.shape_params, self.rank, self.layout, self.backend
        )
        self.make_grid_storage(pygrid)
        pygrid.add_data(self.data)
        return pygrid
