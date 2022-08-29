import numpy as np

import pace.dsl.gt4py_utils as utils
from pace.dsl.stencil import GridIndexing
from pace.physics import PhysicsConfig
from pace.stencils.testing.parallel_translate import ParallelTranslate2Py
from pace.stencils.testing.translate import TranslateFortranData2Py


def transform_dwind_serialized_data(data, grid_indexing: GridIndexing, backend: str):
    max_shape = grid_indexing.domain_full(add=(1, 1, 1))
    # convert single element numpy arrays to scalars
    if data.size == 1:
        data = data.item()
    elif len(data.shape) < 2:
        start1 = 0
        size1 = data.shape[0]
        padded_data = np.zeros(max_shape[2])
        padded_data[start1 : start1 + size1] = data
        data = padded_data
    elif len(data.shape) == 2:
        padded_data = np.zeros(max_shape[0:2])
        start1, start2 = (0, 0)
        size1, size2 = data.shape
        padded_data[start1 : start1 + size1, start2 : start2 + size2] = data
        data = padded_data
    else:
        start1, start2, start3 = 0, 0, 0
        size1, size2, size3 = data.shape
        padded_data = np.zeros(max_shape)
        padded_data[
            start1 : start1 + size1,
            start2 : start2 + size2,
            start3 : start3 + size3,
        ] = data
        data = padded_data
    if isinstance(data, np.ndarray):
        data = utils.make_storage_data(
            data=data,
            origin=grid_indexing.origin,
            shape=data.shape,
            backend=backend,
        )
    return data


class TranslatePhysicsFortranData2Py(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, stencil_factory)
        self.namelist = PhysicsConfig.from_namelist(namelist)

    def transform_physics_serialized_data(self, data, roll_zero, index_order):
        if isinstance(data, np.ndarray):
            n_dim = len(data.shape)
            cn = int(np.sqrt(data.shape[0]))
            if n_dim == 2:
                var_reshape = np.reshape(
                    data, (cn, cn, data.shape[1]), order=index_order
                )
                rearranged = var_reshape[:, :, ::-1]
            elif n_dim == 3:
                npz = data.shape[1]
                n_trac = data.shape[2]
                var_reshape = np.reshape(data, (cn, cn, npz, n_trac), order=index_order)
                rearranged = var_reshape[:, :, ::-1, :]
            elif len(data.flatten()) == 1:
                rearranged = data[0]
            else:
                raise NotImplementedError("Data dimension not supported")
        else:
            return data
        if roll_zero:
            rearranged = np.roll(rearranged, -1, axis=-1)
        return rearranged

    def transform_microphysics_serialized_data(self, data):
        if isinstance(data, np.ndarray):
            n_dim = len(data.shape)
            cn = int(np.sqrt(data.shape[0]))
            npz = data.shape[-1]
            if n_dim == 3:
                var_reshape = np.reshape(data[:, 0, :], (cn, cn, npz))
                rearranged = var_reshape[:, :, :]
            elif len(data.flatten()) == 1:
                rearranged = data[0]
            else:
                raise NotImplementedError("Data dimension not supported")
        else:
            return data
        return rearranged

    def transform_dwind_serialized_data(self, data):
        return transform_dwind_serialized_data(
            data, self.stencil_factory.grid_indexing, self.stencil_factory.backend
        )

    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        for varname in [*self.in_vars["data_vars"]]:
            info = self.in_vars["data_vars"][varname]
            roll_zero = info["in_roll_zero"] if "in_roll_zero" in info else False
            if "serialname" in info:
                serialname = info["serialname"]
            else:
                serialname = varname
            dycore_format = info["dycore"] if "dycore" in info else False
            microph_format = info["microph"] if "microph" in info else False
            dwind_format = info["dwind"] if "dwind" in info else False
            index_order = info["order"] if "order" in info else "C"
            if dycore_format:
                pass
            elif microph_format:
                inputs[serialname] = self.transform_microphysics_serialized_data(
                    inputs[serialname]
                )
            elif dwind_format:
                inputs[serialname] = self.transform_dwind_serialized_data(
                    inputs[serialname]
                )
            else:
                inputs[serialname] = self.transform_physics_serialized_data(
                    inputs[serialname], roll_zero, index_order
                )
        super().make_storage_data_input_vars(inputs, storage_vars=storage_vars)

    def slice_output(self, inputs, out_data=None):
        if out_data is None:
            out_data = inputs
        else:
            out_data.update(inputs)
        out = {}
        for var in self.out_vars.keys():
            info = self.out_vars[var]
            self.update_info(info, inputs)
            manual = info["manual"] if "manual" in info else False
            serialname = info["serialname"] if "serialname" in info else var
            compute_domain = info["compute"] if "compute" in info else True
            if not manual:
                data_result = out_data[var]
                n_dim = len(data_result.shape)
                cn2 = int(data_result.shape[0] - self.grid.halo * 2 - 1) ** 2
                roll_zero = info["out_roll_zero"] if "out_roll_zero" in info else False
                index_order = info["order"] if "order" in info else "C"
                dycore = info["dycore"] if "dycore" in info else False
                data_result.synchronize()
                if n_dim == 3:
                    npz = data_result.shape[2]
                    k_length = info["kend"] if "kend" in info else npz
                    if compute_domain:
                        ds = self.grid.compute_dict()
                    else:
                        ds = self.grid.default_domain_dict()
                    ds.update(info)
                    ij_slice = self.grid.slice_dict(ds)
                    data_compute = np.asarray(data_result)[
                        ij_slice[0],
                        ij_slice[1],
                        :,
                    ]
                    if dycore:
                        if k_length < npz:
                            data_compute = data_compute[:, :, 0:-1]
                        out[serialname] = data_compute
                    else:
                        data_compute = np.reshape(
                            data_compute, (cn2, npz), order=index_order
                        )
                        if k_length < npz:
                            out[serialname] = data_compute[:, ::-1][:, 1:]
                        else:
                            if roll_zero:
                                out[serialname] = np.roll(data_compute[:, ::-1], -1)
                            else:
                                out[serialname] = data_compute[:, ::-1]
                elif n_dim == 2:
                    if compute_domain:
                        ds = self.grid.compute_dict()
                    else:
                        ds = self.grid.default_domain_dict()
                    ds.update(info)
                    ij_slice = self.grid.slice_dict(ds)
                    data_compute = np.asarray(data_result)[ij_slice[0], ij_slice[1]]
                    out[serialname] = data_compute
                else:
                    raise NotImplementedError("Output data dimension not supported")
        return out


class ParallelPhysicsTranslate2Py(ParallelTranslate2Py):
    def __init__(self, rank_grids, namelist, stencil_factory):
        physics_namelist = PhysicsConfig.from_namelist(namelist)
        super().__init__(rank_grids, physics_namelist, stencil_factory)
