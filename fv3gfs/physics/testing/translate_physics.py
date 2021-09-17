from fv3core.testing import TranslateFortranData2Py
import numpy as np
import fv3core.utils.gt4py_utils as utils


class TranslatePhysicsFortranData2Py(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)

    def read_physics_serialized_data(self, serializer, savepoint, variable, roll_zero):
        data = serializer.read(variable, savepoint)
        if isinstance(data, np.ndarray):
            n_dim = len(data.shape)
            cn = int(np.sqrt(data.shape[0]))
            npz = data.shape[1]
            if n_dim == 2:
                var_reshape = np.reshape(data, (cn, cn, npz))
                rearranged = var_reshape[:, :, ::-1]
            elif n_dim == 3:
                n_trac = data.shape[2]
                var_reshape = np.reshape(data, (cn, cn, npz, n_trac))
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

    def read_microphysics_serialized_data(self, serializer, savepoint, variable):
        data = serializer.read(variable, savepoint)
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

    def read_dycore_serialized_data(self, serializer, savepoint, variable):
        data = serializer.read(variable, savepoint)
        if len(data.flatten()) == 1:
            return data[0]
        return data

    def collect_input_data(self, serializer, savepoint):
        input_data = {}
        for varname in [*self.in_vars["data_vars"]] + self.in_vars["parameters"]:
            info = self.in_vars["data_vars"][varname]
            roll_zero = info["in_roll_zero"] if "in_roll_zero" in info else False
            if "serialname" in info:
                serialname = info["serialname"]
            else:
                serialname = varname
            dycore_format = info["dycore"] if "dycore" in info else False
            microph_format = info["microph"] if "microph" in info else False
            if dycore_format:
                input_data[serialname] = self.read_dycore_serialized_data(
                    serializer, savepoint, serialname
                )
            elif microph_format:
                input_data[serialname] = self.read_microphysics_serialized_data(
                    serializer, savepoint, serialname
                )
            else:
                input_data[serialname] = self.read_physics_serialized_data(
                    serializer, savepoint, serialname, roll_zero
                )
        return input_data

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
                npz = data_result.shape[2]
                k_length = info["kend"] if "kend" in info else npz
                roll_zero = info["out_roll_zero"] if "out_roll_zero" in info else False
                index_order = info["order"] if "order" in info else "C"
                data_result.synchronize()
                ds = self.grid.compute_dict()
                ds.update(info)
                if n_dim == 3:
                    ij_slice = self.grid.slice_dict(ds)
                    data_compute = np.asarray(data_result)[
                        ij_slice[0], ij_slice[1], :,
                    ]
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
                else:
                    raise NotImplementedError("Output data dimension not supported")
        return out
