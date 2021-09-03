from fv3core.testing import TranslateFortranData2Py
import numpy as np
import fv3core.utils.gt4py_utils as utils


def read_serialized_data(serializer, savepoint, variable):
    data = serializer.read(variable, savepoint)
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
    return rearranged


class TranslatePhysicsFortranData2Py(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)

    def collect_input_data(self, serializer, savepoint):
        input_data = {}
        for varname in (
            self.serialnames(self.in_vars["data_vars"]) + self.in_vars["parameters"]
        ):
            input_data[varname] = read_serialized_data(serializer, savepoint, varname)
        return input_data

    def slice_output(self, inputs, out_data=None):
        utils.device_sync()
        if out_data is None:
            out_data = inputs
        else:
            out_data.update(inputs)
        out = {}
        for var in self.out_vars.keys():
            info = self.out_vars[var]
            self.update_info(info, inputs)
            serialname = info["serialname"] if "serialname" in info else var
            data_result = out_data[var]
            n_dim = len(data_result.shape)
            cn2 = int(data_result.shape[0] - self.grid.halo * 2 - 1) ** 2
            npz = data_result.shape[2]
            k_length = info["kend"] if "kend" in info else npz
            roll_zero = info["roll_zero"] if "roll_zero" in info else False
            data_result.synchronize()
            if n_dim == 3:
                data_compute = np.asarray(data_result)[
                    self.grid.global_is : self.grid.global_ie + 1,
                    self.grid.global_is : self.grid.global_ie + 1,
                    :,
                ]
                data_compute = np.reshape(data_compute, (cn2, npz))
                if k_length < npz:
                    out[serialname] = data_compute[:, ::-1][:, 1:]
                else:
                    if roll_zero:
                        out[serialname] = np.roll(data_compute[:, ::-1], -1)
                    else:
                        out[serialname] = data_compute[:, ::-1]
        return out
