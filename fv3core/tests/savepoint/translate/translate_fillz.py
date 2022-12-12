import numpy as np

import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.fv3core.stencils.fillz as fillz
import pace.util
from pace.fv3core.testing import TranslateDycoreFortranData2Py
from pace.stencils.testing import pad_field_in_j


class TranslateFillz(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "dp2": {"istart": grid.is_, "iend": grid.ie, "axis": 1},
            "q2tracers": {"istart": grid.is_, "iend": grid.ie, "axis": 1},
        }
        self.in_vars["parameters"] = ["nq"]
        self.out_vars = {
            "q2tracers": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.js,
                "axis": 1,
            }
        }
        self.max_error = 1e-13
        self.ignore_near_zero_errors = {"q2tracers": True}
        self.stencil_factory = stencil_factory

    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        if storage_vars is None:
            storage_vars = self.storage_vars()
        info = storage_vars["dp2"]
        inputs["dp2"] = self.make_storage_data(
            np.squeeze(inputs["dp2"]), istart=info["istart"], axis=info["axis"]
        )
        inputs["tracers"] = {}
        info = storage_vars["q2tracers"]
        for i in range(int(inputs["nq"])):
            inputs["tracers"][utils.tracer_variables[i]] = self.make_storage_data(
                np.squeeze(inputs["q2tracers"][:, :, i]),
                istart=info["istart"],
                axis=info["axis"],
            )
        del inputs["q2tracers"]

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        for name, value in tuple(inputs.items()):
            if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] == 1:
                inputs[name] = self.make_storage_data(
                    pad_field_in_j(
                        value, self.grid.njd, backend=self.stencil_factory.backend
                    )
                )
        for name, value in tuple(inputs["tracers"].items()):
            if hasattr(value, "shape") and len(value.shape) > 1 and value.shape[1] == 1:
                inputs["tracers"][name] = self.make_storage_data(
                    pad_field_in_j(
                        value, self.grid.njd, backend=self.stencil_factory.backend
                    )
                )
        run_fillz = fillz.FillNegativeTracerValues(
            self.stencil_factory,
            self.grid.quantity_factory,
            inputs.pop("nq"),
            inputs["tracers"],
        )
        run_fillz(**inputs)
        ds = self.grid.default_domain_dict()
        ds.update(self.out_vars["q2tracers"])
        tracers = np.zeros((self.grid.nic, self.grid.npz, len(inputs["tracers"])))
        for varname, data in inputs["tracers"].items():
            index = utils.tracer_variables.index(varname)
            tracers[:, :, index] = np.squeeze(data[self.grid.slice_dict(ds)])
        out = {"q2tracers": tracers}
        return out
