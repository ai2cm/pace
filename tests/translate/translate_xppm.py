from .translate import TranslateFortranData2Py, TranslateGrid
import fv3core.stencils.xppm as xppm
import fv3core.utils.gt4py_utils as utils


class TranslateXPPM(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = xppm.compute_flux
        self.in_vars["data_vars"] = {
            "q": {"serialname": "qx", "jstart": "jfirst"},
            "c": {"serialname": "cx", "istart": grid.is_},
        }
        self.in_vars["parameters"] = ["iord", "jfirst", "jlast"]
        self.out_vars = {
            "xflux": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": "jfirst",
                "jend": "jlast",
            }
        }

    def jvars(self, inputs):
        inputs["jfirst"] += TranslateGrid.fpy_model_index_offset
        inputs["jlast"] += TranslateGrid.fpy_model_index_offset
        inputs["jfirst"] = self.grid.global_to_local_y(inputs["jfirst"])
        inputs["jlast"] = self.grid.global_to_local_y(inputs["jlast"])

    def process_inputs(self, inputs):
        self.jvars(inputs)
        self.make_storage_data_input_vars(inputs)

    def compute(self, inputs):
        self.process_inputs(inputs)
        inputs["xflux"] = utils.make_storage_from_shape(inputs["q"].shape, xppm.origin)
        self.compute_func(**inputs)
        return self.slice_output(inputs)


class TranslateXPPM_2(TranslateXPPM):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"]["q"]["serialname"] = "q"
        self.out_vars["xflux"]["serialname"] = "xflux_2"
