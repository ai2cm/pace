from fv3.translate.translate import TranslateFortranData2Py, TranslateGrid
import fv3.stencils.yppm as yppm


class TranslateYPPM(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = yppm.compute_flux
        self.in_vars["data_vars"] = {
            "q": {"istart": "ifirst"},
            "c": {"jstart": grid.js},
        }
        self.in_vars["parameters"] = ["jord", "ifirst", "ilast"]
        self.out_vars = {
            "flux": {
                "istart": "ifirst",
                "iend": "ilast",
                "jstart": grid.js,
                "jend": grid.je + 1,
            }
        }

    def ivars(self, inputs):
        inputs["ifirst"] += TranslateGrid.fpy_model_index_offset
        inputs["ilast"] += TranslateGrid.fpy_model_index_offset
        inputs["ifirst"] = self.grid.global_to_local_x(inputs["ifirst"])
        inputs["ilast"] = self.grid.global_to_local_x(inputs["ilast"])

    def process_inputs(self, inputs):
        self.ivars(inputs)
        self.make_storage_data_input_vars(inputs)

    def compute(self, inputs):
        self.process_inputs(inputs)
        flux = self.compute_func(**inputs)
        return self.slice_output(inputs, {"flux": flux})


class TranslateYPPM_2(TranslateYPPM):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"]["q"]["serialname"] = "q_2"
        self.out_vars["flux"]["serialname"] = "flux_2"
