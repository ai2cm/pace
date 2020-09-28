import fv3core.stencils.ke_c_sw as KE_C_SW

from .translate import TranslateFortranData2Py


class TranslateKE_C_SW(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = KE_C_SW.compute
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "u": {},
            "v": {},
            "ua": {},
            "va": {},
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {
            "ke_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
            "vort_c": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
            },
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        ke_c, vort_c = KE_C_SW.compute(**inputs)
        return self.slice_output(inputs, {"ke_c": ke_c, "vort_c": vort_c})
