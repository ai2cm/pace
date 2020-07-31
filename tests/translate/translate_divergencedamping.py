import fv3core.stencils.divergence_damping as dd
from .translate import TranslateFortranData2Py


class TranslateDivergenceDamping(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = dd.compute
        self.in_vars["data_vars"] = {
            "u": {},
            "v": {},
            "va": {},
            "ptc": {},
            "vort": {},
            "ua": {},
            "divg_d": {},
            "vc": {},
            "uc": {},
            "delpc": {},
            "ke": {},
            "wk": {},
            "nord_col": {},
            "d2_bg": {},
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {
            "vort": {},
            "ke": {"iend": grid.ied + 1, "jend": grid.jed + 1},
            "delpc": {},
        }
        self.max_error = 3.0e-11

    def compute(self, inputs):
        return self.column_split_compute(inputs, {"nord": "nord_col", "d2_bg": "d2_bg"})
