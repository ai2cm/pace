import fv3core._config as spec
from fv3core.stencils.divergence_damping import DivergenceDamping
from fv3core.testing import TranslateFortranData2Py


class TranslateDivergenceDamping(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
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

    def compute_from_storage(self, inputs):
        divdamp = DivergenceDamping(spec.namelist, inputs["nord_col"], inputs["d2_bg"])
        del inputs["nord_col"]
        del inputs["d2_bg"]
        divdamp(**inputs)
        return inputs
