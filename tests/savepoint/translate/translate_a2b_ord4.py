import fv3core._config as spec
from fv3core.stencils.divergence_damping import DivergenceDamping
from fv3core.testing import TranslateFortranData2Py


class TranslateA2B_Ord4(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"wk": {}, "vort": {}, "delpc": {}, "nord_col": {}}
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {"wk": {}, "vort": {}}

    def compute_from_storage(self, inputs):
        divdamp = DivergenceDamping(
            spec.namelist, inputs["nord_col"], inputs["nord_col"]
        )
        del inputs["nord_col"]
        divdamp.vorticity_calc(**inputs)
        return inputs
