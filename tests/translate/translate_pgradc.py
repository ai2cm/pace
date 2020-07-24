from .translate import TranslateFortranData2Py
import fv3core.stencils.pgradc as pgradc


class TranslatePGradC(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = pgradc.compute
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "delpc": {},
            "pkc": grid.default_buffer_k_dict(),
            "gz": grid.default_buffer_k_dict(),
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {"uc": grid.x3d_domain_dict(), "vc": grid.y3d_domain_dict()}
