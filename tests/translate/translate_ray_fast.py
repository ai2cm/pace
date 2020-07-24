from .translate import TranslateFortranData2Py
import fv3core.stencils.ray_fast as ray_fast


class TranslateRay_Fast(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = ray_fast.compute
        self.in_vars["data_vars"] = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "dp": {},
            "pfull": {},
        }
        self.in_vars["parameters"] = ["dt", "ptop", "ks"]
        self.out_vars = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
        }
