import fv3core._config as spec
from fv3core.testing import TranslateFortranData2Py
from fv3gfs.util.stencils.ray_fast import RayleighDamping


class TranslateRay_Fast(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = RayleighDamping(
            self.grid.stencil_factory,
            spec.namelist.rf_cutoff,
            spec.namelist.tau,
            spec.namelist.hydrostatic,
        )
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
