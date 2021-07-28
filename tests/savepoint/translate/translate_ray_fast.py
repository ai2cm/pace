import fv3core._config as spec
from fv3core.stencils.ray_fast import RayleighDamping
from fv3core.testing import TranslateFortranData2Py


class TranslateRay_Fast(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = RayleighDamping(
            self.grid.grid_indexing,
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
