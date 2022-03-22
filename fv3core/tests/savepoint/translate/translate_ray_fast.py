from fv3core.stencils.ray_fast import RayleighDamping
from pace.stencils.testing import TranslateDycoreFortranData2Py


class TranslateRay_Fast(TranslateDycoreFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.compute_func = RayleighDamping(
            stencil_factory,
            namelist.rf_cutoff,
            namelist.tau,
            namelist.hydrostatic,
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
        self.stencil_factory = stencil_factory
