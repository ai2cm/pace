import fv3core.stencils.rayleigh_super as super_ray
from fv3core.testing import ParallelTranslate2Py


class TranslateRayleigh_Super(ParallelTranslate2Py):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.compute_func = super_ray.compute
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "phis": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "ua": {},
            "va": {},
            "pt": {},
            "delz": {},
            "pfull": {},
        }
        self._base.in_vars["parameters"] = ["bdt", "ptop"]
        self._base.out_vars = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "ua": {},
            "va": {},
            "pt": {},
            "delz": {},
        }
