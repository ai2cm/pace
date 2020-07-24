from .parallel_translate import ParallelTranslate2Py
import fv3core.stencils.rayleigh_friction as ray_friction


class TranslateRayleigh_Friction(ParallelTranslate2Py):
    def __init__(self, grids):
        super().__init__(grids)
        self._base.compute_func = ray_friction.compute
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
