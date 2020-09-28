import fv3core.stencils.moist_cv as moist_cv
import fv3core.utils.gt4py_utils as utils

from .translate import TranslateFortranData2Py


class TranslateComputeTotalEnergy(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = moist_cv.compute_total_energy
        self.in_vars["data_vars"] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "w": {},
            "u": {},
            "v": {},
            "delz": {},
            "pt": {},
            "delp": {},
            "qc": {"serialname": "dp1"},
            "peln": {"istart": grid.is_, "jstart": grid.js, "kaxis": 1},
            "pe": {"istart": grid.is_ - 1, "jstart": grid.js - 1, "kaxis": 1},
            "hs": {"serialname": "phis"},
            "te_2d": grid.compute_dict(),
        }
        self.in_vars["parameters"] = ["zvir"]
        self.out_vars = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "te_2d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
        }
