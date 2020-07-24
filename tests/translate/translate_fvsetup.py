from .translate import TranslateFortranData2Py
import fv3core.stencils.moist_cv as moist_cv
import fv3core.utils.gt4py_utils as utils


class TranslateFVSetup(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = moist_cv.fv_setup
        self.in_vars["data_vars"] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "pt": {},
            "pkz": {"istart": grid.is_, "jstart": grid.js},
            "delz": {},
            "delp": {},
            "cappa": {},
            "q_con": {},
            "dp1": {},
            "cvm": {"kstart": grid.is_, "axis": 0},
        }
        self.in_vars["parameters"] = ["zvir"]
        self.out_vars = {
            "cvm": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.je,
                "jend": grid.je,
                "kstart": grid.npz,
                "kend": grid.npz,
            },
            "dp1": {},
            "q_con": {},
            "pkz": grid.compute_dict(),
            "cappa": {},
        }
