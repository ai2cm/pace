import fv3core.stencils.moist_cv as moist_cv
from fv3core.testing import TranslateFortranData2Py


class TranslateLastStep(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = moist_cv.compute_last_step
        self.in_vars["data_vars"] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "pt": {},
            "pkz": {"istart": grid.is_, "jstart": grid.js},
            "gz": {"serialname": "gz1d", "kstart": grid.is_, "axis": 0},
        }
        self.in_vars["parameters"] = ["r_vir", "dtmp"]
        self.out_vars = {
            "gz": {
                "serialname": "gz1d",
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.je,
                "jend": grid.je,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            "pt": {},
        }
