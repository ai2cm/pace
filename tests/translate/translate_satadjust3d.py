from fv3core.stencils.saturation_adjustment import SatAdjust3d
from fv3core.testing import TranslateFortranData2Py


class TranslateSatAdjust3d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        cvar = {"axis": 0, "kstart": 3}
        self.in_vars["data_vars"] = {
            "te": {},
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "hs": {},
            "peln": {"istart": grid.is_, "jstart": grid.js, "kaxis": 1},
            "delp": {},
            "delz": {},
            "q_con": {},
            "pt": {},
            "pkz": {"istart": grid.is_, "jstart": grid.js},
            "cappa": {},
        }
        self.max_error = 2e-14
        # te0 is off by 1e-10 when you do nothing...
        self.in_vars["parameters"] = [
            "r_vir",
            "mdt",
            "fast_mp_consv",
            "last_step",
            "akap",
            "kmp",
        ]
        self.out_vars = {
            "te": {},
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "q_con": {},
            "pt": {},
            "pkz": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "cappa": {},
        }

    def compute_from_storage(self, inputs):
        inputs["kmp"] -= 1
        satadjust3d_obj = SatAdjust3d(inputs["kmp"])
        satadjust3d_obj(**inputs)
        return inputs
