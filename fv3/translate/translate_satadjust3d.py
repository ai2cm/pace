from .translate import TranslateFortranData2Py
import fv3.stencils.saturation_adjustment as satadjust


class TranslateSatAdjust3d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = satadjust.compute
        cvar = {"axis": 0, "kstart": 3}
        self.in_vars["data_vars"] = {
            "te": {},
            "dpln": {"istart": grid.is_, "jstart": grid.js},
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
            "dpln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
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

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["kmp"] -= 1
        self.compute_func(**inputs)
        return self.slice_output(inputs)
