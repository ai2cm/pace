from .translate import TranslateFortranData2Py
import fv3.stencils.moist_cv as moist_cv
import fv3.utils.gt4py_utils as utils


class TranslateMoistCVPlusTe_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = moist_cv.compute_te
        self.in_vars["data_vars"] = {
            "qvapor_js": {},
            "qliquid_js": {},
            "qice_js": {},
            "qrain_js": {},
            "qsnow_js": {},
            "qgraupel_js": {},
            "gz": {"serialname": "gz1d", "kstart": grid.is_, "axis": 0},
            "cvm": {"kstart": grid.is_, "axis": 0},
            "phis": {"serialname": "phism", "istart": grid.is_},
            "delp": {},
            "q_con": {},
            "pt": {},
            "w": {},
            "u": {},
            "v": {},
            "te_2d": grid.compute_dict(),
        }
        for k, v in self.in_vars["data_vars"].items():
            if k not in ["gz", "cvm"]:
                v["axis"] = 1

        self.in_vars["parameters"] = ["r_vir", "j_2d"]
        self.out_vars = {
            "gz": {
                "serialname": "gz1d",
                "istart": grid.is_,
                "iend": grid.ie,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            "cvm": {
                "istart": grid.is_,
                "iend": grid.ie,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            "te_2d": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kstart": grid.npz - 1,
                "kend": grid.npz - 1,
            },
            "q_con": {},
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["j_2d"] += 2
        self.compute_func(**inputs)
        for var in ["gz", "cvm"]:
            inputs[var] = inputs[var][:, inputs["j_2d"] : inputs["j_2d"] + 1, :]
        return self.slice_output(inputs)
