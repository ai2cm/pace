import fv3core.stencils.moist_cv as moist_cv
from fv3core.testing import TranslateFortranData2Py, TranslateGrid


class TranslateMoistCVPlusPkz_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = moist_cv.compute_pkz
        self.in_vars["data_vars"] = {
            "qvapor_js": {},
            "qliquid_js": {},
            "qice_js": {},
            "qrain_js": {},
            "qsnow_js": {},
            "qgraupel_js": {},
            "gz": {"serialname": "gz1d", "kstart": grid.is_, "axis": 0},
            "cvm": {"kstart": grid.is_, "axis": 0},
            "delp": {},
            "delz": {},
            "q_con": {},
            "pkz": {"istart": grid.is_, "jstart": grid.js},
            "pt": {},
            "cappa": {},
        }
        self.write_vars = ["gz", "cvm"]
        for k, v in self.in_vars["data_vars"].items():
            if k not in self.write_vars:
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
            "pkz": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
            },
            "cappa": {},
            "q_con": {},
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["j_2d"] = self.grid.global_to_local_y(
            inputs["j_2d"] + TranslateGrid.fpy_model_index_offset
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)
