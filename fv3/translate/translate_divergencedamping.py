import fv3.stencils.divergence_damping as dd
from .translate_d_sw import TranslateD_SW


class TranslateDivergenceDamping(TranslateD_SW):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "u": {},
            "v": {},
            "va": {},
            "ptc": {},
            "vort": {},
            "ua": {},
            "divg_d": {},
            "vc": {},
            "uc": {},
            "delpc": {},
            "ke": {},
            "wk": {},
            "nord_col": {},
            "d2_bg": {},
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {
            "vort": {},
            "ke": {"iend": grid.ied + 1, "jend": grid.jed + 1},
            "delpc": {},
        }

    def compute(self, inputs):
        return self.column_split_compute(
            inputs, dd.compute, {"nord": "nord_col", "d2_bg": "d2_bg"}
        )

    """
        nord_column = [int(i) for i in inputs['nord_col'][0, 0, :]]
        self.make_storage_data_input_vars(inputs)
        del inputs['nord_col']
        dd.nord_compute(inputs, nord_column)
        return self.slice_output(inputs)
    """
