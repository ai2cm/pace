from .translate import TranslateFortranData2Py
import fv3.stencils.vbke as vbke


class TranslateVbKE(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = vbke.compute
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "vt": {},
            "vb": grid.compute_dict_buffer_2d(),
        }
        self.in_vars["parameters"] = ["dt5", "dt4"]
        self.out_vars = {"vb": grid.compute_dict_buffer_2d()}
