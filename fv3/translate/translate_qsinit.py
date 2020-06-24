from .translate import TranslateFortranData2Py
import fv3.stencils.saturation_adjustment as satadjust


class TranslateQSInit(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = satadjust.compute
        self.in_vars["data_vars"] = {
            "table": {},
            "table2": {},
            "tablew": {},
            "des2": {},
            "desw": {},
        }
        self.out_vars = self.in_vars["data_vars"]

    def compute(self, inputs):
        satadjust.qs_init()
        return satadjust.satmix
