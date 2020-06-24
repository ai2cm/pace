from .translate import TranslateFortranData2Py
import fv3.stencils.neg_adj3 as neg_adj3


class TranslateNeg_Adj3(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = neg_adj3.compute
        self.in_vars["data_vars"] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "pt": {},
            "delp": {},
            "delz": {},
            "peln": {"istart": grid.is_, "jstart": grid.js, "kaxis": 1},
        }
        self.in_vars["parameters"] = []
        self.out_vars = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            # "pt": {},
        }
