import fv3core.stencils.c2l_ord as c2l_ord
from fv3core.testing import TranslateFortranData2Py


class TranslateC2L_Ord2(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = c2l_ord.compute_ord2
        self.in_vars["data_vars"] = {"u": {}, "v": {}, "ua": {}, "va": {}}
        self.in_vars["parameters"] = []  # do_halo
        self.out_vars = {"ua": {}, "va": {}}
