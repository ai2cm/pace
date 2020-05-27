from .translate import TranslateFortranData2Py
import fv3.stencils.c2l_ord as c2l_ord


class TranslateCubedToLatLon(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = c2l_ord.compute_cubed_to_latlon
        self.in_vars["data_vars"] = {"u": {}, "v": {}, "ua": {}, "va": {}}
        self.in_vars["parameters"] = []  # do_halo
        self.out_vars = {"ua": {}, "va": {}}
