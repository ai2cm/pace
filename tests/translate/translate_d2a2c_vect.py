import fv3core.stencils.d2a2c_vect as d2a2c_vect
from fv3core.testing import TranslateFortranData2Py


class TranslateD2A2C_Vect(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = d2a2c_vect.compute
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "u": {},
            "v": {},
            "ua": {},
            "va": {},
            "utc": {},
            "vtc": {},
        }
        self.in_vars["parameters"] = ["dord4"]
        self.out_vars = {
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "ua": {},
            "va": {},
            "utc": {},
            "vtc": {},
        }
        # TODO: This seems to be needed primarily for the edge_interpolate_4
        # methods, can we rejigger the order of operations to make it match to
        # more precision?
        self.max_error = 2e-10
