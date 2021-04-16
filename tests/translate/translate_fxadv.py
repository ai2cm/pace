import fv3core.stencils.fxadv as fxadv
from fv3core.testing import TranslateFortranData2Py


class TranslateFxAdv(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        utinfo = grid.x3d_domain_dict()
        vtinfo = grid.y3d_domain_dict()
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "ut": utinfo,
            "vt": vtinfo,
            "xfx_adv": grid.x3d_compute_domain_y_dict(),
            "crx_adv": grid.x3d_compute_domain_y_dict(),
            "yfx_adv": grid.y3d_compute_domain_x_dict(),
            "cry_adv": grid.y3d_compute_domain_x_dict(),
        }
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {
            "ut": utinfo,
            "vt": vtinfo,
        }
        for var in ["xfx_adv", "crx_adv", "yfx_adv", "cry_adv"]:
            self.out_vars[var] = self.in_vars["data_vars"][var]

    def compute_from_storage(self, inputs):
        fxadv.compute(**inputs)
        return inputs
