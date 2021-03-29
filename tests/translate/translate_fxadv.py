import fv3core.stencils.fxadv as fxadv
import fv3core.utils.gt4py_utils as utils
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
            "ra_x": {"istart": grid.is_, "iend": grid.ie},
            "ra_y": {"jstart": grid.js, "jend": grid.je},
            "ut": utinfo,
            "vt": vtinfo,
        }
        for var in ["xfx_adv", "crx_adv", "yfx_adv", "cry_adv"]:
            self.out_vars[var] = self.in_vars["data_vars"][var]

    def compute_from_storage(self, inputs):
        grid = self.grid
        inputs["ra_x"] = utils.make_storage_from_shape(
            inputs["uc"].shape, grid.compute_origin()
        )
        inputs["ra_y"] = utils.make_storage_from_shape(
            inputs["vc"].shape, grid.compute_origin()
        )
        fxadv.compute(**inputs)
        return inputs
