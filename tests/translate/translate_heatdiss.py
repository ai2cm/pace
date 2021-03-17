import fv3core.stencils.d_sw as d_sw
from fv3core.testing import TranslateFortranData2Py


class TranslateHeatDiss(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "fx2": {},
            "fy2": {},
            "w": {},
            "dw": {},
            "heat_source": {},
            "diss_est": {},
        }
        self.in_vars["parameters"] = ["dd8"]
        self.out_vars = {
            "heat_source": grid.compute_dict(),
            "diss_est": grid.compute_dict(),
            "dw": grid.compute_dict(),
        }

    def compute_from_storage(self, inputs):
        inputs["rarea"] = self.grid.rarea
        inputs["origin"] = self.grid.compute_origin()
        inputs["domain"] = self.grid.domain_shape_compute()
        d_sw.heat_diss(**inputs)
        return inputs
