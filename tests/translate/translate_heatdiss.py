from .translate import TranslateFortranData2Py
import fv3core.stencils.heatdiss as heatdiss


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

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        heatdiss.compute(**inputs)
        return self.slice_output(inputs)
