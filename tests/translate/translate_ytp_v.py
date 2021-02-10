import fv3core.stencils.ytp_v as ytpv
from fv3core.testing import TranslateFortranData2Py


class TranslateYTP_V(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        c_info = self.grid.compute_dict_buffer_2d()
        c_info["serialname"] = "vb"
        flux_info = self.grid.compute_dict_buffer_2d()
        flux_info["serialname"] = "ub"
        self.in_vars["data_vars"] = {"c": c_info, "u": {}, "v": {}, "flux": flux_info}
        self.in_vars["parameters"] = []
        self.out_vars = {"flux": flux_info}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        ytpv.compute(inputs["c"], inputs["v"], inputs["flux"])
        return self.slice_output(inputs)
