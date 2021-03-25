import fv3core.stencils.xtp_u as xtpu

from .translate_ytp_v import TranslateYTP_V


class TranslateXTP_U(TranslateYTP_V):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"]["c"]["serialname"] = "ub"
        self.in_vars["data_vars"]["flux"]["serialname"] = "vb"

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        xtpu.compute(inputs["c"], inputs["u"], inputs["flux"])
        return self.slice_output(inputs)
