from .translate_ytp_v import TranslateYTP_V
import fv3.stencils.xtp_u as xtpu


class TranslateXTP_U(TranslateYTP_V):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"]["c"]["serialname"] = "ub"
        self.in_vars["data_vars"]["flux"]["serialname"] = "vb"

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        xtpu.compute(**inputs)
        return self.slice_output(inputs)
