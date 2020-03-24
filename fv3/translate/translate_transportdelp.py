from .translate import TranslateFortranData2Py
import fv3.stencils.transportdelp as TransportDelp


class TranslateTransportDelp(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = TransportDelp.compute
        self.in_vars["data_vars"] = {
            "delp": {},
            "pt": {},
            "w": {},
            "utc": {},
            "vtc": {},
        }
        self.out_vars = {
            "delpc": {},
            "ptc": {},
            "wc": {},
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        delpc, ptc, wc = self.compute_func(**inputs)
        return self.slice_output(inputs, {"delpc": delpc, "ptc": ptc, "wc": wc})
