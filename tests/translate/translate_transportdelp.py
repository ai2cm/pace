import fv3core.stencils.transportdelp as TransportDelp

from .translate import TranslateFortranData2Py


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
            "wc": {},
        }
        self.out_vars = {"delpc": {}, "ptc": {}, "wc": {}}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        delpc, ptc = self.compute_func(**inputs)
        return self.slice_output(inputs, {"delpc": delpc, "ptc": ptc})
