import fv3core.stencils.updatedzc as updatedzc

from .translate import TranslateFortranData2Py


class TranslateUpdateDzC(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = updatedzc.compute
        self.in_vars["data_vars"] = {
            "dp_ref": {"serialname": "dp0"},
            "zs": {},
            "ut": {"serialname": "utc"},
            "vt": {"serialname": "vtc"},
            "gz_in": {"serialname": "gz"},
            "ws3": {"serialname": "ws"},
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {
            "gz": grid.default_buffer_k_dict(),
            "ws": {"kstart": -1, "kend": None},
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        gz, ws = self.compute_func(**inputs)
        return self.slice_output(inputs, {"gz": gz, "ws": ws})
