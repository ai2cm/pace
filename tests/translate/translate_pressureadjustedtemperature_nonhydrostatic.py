from .translate import TranslateFortranData2Py
import fv3core.stencils.temperature_adjust as temperature_adjust


class TranslatePressureAdjustedTemperature_NonHydrostatic(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = temperature_adjust.compute
        self.in_vars["data_vars"] = {
            "cappa": {},
            "delp": {},
            "delz": {},
            "pt": {},
            "heat_source": {"serialname": "heat_source_dyn"},
            "pkz": grid.compute_dict(),
        }
        self.in_vars["parameters"] = ["bdt", "n_con"]
        self.out_vars = {"pt": {}, "pkz": grid.compute_dict()}
