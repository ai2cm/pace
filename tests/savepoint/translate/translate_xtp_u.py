import fv3core._config as spec
import fv3core.stencils.xtp_u as xtp_u

from .translate_ytp_v import TranslateYTP_V


class TranslateXTP_U(TranslateYTP_V):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"]["u"] = {}
        self.in_vars["data_vars"]["c"]["serialname"] = "ub"
        self.in_vars["data_vars"]["flux"]["serialname"] = "vb"

    def compute_from_storage(self, inputs):
        xtp_obj = xtp_u.XTP_U(
            grid_indexing=self.grid.grid_indexing,
            grid_data=self.grid.grid_data,
            grid_type=spec.namelist.grid_type,
            iord=spec.namelist.hord_mt,
        )
        xtp_obj(inputs["c"], inputs["u"], inputs["flux"])
        return inputs
