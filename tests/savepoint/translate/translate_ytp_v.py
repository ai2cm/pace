import fv3core._config as spec
import fv3core.stencils.ytp_v as ytp_v
from fv3core.testing import TranslateFortranData2Py


class TranslateYTP_V(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        c_info = self.grid.compute_dict_buffer_2d()
        c_info["serialname"] = "vb"
        flux_info = self.grid.compute_dict_buffer_2d()
        flux_info["serialname"] = "ub"
        self.in_vars["data_vars"] = {"c": c_info, "v": {}, "flux": flux_info}
        self.in_vars["parameters"] = []
        self.out_vars = {"flux": flux_info}

    def compute_from_storage(self, inputs):
        ytp_obj = ytp_v.YTP_V(
            grid_indexing=spec.grid.grid_indexing,
            dy=spec.grid.dy,
            dya=spec.grid.dya,
            rdy=spec.grid.rdy,
            grid_type=spec.namelist.grid_type,
            jord=spec.namelist.hord_mt,
        )
        ytp_obj(inputs["c"], inputs["v"], inputs["flux"])
        return inputs
