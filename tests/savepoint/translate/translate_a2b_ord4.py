import fv3core._config as spec
from fv3core.stencils.divergence_damping import DivergenceDamping
from fv3core.testing import TranslateFortranData2Py


class TranslateA2B_Ord4(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"wk": {}, "vort": {}, "delpc": {}, "nord_col": {}}
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {"wk": {}, "vort": {}}

    def compute_from_storage(self, inputs):
        divdamp = DivergenceDamping(
            self.grid.grid_indexing,
            self.grid.agrid1,
            self.grid.agrid2,
            self.grid.bgrid1,
            self.grid.bgrid2,
            self.grid.dxa,
            self.grid.dya,
            self.grid.edge_n,
            self.grid.edge_s,
            self.grid.edge_e,
            self.grid.edge_w,
            self.grid.nested,
            self.grid.stretched_grid,
            self.grid.da_min,
            self.grid.da_min_c,
            self.grid.divg_u,
            self.grid.divg_v,
            self.grid.rarea_c,
            self.grid.sin_sg1,
            self.grid.sin_sg2,
            self.grid.sin_sg3,
            self.grid.sin_sg4,
            self.grid.cosa_u,
            self.grid.cosa_v,
            self.grid.sina_u,
            self.grid.sina_v,
            self.grid.dxc,
            self.grid.dyc,
            spec.namelist.dddmp,
            spec.namelist.d4_bg,
            spec.namelist.nord,
            spec.namelist.grid_type,
            inputs["nord_col"],
            inputs["nord_col"],
        )
        del inputs["nord_col"]
        divdamp.vorticity_calc(**inputs)
        return inputs
