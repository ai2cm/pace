from fv3core.stencils.divergence_damping import DivergenceDamping
from pace.stencils.testing import TranslateFortranData2Py


class TranslateA2B_Ord4(TranslateFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {"wk": {}, "vort": {}, "delpc": {}, "nord_col": {}}
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = {"wk": {}, "vort": {}}
        self.namelist = namelist
        self.stencil_factory = stencil_factory

    def compute_from_storage(self, inputs):
        divdamp = DivergenceDamping(
            self.stencil_factory,
            self.grid.grid_data,
            self.grid.damping_coefficients,
            self.grid.nested,
            self.grid.stretched_grid,
            self.namelist.dddmp,
            self.namelist.d4_bg,
            self.namelist.nord,
            self.namelist.grid_type,
            inputs["nord_col"],
            inputs["nord_col"],
        )
        del inputs["nord_col"]
        divdamp._vorticity_calc(**inputs)
        return inputs
