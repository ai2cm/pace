from fv3core.testing import TranslateFortranData2Py
from fv3gfs.util.stencils.pk3_halo import PK3Halo


class TranslatePK3_Halo(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = PK3Halo(grid.stencil_factory)
        self.in_vars["data_vars"] = {"pk3": {}, "delp": {}}
        self.in_vars["parameters"] = ["akap", "ptop"]
        self.out_vars = {"pk3": {"kend": grid.npz + 1}}
