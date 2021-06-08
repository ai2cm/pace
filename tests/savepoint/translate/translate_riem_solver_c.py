import fv3core._config as spec
from fv3core.stencils.riem_solver_c import RiemannSolverC
from fv3core.testing import TranslateFortranData2Py


class TranslateRiem_Solver_C(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = RiemannSolverC(spec.namelist)
        self.in_vars["data_vars"] = {
            "cappa": {},
            "hs": {},
            "w3": {},
            "ptc": {},
            "q_con": {},
            "delpc": {},
            "gz": {},
            "pef": {},
            "ws": {},
        }
        self.in_vars["parameters"] = ["dt2", "ptop"]
        self.out_vars = {"pef": {"kend": grid.npz}, "gz": {"kend": grid.npz}}
        self.max_error = 5e-14
