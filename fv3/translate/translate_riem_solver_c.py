from .translate import TranslateFortranData2Py
import fv3.stencils.riem_solver_c as riem_solver_c


class TranslateRiem_Solver_C(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = riem_solver_c.compute
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
        self.in_vars["parameters"] = ["dt2", "akap", "ptop", "ms"]
        self.out_vars = {
            "pef": {"kend": grid.npz},
            "gz": {"kend": grid.npz},
        }
