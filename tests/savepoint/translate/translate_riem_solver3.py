import fv3core._config as spec
from fv3core.stencils.riem_solver3 import RiemannSolver3
from fv3core.testing import TranslateFortranData2Py


class TranslateRiem_Solver3(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = RiemannSolver3(
            self.grid.grid_indexing,
            spec.namelist.p_fac,
            spec.namelist.a_imp,
            spec.namelist.use_logp,
            spec.namelist.beta,
        )
        self.in_vars["data_vars"] = {
            "cappa": {},
            "zs": {},
            "w": {},
            "delz": {},
            "q_con": {},
            "delp": {},
            "pt": {},
            "zh": {},
            "pe": {"istart": grid.is_ - 1, "jstart": grid.js - 1, "kaxis": 1},
            "ppe": {},
            "pk3": {},
            "pk": {},
            "peln": {"istart": grid.is_, "jstart": grid.js, "kaxis": 1},
            "wsd": {"istart": grid.is_, "jstart": grid.js},
        }
        self.in_vars["parameters"] = ["dt", "ptop", "last_call"]
        self.out_vars = {
            "zh": {"kend": grid.npz},
            "w": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "ppe": {"kend": grid.npz},
            "delz": {},
            "pk": grid.compute_buffer_k_dict(),
            "pk3": grid.default_buffer_k_dict(),
        }
