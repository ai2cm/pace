import pace.dsl
import pace.util
from pace.fv3core import _config as spec
from pace.fv3core.stencils.riem_solver3 import RiemannSolver3
from pace.fv3core.testing import TranslateDycoreFortranData2Py


class TranslateRiem_Solver3(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.riemann_solver_3 = RiemannSolver3(
            stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            config=spec.RiemannConfig(
                p_fac=namelist.p_fac,
                a_imp=namelist.a_imp,
                use_logp=namelist.use_logp,
                beta=namelist.beta,
            ),
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
        self.stencil_factory = stencil_factory

    def compute_func(self, **kwargs):
        kwargs["last_call"] = bool(kwargs["last_call"])
        return self.riemann_solver_3(**kwargs)
