import pace.dsl
import pace.util
from pace.fv3core.stencils.riem_solver_c import RiemannSolverC
from pace.fv3core.testing import TranslateDycoreFortranData2Py


class TranslateRiem_Solver_C(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.compute_func = RiemannSolverC(  # type: ignore
            stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            p_fac=namelist.p_fac,
        )
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
        self.stencil_factory = stencil_factory
