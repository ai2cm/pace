import fv3core.stencils.updatedzc as updatedzc
import pace.dsl
import pace.util
from pace.stencils.testing import TranslateDycoreFortranData2Py


class TranslateUpdateDzC(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        update_gz_on_c_grid = updatedzc.UpdateGeopotentialHeightOnCGrid(
            self.stencil_factory, grid.area
        )

        def compute(**kwargs):
            kwargs["dt"] = kwargs.pop("dt2")
            update_gz_on_c_grid(**kwargs)

        self.compute_func = compute
        self.in_vars["data_vars"] = {
            "dp_ref": {"serialname": "dp0"},
            "zs": {},
            "ut": {"serialname": "utc"},
            "vt": {"serialname": "vtc"},
            "gz": {},
            "ws": {},
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {
            "gz": grid.default_buffer_k_dict(),
            "ws": {"kstart": -1, "kend": None},
        }
