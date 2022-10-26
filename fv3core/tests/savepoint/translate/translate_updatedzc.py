import pace.dsl
import pace.fv3core.stencils.updatedzc as updatedzc
import pace.util
from pace.fv3core.testing import TranslateDycoreFortranData2Py


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
            self.stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            area=grid.grid_data.area,
            dp_ref=grid.grid_data.dp_ref,
        )

        def compute(**kwargs):
            kwargs["dt"] = kwargs.pop("dt2")
            update_gz_on_c_grid(**kwargs)

        self.compute_func = compute
        self.in_vars["data_vars"] = {
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
