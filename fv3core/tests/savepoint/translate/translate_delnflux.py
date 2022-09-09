import pace.dsl
import pace.fv3core.stencils.delnflux as delnflux
import pace.util
from pace.fv3core.testing import TranslateDycoreFortranData2Py


class TranslateDelnFlux(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "q": {},
            "fx": grid.x3d_compute_dict(),
            "fy": grid.y3d_compute_dict(),
            "damp_c": {},
            "nord_column": {},
            "mass": {},
        }
        self.in_vars["parameters"] = []
        self.out_vars = {"fx": grid.x3d_compute_dict(), "fy": grid.y3d_compute_dict()}
        self.stencil_factory = stencil_factory

    # If use_sg is defined -- 'dx', 'dy', 'rdxc', 'rdyc', 'sin_sg needed
    def compute(self, inputs):
        if "mass" not in inputs:
            inputs["mass"] = None
        self.make_storage_data_input_vars(inputs)
        self.compute_func = delnflux.DelnFlux(
            self.stencil_factory,
            self.grid.damping_coefficients,
            self.grid.rarea,
            inputs.pop("nord_column"),
            inputs.pop("damp_c"),
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)


class TranslateDelnFlux_2(TranslateDelnFlux):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        del self.in_vars["data_vars"]["mass"]
