import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.util
from fv3core.stencils import xppm
from pace.stencils.testing import TranslateDycoreFortranData2Py, TranslateGrid


class TranslateXPPM(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "q": {"serialname": "qx", "jstart": "jfirst"},
            "c": {"serialname": "cx", "istart": grid.is_},
        }
        self.in_vars["parameters"] = ["iord", "jfirst", "jlast"]
        self.out_vars = {
            "xflux": {
                "istart": grid.is_,
                "iend": grid.ie + 1,
                "jstart": "jfirst",
                "jend": "jlast",
            }
        }
        self.stencil_factory = stencil_factory

    def jvars(self, inputs):
        inputs["jfirst"] += TranslateGrid.fpy_model_index_offset
        inputs["jlast"] += TranslateGrid.fpy_model_index_offset
        inputs["jfirst"] = self.grid.global_to_local_y(inputs["jfirst"])
        inputs["jlast"] = self.grid.global_to_local_y(inputs["jlast"])

    def process_inputs(self, inputs):
        self.jvars(inputs)
        self.make_storage_data_input_vars(inputs)

    def compute(self, inputs):
        self.process_inputs(inputs)
        inputs["xflux"] = utils.make_storage_from_shape(
            inputs["q"].shape, backend=self.stencil_factory.backend
        )
        origin = self.grid.grid_indexing.origin_compute()
        domain = self.grid.grid_indexing.domain_compute(add=(1, 1, 0))
        self.compute_func = xppm.XPiecewiseParabolic(
            stencil_factory=self.stencil_factory,
            dxa=self.grid.dxa,
            grid_type=self.grid.grid_type,
            iord=int(inputs["iord"]),
            origin=(origin[0], int(inputs["jfirst"]), origin[2]),
            domain=(domain[0], int(inputs["jlast"] - inputs["jfirst"] + 1), domain[2]),
        )
        self.compute_func(inputs["q"], inputs["c"], inputs["xflux"])
        return self.slice_output(inputs)


class TranslateXPPM_2(TranslateXPPM):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"]["q"]["serialname"] = "q"
        self.out_vars["xflux"]["serialname"] = "xflux_2"
