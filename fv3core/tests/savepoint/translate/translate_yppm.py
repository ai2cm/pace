import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.util
from pace.fv3core.stencils import yppm
from pace.fv3core.testing import TranslateDycoreFortranData2Py
from pace.stencils.testing import TranslateGrid


class TranslateYPPM(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "q": {"istart": "ifirst"},
            "c": {"jstart": grid.js},
        }
        self.in_vars["parameters"] = ["jord", "ifirst", "ilast"]
        self.out_vars = {
            "flux": {
                "istart": "ifirst",
                "iend": "ilast",
                "jstart": grid.js,
                "jend": grid.je + 1,
            }
        }
        self.grid = grid
        self.stencil_factory = stencil_factory

    def ivars(self, inputs):
        inputs["ifirst"] += TranslateGrid.fpy_model_index_offset
        inputs["ilast"] += TranslateGrid.fpy_model_index_offset
        inputs["ifirst"] = self.grid.global_to_local_x(inputs["ifirst"])
        inputs["ilast"] = self.grid.global_to_local_x(inputs["ilast"])

    def process_inputs(self, inputs):
        self.ivars(inputs)
        self.make_storage_data_input_vars(inputs)
        inputs["flux"] = utils.make_storage_from_shape(
            inputs["q"].shape, backend=self.stencil_factory.backend
        )

    def compute(self, inputs):
        self.process_inputs(inputs)
        origin = self.grid.grid_indexing.origin_compute()
        domain = self.grid.grid_indexing.domain_compute(add=(1, 1, 0))
        self.compute_func = yppm.YPiecewiseParabolic(
            stencil_factory=self.stencil_factory,
            dya=self.grid.dya,
            grid_type=self.grid.grid_type,
            jord=int(inputs["jord"]),
            origin=(inputs["ifirst"], origin[1], origin[2]),
            domain=(inputs["ilast"] - inputs["ifirst"] + 1, domain[1], domain[2]),
        )
        self.compute_func(inputs["q"], inputs["c"], inputs["flux"])
        return self.slice_output(inputs)


class TranslateYPPM_2(TranslateYPPM):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"]["q"]["serialname"] = "q_2"
        self.out_vars["flux"]["serialname"] = "flux_2"
