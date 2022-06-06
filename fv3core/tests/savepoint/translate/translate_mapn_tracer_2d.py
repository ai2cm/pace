import fv3core.stencils.mapn_tracer as MapN_Tracer
import pace.dsl
import pace.util
from pace.stencils.testing import (
    TranslateDycoreFortranData2Py,
    TranslateGrid,
    pad_field_in_j,
)


class TranslateMapN_Tracer_2d(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "pe1": {"istart": grid.is_, "iend": grid.ie - 2, "axis": 1},
            "pe2": {"istart": grid.is_, "iend": grid.ie - 2, "axis": 1},
            "dp2": {"istart": grid.is_, "iend": grid.ie - 2, "axis": 1},
            "tracers": {"serialname": "qtracers"},
        }
        self.in_vars["parameters"] = ["j_2d", "nq", "q_min"]
        self.out_vars = {"tracers": {"serialname": "qtracers"}}

        self.is_ = grid.is_
        self.ie = grid.ie
        self.max_error = 3.5e-11
        self.near_zero = 7e-17
        self.ignore_near_zero_errors["qtracers"] = True
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    def compute(self, inputs):
        self.setup(inputs)
        inputs["j_2d"] = self.grid.global_to_local_y(
            inputs["j_2d"] + TranslateGrid.fpy_model_index_offset
        )
        inputs["i1"] = self.is_
        inputs["i2"] = self.ie
        inputs["j1"] = inputs["j_2d"]
        inputs["j2"] = inputs["j_2d"]
        del inputs["j_2d"]
        inputs["pe1"] = self.make_storage_data(
            pad_field_in_j(
                inputs["pe1"], self.grid.njd, backend=self.stencil_factory.backend
            )
        )
        inputs["pe2"] = self.make_storage_data(
            pad_field_in_j(
                inputs["pe2"], self.grid.njd, backend=self.stencil_factory.backend
            )
        )
        inputs["dp2"] = self.make_storage_data(
            pad_field_in_j(
                inputs["dp2"], self.grid.njd, backend=self.stencil_factory.backend
            )
        )
        inputs["kord"] = abs(self.namelist.kord_tr)
        self.compute_func = MapN_Tracer.MapNTracer(
            self.stencil_factory,
            inputs.pop("kord"),
            inputs.pop("nq"),
            inputs.pop("i1"),
            inputs.pop("i2"),
            inputs.pop("j1"),
            inputs.pop("j2"),
            fill=self.namelist.fill,
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)
