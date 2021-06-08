import fv3core._config as spec
import fv3core.stencils.mapn_tracer as MapN_Tracer
from fv3core.testing import TranslateFortranData2Py, TranslateGrid, pad_field_in_j


class TranslateMapN_Tracer_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
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
            pad_field_in_j(inputs["pe1"], self.grid.njd)
        )
        inputs["pe2"] = self.make_storage_data(
            pad_field_in_j(inputs["pe2"], self.grid.njd)
        )
        inputs["dp2"] = self.make_storage_data(
            pad_field_in_j(inputs["dp2"], self.grid.njd)
        )
        inputs["kord"] = abs(spec.namelist.kord_tr)
        self.compute_func = MapN_Tracer.MapNTracer(
            inputs.pop("kord"),
            inputs.pop("nq"),
            inputs.pop("i1"),
            inputs.pop("i2"),
            inputs.pop("j1"),
            inputs.pop("j2"),
        )
        self.compute_func(**inputs)
        return self.slice_output(inputs)
