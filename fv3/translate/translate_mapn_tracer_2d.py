from .translate import TranslateFortranData2Py, TranslateGrid
import fv3.stencils.mapn_tracer as MapN_Tracer
import fv3._config as spec


class TranslateMapN_Tracer_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = MapN_Tracer.compute
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
        self.max_error = 1e-13

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["j_2d"] = self.grid.global_to_local_y(
            inputs["j_2d"] + TranslateGrid.fpy_model_index_offset
        )
        inputs["i1"] = self.is_
        inputs["i2"] = self.ie
        inputs["kord"] = abs(spec.namelist["kord_tr"])
        self.compute_func(**inputs)
        return self.slice_output(inputs,)
