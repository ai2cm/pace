from typing import Any, Dict

import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.util
from pace.fv3core.stencils.neg_adj3 import AdjustNegativeTracerMixingRatio
from pace.fv3core.testing import TranslateDycoreFortranData2Py


class TranslateNeg_Adj3(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            "pt": {},
            "delp": {},
            "delz": {},
            "peln": {"istart": grid.is_, "jstart": grid.js, "kaxis": 1},
        }
        self.in_vars["parameters"] = []
        self.out_vars: Dict[str, Any] = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "qcld": {},
            # "pt": {},
        }
        for qvar in utils.tracer_variables:
            self.ignore_near_zero_errors[qvar] = True
        self.stencil_factory = stencil_factory
        self.namelist = namelist  # type: ignore

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        compute_fn = AdjustNegativeTracerMixingRatio(
            self.stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            check_negative=self.namelist.check_negative,
            hydrostatic=self.namelist.hydrostatic,
        )
        compute_fn(
            inputs["qvapor"],
            inputs["qliquid"],
            inputs["qrain"],
            inputs["qsnow"],
            inputs["qice"],
            inputs["qgraupel"],
            inputs["qcld"],
            inputs["pt"],
            inputs["delp"],
        )
        return self.slice_output(inputs)
