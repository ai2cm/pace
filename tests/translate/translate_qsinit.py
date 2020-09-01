from .translate import TranslateFortranData2Py
import fv3core.stencils.saturation_adjustment as satadjust
import fv3core.utils.gt4py_utils as utils
import numpy as np


class TranslateQSInit(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = satadjust.compute_q_tables
        self.in_vars["data_vars"] = {
            "table": {},
            "table2": {},
            "tablew": {},
            "des2": {},
            "desw": {},
        }
        self.out_vars = self.in_vars["data_vars"]
        self.maxshape = (1, 1, satadjust.QS_LENGTH)

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["index"] = utils.make_storage_data_from_1d(
            np.arange(satadjust.QS_LENGTH), self.maxshape, origin=(0, 0, 0)
        )
        kwargs = {"origin": (0, 0, 0), "domain": self.maxshape}
        self.compute_func(**inputs, **kwargs)
        for k, v in inputs.items():
            if v.shape == self.maxshape:
                inputs[k] = np.squeeze(v)
        return inputs
