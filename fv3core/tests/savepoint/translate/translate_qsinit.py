import numpy as np

import fv3core.stencils.saturation_adjustment as satadjust
import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.util
from pace.stencils.testing import TranslateDycoreFortranData2Py


class TranslateQSInit(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "table": {},
            "table2": {},
            "tablew": {},
            "des2": {},
            "desw": {},
        }
        self.out_vars = self.in_vars["data_vars"]
        self.maxshape = (1, 1, satadjust.QS_LENGTH)
        self.write_vars = list(self.in_vars["data_vars"].keys())
        self.stencil_factory = stencil_factory
        self._compute_q_tables_stencil = self.stencil_factory.from_origin_domain(
            satadjust.compute_q_tables, origin=(0, 0, 0), domain=self.maxshape
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        index = np.arange(satadjust.QS_LENGTH)
        inputs["index"] = utils.make_storage_data(
            index,
            self.maxshape,
            origin=(0, 0, 0),
            read_only=False,
            backend=self.stencil_factory.backend,
        )
        self._compute_q_tables_stencil(**inputs)
        utils.device_sync(backend=self.stencil_factory.backend)
        for k, v in inputs.items():
            if v.shape == self.maxshape:
                inputs[k] = np.squeeze(v)
        return inputs
