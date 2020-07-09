from .parallel_translate import ParallelTranslate
import fv3.stencils.tracer_2d_1l as tracer_2d_1l
import fv3.utils.gt4py_utils as utils
import fv3util
import pytest


class TranslateTracer2D1L(ParallelTranslate):
    inputs = {
        "tracers": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "kg/m^2",
        }
    }

    def __init__(self, grids):
        super().__init__(grids)
        self._base.compute_func = tracer_2d_1l.compute
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "tracers": {},
            "dp1": {},
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
        }
        self._base.in_vars["parameters"] = ["nq", "mdt"]
        self._base.out_vars = self._base.in_vars["data_vars"]

    def collect_input_data(self, serializer, savepoint):
        input_data = self._base.collect_input_data(serializer, savepoint)
        return input_data

    def compute_parallel(self, inputs, communicator):
        inputs["comm"] = communicator

        self._base.make_storage_data_input_vars(inputs)
        properties = self.inputs["tracers"]
        for name in utils.tracer_variables:
            self.grid.quantity_dict_update(
                inputs["tracers"],
                name,
                dims=properties["dims"],
                units=properties["units"],
            )
        self._base.compute_func(**inputs)
        for name in utils.tracer_variables:
            del inputs["tracers"][name + "_quantity"]
        return self._base.slice_output(inputs)

    def compute_sequential(self, a, b):
        pytest.skip(
            f"{self.__class__} only has a mpirun implementation, not running in mock-parallel"
        )
