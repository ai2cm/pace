from .parallel_translate import ParallelTranslate2PyState
import fv3.stencils.fv_dynamics as fv_dynamics
import fv3util


class TranslateFVDynamics_KLoopPostRemap(ParallelTranslate2PyState):
    inputs = {
        "omga": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        }
    }

    def __init__(self, grids):
        super().__init__(grids)
        self._base.compute_func = fv_dynamics.post_remap
        self._base.in_vars["data_vars"] = {
            "delz": {},
            "w": {},
            "delp": {},
            "omga": {},
        }
        self._base.in_vars["parameters"] = []
        self._base.out_vars = {"omga": {}}
