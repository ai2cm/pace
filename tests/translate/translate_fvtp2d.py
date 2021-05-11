import fv3core._config as spec
import fv3core.utils.gt4py_utils as utils
from fv3core.stencils.fvtp2d import FiniteVolumeTransport
from fv3core.testing import TranslateFortranData2Py


class TranslateFvTp2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "q": {},
            "mass": {},
            "damp_c": {},
            "nord": {"serialname": "nord_column"},
            "crx": {"istart": grid.is_},
            "cry": {"jstart": grid.js},
            "xfx": {"istart": grid.is_},
            "yfx": {"jstart": grid.js},
            "mfx": grid.x3d_compute_dict(),
            "mfy": grid.y3d_compute_dict(),
        }
        # 'fx': grid.x3d_compute_dict(),'fy': grid.y3d_compute_dict(),
        self.in_vars["parameters"] = ["hord"]
        self.out_vars = {
            "q": {},
            "fx": grid.x3d_compute_dict(),
            "fy": grid.y3d_compute_dict(),
        }

    # use_sg -- 'dx', 'dy', 'rdxc', 'rdyc', 'sin_sg needed
    def compute_from_storage(self, inputs):
        inputs["fx"] = utils.make_storage_from_shape(
            self.maxshape, self.grid.full_origin()
        )
        inputs["fy"] = utils.make_storage_from_shape(
            self.maxshape, self.grid.full_origin()
        )
        for optional_arg in ["mass", "mfx", "mfy"]:
            if optional_arg not in inputs:
                inputs[optional_arg] = None
        self.compute_func = utils.cached_stencil_class(FiniteVolumeTransport)(
            spec.namelist,
            int(inputs["hord"]),
            inputs.pop("nord"),
            inputs.pop("damp_c"),
            cache_key="regression-test",
        )
        del inputs["hord"]
        inputs["x_area_flux"] = inputs.pop("xfx")
        inputs["y_area_flux"] = inputs.pop("yfx")
        self.compute_func(**inputs)
        return inputs


class TranslateFvTp2d_2(TranslateFvTp2d):
    def __init__(self, grid):
        super().__init__(grid)
        del self.in_vars["data_vars"]["mass"]
        del self.in_vars["data_vars"]["mfx"]
        del self.in_vars["data_vars"]["mfy"]
