import pace.dsl.gt4py_utils as utils
from fv3core.stencils.fvtp2d import (
    FiniteVolumeTransport,
    PreAllocatedCopiedCornersFactory,
)
from fv3core.testing import TranslateFortranData2Py
from pace.util import X_DIM, Y_DIM, Z_DIM


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
            "x_area_flux": {"istart": grid.is_, "serialname": "xfx"},
            "y_area_flux": {"jstart": grid.js, "serialname": "yfx"},
            "x_mass_flux": grid.x3d_compute_dict(),
            "y_mass_flux": grid.y3d_compute_dict(),
        }
        self.in_vars["data_vars"]["x_mass_flux"]["serialname"] = "mfx"
        self.in_vars["data_vars"]["y_mass_flux"]["serialname"] = "mfy"
        # 'fx': grid.x3d_compute_dict(),'fy': grid.y3d_compute_dict(),
        self.in_vars["parameters"] = ["hord"]
        self.out_vars = {
            "q": {},
            "q_x_flux": grid.x3d_compute_dict(),
            "q_y_flux": grid.y3d_compute_dict(),
        }
        self.out_vars["q_x_flux"]["serialname"] = "fx"
        self.out_vars["q_y_flux"]["serialname"] = "fy"

    # use_sg -- 'dx', 'dy', 'rdxc', 'rdyc', 'sin_sg needed
    def compute_from_storage(self, inputs):
        inputs["q_x_flux"] = utils.make_storage_from_shape(
            self.maxshape,
            self.grid.full_origin(),
            backend=self.grid.stencil_factory.backend,
        )
        inputs["q_y_flux"] = utils.make_storage_from_shape(
            self.maxshape,
            self.grid.full_origin(),
            backend=self.grid.stencil_factory.backend,
        )
        for optional_arg in ["mass"]:
            if optional_arg not in inputs:
                inputs[optional_arg] = None
        self.compute_func = FiniteVolumeTransport(
            stencil_factory=self.grid.stencil_factory,
            grid_data=self.grid.grid_data,
            damping_coefficients=self.grid.damping_coefficients,
            grid_type=self.grid.grid_type,
            hord=int(inputs["hord"]),
            nord=inputs.pop("nord"),
            damp_c=inputs.pop("damp_c"),
        )
        del inputs["hord"]
        q_storage = inputs["q"]
        factory = PreAllocatedCopiedCornersFactory(
            self.grid.stencil_factory, dims=[X_DIM, Y_DIM, Z_DIM], y_temporary=None
        )
        inputs["q"] = factory(q_storage)
        self.compute_func(**inputs)
        inputs["q"] = q_storage
        return inputs


class TranslateFvTp2d_2(TranslateFvTp2d):
    def __init__(self, grid):
        super().__init__(grid)
        del self.in_vars["data_vars"]["mass"]
        del self.in_vars["data_vars"]["x_mass_flux"]
        del self.in_vars["data_vars"]["y_mass_flux"]
