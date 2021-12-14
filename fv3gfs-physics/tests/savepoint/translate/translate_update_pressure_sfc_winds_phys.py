from fv3gfs.physics.stencils.fv_update_phys import update_pressure_and_surface_winds
from fv3gfs.physics.testing import TranslatePhysicsFortranData2Py
from pace.util.constants import KAPPA


class TranslatePhysUpdatePressureSurfaceWinds(TranslatePhysicsFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "peln": {"dycore": True, "istart": grid.is_, "jstart": grid.js, "kaxis": 1},
            "pk": {
                "dycore": True,
            },
            "delp": {
                "dycore": True,
            },
            "pe": {
                "dycore": True,
                "istart": grid.is_ - 1,
                "jstart": grid.js - 1,
                "kaxis": 1,
            },
            "ps": {"dycore": True},
            "ua": {
                "dycore": True,
            },
            "va": {
                "dycore": True,
            },
            "u_srf": {
                "dycore": True,
            },
            "v_srf": {
                "dycore": True,
            },
        }

        self.out_vars = {
            "pk": self.in_vars["data_vars"]["pk"],
            "ps": {"compute": False},
            "u_srf": {},
            "v_srf": {},
        }
        origin = self.grid.grid_indexing.origin_compute()
        domain = self.grid.grid_indexing.domain_compute(add=(0, 0, 1))
        self.compute_func = grid.stencil_factory.from_origin_domain(
            update_pressure_and_surface_winds, origin=origin, domain=domain
        )

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["KAPPA"] = KAPPA
        self.compute_func(**inputs)
        out = self.slice_output(inputs)
        return out
