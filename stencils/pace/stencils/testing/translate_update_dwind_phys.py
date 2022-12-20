import numpy as np

import pace.util
from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py
from pace.stencils.update_dwind_phys import AGrid2DGridPhysics
from pace.util.utils import safe_assign_array


class TranslateUpdateDWindsPhys(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "u": {"dwind": True},
            "u_dt": {"dwind": True},
            "v": {"dwind": True},
            "v_dt": {"dwind": True},
        }
        self.out_vars = {
            "u": {"dwind": True, "kend": namelist.npz - 1},
            "v": {"dwind": True, "kend": namelist.npz - 1},
        }
        self.namelist = namelist
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        partitioner = pace.util.TilePartitioner(self.namelist.layout)
        self.compute_func = AGrid2DGridPhysics(
            self.stencil_factory,
            self.grid.quantity_factory,
            partitioner,
            self.grid.rank,
            self.namelist,
            grid_info=self.grid.driver_grid_data,
        )
        self.compute_func(**inputs)
        out = {}
        # This alloc then copy pattern is requried to deal transparently with
        # arrays on different device
        out["u"] = np.empty_like(inputs["u"][self.grid.y3d_domain_interface()])
        out["v"] = np.empty_like(inputs["v"][self.grid.x3d_domain_interface()])
        safe_assign_array(out["u"], inputs["u"][self.grid.y3d_domain_interface()])
        safe_assign_array(out["v"], inputs["v"][self.grid.x3d_domain_interface()])
        return out
