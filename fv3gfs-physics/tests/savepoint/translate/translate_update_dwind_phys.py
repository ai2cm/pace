import numpy as np

import fv3core._config as spec
import pace.util
from fv3gfs.physics.stencils.update_dwind_phys import AGrid2DGridPhysics
from fv3gfs.physics.testing import TranslatePhysicsFortranData2Py


class TranslateUpdateDWindsPhys(TranslatePhysicsFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)

        self.in_vars["data_vars"] = {
            "edge_vect_e": {"dwind": True},
            "edge_vect_n": {"dwind": True, "axis": 0},
            "edge_vect_s": {"dwind": True, "axis": 0},
            "edge_vect_w": {"dwind": True},
            "u": {"dwind": True},
            "u_dt": {"dwind": True},
            "v": {"dwind": True},
            "v_dt": {"dwind": True},
            "vlat": {"dwind": True},
            "vlon": {"dwind": True},
            "es": {"dwind": True},
            "ew": {"dwind": True},
        }
        self.out_vars = {
            "u": {"dwind": True, "kend": grid.npz - 1},
            "v": {"dwind": True, "kend": grid.npz - 1},
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        del inputs["vlat"]
        del inputs["vlon"]
        del inputs["es"]
        del inputs["ew"]
        del inputs["es1_2"]
        del inputs["es2_2"]
        del inputs["es3_2"]
        del inputs["ew1_1"]
        del inputs["ew2_1"]
        del inputs["ew3_1"]
        grid_names = [
            "vlon1",
            "vlon2",
            "vlon3",
            "vlat1",
            "vlat2",
            "vlat3",
            "edge_vect_w",
            "edge_vect_e",
            "edge_vect_s",
            "edge_vect_n",
            "es1_1",
            "es2_1",
            "es3_1",
            "ew1_2",
            "ew2_2",
            "ew3_2",
        ]
        grid_info = {}
        for var in grid_names:
            grid_info[var] = inputs.pop(var)
        partitioner = pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(spec.namelist.layout)
        )
        self.compute_func = AGrid2DGridPhysics(
            self.grid.stencil_factory,
            partitioner,
            self.grid.rank,
            spec.namelist,
            grid_info,
        )
        self.compute_func(**inputs)
        out = {}
        out["u"] = np.asarray(inputs["u"])[self.grid.y3d_domain_interface()]
        out["v"] = np.asarray(inputs["v"])[self.grid.x3d_domain_interface()]
        return out
