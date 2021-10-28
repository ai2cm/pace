import fv3core._config as spec
from fv3core.stencils import pe_halo
from fv3core.testing import TranslateFortranData2Py
from fv3core.utils.grid import axis_offsets


class TranslatePE_Halo(TranslateFortranData2Py):
    def __init__(self, grid):

        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "delp": {},
        }
        self.in_vars["parameters"] = ["ptop"]
        self.out_vars = {"pe": self.in_vars["data_vars"]["pe"]}
        ax_offsets_pe = axis_offsets(
            self.grid.grid_indexing,
            self.grid.grid_indexing.origin_full(),
            self.grid.grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self.compute_func = spec.grid.stencil_factory.from_origin_domain(
            pe_halo.edge_pe,
            origin=self.grid.grid_indexing.origin_full(),
            domain=self.grid.grid_indexing.domain_full(add=(0, 0, 1)),
            externals={**ax_offsets_pe},
        )
