from fv3core.stencils import pe_halo
from pace.stencils.testing import TranslateDycoreFortranData2Py


class TranslatePE_Halo(TranslateDycoreFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):

        super().__init__(grid, namelist, stencil_factory)
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
        self.stencil_factory = stencil_factory
        ax_offsets_pe = self.stencil_factory.grid_indexing.axis_offsets(
            self.stencil_factory.grid_indexing.origin_full(),
            self.stencil_factory.grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self.compute_func = stencil_factory.from_origin_domain(
            pe_halo.edge_pe,
            origin=self.stencil_factory.grid_indexing.origin_full(),
            domain=self.stencil_factory.grid_indexing.domain_full(add=(0, 0, 1)),
            externals={**ax_offsets_pe},
            skip_passes=("PruneKCacheFills",),
        )
