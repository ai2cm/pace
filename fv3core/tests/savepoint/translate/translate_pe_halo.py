import pace.dsl
import pace.util
from pace.fv3core.stencils import pe_halo
from pace.fv3core.testing import TranslateDycoreFortranData2Py


class PE_Halo_Wrapper:
    def __init__(self, stencil_factory) -> None:
        ax_offsets_pe = stencil_factory.grid_indexing.axis_offsets(
            stencil_factory.grid_indexing.origin_full(),
            stencil_factory.grid_indexing.domain_full(add=(0, 0, 1)),
        )
        self._stencil = stencil_factory.from_origin_domain(
            pe_halo.edge_pe,
            origin=stencil_factory.grid_indexing.origin_full(),
            domain=stencil_factory.grid_indexing.domain_full(add=(0, 0, 1)),
            externals={**ax_offsets_pe},
            skip_passes=("PruneKCacheFills",),
        )

    def __call__(self, pe, delp, ptop):
        self._stencil(pe, delp, ptop)


class TranslatePE_Halo(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):

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
        stencil_class = PE_Halo_Wrapper(self.stencil_factory)
        self.compute_func = stencil_class  # type: ignore
