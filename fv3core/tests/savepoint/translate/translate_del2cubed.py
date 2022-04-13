from fv3core.stencils.del2cubed import HyperdiffusionDamping
from pace.stencils.testing import TranslateDycoreFortranData2Py


class TranslateDel2Cubed(TranslateDycoreFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {"qdel": {}}
        self.in_vars["parameters"] = ["nmax", "cd"]
        self.out_vars = {"qdel": {}}
        self.stencil_factory = stencil_factory

    def compute_from_storage(self, inputs):
        hyperdiffusion = HyperdiffusionDamping(
            self.stencil_factory,
            self.grid.damping_coefficients,
            self.grid.rarea,
            inputs.pop("nmax"),
        )
        hyperdiffusion(**inputs)
        return inputs
