from fv3core.stencils.del2cubed import HyperdiffusionDamping
from fv3core.testing import TranslateFortranData2Py


class TranslateDel2Cubed(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"qdel": {}}
        self.in_vars["parameters"] = ["nmax", "cd"]
        self.out_vars = {"qdel": {}}

    def compute_from_storage(self, inputs):
        hyperdiffusion = HyperdiffusionDamping(self.grid, inputs.pop("nmax"))
        hyperdiffusion(**inputs)
        return inputs
