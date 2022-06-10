import numpy as np

import fv3core.stencils.updatedzd
import pace.dsl
import pace.util
from fv3core.stencils import d_sw
from pace.stencils.testing import TranslateDycoreFortranData2Py


class TranslateUpdateDzD(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "dp0": {},  # column var
            "surface_height": {"serialname": "zs"},
            "height": {"kend": grid.npz + 1},
            "courant_number_x": grid.x3d_compute_domain_y_dict(),
            "courant_number_y": grid.y3d_compute_domain_x_dict(),
            "x_area_flux": grid.x3d_compute_domain_y_dict(),
            "y_area_flux": grid.y3d_compute_domain_x_dict(),
            "ws": grid.compute_dict(),
        }
        self.in_vars["data_vars"]["courant_number_x"]["serialname"] = "crx"
        self.in_vars["data_vars"]["courant_number_y"]["serialname"] = "cry"
        self.in_vars["data_vars"]["x_area_flux"]["serialname"] = "xfx"
        self.in_vars["data_vars"]["y_area_flux"]["serialname"] = "yfx"
        self.in_vars["data_vars"]["y_area_flux"]["serialname"] = "yfx"
        self.in_vars["data_vars"]["height"]["serialname"] = "zh"
        self.in_vars["data_vars"]["ws"]["serialname"] = "wsd"

        self.in_vars["parameters"] = ["dt"]
        out_vars = [
            "height",
            "courant_number_x",
            "courant_number_y",
            "x_area_flux",
            "y_area_flux",
            "ws",
        ]
        self.out_vars = {}
        for v in out_vars:
            self.out_vars[v] = self.in_vars["data_vars"][v]
        self.out_vars["ws"]["kstart"] = grid.npz
        self.out_vars["ws"]["kend"] = None
        self.stencil_factory = stencil_factory
        self.namelist = namelist

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.updatedzd = fv3core.stencils.updatedzd.UpdateHeightOnDGrid(
            self.stencil_factory,
            self.grid.damping_coefficients,
            self.grid.grid_data,
            self.grid.grid_type,
            self.namelist.hord_tm,
            inputs.pop("dp0"),
            d_sw.get_column_namelist(
                self.namelist, self.grid.npz, backend=self.stencil_factory.backend
            ),
        )
        self.updatedzd(**inputs)
        outputs = self.slice_output(inputs)
        outputs["zh"] = self.subset_output("zh", outputs["zh"])
        return outputs

    def subset_output(self, varname: str, output: np.ndarray) -> np.ndarray:
        """
        Given an output array, return the slice of the array which we'd
        like to validate against reference data
        """
        return self.updatedzd.subset_output(varname, output)
