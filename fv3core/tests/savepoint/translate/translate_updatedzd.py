import numpy as np

import pace.dsl
import pace.fv3core
import pace.fv3core.stencils.updatedzd
import pace.util
from pace.fv3core.stencils import d_sw
from pace.fv3core.testing import TranslateDycoreFortranData2Py
from pace.fv3core.utils.functional_validation import get_subset_func


class TranslateUpdateDzD(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
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
        self.namelist = pace.fv3core.DynamicalCoreConfig.from_namelist(namelist)
        self._subset = get_subset_func(
            self.grid.grid_indexing,
            dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
            n_halo=((0, 0), (0, 0)),
        )
        self.ignore_near_zero_errors = {"zh": True, "wsd": True}
        self.near_zero = 1e-30

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        self.updatedzd = pace.fv3core.stencils.updatedzd.UpdateHeightOnDGrid(
            self.stencil_factory,
            self.grid.quantity_factory,
            self.grid.damping_coefficients,
            self.grid.grid_data,
            self.grid.grid_type,
            self.namelist.hord_tm,
            column_namelist=d_sw.get_column_namelist(
                self.namelist, quantity_factory=self.grid.quantity_factory
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
        if varname in ["zh", "height"]:
            return self._subset(output)
        else:
            return output
