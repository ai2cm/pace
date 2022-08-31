from typing import Any, Dict

import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.util
from pace.fv3core.testing import TranslateDycoreFortranData2Py
from pace.stencils import corners


class TranslateFill4Corners(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {"q4c": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars: Dict[str, Any] = {"q4c": {}}
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        origin = self.grid.full_origin()
        domain = self.grid.domain_shape_full()
        axes_offsets = self.stencil_factory.grid_indexing.axis_offsets(origin, domain)
        if inputs["dir"] == 1:
            stencil = self.stencil_factory.from_origin_domain(
                corners.fill_corners_2cells_x_stencil,
                externals=axes_offsets,
                origin=origin,
                domain=domain,
            )
        elif inputs["dir"] == 2:
            stencil = self.stencil_factory.from_origin_domain(
                corners.fill_corners_2cells_y_stencil,
                externals=axes_offsets,
                origin=origin,
                domain=domain,
            )
        stencil(inputs["q4c"], inputs["q4c"])
        return self.slice_output(inputs, {"q4c": inputs["q4c"]})


class TranslateFillCorners(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {"divg_d": {}, "nord_col": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"divg_d": {"iend": grid.ied + 1, "jend": grid.jed + 1}}
        self.stencil_factory = stencil_factory

    def compute_from_storage(self, inputs):
        nord_column = inputs["nord_col"][:]
        utils.device_sync(backend=self.stencil_factory.backend)
        for nord in utils.unique(nord_column):
            if nord != 0:
                ki = [i for i in range(self.grid.npz) if nord_column[i] == nord]
                origin = (self.grid.isd, self.grid.jsd, ki[0])
                domain = (self.grid.nid + 1, self.grid.njd + 1, len(ki))
                if inputs["dir"] == 1:
                    fill_corners = corners.FillCornersBGrid(
                        "x",
                        origin=origin,
                        domain=domain,
                        stencil_factory=self.stencil_factory,
                    )

                    fill_corners(
                        inputs["divg_d"],
                    )
                elif inputs["dir"] == 2:
                    fill_corners = corners.FillCornersBGrid(
                        "y",
                        origin=origin,
                        domain=domain,
                        stencil_factory=self.stencil_factory,
                    )
                    fill_corners(
                        inputs["divg_d"],
                    )
                else:
                    raise ValueError("Invalid input")
        return inputs


class TranslateCopyCorners(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {"q": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars: Dict[str, Any] = {"q": {}}
        self._copy_corners_x = corners.CopyCorners("x", stencil_factory=stencil_factory)
        self._copy_corners_y = corners.CopyCorners("y", stencil_factory=stencil_factory)
        self.stencil_factory = stencil_factory

    def compute_from_storage(self, inputs):
        if inputs["dir"] == 1:
            self._copy_corners_x(inputs["q"])
        elif inputs["dir"] == 2:
            self._copy_corners_y(inputs["q"])
        else:
            raise ValueError("Invalid input")
        return inputs


class FillCornersVector_Wrapper:
    def __init__(self, stencil_factory, axes_offsets, origin, domain):
        self.stencil = stencil_factory.from_origin_domain(
            corners.fill_corners_dgrid_defn,
            externals=axes_offsets,
            origin=origin,
            domain=domain,
        )

    def __call__(
        self,
        x_in,
        x_out,
        y_in,
        y_out,
        mysign,
    ):
        self.stencil(x_in, x_out, y_in, y_out, mysign)


class TranslateFillCornersVector(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {"vc": {}, "uc": {}, "nord_col": {}}
        self.out_vars: Dict[str, Any] = {
            "vc": grid.y3d_domain_dict(),
            "uc": grid.x3d_domain_dict(),
        }
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        nord_column = inputs["nord_col"][:]
        self.make_storage_data_input_vars(inputs)
        for nord in utils.unique(nord_column):
            if nord != 0:
                ki = [k for k in range(self.grid.npz) if nord_column[0, 0, k] == nord]
                origin = (self.grid.isd, self.grid.jsd, ki[0])
                domain = (self.grid.nid + 1, self.grid.njd + 1, len(ki))
                axes_offsets = self.stencil_factory.grid_indexing.axis_offsets(
                    origin, domain
                )
                vector_corner_fill = FillCornersVector_Wrapper(
                    self.stencil_factory, axes_offsets, origin, domain
                )
                vector_corner_fill(
                    inputs["vc"], inputs["vc"], inputs["uc"], inputs["uc"], -1.0
                )
        return self.slice_output(inputs)
