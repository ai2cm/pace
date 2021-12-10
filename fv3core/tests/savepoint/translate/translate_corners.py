import pace.dsl.gt4py_utils as utils
from fv3core.testing import TranslateFortranData2Py
from fv3core.utils import corners
from fv3core.utils.grid import axis_offsets


class TranslateFill4Corners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q4c": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q4c": {}}

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        origin = self.grid.full_origin()
        domain = self.grid.domain_shape_full()
        axes_offsets = axis_offsets(self.grid, origin, domain)
        if inputs["dir"] == 1:
            stencil = self.grid.stencil_factory.from_origin_domain(
                corners.fill_corners_2cells_x_stencil,
                externals=axes_offsets,
                origin=origin,
                domain=domain,
            )
        elif inputs["dir"] == 2:
            stencil = self.grid.stencil_factory.from_origin_domain(
                corners.fill_corners_2cells_y_stencil,
                externals=axes_offsets,
                origin=origin,
                domain=domain,
            )
        stencil(inputs["q4c"], inputs["q4c"])
        return self.slice_output(inputs, {"q4c": inputs["q4c"]})


class TranslateFillCorners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"divg_d": {}, "nord_col": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"divg_d": {"iend": grid.ied + 1, "jend": grid.jed + 1}}

    def compute_from_storage(self, inputs):
        nord_column = inputs["nord_col"][:]
        utils.device_sync(backend=self.grid.stencil_factory.backend)
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
                        stencil_factory=self.grid.stencil_factory,
                    )

                    fill_corners(
                        inputs["divg_d"],
                    )
                elif inputs["dir"] == 2:
                    fill_corners = corners.FillCornersBGrid(
                        "y",
                        origin=origin,
                        domain=domain,
                        stencil_factory=self.grid.stencil_factory,
                    )
                    fill_corners(
                        inputs["divg_d"],
                    )
                else:
                    raise ValueError("Invalid input")
        return inputs


class TranslateCopyCorners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q": {}}
        self._copy_corners_x = corners.CopyCorners(
            "x", stencil_factory=self.grid.stencil_factory
        )
        self._copy_corners_y = corners.CopyCorners(
            "y", stencil_factory=self.grid.stencil_factory
        )

    def compute_from_storage(self, inputs):
        if inputs["dir"] == 1:
            self._copy_corners_x(inputs["q"])
        elif inputs["dir"] == 2:
            self._copy_corners_y(inputs["q"])
        else:
            raise ValueError("Invalid input")
        return inputs


class TranslateFillCornersVector(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"vc": {}, "uc": {}, "nord_col": {}}
        self.out_vars = {"vc": grid.y3d_domain_dict(), "uc": grid.x3d_domain_dict()}

    def compute(self, inputs):
        nord_column = inputs["nord_col"][:]
        self.make_storage_data_input_vars(inputs)
        for nord in utils.unique(nord_column):
            if nord != 0:
                ki = [k for k in range(self.grid.npz) if nord_column[0, 0, k] == nord]
                origin = (self.grid.isd, self.grid.jsd, ki[0])
                domain = (self.grid.nid + 1, self.grid.njd + 1, len(ki))
                axes_offsets = axis_offsets(self.grid.grid_indexing, origin, domain)
                vector_corner_fill = self.grid.stencil_factory.from_origin_domain(
                    corners.fill_corners_dgrid_defn,
                    externals=axes_offsets,
                    origin=origin,
                    domain=domain,
                )
                vector_corner_fill(
                    inputs["vc"], inputs["vc"], inputs["uc"], inputs["uc"], -1.0
                )
        return self.slice_output(inputs)
