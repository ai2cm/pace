from typing import Any, Dict

import numpy as np

import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.util
from pace.fv3core.testing import MapSingleFactory, TranslateDycoreFortranData2Py
from pace.stencils.testing import TranslateGrid, pad_field_in_j


class TranslateSingleJ(TranslateDycoreFortranData2Py):
    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        if storage_vars is None:
            storage_vars = self.storage_vars()
        for d, info in storage_vars.items():
            serialname = info["serialname"] if "serialname" in info else d
            shapes = np.squeeze(inputs[serialname]).shape
            if len(shapes) == 2:
                # suppress j
                dummy_axes = [1]
            elif len(shapes) == 1:
                # suppress j and k
                dummy_axes = [1, 2]
            else:
                dummy_axes = None
            info["dummy_axes"] = dummy_axes
        super().make_storage_data_input_vars(inputs, storage_vars)


class TranslateMap1_PPM_2d(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.compute_func = MapSingleFactory(  # type: ignore
            stencil_factory, grid.quantity_factory
        )
        self.in_vars["data_vars"] = {
            "q1": {"serialname": "var_in"},
            "pe1": {"istart": 3, "iend": grid.ie - 2, "axis": 1},
            "pe2": {"istart": 3, "iend": grid.ie - 2, "axis": 1},
            "qs": {"serialname": "ws_1d", "kstart": grid.is_, "axis": 0},
        }
        self.in_vars["parameters"] = ["j_2d", "i1", "i2", "mode", "kord"]
        self.out_vars: Dict[str, Any] = {"var_inout": {}}
        self.max_error = 5e-13
        self.write_vars = ["qs"]
        self.nj = self.maxshape[1]
        self.stencil_factory = stencil_factory

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        if "qs" in inputs:
            qs_field = utils.make_storage_from_shape(
                self.maxshape[0:2],
                origin=(0, 0),
                backend=self.stencil_factory.backend,
            )
            qs_field[:, :] = inputs["qs"][:, :, 0]
            inputs["qs"] = qs_field
            if inputs["qs"].shape[1] == 1:
                inputs["qs"] = utils.tile(inputs["qs"][:, 0], [self.nj, 1]).transpose(
                    1, 0
                )
        inputs["i1"] = self.grid.global_to_local_x(
            inputs["i1"] + TranslateGrid.fpy_model_index_offset
        )
        inputs["i2"] = self.grid.global_to_local_x(
            inputs["i2"] + TranslateGrid.fpy_model_index_offset
        )
        inputs["qmin"] = 0.0
        inputs["j_2d"] = self.grid.global_to_local_y(
            inputs["j_2d"] + TranslateGrid.fpy_model_index_offset
        )
        if inputs["q1"].shape[1] == 1:
            # some test cases are on singleton j-slices,
            # so for those calls these are zero and not n_halo
            inputs["j1"] = 0
            inputs["j2"] = 0
        else:
            inputs["j1"] = inputs["j_2d"]
            inputs["j2"] = inputs["j_2d"]
            if inputs["pe1"].shape[1] == 1:
                inputs["pe1"] = self.make_storage_data(
                    pad_field_in_j(
                        inputs["pe1"],
                        self.nj,
                        backend=self.stencil_factory.backend,
                    )
                )
            if inputs["pe2"].shape[1] == 1:
                inputs["pe2"] = self.make_storage_data(
                    pad_field_in_j(
                        inputs["pe2"],
                        self.nj,
                        backend=self.stencil_factory.backend,
                    )
                )
        del inputs["j_2d"]
        self.compute_func(**inputs)
        return self.slice_output(inputs, {"var_inout": inputs["q1"]})


class TranslateMap1_PPM_2d_3(TranslateMap1_PPM_2d):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"]["pe1"]["serialname"] = "pe1_2"
        self.in_vars["data_vars"]["pe2"]["serialname"] = "pe2_2"
        self.in_vars["data_vars"]["q1"]["serialname"] = "var_in_3"
        self.out_vars = {
            "var_inout": {
                "serialname": "var_inout_3",
                "istart": 0,
                "iend": grid.ied + 1,
            }
        }


class TranslateMap1_PPM_2d_2(TranslateMap1_PPM_2d):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"]["pe1"]["serialname"] = "pe1_2"
        self.in_vars["data_vars"]["pe2"]["serialname"] = "pe2_2"
        self.in_vars["data_vars"]["q1"]["serialname"] = "var_in_2"
        self.out_vars = {
            "var_inout": {
                "serialname": "var_inout_2",
                "jstart": 0,
                "jend": grid.jed + 1,
            }
        }
        self.max_error = 2e-14
