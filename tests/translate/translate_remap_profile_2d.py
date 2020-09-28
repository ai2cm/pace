import numpy as np

import fv3core.stencils.remap_profile as Profile

from .translate import TranslateFortranData2Py


class TranslateCS_Profile_2d(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = Profile.compute
        self.in_vars["data_vars"] = {
            "qs": {"serialname": "qs_column"},
            "a4_1": {"serialname": "q4_1"},
            "a4_2": {"serialname": "q4_2"},
            "a4_3": {"serialname": "q4_3"},
            "a4_4": {"serialname": "q4_4"},
            "delp": {"serialname": "dp1_2d"},
        }
        self.in_vars["parameters"] = ["km", "i1", "i2", "iv", "kord"]
        self.out_vars = {
            "a4_1": {"serialname": "q4_1", "istart": 0, "iend": grid.ie - 2},
            "a4_2": {"serialname": "q4_2", "istart": 0, "iend": grid.ie - 2},
            "a4_3": {"serialname": "q4_3", "istart": 0, "iend": grid.ie - 2},
            "a4_4": {"serialname": "q4_4", "istart": 0, "iend": grid.ie - 2},
        }

    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        if storage_vars is None:
            storage_vars = self.storage_vars()
        for p in self.in_vars["parameters"]:
            if type(inputs[p]) in [np.int64, np.int32]:
                inputs[p] = int(inputs[p])
        for d, info in storage_vars.items():
            serialname = info["serialname"] if "serialname" in info else d
            self.update_info(info, inputs)
            istart, jstart, kstart = self.collect_start_indices(
                inputs[serialname].shape, info
            )

            shapes = np.squeeze(inputs[serialname]).shape
            if len(shapes) == 2:
                # suppress j
                dummy_axes = [1]
            elif len(shapes) == 1:
                # suppress j and k
                dummy_axes = [1, 2]
            else:
                dummy_axes = None

            inputs[d] = self.make_storage_data(
                np.squeeze(inputs[serialname]),
                istart=istart,
                jstart=jstart,
                kstart=kstart,
                dummy_axes=dummy_axes,
            )
            if d != serialname:
                del inputs[serialname]

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        inputs["i1"] = self.grid.global_to_local_x(inputs["i1"] - 1)
        inputs["i2"] = self.grid.global_to_local_x(inputs["i2"] - 1)
        inputs["jslice"] = slice(0, 1)

        q4_1, q4_2, q4_3, q4_4 = self.compute_func(**inputs)
        return self.slice_output(
            inputs, {"q4_1": q4_1, "q4_2": q4_2, "q4_3": q4_3, "q4_4": q4_4}
        )


class TranslateCS_Profile_2d_2(TranslateCS_Profile_2d):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = Profile.compute
        self.in_vars["data_vars"] = {
            "qs": {"serialname": "qs_column_2"},
            "a4_1": {"serialname": "q4_1_2"},
            "a4_2": {"serialname": "q4_2_2"},
            "a4_3": {"serialname": "q4_3_2"},
            "a4_4": {"serialname": "q4_4_2"},
            "delp": {"serialname": "dp1_2d_2"},
        }
        self.in_vars["parameters"] = ["km", "i1", "i2", "iv", "kord"]
        self.out_vars = {
            "a4_1": {"serialname": "q4_1_2", "istart": 0, "iend": grid.ie - 3},
            "a4_2": {"serialname": "q4_2_2", "istart": 0, "iend": grid.ie - 3},
            "a4_3": {"serialname": "q4_3_2", "istart": 0, "iend": grid.ie - 3},
            "a4_4": {"serialname": "q4_4_2", "istart": 0, "iend": grid.ie - 3},
        }
