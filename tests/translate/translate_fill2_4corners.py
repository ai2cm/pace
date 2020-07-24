from .translate import TranslateFortranData2Py
from fv3core.utils.corners import fill2_4corners


class TranslateFill2_4Corners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q1c": {}, "q2c": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q1c": {}, "q2c": {}}

    def compute(self, inputs):

        if inputs["dir"] == 1:
            direction = "x"
        if inputs["dir"] == 2:
            direction = "y"
        fill2_4corners(inputs["q1c"], inputs["q2c"], direction, self.grid)

        return {"q1c": inputs["q1c"], "q2c": inputs["q2c"]}
