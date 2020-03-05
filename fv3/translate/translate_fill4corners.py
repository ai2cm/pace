from .translate import TranslateFortranData2Py
from ..utils.corners import fill_4corners


class TranslateFill4Corners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {"q4c": {}}
        self.in_vars["parameters"] = ["dir"]
        self.out_vars = {"q4c": {}}

    def compute(self, inputs):

        if inputs["dir"] == 1:
            direction = "x"
        if inputs["dir"] == 2:
            direction = "y"
        fill_4corners(inputs["q4c"], direction, self.grid)

        return {"q4c": inputs["q4c"]}
