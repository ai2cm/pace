from .translate import TranslateFortranData2Py
from ..utils.corners import copy_corners


class TranslateCopyCorners(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars['data_vars'] = {'q': {}}
        self.in_vars['parameters'] = ['dir']
        self.out_vars = {'q': {}}

    def compute(self, inputs):
        if inputs['dir'] == 1:
            direction = 'x'
        if inputs['dir'] == 2:
            direction = 'y'
        copy_corners(inputs['q'], direction, self.grid)
        return {'q': inputs['q']}
