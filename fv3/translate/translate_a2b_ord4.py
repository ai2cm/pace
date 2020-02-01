from .translate import TranslateFortranData2Py
import fv3.stencils.divergence_damping as dd
import fv3.utils.gt4py_utils as utils
from .translate_d_sw import TranslateD_SW

class TranslateA2B_Ord4(TranslateD_SW):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars['data_vars'] = {'wk': {}, 'vort': {}, 'delpc': {}, 'nord_col': {}}
        self.in_vars['parameters'] = ['dt']
        self.out_vars = {'wk': {}, 'vort': {}}

    def compute(self, inputs):
        return self.column_split_compute(inputs,  dd.vorticity_calc, {'nord': 'nord_col'})
