from fv3.translate.translate import TranslateFortranData2Py
import fv3.stencils.fvtp2d as fvtp2d
from fv3._config import grid
import numpy as np
import fv3.utils.gt4py_utils as utils
import fv3.stencils.d_sw as d_sw
from .translate_d_sw import TranslateD_SW


class TranslateFvTp2d(TranslateD_SW):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars['data_vars'] = {'q': {}, 'mass': {}, 
                                     'damp_c':{}, 'nord_column':{},
                                     'crx': {'istart': grid.is_}, 'cry': {'jstart': grid.js},
                                     'xfx': {'istart': grid.is_}, 'yfx': {'jstart': grid.js},
                                     'ra_x': {'istart': grid.is_}, 'ra_y': {'jstart': grid.js},
                                     'mfx': grid.x3d_compute_dict(), 'mfy': grid.y3d_compute_dict()}
        # 'fx': grid.x3d_compute_dict(),'fy': grid.y3d_compute_dict(),
        self.in_vars['parameters'] = ['hord']
        self.out_vars = {'q': {}, 'fx': grid.x3d_compute_dict(), 'fy': grid.y3d_compute_dict()}

    # use_sg -- 'dx', 'dy', 'rdxc', 'rdyc', 'sin_sg needed
    def compute(self, inputs):
        
        #column_info = {'nord': self.column_namelist_vals('nord_column', inputs), 'damp_c':self.column_namelist_vals('damp_c', inputs)}
        #column_info['nord'] = [int(i) for i in column_info['nord']]
        #self.make_storage_data_input_vars(inputs)
        #del inputs['nord_column']
        #del inputs['damp_c']
        #d_sw.d_sw_ksplit(fvtp2d.compute_no_sg, inputs, column_info, list(self.out_vars.keys()), self.grid)
        ##fvtp2d.compute(inputs, nord_column)
        #return self.slice_output(inputs)
        for optional_arg in ['mass', 'mfx', 'mfy']:
            if optional_arg not in inputs:
                inputs[optional_arg] = None
        return self.nord_column_split_compute(inputs, fvtp2d.compute_no_sg)
class TranslateFvTp2d_2(TranslateFvTp2d):
    def __init__(self, grid):
        super().__init__(grid)
        del self.in_vars['data_vars']['mass']
        del self.in_vars['data_vars']['mfx']
        del self.in_vars['data_vars']['mfy']
