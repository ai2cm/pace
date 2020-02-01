from fv3.translate.translate import TranslateFortranData2Py
import fv3.stencils.delnflux as delnflux
from .translate_d_sw import TranslateD_SW

class TranslateDelnFlux(TranslateD_SW):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars['data_vars'] = {'q': {}, 'fx': grid.x3d_compute_dict(), 'fy': grid.y3d_compute_dict(),
                                     'damp_c': {}, 'nord_column': {}, 'mass': {}}
        self.in_vars['parameters'] = []
        self.out_vars = {'fx': grid.x3d_compute_dict(), 'fy': grid.y3d_compute_dict()}

    # use_sg -- 'dx', 'dy', 'rdxc', 'rdyc', 'sin_sg needed
    def compute(self, inputs):
        '''
        column_info = {'nord': self.column_namelist_vals('nord_column', inputs), 'damp_c':self.column_namelist_vals('damp_c', inputs)}
        column_info['nord'] = [int(i) for i in column_info['nord']]
        self.make_storage_data_input_vars(inputs)
        del inputs['nord_column']
        del inputs['damp_c']
        if 'mass' not in inputs:
            inputs['mass'] = None
            
        d_sw.d_sw_ksplit(delnflux.compute_delnflux_no_sg, inputs, column_info, list(self.out_vars.keys()), self.grid)
        #delnflux.compute_delnflux(inputs, column_info)
        return self.slice_output(inputs)
        '''
        if 'mass' not in inputs:
            inputs['mass'] = None
        return self.nord_column_split_compute(inputs, delnflux.compute_delnflux_no_sg)

class TranslateDelnFlux_2(TranslateDelnFlux):
    def __init__(self, grid):
        super().__init__(grid)
        del self.in_vars['data_vars']['mass']
