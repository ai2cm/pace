from fv3.translate.translate import TranslateFortranData2Py
import fv3.stencils.d_sw as d_sw
import fv3.utils.gt4py_utils as utils


class TranslateD_SW(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.max_error = 2e-13 # propagated error from vt roundoff error in FxAdv
        self.in_vars['data_vars'] = {'uc': grid.x3d_domain_dict(), 'vc': grid.y3d_domain_dict(), 'w':{}, 'delpc':{}, 'delp': {},
                                     'u': grid.y3d_domain_dict(), 'v' :  grid.x3d_domain_dict(), 
                                     'xfx': grid.x3d_compute_domain_y_dict(),
                                     'crx': grid.x3d_compute_domain_y_dict(),
                                     'yfx': grid.y3d_compute_domain_x_dict(),
                                     'cry': grid.y3d_compute_domain_x_dict(),
                                     'mfx': grid.x3d_compute_dict(),
                                     'mfy': grid.y3d_compute_dict(),
                                     'cx': grid.x3d_compute_domain_y_dict(),
                                     'cy': grid.y3d_compute_domain_x_dict(),
                                     'heat_source': {},
                                     'diss_est': {},
                                     'q_con': {}, 'pt': {}, 'ptc': {}, 'ua': {}, 'va': {}, 'zh': {},
                                     'divgd': grid.default_dict_buffer_2d()
        }
        for name, info in self.in_vars['data_vars'].items():
            info['serialname'] = name + 'd'
        self.in_vars['parameters'] = ['dt']
        self.out_vars = self.in_vars['data_vars'].copy()
        del self.out_vars['zh']

       
        #self.out_vars['damp_vt'] = {}
        #self.out_vars['nord_v'] = {}
    # use_sg -- 'dx', 'dy', 'rdxc', 'rdyc', 'sin_sg needed
    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        
        #ref = inputs['keco']
        
        for v in ['utco', 'vtco', 'keco','uco', 'uvort','kex', 'kevort', 'vco', 'ubkey','vbkey', 'fyh', 'uts','utafter']:
            if v in inputs:
                del(inputs[v])
        #namelist {'ke_bg': 0.0, 'd_con': 1.0, 'nord': 2, 'd2_divg': 0.0, 'nord_v': 2, 'nord_w': 2, 'nord_t': 2, 'damp_vt': 0.06, 'damp_w': 0.06, 'damp_t': 0.06} (55, 55, 64)
       
        d_sw.compute(**inputs)
       
        #raise Exception('stop')
        '''
        print('utcalc', ut[2,3,47])
        for i in range(3,51):
            for j in range(3,51):
                for k in range(2,63):
                    r = ref[i, j, k]
                    c = ke[i, j, k]
                    if r != c:
                        print('bad', i, j, k, c, r, r-c)
        raise Exception('stop')
        '''
        return self.slice_output(inputs)


    # For child tranlate tests where nord and damp column processing applies
    def nord_column_split_compute(self, inputs, func):
        return self.column_split_compute(inputs, func, {'nord': 'nord_column', 'damp_c': 'damp_c'})
        '''
        column_info = {'nord': self.column_namelist_vals('nord_column', inputs), 'damp_c': self.column_namelist_vals('damp_c', inputs)}
        column_info['nord'] = [int(i) for i in column_info['nord']]
        self.make_storage_data_input_vars(inputs)
        del inputs['nord_column']
        del inputs['damp_c']
        d_sw.d_sw_ksplit(func, inputs, column_info, list(self.out_vars.keys()), self.grid)
        return self.slice_output(inputs)
        '''
    def column_split_compute(self, inputs, func, info_mapping):
        column_info = {}
        for pyfunc_var, serialbox_var in info_mapping.items():
            column_info[pyfunc_var] = self.column_namelist_vals(serialbox_var, inputs)
        self.make_storage_data_input_vars(inputs)
        for k in info_mapping.values():
            del inputs[k]
        outputs = {}
        for outvar, info in self.out_vars.items():
            if outvar not in inputs:
                outputs[outvar] = utils.make_storage_from_shape(self.maxshape, self.grid.default_origin())
            else:
                outputs[outvar] = inputs[outvar]
        d_sw.d_sw_ksplit(func, inputs, column_info, outputs, self.grid)
        return self.slice_output(outputs)
