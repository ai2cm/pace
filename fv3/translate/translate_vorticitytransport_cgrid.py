from .translate import TranslateFortranData2Py
import fv3.stencils.vorticitytransport_cgrid as VorticityTransport_Cgrid 


class TranslateVorticityTransport_Cgrid(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = VorticityTransport_Cgrid.compute
        self.in_vars['data_vars'] = {'uc': {}, 
                                     'vc': {}, 
                                     'vort_c': {'istart': grid.is_-1, 'iend': grid.ie+1,  'jstart': grid.js-1, 'jend': grid.je+1}, 
                                     'ke_c': {'istart': grid.is_-1, 'iend': grid.ie+1,  'jstart': grid.js-1, 'jend': grid.je+1}, 
                                     'u': {}, 
                                     'v': {}, 
                                     'fxv': {'istart': grid.is_-1, 'iend': grid.ie+2,  'jstart': grid.js-1, 'jend': grid.je+1}, 
                                     'fyv': {'istart': grid.is_-1, 'iend': grid.ie+1,  'jstart': grid.js-1, 'jend': grid.je+2}}
        self.in_vars['parameters'] = ['dt2']
        self.out_vars = {'uc': grid.x3d_domain_dict(), 'vc': grid.y3d_domain_dict()}

