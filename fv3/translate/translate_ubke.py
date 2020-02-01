from .translate import TranslateFortranData2Py
import fv3.stencils.ubke as ubke

class TranslateUbKE(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.compute_func = ubke.compute
        self.in_vars['data_vars'] = {'uc': {}, 'vc': {}, 'ut': {}, 'ub': grid.compute_dict_buffer_2d()}
        self.in_vars['parameters'] = ['dt5', 'dt4']
        self.out_vars = {'ub': grid.compute_dict_buffer_2d()}
