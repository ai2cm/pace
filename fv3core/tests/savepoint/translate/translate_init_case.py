import fv3core._config as spec
import fv3core.utils.baroclinic_initialization as baroclinic_init
from fv3core.testing import TranslateFortranData2Py, ParallelTranslate
import numpy as np
import fv3core._config as spec
class TranslateInitCase(TranslateFortranData2Py):
    
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            #"u": {},
            #"v": {},
            #"uc": {},
            #"vc": {},
            #"ua": {},
            #"va": {},
            #"w": {},
            #"pt": {},
            #"delp": {},
            #"q4d": {},
            #"phis":{},
            #"ps": {},
            #"delz": {},
            "ak": {},
            "bk": {},
            #"pe": {"istart": grid.is_ - 1, "jstart": grid.js - 1, "kaxis": 1},
            #"peln": {"istart": grid.is_, "jstart": grid.js, "kaxis": 1},
            #"pk": grid.compute_buffer_k_dict(),
            #"ze0": grid.compute_dict(),
         
        }
        self.in_vars["parameters"] = ["ptop"]
        self.out_vars = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            #"uc": grid.x3d_domain_dict(),
            #"vc": grid.y3d_domain_dict(),
            #"ua": {},
            #"va": {},
            "w": {},
            "pt": {},
            "delp": {},
            #"q4d": {},
            "phis": {},
            "delz": {},
            "ps": {"kstart": grid.npz, "kend": grid.npz},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "pk": grid.compute_buffer_k_dict(),
            "pkz": grid.compute_dict(),
            "fC": { "iend": grid.ied + 1, "jend": grid.jed + 1,"kstart": grid.npz, "kend": grid.npz},
            "f0": {"kstart": grid.npz, "kend": grid.npz},
        }
        self.ignore_near_zero_errors = {}
        for var in ['u', 'v']:
            self.ignore_near_zero_errors[var] = {'near_zero': 1e-13}
    def compute(self, inputs):
        #self.make_storage_data_input_vars(inputs)
        #baroclinic_init(inputs["ak"], inputs["bk"], self.grid)
        #return self.slice_output(inputs)
        #return inputs
        #for v in  self.in_vars["data_vars"]:
        #    inputs[v] = inputs[v].data
        full_shape = self.grid.domain_shape_full(add=(1, 1, 1))
        for variable in ["qvapor", "pe", "peln", "pk", "pkz", "pt", "delz", "w", "u", "v", "delp"]:
            inputs[variable] = np.zeros(full_shape)
        for var2d in ["ps", "phis", "fC", "f0"]:
            inputs[var2d] = np.zeros(full_shape[0:2])
        for zvar in ["eta", "eta_v"]:
            inputs[zvar] = np.zeros(self.grid.npz+1)
        namelist = spec.namelist
        grid_vars = {
            "lon":self.grid.bgrid1.data,
            "lat": self.grid.bgrid2.data,
            "lon_agrid" : self.grid.agrid1.data,
            "lat_agrid" : self.grid.agrid2.data,
            "ee1": self.grid.ee1.data,
            "ee2": self.grid.ee2.data,
            "es1": self.grid.es1.data,
            "ew2": self.grid.ew2.data
        }
    
        baroclinic_init.init_case(**inputs, **grid_vars, adiabatic=namelist.adiabatic, hydrostatic=namelist.hydrostatic, moist_phys=namelist.moist_phys)
        return self.slice_output(inputs)

class TranslateInitPreJab(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "ak": {},
            "bk": {},
            "delp": {}
         
        }
        self.in_vars["parameters"] = [ "ptop"]
        self.out_vars = {
            "delp": {},
            "qvapor": {},
            "ps": {"kstart": grid.npz, "kend": grid.npz},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "pk": grid.compute_buffer_k_dict(),
            "pkz": grid.compute_dict(),
            "eta": {"istart":0, "iend":0, "jstart":0, "jend":0},
            "eta_v": {"istart":0, "iend":0, "jstart":0, "jend":0}
            
        }
      
    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        for k, v in inputs.items():
            if k != 'ptop':
                inputs[k] = v.data
        full_shape = self.grid.domain_shape_full(add=(1, 1, 1))
        for variable in ["qvapor", "pe", "peln", "pk", "pkz"]:
            inputs[variable] = np.zeros(full_shape)
        inputs["ps"] = np.zeros(full_shape[0:2])
        for zvar in ["eta", "eta_v"]:
            inputs[zvar] = np.zeros(self.grid.npz+1)
        baroclinic_init.setup_pressure_fields(**inputs, lat_agrid=self.grid.agrid2.data[:-1, :-1], adiabatic=spec.namelist.adiabatic)
        return self.slice_output(inputs)

class TranslateJablonowskiBaroclinic(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "delp": {},
            "eta_v": {"istart":0, "iend":0, "jstart":0, "jend":0},
            "eta": {"istart":0, "iend":0, "jstart":0, "jend":0},
            "qvapor": {},
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
        }
                 
        self.in_vars["parameters"] = ["ptop"]
        self.out_vars = {
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "pt": {},
            "phis": {},
            "delz": {},
        
        }
        self.ignore_near_zero_errors = {}
        for var in ['u', 'v']:
            self.ignore_near_zero_errors[var] = {'near_zero': 2e-13}
       
        self.max_error = 1e-13
        
    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        # testing just numpy arrays for this
        for k, v in inputs.items():
            if k != 'ptop':
                inputs[k] = v.data
        full_shape = self.grid.domain_shape_full(add=(1, 1, 1))
        for variable in ["u", "v", "pt", "delz", "w"]:
            inputs[variable] = np.zeros(full_shape)
        for var2d in ["phis"]:
            inputs[var2d] = np.zeros(full_shape[0:2])
        
        grid_vars = {
            "lon":self.grid.bgrid1.data,
            "lat": self.grid.bgrid2.data,
            "lon_agrid" : self.grid.agrid1.data,
            "lat_agrid" : self.grid.agrid2.data,
            "ee1": self.grid.ee1.data,
            "ee2": self.grid.ee2.data,
            "es1": self.grid.es1.data,
            "ew2": self.grid.ew2.data
        }
        inputs["w"][:] = 1e30
        baroclinic_init.baroclinic_initialization(**inputs, **grid_vars, adiabatic=spec.namelist.adiabatic,hydrostatic=spec.namelist.hydrostatic)
        return self.slice_output(inputs)

class TranslatePVarAuxiliaryPressureVars(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "delp": {},
            "delz": {},
            "pt": {},
            "ps": {"kstart": grid.npz, "kend": grid.npz},
            "qvapor": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz,
                "kaxis": 1,
            },
            "pk": grid.compute_buffer_k_dict(),
            "pkz": grid.compute_dict(),
        }
                 
        self.in_vars["parameters"] = ["ptop"]
        self.out_vars = {}
        for var in ["delz", "delp", "ps", "pe", "peln", "pk", "pkz"]:
            self.out_vars[var] =  self.in_vars["data_vars"][var]
        
    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        # testing just numpy arrays for this
        for k, v in inputs.items():
            if k != 'ptop':
                inputs[k] = v.data
    
        namelist = spec.namelist
        baroclinic_init.p_var(**inputs, moist_phys=namelist.moist_phys, make_nh=(not namelist.hydrostatic), hydrostatic=namelist.hydrostatic)
        return self.slice_output(inputs)

