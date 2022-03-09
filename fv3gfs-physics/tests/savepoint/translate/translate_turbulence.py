import copy

import numpy as np

import pace.dsl.gt4py_utils as utils
import pace.util
from fv3gfs.physics.stencils.turbulence import Turbulence
from fv3gfs.physics.stencils.physics import PhysicsState
from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py

class TranslateTurbulence(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)

        self.in_vars["data_vars"] = {
            "qvapor": {"serialname" : "qvapor", "turb" : True},
            "qliquid" : {"serialname" : "qliquid", "turb" : True},
            "qice" : {"serialname" : "qice", "turb" : True},
            "qrain" : {"serialname" : "qrain", "turb" : True},
            "qsnow" : {"serialname" : "qsnow", "turb" : True},
            "qgraupel" : {"serialname" : "qgraupel", "turb" : True},
            "ua": {"serialname": "ua", "turb": True},
            "va": {"serialname": "va", "turb": True},
            "pt": {"serialname": "pt", "turb": True},
            "qo3mr" : {"serialname" : "qo3mr", "turb" : True},
            "qsgs_tke" : {"serialname" : "qsgs_tke", "turb" : True},
            "delprsi" : {"serialname" : "delprsi", "turb" : True},
        }

        self.out_vars = {
            "dv" : {"serialname": "dv", "kend": namelist.npz - 1},
            "du" : {"serialname": "du", "kend": namelist.npz - 1},
            "tdt" : {"serialname": "tdt", "kend": namelist.npz - 1},
            "rtg" : {"serialname": "rtg"}, # MultiD storage 8
            "kpbl" : {"serialname": "kpbl",},
            "dusfc" : {"serialname": "dusfc",},
            "dvsfc" : {"serialname": "dvsfc",},
            "dtsfc" : {"serialname": "dtsfc",},
            "dqsfc" : {"serialname": "dqsfc",},
            "hpbl" : {"serialname": "hpbl",},
        }
        self.namelist = namelist
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
    
    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        storage = utils.make_storage_from_shape(
            self.grid_indexing.domain_full(add=(1, 1, 1)),
            origin=self.grid_indexing.origin_compute(),
            init=True,
            backend=self.stencil_factory.backend,
        )