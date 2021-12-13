import copy

import numpy as np

import fv3core._config as spec
import pace.dsl.gt4py_utils as utils
from fv3gfs.physics.stencils.microphysics import Microphysics
from fv3gfs.physics.stencils.physics import PhysicsState
from fv3gfs.physics.testing import TranslatePhysicsFortranData2Py


class TranslateMicroph(TranslatePhysicsFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)

        self.in_vars["data_vars"] = {
            "qvapor": {"serialname": "mph_qv1", "microph": True},
            "qliquid": {"serialname": "mph_ql1", "microph": True},
            "qrain": {"serialname": "mph_qr1", "microph": True},
            "qice": {"serialname": "mph_qi1", "microph": True},
            "qsnow": {"serialname": "mph_qs1", "microph": True},
            "qgraupel": {"serialname": "mph_qg1", "microph": True},
            "qcld": {"serialname": "mph_qa1", "microph": True},
            "ua": {"serialname": "mph_uin", "microph": True},
            "va": {"serialname": "mph_vin", "microph": True},
            "delprsi": {"serialname": "mph_delp", "microph": True},
            "wmp": {"serialname": "mph_w", "microph": True},
            "delz": {"serialname": "mph_dz", "microph": True},
            "pt": {"serialname": "mph_pt", "microph": True},
        }

        self.out_vars = {
            "pt_dt": {"serialname": "mph_pt_dt", "kend": grid.npz - 1},
            "qv_dt": {"serialname": "mph_qv_dt", "kend": grid.npz - 1},
            "ql_dt": {"serialname": "mph_ql_dt", "kend": grid.npz - 1},
            "qr_dt": {"serialname": "mph_qr_dt", "kend": grid.npz - 1},
            "qi_dt": {"serialname": "mph_qi_dt", "kend": grid.npz - 1},
            "qs_dt": {"serialname": "mph_qs_dt", "kend": grid.npz - 1},
            "qg_dt": {"serialname": "mph_qg_dt", "kend": grid.npz - 1},
            "qa_dt": {"serialname": "mph_qa_dt", "kend": grid.npz - 1},
            "udt": {"serialname": "mph_udt", "kend": grid.npz - 1},
            "vdt": {"serialname": "mph_vdt", "kend": grid.npz - 1},
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        storage = utils.make_storage_from_shape(
            self.grid.domain_shape_full(add=(1, 1, 1)),
            origin=self.grid.compute_origin(),
            init=True,
            backend=self.grid.stencil_factory.backend,
        )
        inputs["qo3mr"] = copy.deepcopy(storage)
        inputs["qsgs_tke"] = copy.deepcopy(storage)
        inputs["w"] = copy.deepcopy(storage)
        inputs["delp"] = copy.deepcopy(storage)
        inputs["phii"] = copy.deepcopy(storage)
        inputs["phil"] = copy.deepcopy(storage)
        inputs["dz"] = copy.deepcopy(storage)
        inputs["qvapor_t1"] = copy.deepcopy(storage)
        inputs["qliquid_t1"] = copy.deepcopy(storage)
        inputs["qrain_t1"] = copy.deepcopy(storage)
        inputs["qsnow_t1"] = copy.deepcopy(storage)
        inputs["qice_t1"] = copy.deepcopy(storage)
        inputs["qgraupel_t1"] = copy.deepcopy(storage)
        inputs["qcld_t1"] = copy.deepcopy(storage)
        inputs["pt_t1"] = copy.deepcopy(storage)
        inputs["ua_t1"] = copy.deepcopy(storage)
        inputs["va_t1"] = copy.deepcopy(storage)
        inputs["omga"] = copy.deepcopy(storage)
        physics_state = PhysicsState(**inputs)
        microphysics = Microphysics(
            self.grid.stencil_factory, self.grid.grid_data, spec.namelist
        )
        microph_state = physics_state.microphysics(storage)
        microphysics(microph_state)
        inputs["pt_dt"] = microph_state.pt_dt
        inputs["qv_dt"] = microph_state.qv_dt
        inputs["ql_dt"] = microph_state.ql_dt
        inputs["qr_dt"] = microph_state.qr_dt
        inputs["qi_dt"] = microph_state.qi_dt
        inputs["qs_dt"] = microph_state.qs_dt
        inputs["qg_dt"] = microph_state.qg_dt
        inputs["qa_dt"] = microph_state.qa_dt
        inputs["udt"] = microph_state.udt
        inputs["vdt"] = microph_state.vdt
        out = self.slice_output(inputs)
        # microphysics data is already reversed
        for key in out.keys():  # fortran data has dimension 1 in the 2nd axis
            out[key] = out[key][:, np.newaxis, ::-1]
        return out
