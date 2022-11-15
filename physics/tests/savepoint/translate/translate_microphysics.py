import copy

import numpy as np

import pace.dsl.gt4py_utils as utils
import pace.util
from pace.dsl.typing import Float
from pace.physics.stencils.microphysics import Microphysics
from pace.physics.stencils.physics import PhysicsState
from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py


class TranslateMicroph(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
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
            "land": {"serialname": "mph_land", "microph": True},
        }

        self.out_vars = {
            "pt_dt": {"serialname": "mph_pt_dt", "kend": namelist.npz - 1},
            "qv_dt": {"serialname": "mph_qv_dt", "kend": namelist.npz - 1},
            "ql_dt": {"serialname": "mph_ql_dt", "kend": namelist.npz - 1},
            "qr_dt": {"serialname": "mph_qr_dt", "kend": namelist.npz - 1},
            "qi_dt": {"serialname": "mph_qi_dt", "kend": namelist.npz - 1},
            "qs_dt": {"serialname": "mph_qs_dt", "kend": namelist.npz - 1},
            "qg_dt": {"serialname": "mph_qg_dt", "kend": namelist.npz - 1},
            "qa_dt": {"serialname": "mph_qa_dt", "kend": namelist.npz - 1},
            "udt": {"serialname": "mph_udt", "kend": namelist.npz - 1},
            "vdt": {"serialname": "mph_vdt", "kend": namelist.npz - 1},
        }
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        storage = utils.make_storage_from_shape(
            self.grid_indexing.domain_full(add=(1, 1, 1)),
            origin=self.grid_indexing.origin_compute(),
            backend=self.stencil_factory.backend,
        )
        inputs["qo3mr"] = copy.deepcopy(storage)
        inputs["qsgs_tke"] = copy.deepcopy(storage)
        inputs["w"] = copy.deepcopy(storage)
        inputs["delp"] = copy.deepcopy(storage)
        inputs["phii"] = copy.deepcopy(storage)
        inputs["phil"] = copy.deepcopy(storage)
        inputs["dz"] = copy.deepcopy(storage)
        inputs["physics_updated_specific_humidity"] = copy.deepcopy(storage)
        inputs["physics_updated_qliquid"] = copy.deepcopy(storage)
        inputs["physics_updated_qrain"] = copy.deepcopy(storage)
        inputs["physics_updated_qsnow"] = copy.deepcopy(storage)
        inputs["physics_updated_qice"] = copy.deepcopy(storage)
        inputs["physics_updated_qgraupel"] = copy.deepcopy(storage)
        inputs["physics_updated_cloud_fraction"] = copy.deepcopy(storage)
        inputs["physics_updated_pt"] = copy.deepcopy(storage)
        inputs["physics_updated_ua"] = copy.deepcopy(storage)
        inputs["physics_updated_va"] = copy.deepcopy(storage)
        inputs["omga"] = copy.deepcopy(storage)
        inputs["prsi"] = copy.deepcopy(storage)
        inputs["prsik"] = copy.deepcopy(storage)
        sizer = pace.util.SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        quantity_factory = pace.util.QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )
        physics_state = PhysicsState.init_from_storages(
            inputs,
            sizer=sizer,
            quantity_factory=quantity_factory,
            active_packages=["microphysics"],
        )
        microphysics = Microphysics(
            self.stencil_factory, self.grid.grid_data, self.namelist
        )
        microph_state = physics_state.microphysics
        microphysics(microph_state, timestep=Float(self.namelist.dt_atmos))
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
