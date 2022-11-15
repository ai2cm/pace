import copy

import pace.dsl.gt4py_utils as utils
import pace.util as util
from pace.physics.stencils.physics import Physics, PhysicsState
from pace.stencils import update_atmos_state
from pace.stencils.testing.translate_physics import TranslatePhysicsFortranData2Py


class TranslateGFSPhysicsDriver(TranslatePhysicsFortranData2Py):
    def __init__(self, grid, namelist, stencil_factory):
        super().__init__(grid, namelist, stencil_factory)
        # using top level namelist rather than PhysicsConfig
        # because DycoreToPhysics needs some dycore info
        self.namelist = namelist
        self.in_vars["data_vars"] = {
            "qvapor": {"dycore": True},
            "qliquid": {"dycore": True},
            "qrain": {"dycore": True},
            "qsnow": {"dycore": True},
            "qice": {"dycore": True},
            "qgraupel": {"dycore": True},
            "qo3mr": {"dycore": True},
            "qsgs_tke": {"dycore": True},
            "qcld": {"dycore": True},
            "pt": {"dycore": True},
            "delp": {"dycore": True},
            "delz": {"dycore": True},
            "ua": {"dycore": True},
            "va": {"dycore": True},
            "w": {"dycore": True},
            "omga": {"dycore": True},
        }
        self.out_vars = {
            "gt0": {
                "serialname": "IPD_gt0",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "gu0": {
                "serialname": "IPD_gu0",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "gv0": {
                "serialname": "IPD_gv0",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qvapor": {
                "serialname": "IPD_qvapor",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qliquid": {
                "serialname": "IPD_qliquid",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qrain": {
                "serialname": "IPD_rain",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qice": {
                "serialname": "IPD_qice",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qsnow": {
                "serialname": "IPD_snow",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qgraupel": {
                "serialname": "IPD_qgraupel",
                "kend": namelist.npz - 1,
                "order": "F",
            },
            "qcld": {
                "serialname": "IPD_qcld",
                "kend": namelist.npz - 1,
                "order": "F",
            },
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
        inputs["delprsi"] = copy.deepcopy(storage)
        inputs["phii"] = copy.deepcopy(storage)
        inputs["phil"] = copy.deepcopy(storage)
        inputs["dz"] = copy.deepcopy(storage)
        inputs["wmp"] = copy.deepcopy(storage)
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
        inputs["prsi"] = copy.deepcopy(storage)
        inputs["prsik"] = copy.deepcopy(storage)
        # When we start doing standard case physics driver test,
        # land will need to be added as part of the savepoint.
        inputs["land"] = utils.make_storage_from_shape(
            self.grid_indexing.domain_full(add=(1, 1, 1))[0:2],
            origin=self.grid_indexing.origin_compute()[0:2],
            backend=self.stencil_factory.backend,
        )
        sizer = util.SubtileGridSizer.from_tile_params(
            nx_tile=self.namelist.npx - 1,
            ny_tile=self.namelist.npy - 1,
            nz=self.namelist.npz,
            n_halo=3,
            extra_dim_lengths={},
            layout=self.namelist.layout,
        )

        quantity_factory = util.QuantityFactory.from_backend(
            sizer, self.stencil_factory.backend
        )
        active_packages = ["microphysics"]
        physics_state = PhysicsState(
            **inputs,
            quantity_factory=quantity_factory,
            active_packages=active_packages,
        )
        physics = Physics(
            self.stencil_factory,
            self.grid.grid_data,
            self.namelist,
            active_packages=active_packages,
        )
        # TODO, self.namelist doesn't have fv_sg_adj because it is PhysicsConfig
        # either move where GFSPhysicsDriver starts, or pass the full namelist or
        # get around this issue another way. Setting do_dry_convective_adjustment
        # to False for now (we don't run this on a case where it is True yet)
        dycore_to_physics = update_atmos_state.DycoreToPhysics(
            self.stencil_factory,
            self.grid.quantity_factory,
            self.namelist,
            do_dry_convective_adjust=False,
            dycore_only=self.namelist.dycore_only,
        )
        dycore_to_physics(dycore_state=physics_state, physics_state=physics_state)
        physics._atmos_phys_driver_statein(
            physics_state.prsik,
            physics_state.phii,
            physics_state.prsi,
            physics_state.delz,
            physics_state.delp,
            physics_state.qvapor,
            physics_state.qliquid,
            physics_state.qrain,
            physics_state.qice,
            physics_state.qsnow,
            physics_state.qgraupel,
            physics_state.qo3mr,
            physics_state.qsgs_tke,
            physics_state.qcld,
            physics_state.pt,
            physics._dm3d,
        )
        physics._get_prs_fv3(
            physics_state.phii,
            physics_state.prsi,
            physics_state.pt,
            physics_state.qvapor,
            physics_state.delprsi,
            physics._del_gz,
        )
        # If PBL scheme is present, physics_state should be updated here
        physics._get_phi_fv3(
            physics_state.pt,
            physics_state.qvapor,
            physics._del_gz,
            physics_state.phii,
            physics_state.phil,
        )
        physics._prepare_microphysics(
            physics_state.dz,
            physics_state.phii,
            physics_state.wmp,
            physics_state.omga,
            physics_state.qvapor,
            physics_state.pt,
            physics_state.delp,
            physics_state.microphysics.udt,
            physics_state.microphysics.vdt,
            physics_state.microphysics.pt_dt,
            physics_state.microphysics.qv_dt,
            physics_state.microphysics.ql_dt,
            physics_state.microphysics.qr_dt,
            physics_state.microphysics.qi_dt,
            physics_state.microphysics.qs_dt,
            physics_state.microphysics.qg_dt,
            physics_state.microphysics.qa_dt,
        )
        microph_state = physics_state.microphysics
        physics._microphysics(microph_state, float(self.namelist.dt_atmos))
        # Fortran uses IPD interface, here we use physics_updated_<var>
        # to denote the updated field
        physics._update_physics_state_with_tendencies(
            physics_state.qvapor,
            physics_state.qliquid,
            physics_state.qrain,
            physics_state.qice,
            physics_state.qsnow,
            physics_state.qgraupel,
            physics_state.qcld,
            physics_state.pt,
            physics_state.ua,
            physics_state.va,
            microph_state.qv_dt,
            microph_state.ql_dt,
            microph_state.qr_dt,
            microph_state.qi_dt,
            microph_state.qs_dt,
            microph_state.qg_dt,
            microph_state.qa_dt,
            microph_state.pt_dt,
            microph_state.udt,
            microph_state.vdt,
            physics_state.physics_updated_specific_humidity,
            physics_state.physics_updated_qliquid,
            physics_state.physics_updated_qrain,
            physics_state.physics_updated_qice,
            physics_state.physics_updated_qsnow,
            physics_state.physics_updated_qgraupel,
            physics_state.physics_updated_cloud_fraction,
            physics_state.physics_updated_pt,
            physics_state.physics_updated_ua,
            physics_state.physics_updated_va,
            float(self.namelist.dt_atmos),
        )
        inputs["gt0"] = physics_state.physics_updated_pt
        inputs["gu0"] = physics_state.physics_updated_ua
        inputs["gv0"] = physics_state.physics_updated_va
        inputs["qvapor"] = physics_state.physics_updated_specific_humidity
        inputs["qliquid"] = physics_state.physics_updated_qliquid
        inputs["qrain"] = physics_state.physics_updated_qrain
        inputs["qice"] = physics_state.physics_updated_qice
        inputs["qsnow"] = physics_state.physics_updated_qsnow
        inputs["qgraupel"] = physics_state.physics_updated_qgraupel
        inputs["qcld"] = physics_state.physics_updated_cloud_fraction
        return self.slice_output(inputs)
