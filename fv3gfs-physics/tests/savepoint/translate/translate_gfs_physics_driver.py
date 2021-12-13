import copy

from mpi4py import MPI

import fv3core._config as spec
import pace.dsl.gt4py_utils as utils
import pace.util as util
from fv3gfs.physics.stencils.physics import Physics, PhysicsState
from fv3gfs.physics.testing import TranslatePhysicsFortranData2Py
from pace.dsl.typing import Float


class TranslateGFSPhysicsDriver(TranslatePhysicsFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)

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
                "kend": grid.npz - 1,
                "order": "F",
            },
            "gu0": {
                "serialname": "IPD_gu0",
                "kend": grid.npz - 1,
                "order": "F",
            },
            "gv0": {
                "serialname": "IPD_gv0",
                "kend": grid.npz - 1,
                "order": "F",
            },
            "qvapor": {
                "serialname": "IPD_qvapor",
                "kend": grid.npz - 1,
                "order": "F",
            },
            "qliquid": {
                "serialname": "IPD_qliquid",
                "kend": grid.npz - 1,
                "order": "F",
            },
            "qrain": {
                "serialname": "IPD_rain",
                "kend": grid.npz - 1,
                "order": "F",
            },
            "qice": {
                "serialname": "IPD_qice",
                "kend": grid.npz - 1,
                "order": "F",
            },
            "qsnow": {
                "serialname": "IPD_snow",
                "kend": grid.npz - 1,
                "order": "F",
            },
            "qgraupel": {
                "serialname": "IPD_qgraupel",
                "kend": grid.npz - 1,
                "order": "F",
            },
            "qcld": {
                "serialname": "IPD_qcld",
                "kend": grid.npz - 1,
                "order": "F",
            },
        }

    def compute(self, inputs):
        self.make_storage_data_input_vars(inputs)
        storage = utils.make_storage_from_shape(
            self.grid.domain_shape_full(add=(1, 1, 1)),
            origin=self.grid.compute_origin(),
            init=True,
            backend=self.grid.stencil_factory.backend,
        )
        inputs["delprsi"] = copy.deepcopy(storage)
        inputs["phii"] = copy.deepcopy(storage)
        inputs["phil"] = copy.deepcopy(storage)
        inputs["dz"] = copy.deepcopy(storage)
        inputs["wmp"] = copy.deepcopy(storage)
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
        physics_state = PhysicsState(**inputs)
        # make mock communicator, this is not used
        # but needs to be passed as type CubedSphereCommunicator
        comm = MPI.COMM_WORLD
        layout = [1, 1]
        partitioner = util.CubedSpherePartitioner(util.TilePartitioner(layout))
        communicator = util.CubedSphereCommunicator(comm, partitioner)
        grid_info = {}  # pass empty grid info, they are not in used for this test
        grid_info["vlon1"] = 0
        grid_info["vlon2"] = 0
        grid_info["vlon3"] = 0
        grid_info["vlat1"] = 0
        grid_info["vlat2"] = 0
        grid_info["vlat3"] = 0
        grid_info["edge_vect_w"] = 0
        grid_info["edge_vect_e"] = 0
        grid_info["edge_vect_s"] = 0
        grid_info["edge_vect_n"] = 0
        grid_info["es1_1"] = 0
        grid_info["es2_1"] = 0
        grid_info["es3_1"] = 0
        grid_info["ew1_2"] = 0
        grid_info["ew2_2"] = 0
        grid_info["ew3_2"] = 0
        physics = Physics(
            self.grid.stencil_factory,
            self.grid.grid_data,
            spec.namelist,
            communicator,
            partitioner,
            self.grid.rank,
            grid_info,
        )
        physics._atmos_phys_driver_statein(
            physics._prsik,
            physics_state.phii,
            physics._prsi,
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
            physics._prsi,
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
        )
        microph_state = physics_state.microphysics(storage)
        physics._microphysics(microph_state)
        # Fortran uses IPD interface, here we use var_t1 to denote the updated field
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
            physics_state.qvapor_t1,
            physics_state.qliquid_t1,
            physics_state.qrain_t1,
            physics_state.qice_t1,
            physics_state.qsnow_t1,
            physics_state.qgraupel_t1,
            physics_state.qcld_t1,
            physics_state.pt_t1,
            physics_state.ua_t1,
            physics_state.va_t1,
            Float(physics._dt_atmos),
        )
        inputs["gt0"] = physics_state.pt_t1
        inputs["gu0"] = physics_state.ua_t1
        inputs["gv0"] = physics_state.va_t1
        inputs["qvapor"] = physics_state.qvapor_t1
        inputs["qliquid"] = physics_state.qliquid_t1
        inputs["qrain"] = physics_state.qrain_t1
        inputs["qice"] = physics_state.qice_t1
        inputs["qsnow"] = physics_state.qsnow_t1
        inputs["qgraupel"] = physics_state.qgraupel_t1
        inputs["qcld"] = physics_state.qcld_t1
        return self.slice_output(inputs)
