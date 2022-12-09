from datetime import timedelta
from typing import Dict

import f90nml
import numpy as np

import pace.util
from pace import fv3core


class GeosDycoreWrapper:
    """
    Provides an interface for the Geos model to access the Pace dycore.
    Takes numpy arrays as inputs, returns a dictionary of numpy arrays as outputs
    """

    def __init__(self, namelist: f90nml.Namelist, comm: pace.util.Comm, backend: str):
        self.timer = pace.util.Timer()
        self.namelist = namelist

        self.dycore_config = fv3core.DynamicalCoreConfig.from_f90nml(self.namelist)

        self.layout = self.dycore_config.layout
        partitioner = pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(self.layout)
        )
        self.communicator = pace.util.CubedSphereCommunicator(
            comm, partitioner, timer=self.timer
        )

        sizer = pace.util.SubtileGridSizer.from_namelist(
            self.namelist, partitioner.tile, self.communicator.tile.rank
        )
        quantity_factory = pace.util.QuantityFactory.from_backend(
            sizer=sizer, backend=backend
        )

        # set up the metric terms and grid data
        metric_terms = pace.util.grid.MetricTerms(
            quantity_factory=quantity_factory, communicator=self.communicator
        )
        grid_data = pace.util.grid.GridData.new_from_metric_terms(metric_terms)

        stencil_config = pace.dsl.stencil.StencilConfig(
            compilation_config=pace.dsl.stencil.CompilationConfig(
                backend=backend, rebuild=False, validate_args=False
            ),
        )

        self._grid_indexing = pace.dsl.stencil.GridIndexing.from_sizer_and_communicator(
            sizer=sizer, cube=self.communicator
        )
        stencil_factory = pace.dsl.StencilFactory(
            config=stencil_config, grid_indexing=self._grid_indexing
        )

        self.dycore_state = fv3core.DycoreState.init_zeros(
            quantity_factory=quantity_factory
        )

        self.dycore_state.bdt = float(namelist["dt_atmos"])
        if "fv_core_nml" in namelist.keys():
            self.dycore_state.bdt = (
                float(namelist["dt_atmos"]) / namelist["fv_core_nml"]["k_split"]
            )
        elif "dycore_config" in namelist.keys():
            self.dycore_state.bdt = (
                float(namelist["dt_atmos"]) / namelist["dycore_config"]["k_split"]
            )
        else:
            raise KeyError("Cannot find k_split in namelist")

        damping_coefficients = pace.util.grid.DampingCoefficients.new_from_metric_terms(
            metric_terms
        )

        self.dynamical_core = fv3core.DynamicalCore(
            comm=self.communicator,
            grid_data=grid_data,
            stencil_factory=stencil_factory,
            quantity_factory=quantity_factory,
            damping_coefficients=damping_coefficients,
            config=self.dycore_config,
            timestep=timedelta(seconds=self.dycore_config.dt_atmos),
            phis=self.dycore_state.phis,
            state=self.dycore_state,
        )

        self.output_dict: Dict[str, np.ndarray] = {}
        self._allocate_output_dir()

    def __call__(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        delz: np.ndarray,
        pt: np.ndarray,
        delp: np.ndarray,
        q: np.ndarray,
        ps: np.ndarray,
        pe: np.ndarray,
        pk: np.ndarray,
        peln: np.ndarray,
        pkz: np.ndarray,
        phis: np.ndarray,
        q_con: np.ndarray,
        omga: np.ndarray,
        ua: np.ndarray,
        va: np.ndarray,
        uc: np.ndarray,
        vc: np.ndarray,
        mfxd: np.ndarray,
        mfyd: np.ndarray,
        cxd: np.ndarray,
        cyd: np.ndarray,
        diss_estd: np.ndarray,
    ) -> Dict[str, np.ndarray]:

        with self.timer.clock("move_to_pace"):
            self.dycore_state = self._put_fortran_data_in_dycore(
                u,
                v,
                w,
                delz,
                pt,
                delp,
                q,
                ps,
                pe,
                pk,
                peln,
                pkz,
                phis,
                q_con,
                omga,
                ua,
                va,
                uc,
                vc,
                mfxd,
                mfyd,
                cxd,
                cyd,
                diss_estd,
            )

        with self.timer.clock("dycore"):
            self.dynamical_core.step_dynamics(state=self.dycore_state, timer=self.timer)

        with self.timer.clock("move_to_fortran"):
            self.output_dict = self._prep_outputs_for_geos()

        return self.output_dict

    def _put_fortran_data_in_dycore(
        self,
        u: np.ndarray,
        v: np.ndarray,
        w: np.ndarray,
        delz: np.ndarray,
        pt: np.ndarray,
        delp: np.ndarray,
        q: np.ndarray,
        ps: np.ndarray,
        pe: np.ndarray,
        pk: np.ndarray,
        peln: np.ndarray,
        pkz: np.ndarray,
        phis: np.ndarray,
        q_con: np.ndarray,
        omga: np.ndarray,
        ua: np.ndarray,
        va: np.ndarray,
        uc: np.ndarray,
        vc: np.ndarray,
        mfxd: np.ndarray,
        mfyd: np.ndarray,
        cxd: np.ndarray,
        cyd: np.ndarray,
        diss_estd: np.ndarray,
    ) -> fv3core.DycoreState:

        isc = self._grid_indexing.isc
        jsc = self._grid_indexing.jsc
        iec = self._grid_indexing.iec + 1
        jec = self._grid_indexing.jec + 1

        state = self.dycore_state

        # Assign compute domain:
        pace.util.utils.safe_assign_array(state.u.view[:], u[isc:iec, jsc : jec + 1, :])
        pace.util.utils.safe_assign_array(state.v.view[:], v[isc : iec + 1, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.w.view[:], w[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.ua.view[:], ua[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.va.view[:], va[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(
            state.uc.view[:], uc[isc : iec + 1, jsc:jec, :]
        )
        pace.util.utils.safe_assign_array(
            state.vc.view[:], vc[isc:iec, jsc : jec + 1, :]
        )

        pace.util.utils.safe_assign_array(state.delz.view[:], delz[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.pt.view[:], pt[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.delp.view[:], delp[isc:iec, jsc:jec, :])

        pace.util.utils.safe_assign_array(state.mfxd.view[:], mfxd)
        pace.util.utils.safe_assign_array(state.mfyd.view[:], mfyd)
        pace.util.utils.safe_assign_array(state.cxd.view[:], cxd[:, jsc:jec, :])
        pace.util.utils.safe_assign_array(state.cyd.view[:], cyd[isc:iec, :, :])

        pace.util.utils.safe_assign_array(state.ps.view[:], ps[isc:iec, jsc:jec])
        pace.util.utils.safe_assign_array(
            state.pe.data[isc - 1 : iec + 1, jsc - 1 : jec + 1, :], pe
        )
        pace.util.utils.safe_assign_array(state.pk.view[:], pk)
        pace.util.utils.safe_assign_array(state.peln.view[:], peln)
        pace.util.utils.safe_assign_array(state.pkz.view[:], pkz)
        pace.util.utils.safe_assign_array(state.phis.view[:], phis[isc:iec, jsc:jec])
        pace.util.utils.safe_assign_array(
            state.q_con.view[:], q_con[isc:iec, jsc:jec, :]
        )
        pace.util.utils.safe_assign_array(state.omga.view[:], omga[isc:iec, jsc:jec, :])
        pace.util.utils.safe_assign_array(
            state.diss_estd.view[:], diss_estd[isc:iec, jsc:jec, :]
        )

        # tracer quantities should be a 4d array in order:
        # vapor, liquid, ice, rain, snow, graupel, cloud
        pace.util.utils.safe_assign_array(
            state.qvapor.view[:], q[isc:iec, jsc:jec, :, 0]
        )
        pace.util.utils.safe_assign_array(
            state.qliquid.view[:], q[isc:iec, jsc:jec, :, 1]
        )
        pace.util.utils.safe_assign_array(state.qice.view[:], q[isc:iec, jsc:jec, :, 2])
        pace.util.utils.safe_assign_array(
            state.qrain.view[:], q[isc:iec, jsc:jec, :, 3]
        )
        pace.util.utils.safe_assign_array(
            state.qsnow.view[:], q[isc:iec, jsc:jec, :, 4]
        )
        pace.util.utils.safe_assign_array(
            state.qgraupel.view[:], q[isc:iec, jsc:jec, :, 5]
        )
        pace.util.utils.safe_assign_array(state.qcld.view[:], q[isc:iec, jsc:jec, :, 6])

        return state

    def _prep_outputs_for_geos(self) -> Dict[str, np.ndarray]:

        output_dict = self.output_dict
        isc = self._grid_indexing.isc
        jsc = self._grid_indexing.jsc
        iec = self._grid_indexing.iec + 1
        jec = self._grid_indexing.jec + 1

        pace.util.utils.safe_assign_array(
            output_dict["u"], self.dycore_state.u.data[:-1, :, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["v"], self.dycore_state.v.data[:, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["w"], self.dycore_state.w.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["ua"], self.dycore_state.ua.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["va"], self.dycore_state.va.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["uc"], self.dycore_state.uc.data[:, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["vc"], self.dycore_state.vc.data[:-1, :, :-1]
        )

        pace.util.utils.safe_assign_array(
            output_dict["delz"], self.dycore_state.delz.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["pt"], self.dycore_state.pt.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["delp"], self.dycore_state.delp.data[:-1, :-1, :-1]
        )

        pace.util.utils.safe_assign_array(
            output_dict["mfxd"],
            self.dycore_state.mfxd.data[isc : iec + 1, jsc:jec, :-1],
        )
        pace.util.utils.safe_assign_array(
            output_dict["mfyd"],
            self.dycore_state.mfyd.data[isc:iec, jsc : jec + 1, :-1],
        )
        pace.util.utils.safe_assign_array(
            output_dict["cxd"], self.dycore_state.cxd.data[isc : iec + 1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["cyd"], self.dycore_state.cyd.data[:-1, jsc : jec + 1, :-1]
        )

        pace.util.utils.safe_assign_array(
            output_dict["ps"], self.dycore_state.ps.data[:-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["pe"],
            self.dycore_state.pe.data[isc - 1 : iec + 1, jsc - 1 : jec + 1, :],
        )
        pace.util.utils.safe_assign_array(
            output_dict["pk"], self.dycore_state.pk.data[isc:iec, jsc:jec, :]
        )
        pace.util.utils.safe_assign_array(
            output_dict["peln"], self.dycore_state.peln.data[isc:iec, jsc:jec, :]
        )
        pace.util.utils.safe_assign_array(
            output_dict["pkz"], self.dycore_state.pkz.data[isc:iec, jsc:jec, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["phis"], self.dycore_state.phis.data[:-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["q_con"], self.dycore_state.q_con.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["omga"], self.dycore_state.omga.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["diss_estd"], self.dycore_state.diss_estd.data[:-1, :-1, :-1]
        )

        pace.util.utils.safe_assign_array(
            output_dict["qvapor"], self.dycore_state.qvapor.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["qliquid"], self.dycore_state.qliquid.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["qice"], self.dycore_state.qice.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["qrain"], self.dycore_state.qrain.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["qsnow"], self.dycore_state.qsnow.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["qgraupel"], self.dycore_state.qgraupel.data[:-1, :-1, :-1]
        )
        pace.util.utils.safe_assign_array(
            output_dict["qcld"], self.dycore_state.qcld.data[:-1, :-1, :-1]
        )

        return output_dict

    def _allocate_output_dir(self):

        nhalo = self._grid_indexing.n_halo
        shape_centered = self._grid_indexing.domain_full(add=(0, 0, 0))
        shape_x_interface = self._grid_indexing.domain_full(add=(1, 0, 0))
        shape_y_interface = self._grid_indexing.domain_full(add=(0, 1, 0))
        shape_z_interface = self._grid_indexing.domain_full(add=(0, 0, 1))
        shape_2d = shape_centered[:-1]

        self.output_dict["u"] = np.empty((shape_y_interface))
        self.output_dict["v"] = np.empty((shape_x_interface))
        self.output_dict["w"] = np.empty((shape_centered))
        self.output_dict["ua"] = np.empty((shape_centered))
        self.output_dict["va"] = np.empty((shape_centered))
        self.output_dict["uc"] = np.empty((shape_x_interface))
        self.output_dict["vc"] = np.empty((shape_y_interface))

        self.output_dict["delz"] = np.empty((shape_centered))
        self.output_dict["pt"] = np.empty((shape_centered))
        self.output_dict["delp"] = np.empty((shape_centered))

        self.output_dict["mfxd"] = np.empty(
            (self._grid_indexing.domain_full(add=(1 - 2 * nhalo, -2 * nhalo, 0)))
        )
        self.output_dict["mfyd"] = np.empty(
            (self._grid_indexing.domain_full(add=(-2 * nhalo, 1 - 2 * nhalo, 0)))
        )
        self.output_dict["cxd"] = np.empty(
            (self._grid_indexing.domain_full(add=(1 - 2 * nhalo, 0, 0)))
        )
        self.output_dict["cyd"] = np.empty(
            (self._grid_indexing.domain_full(add=(0, 1 - 2 * nhalo, 0)))
        )

        self.output_dict["ps"] = np.empty((shape_2d))
        self.output_dict["pe"] = np.empty(
            (self._grid_indexing.domain_full(add=(2 - 2 * nhalo, 2 - 2 * nhalo, 1)))
        )
        self.output_dict["pk"] = np.empty(
            (self._grid_indexing.domain_full(add=(-2 * nhalo, -2 * nhalo, 1)))
        )
        self.output_dict["peln"] = np.empty(
            (self._grid_indexing.domain_full(add=(-2 * nhalo, -2 * nhalo, 1)))
        )
        self.output_dict["pkz"] = np.empty(
            (self._grid_indexing.domain_full(add=(-2 * nhalo, -2 * nhalo, 0)))
        )
        self.output_dict["phis"] = np.empty((shape_2d))
        self.output_dict["q_con"] = np.empty((shape_centered))
        self.output_dict["omga"] = np.empty((shape_centered))
        self.output_dict["diss_estd"] = np.empty((shape_centered))

        self.output_dict["qvapor"] = np.empty((shape_centered))
        self.output_dict["qliquid"] = np.empty((shape_centered))
        self.output_dict["qice"] = np.empty((shape_centered))
        self.output_dict["qrain"] = np.empty((shape_centered))
        self.output_dict["qsnow"] = np.empty((shape_centered))
        self.output_dict["qgraupel"] = np.empty((shape_centered))
        self.output_dict["qcld"] = np.empty((shape_centered))
