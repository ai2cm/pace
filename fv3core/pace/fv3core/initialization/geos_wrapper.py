from datetime import timedelta

import f90nml
import numpy as np

import pace.util
from pace import fv3core


class GeosDycoreWrapper:
    def __init__(self, namelist: f90nml.Namelist, comm: pace.util.Comm, backend: str):
        self.namelist = namelist

        self.dycore_config = fv3core.DynamicalCoreConfig.from_f90nml(self.namelist)

        self.layout = self.dycore_config.layout
        partitioner = pace.util.CubedSpherePartitioner(
            pace.util.TilePartitioner(self.layout)
        )
        self.communicator = pace.util.CubedSphereCommunicator(comm, partitioner)

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

    def __call__(
        self,
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
    ):

        self._put_fortran_data_in_dycore(
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

        self.dynamical_core.step_dynamics(
            state=self.dycore_state,
        )
        return self._outputs_for_geos()

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
    ):
        isc = self._grid_indexing.isc
        jsc = self._grid_indexing.jsc
        iec = self._grid_indexing.iec + 1
        jec = self._grid_indexing.jec + 1

        # Assign compute domain:
        self.dycore_state.u.view[:] = u[isc:iec, jsc : jec + 1, :-1]
        self.dycore_state.v.view[:] = v[isc : iec + 1, jsc:jec, :-1]
        self.dycore_state.w.view[:] = w[isc:iec, jsc:jec, :-1]
        self.dycore_state.ua.view[:] = ua[isc:iec, jsc:jec, :-1]
        self.dycore_state.va.view[:] = va[isc:iec, jsc:jec, :-1]
        self.dycore_state.uc.view[:] = uc[isc : iec + 1, jsc:jec, :-1]
        self.dycore_state.vc.view[:] = vc[isc:iec, jsc : jec + 1, :-1]

        self.dycore_state.delz.view[:] = delz[isc:iec, jsc:jec, :-1]
        self.dycore_state.pt.view[:] = pt[isc:iec, jsc:jec, :-1]
        self.dycore_state.delp.view[:] = delp[isc:iec, jsc:jec, :-1]

        self.dycore_state.mfxd.view[:] = mfxd[isc : iec + 1, jsc:jec, :-1]
        self.dycore_state.mfyd.view[:] = mfyd[isc:iec, jsc : jec + 1, :-1]
        self.dycore_state.cxd.view[:] = cxd[isc : iec + 1, jsc:jec, :-1]
        self.dycore_state.cyd.view[:] = cyd[isc:iec, jsc : jec + 1, :-1]

        self.dycore_state.ps.view[:] = ps[isc:iec, jsc:jec]
        self.dycore_state.pe.view[:] = pe[isc:iec, jsc:jec, :]
        self.dycore_state.pk.view[:] = pk[isc:iec, jsc:jec, :]
        self.dycore_state.peln.view[:] = peln[isc:iec, jsc:jec, :]
        self.dycore_state.pkz.view[:] = pkz[isc:iec, jsc:jec, :-1]
        self.dycore_state.phis.view[:] = phis[isc:iec, jsc:jec]
        self.dycore_state.q_con.view[:] = q_con[isc:iec, jsc:jec, :-1]
        self.dycore_state.omga.view[:] = omga[isc:iec, jsc:jec, :-1]
        self.dycore_state.diss_estd.view[:] = diss_estd[isc:iec, jsc:jec, :-1]

        # tracer quantities should be a 4d array in order:
        # vapor, liquid, ice, rain, snow, graupel, cloud
        self.dycore_state.qvapor.view[:] = q[isc:iec, jsc:jec, :-1, 0]
        self.dycore_state.qliquid.view[:] = q[isc:iec, jsc:jec, :-1, 1]
        self.dycore_state.qice.view[:] = q[isc:iec, jsc:jec, :-1, 2]
        self.dycore_state.qrain.view[:] = q[isc:iec, jsc:jec, :-1, 3]
        self.dycore_state.qsnow.view[:] = q[isc:iec, jsc:jec, :-1, 4]
        self.dycore_state.qgraupel.view[:] = q[isc:iec, jsc:jec, :-1, 5]
        if self.namelist["dycore_config"]["nwat"] > 6:
            self.dycore_state.qcld.view[:] = q[isc:iec, jsc:jec, :-1, 6]

    def _outputs_for_geos(self):
        out_state = {}
        out_state["u"] = self.dycore_state.u.data[:]
        out_state["v"] = self.dycore_state.v.data[:]
        out_state["w"] = self.dycore_state.w.data[:]
        out_state["ua"] = self.dycore_state.ua.data[:]
        out_state["va"] = self.dycore_state.va.data[:]
        out_state["uc"] = self.dycore_state.uc.data[:]
        out_state["vc"] = self.dycore_state.vc.data[:]

        out_state["delc"] = self.dycore_state.delz.data[:]
        out_state["pt"] = self.dycore_state.pt.data[:]
        out_state["delp"] = self.dycore_state.delp.data[:]

        out_state["mfxd"] = self.dycore_state.mfxd.data[:]
        out_state["mfyd"] = self.dycore_state.mfyd.data[:]
        out_state["cxd"] = self.dycore_state.cxd.data[:]
        out_state["cyd"] = self.dycore_state.cyd.data[:]

        out_state["ps"] = self.dycore_state.ps.data[:]
        out_state["pe"] = self.dycore_state.pe.data[:]
        out_state["pk"] = self.dycore_state.pk.data[:]
        out_state["peln"] = self.dycore_state.peln.data[:]
        out_state["pkz"] = self.dycore_state.pkz.data[:]
        out_state["phis"] = self.dycore_state.phis.data[:]
        out_state["q_con"] = self.dycore_state.q_con.data[:]
        out_state["omga"] = self.dycore_state.omga.data[:]
        out_state["diss_estd"] = self.dycore_state.diss_estd.data[:]

        out_state["qvapor"] = self.dycore_state.qvapor.data[:]
        out_state["qliquid"] = self.dycore_state.qliquid.data[:]
        out_state["qice"] = self.dycore_state.qice.data[:]
        out_state["qrain"] = self.dycore_state.qrain.data[:]
        out_state["qsnow"] = self.dycore_state.qsnow.data[:]
        out_state["qgraupel"] = self.dycore_state.qgraupel.data[:]
        out_state["qcld"] = self.dycore_state.qcld.data[:]

        return out_state
