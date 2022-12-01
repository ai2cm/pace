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

        self.output_dictionary: Dict[str, np.ndarray] = {}

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
    ) -> dict:

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

        self._prep_outputs_for_geos()

        return self.output_dictionary

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

    def _prep_outputs_for_geos(self):
        self.output_dictionary["u"] = self.dycore_state.u.data[:]
        self.output_dictionary["v"] = self.dycore_state.v.data[:]
        self.output_dictionary["w"] = self.dycore_state.w.data[:]
        self.output_dictionary["ua"] = self.dycore_state.ua.data[:]
        self.output_dictionary["va"] = self.dycore_state.va.data[:]
        self.output_dictionary["uc"] = self.dycore_state.uc.data[:]
        self.output_dictionary["vc"] = self.dycore_state.vc.data[:]

        self.output_dictionary["delc"] = self.dycore_state.delz.data[:]
        self.output_dictionary["pt"] = self.dycore_state.pt.data[:]
        self.output_dictionary["delp"] = self.dycore_state.delp.data[:]

        self.output_dictionary["mfxd"] = self.dycore_state.mfxd.data[:]
        self.output_dictionary["mfyd"] = self.dycore_state.mfyd.data[:]
        self.output_dictionary["cxd"] = self.dycore_state.cxd.data[:]
        self.output_dictionary["cyd"] = self.dycore_state.cyd.data[:]

        self.output_dictionary["ps"] = self.dycore_state.ps.data[:]
        self.output_dictionary["pe"] = self.dycore_state.pe.data[:]
        self.output_dictionary["pk"] = self.dycore_state.pk.data[:]
        self.output_dictionary["peln"] = self.dycore_state.peln.data[:]
        self.output_dictionary["pkz"] = self.dycore_state.pkz.data[:]
        self.output_dictionary["phis"] = self.dycore_state.phis.data[:]
        self.output_dictionary["q_con"] = self.dycore_state.q_con.data[:]
        self.output_dictionary["omga"] = self.dycore_state.omga.data[:]
        self.output_dictionary["diss_estd"] = self.dycore_state.diss_estd.data[:]

        self.output_dictionary["qvapor"] = self.dycore_state.qvapor.data[:]
        self.output_dictionary["qliquid"] = self.dycore_state.qliquid.data[:]
        self.output_dictionary["qice"] = self.dycore_state.qice.data[:]
        self.output_dictionary["qrain"] = self.dycore_state.qrain.data[:]
        self.output_dictionary["qsnow"] = self.dycore_state.qsnow.data[:]
        self.output_dictionary["qgraupel"] = self.dycore_state.qgraupel.data[:]
        self.output_dictionary["qcld"] = self.dycore_state.qcld.data[:]
