import math

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.stencils.fvtp2d as fvtp2d
import fv3core.utils
import fv3core.utils.global_config as global_config
import fv3core.utils.gt4py_utils as utils
from fv3core.stencils.basic_operations import copy_stencil
from fv3core.stencils.updatedzd import ra_stencil_update
from fv3core.utils.typing import FloatField, FloatFieldIJ


@gtscript.function
def flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is : local_ie + 2, local_js - 3 : local_je + 4]):
        xfx = (
            cx * dxa[-1, 0] * dy * sin_sg3[-1, 0] if cx > 0 else cx * dxa * dy * sin_sg1
        )
    return xfx


@gtscript.function
def flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is - 3 : local_ie + 4, local_js : local_je + 2]):
        yfx = (
            cy * dya[0, -1] * dx * sin_sg4[0, -1] if cy > 0 else cy * dya * dx * sin_sg2
        )
    return yfx


def flux_compute(
    cx: FloatField,
    cy: FloatField,
    dxa: FloatFieldIJ,
    dya: FloatFieldIJ,
    dx: FloatFieldIJ,
    dy: FloatFieldIJ,
    sin_sg1: FloatFieldIJ,
    sin_sg2: FloatFieldIJ,
    sin_sg3: FloatFieldIJ,
    sin_sg4: FloatFieldIJ,
    xfx: FloatField,
    yfx: FloatField,
):
    with computation(PARALLEL), interval(...):
        xfx = flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx)
        yfx = flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx)


def cmax_multiply_by_frac(
    cxd: FloatField,
    xfx: FloatField,
    mfxd: FloatField,
    cyd: FloatField,
    yfx: FloatField,
    mfyd: FloatField,
    nsplt: int,
):
    """multiply all other inputs in-place by frac."""
    with computation(PARALLEL), interval(...):
        frac = 1.0 / nsplt
        cxd = cxd * frac
        xfx = xfx * frac
        mfxd = mfxd * frac
        cyd = cyd * frac
        yfx = yfx * frac
        mfyd = mfyd * frac


def cmax_stencil1(cx: FloatField, cy: FloatField, cmax: FloatField):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy))


def cmax_stencil2(
    cx: FloatField, cy: FloatField, sin_sg5: FloatField, cmax: FloatField
):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy)) + 1.0 - sin_sg5


def dp_fluxadjustment(
    dp1: FloatField,
    mfx: FloatField,
    mfy: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
):
    with computation(PARALLEL), interval(...):
        dp2 = dp1 + (mfx - mfx[1, 0, 0] + mfy - mfy[0, 1, 0]) * rarea


def loop_temporaries_copy(
    tmp_dp1_orig: FloatField,
    q: FloatField,
    dp1: FloatField,
    tmp_qn2: FloatField,
):
    with computation(PARALLEL), interval(...):
        dp1 = tmp_dp1_orig
        tmp_qn2 = q


@gtscript.function
def adjustment(q, dp1, fx, fy, rarea, dp2):
    return (q * dp1 + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / dp2


def q_adjust(
    q: FloatField,
    dp1: FloatField,
    fx: FloatField,
    fy: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
):
    with computation(PARALLEL), interval(...):
        q = adjustment(q, dp1, fx, fy, rarea, dp2)


def q_adjustments(
    q: FloatField,
    qset: FloatField,
    dp1: FloatField,
    fx: FloatField,
    fy: FloatField,
    rarea: FloatFieldIJ,
    dp2: FloatField,
    it: int,
    nsplt: int,
):
    with computation(PARALLEL), interval(...):
        if it < nsplt - 1:
            q = adjustment(q, dp1, fx, fy, rarea, dp2)
        else:
            qset = adjustment(q, dp1, fx, fy, rarea, dp2)


class Tracer2D1L:
    def __init__(self, namelist):
        self.grid = spec.grid
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        origin = self.grid.compute_origin()
        self._tmp_xfx = utils.make_storage_from_shape(shape, origin)
        self._tmp_yfx = utils.make_storage_from_shape(shape, origin)
        self._tmp_ra_x = utils.make_storage_from_shape(shape, origin)
        self._tmp_ra_y = utils.make_storage_from_shape(shape, origin)
        self._tmp_fx = utils.make_storage_from_shape(shape, origin)
        self._tmp_fy = utils.make_storage_from_shape(shape, origin)
        self._tmp_dp2 = utils.make_storage_from_shape(shape, origin)
        self._tmp_dp1_orig = utils.make_storage_from_shape(shape, origin)
        self._tmp_qn2 = self.grid.quantity_wrap(
            utils.make_storage_from_shape(shape, origin),
            units="kg/m^2",
        )
        stencil_kwargs = {
            "backend": global_config.get_backend(),
            "rebuild": global_config.get_rebuild(),
            "externals": fv3core.utils.axis_offsets(
                self.grid, self.grid.full_origin(), self.grid.domain_shape_full()
            ),
        }
        stencil_wrapper = gtscript.stencil(**stencil_kwargs)
        self.stencil_flux_compute = stencil_wrapper(flux_compute)
        self.stencil_ra_update = stencil_wrapper(ra_stencil_update.func)
        self.stencil_cmax_multiply_by_frac = stencil_wrapper(cmax_multiply_by_frac)
        self.stencil_copy_field = stencil_wrapper(copy_stencil.func)
        self.stencil_loop_temporaries_copy = stencil_wrapper(loop_temporaries_copy)
        self.stencil_dp_fluxadjustment = stencil_wrapper(dp_fluxadjustment)
        self.stencil_q_adjustments = stencil_wrapper(q_adjustments)
        self.stencil_q_adjust = stencil_wrapper(q_adjust)
        self.fvtp2d_obj = fvtp2d.FvTp2d(spec.namelist, spec.namelist.hord_tr)
        # If use AllReduce, will need something like this:
        # self._tmp_cmax = utils.make_storage_from_shape(shape, origin)
        # self.stencil_cmax_1 = stencil_wrapper(cmax_stencil1)
        # self.stencil_max_2 = stencil_wrapper(cmax_stencil2)

    def __call__(self, comm, tracers, dp1, mfxd, mfyd, cxd, cyd, mdt, nq):
        grid = self.grid
        # start HALO update on q (in dyn_core in fortran -- just has started when
        # this function is called...)
        self.stencil_flux_compute(
            cxd,
            cyd,
            grid.dxa,
            grid.dya,
            grid.dx,
            grid.dy,
            grid.sin_sg1,
            grid.sin_sg2,
            grid.sin_sg3,
            grid.sin_sg4,
            self._tmp_xfx,
            self._tmp_yfx,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(add=(1, 1, 0)),
        )

        # # TODO for if we end up using the Allreduce and compute cmax globally
        # (or locally). For now, hardcoded.
        # split = int(grid.npz / 6)
        # self.stencil_cmax_1(
        #     cxd, cyd, self._tmp_cmax, origin=grid.compute_origin(),
        #     domain=(grid.nic, grid.njc, split)
        # )
        # self.stencil_cmax_2(
        #     cxd,
        #     cyd,
        #     grid.sin_sg5,
        #     self._tmp_cmax,
        #     origin=(grid.is_, grid.js, split),
        #     domain=(grid.nic, grid.njc, grid.npz - split + 1),
        # )
        # cmax_flat = np.amax(self._tmp_cmax, axis=(0, 1))
        # # cmax_flat is a gt4py storage still, but of dimension [npz+1]...

        # cmax_max_all_ranks = cmax_flat.data
        # # TODO mpi allreduce...
        # # comm.Allreduce(cmax_flat, cmax_max_all_ranks, op=MPI.MAX)

        cmax_max_all_ranks = 2.0
        nsplt = math.floor(1.0 + cmax_max_all_ranks)
        # NOTE: cmax is not usually a single value, it varies with k, if return to
        # that, make nsplt a column as well

        if nsplt > 1.0:
            self.stencil_cmax_multiply_by_frac(
                cxd,
                self._tmp_xfx,
                mfxd,
                cyd,
                self._tmp_yfx,
                mfyd,
                nsplt,
                origin=grid.full_origin(),
                domain=grid.domain_shape_full(add=(1, 1, 0)),
            )

        if global_config.get_do_halo_exchange():
            for qname in utils.tracer_variables[0:nq]:
                q = tracers[qname + "_quantity"]
                comm.halo_update(q, n_points=utils.halo)

        self.stencil_ra_update(
            grid.area,
            self._tmp_xfx,
            self._tmp_ra_x,
            self._tmp_yfx,
            self._tmp_ra_y,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(),
        )
        # TODO: Revisit: the loops over q and nsplt have two inefficient options
        # duplicating storages/stencil calls, return to this, maybe you have more
        # options now, or maybe the one chosen here is the worse one.

        self.stencil_copy_field(
            dp1,
            self._tmp_dp1_orig,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(),
        )
        for qname in utils.tracer_variables[0:nq]:
            q = tracers[qname + "_quantity"]
            self.stencil_loop_temporaries_copy(
                self._tmp_dp1_orig,
                q.storage,
                dp1,
                self._tmp_qn2.storage,
                origin=grid.full_origin(),
                domain=grid.domain_shape_full(),
            )
            for it in range(int(nsplt)):
                self.stencil_dp_fluxadjustment(
                    dp1,
                    mfxd,
                    mfyd,
                    grid.rarea,
                    self._tmp_dp2,
                    origin=grid.compute_origin(),
                    domain=grid.domain_shape_compute(),
                )
                if nsplt != 1:
                    self.fvtp2d_obj(
                        self._tmp_qn2.storage,
                        cxd,
                        cyd,
                        self._tmp_xfx,
                        self._tmp_yfx,
                        self._tmp_ra_x,
                        self._tmp_ra_y,
                        self._tmp_fx,
                        self._tmp_fy,
                        mfx=mfxd,
                        mfy=mfyd,
                    )

                    self.stencil_q_adjustments(
                        self._tmp_qn2.storage,
                        q.storage,
                        dp1,
                        self._tmp_fx,
                        self._tmp_fy,
                        grid.rarea,
                        self._tmp_dp2,
                        it,
                        nsplt,
                        origin=grid.compute_origin(),
                        domain=grid.domain_shape_compute(),
                    )
                else:
                    self.fvtp2d_obj(
                        q.storage,
                        cxd,
                        cyd,
                        self._tmp_xfx,
                        self._tmp_yfx,
                        self._tmp_ra_x,
                        self._tmp_ra_y,
                        self._tmp_fx,
                        self._tmp_fy,
                        mfx=mfxd,
                        mfy=mfyd,
                    )
                    self.stencil_q_adjust(
                        q.storage,
                        dp1,
                        self._tmp_fx,
                        self._tmp_fy,
                        grid.rarea,
                        self._tmp_dp2,
                        origin=grid.compute_origin(),
                        domain=grid.domain_shape_compute(),
                    )

                if it < nsplt - 1:
                    self.stencil_copy_field(
                        self._tmp_dp2,
                        dp1,
                        origin=grid.compute_origin(),
                        domain=grid.domain_shape_compute(),
                    )
                    if global_config.get_do_halo_exchange():
                        comm.halo_update(self._tmp_qn2, n_points=utils.halo)
