#!/usr/bin/env python3
import math

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.fvtp2d as fvtp2d
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy, copy_stencil
from fv3core.stencils.updatedzd import ra_x_stencil, ra_y_stencil


sd = utils.sd


@gtstencil()
def flux_x(cx: sd, dxa: sd, dy: sd, sin_sg3: sd, sin_sg1: sd, xfx: sd):
    with computation(PARALLEL), interval(...):
        xfx[0, 0, 0] = (
            cx * dxa[-1, 0, 0] * dy * sin_sg3[-1, 0, 0]
            if cx > 0
            else cx * dxa * dy * sin_sg1
        )


@gtstencil()
def flux_y(cy: sd, dya: sd, dx: sd, sin_sg4: sd, sin_sg2: sd, yfx: sd):
    with computation(PARALLEL), interval(...):
        yfx[0, 0, 0] = (
            cy * dya[0, -1, 0] * dx * sin_sg4[0, -1, 0]
            if cy > 0
            else cy * dya * dx * sin_sg2
        )


@gtscript.function
def mult_frac(var, frac):
    var_tmp = var
    return var_tmp * frac


@gtstencil()
def cmax_split_vars(
    cxd: sd, xfx: sd, mfxd: sd, cyd: sd, yfx: sd, mfyd: sd, frac: float
):
    with computation(PARALLEL), interval(...):
        cxd = mult_frac(cxd, frac)
        xfx = mult_frac(xfx, frac)
        mfxd = mult_frac(mfxd, frac)
        cyd = mult_frac(cyd, frac)
        yfx = mult_frac(yfx, frac)
        mfyd = mult_frac(mfyd, frac)


@gtstencil()
def cmax_stencil1(cx: sd, cy: sd, cmax: sd):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy))


@gtstencil()
def cmax_stencil2(cx: sd, cy: sd, sin_sg5: sd, cmax: sd):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy)) + 1.0 - sin_sg5


@gtstencil()
def dp_fluxadjustment(dp1: sd, mfx: sd, mfy: sd, rarea: sd, dp2: sd):
    with computation(PARALLEL), interval(...):
        dp2 = dp1 + (mfx - mfx[1, 0, 0] + mfy - mfy[0, 1, 0]) * rarea


@gtscript.function
def adjustment(q, dp1, fx, fy, rarea, dp2):
    return (q * dp1 + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / dp2


@gtstencil()
def q_adjust(q: sd, dp1: sd, fx: sd, fy: sd, rarea: sd, dp2: sd):
    with computation(PARALLEL), interval(...):
        q = adjustment(q, dp1, fx, fy, rarea, dp2)


@gtstencil()
def q_other_adjust(q: sd, qset: sd, dp1: sd, fx: sd, fy: sd, rarea: sd, dp2: sd):
    with computation(PARALLEL), interval(...):
        qset = adjustment(q, dp1, fx, fy, rarea, dp2)


def compute(comm, tracers, dp1, mfxd, mfyd, cxd, cyd, mdt, nq):
    grid = spec.grid
    shape = mfxd.data.shape
    # start HALO update on q (in dyn_core in fortran -- just has started when this function is called...)
    xfx = utils.make_storage_from_shape(shape, origin=grid.compute_x_origin())
    yfx = utils.make_storage_from_shape(shape, origin=grid.compute_y_origin())
    fx = utils.make_storage_from_shape(shape, origin=grid.compute_origin())
    fy = utils.make_storage_from_shape(shape, origin=grid.compute_origin())
    ra_x = utils.make_storage_from_shape(shape, origin=grid.compute_x_origin())
    ra_y = utils.make_storage_from_shape(shape, origin=grid.compute_y_origin())
    cmax = utils.make_storage_from_shape(shape, origin=grid.compute_origin())
    dp2 = utils.make_storage_from_shape(shape, origin=grid.compute_origin())
    flux_x(
        cxd,
        grid.dxa,
        grid.dy,
        grid.sin_sg3,
        grid.sin_sg1,
        xfx,
        origin=grid.compute_x_origin(),
        domain=grid.domain_y_compute_xbuffer(),
    )
    flux_y(
        cyd,
        grid.dya,
        grid.dx,
        grid.sin_sg4,
        grid.sin_sg2,
        yfx,
        origin=grid.compute_y_origin(),
        domain=grid.domain_x_compute_ybuffer(),
    )
    """
    # TODO for if we end up using the Allreduce and compute cmax globally (or locally). For now, hardcoded
    split = int(grid.npz / 6)
    cmax_stencil1(
        cxd, cyd, cmax, origin=grid.compute_origin(), domain=(grid.nic, grid.njc, split)
    )
    cmax_stencil2(
        cxd,
        cyd,
        grid.sin_sg5,
        cmax,
        origin=(grid.is_, grid.js, split),
        domain=(grid.nic, grid.njc, grid.npz - split + 1),
    )
    cmax_flat = np.amax(cmax, axis=(0, 1))
    # cmax_flat is a gt4py storage still, but of dimension [npz+1]...

    cmax_max_all_ranks = cmax_flat.data
    # TODO mpi allreduce.... can we not?
    # comm.Allreduce(cmax_flat, cmax_max_all_ranks, op=MPI.MAX)
    """
    cmax_max_all_ranks = 2.0
    nsplt = math.floor(1.0 + cmax_max_all_ranks)
    # NOTE cmax is not usually a single value, it varies with k, if return to that, make nsplt a column as well and compute frac inside cmax_split_vars
    # nsplt3d = utils.make_storage_from_shape(cyd.shape, origin=grid.compute_origin())
    # nsplt3d[:] = nsplt
    frac = 1.0
    if nsplt > 1.0:
        frac = 1.0 / nsplt
    cmax_split_vars(
        cxd,
        xfx,
        mfxd,
        cyd,
        yfx,
        mfyd,
        frac,
        origin=grid.default_origin(),
        domain=grid.domain_shape_buffer_1cell(),
    )

    # complete HALO update on q
    for qname in utils.tracer_variables[0:nq]:
        q = tracers[qname + "_quantity"]
        comm.halo_update(q, n_points=utils.halo)

    ra_x_stencil(
        grid.area,
        xfx,
        ra_x,
        origin=grid.compute_x_origin(),
        domain=grid.domain_y_compute_x(),
    )
    ra_y_stencil(
        grid.area,
        yfx,
        ra_y,
        origin=grid.compute_y_origin(),
        domain=grid.domain_x_compute_y(),
    )

    # TODO revisit: the loops over q and nsplt have two inefficient options duplicating storages/stencil calls,
    # return to this, maybe you have more options now, or maybe the one chosen here is the worse one

    dp1_orig = copy(
        dp1, origin=grid.default_origin(), domain=grid.domain_shape_standard()
    )
    for qname in utils.tracer_variables[0:nq]:
        q = tracers[qname + "_quantity"]
        # handling the q and it loop switching
        copy_stencil(
            dp1_orig,
            dp1,
            origin=grid.default_origin(),
            domain=grid.domain_shape_standard(),
        )
        for it in range(int(nsplt)):
            dp_fluxadjustment(
                dp1,
                mfxd,
                mfyd,
                grid.rarea,
                dp2,
                origin=grid.compute_origin(),
                domain=grid.domain_shape_compute(),
            )
            if nsplt != 1:
                if it == 0:
                    # TODO 1d
                    qn2 = grid.quantity_wrap(
                        copy(
                            q.storage,
                            origin=grid.default_origin(),
                            domain=grid.domain_shape_standard(),
                        ),
                        units="kg/m^2",
                    )

                fvtp2d.compute_no_sg(
                    qn2.storage,
                    cxd,
                    cyd,
                    spec.namelist.hord_tr,
                    xfx,
                    yfx,
                    ra_x,
                    ra_y,
                    fx,
                    fy,
                    mfx=mfxd,
                    mfy=mfyd,
                )
                if it < nsplt - 1:
                    q_adjust(
                        qn2.storage,
                        dp1,
                        fx,
                        fy,
                        grid.rarea,
                        dp2,
                        origin=grid.compute_origin(),
                        domain=grid.domain_shape_compute(),
                    )
                else:
                    q_other_adjust(
                        qn2.storage,
                        q.storage,
                        dp1,
                        fx,
                        fy,
                        grid.rarea,
                        dp2,
                        origin=grid.compute_origin(),
                        domain=grid.domain_shape_compute(),
                    )
            else:
                fvtp2d.compute_no_sg(
                    q.storage,
                    cxd,
                    cyd,
                    spec.namelist.hord_tr,
                    xfx,
                    yfx,
                    ra_x,
                    ra_y,
                    fx,
                    fy,
                    mfx=mfxd,
                    mfy=mfyd,
                )
                q_adjust(
                    q.storage,
                    dp1,
                    fx,
                    fy,
                    grid.rarea,
                    dp2,
                    origin=grid.compute_origin(),
                    domain=grid.domain_shape_compute(),
                )

            if it < nsplt - 1:
                copy_stencil(
                    dp2,
                    dp1,
                    origin=grid.compute_origin(),
                    domain=grid.domain_shape_compute(),
                )
                comm.halo_update(qn2, n_points=utils.halo)
