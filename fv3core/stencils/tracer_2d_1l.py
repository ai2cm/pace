import math

import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.stencils.fvtp2d as fvtp2d
import fv3core.utils.global_config as global_config
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy, copy_stencil
from fv3core.stencils.updatedzd import ra_stencil_update
from fv3core.utils.typing import FloatField


@gtscript.function
def flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is : local_ie + 2, local_js - 3 : local_je + 4]):
        xfx = (
            cx * dxa[-1, 0, 0] * dy * sin_sg3[-1, 0, 0]
            if cx > 0
            else cx * dxa * dy * sin_sg1
        )
    return xfx


@gtscript.function
def flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx):
    from __externals__ import local_ie, local_is, local_je, local_js

    with horizontal(region[local_is - 3 : local_ie + 4, local_js : local_je + 2]):
        yfx = (
            cy * dya[0, -1, 0] * dx * sin_sg4[0, -1, 0]
            if cy > 0
            else cy * dya * dx * sin_sg2
        )
    return yfx


@gtstencil()
def flux_compute(
    cx: FloatField,
    cy: FloatField,
    dxa: FloatField,
    dya: FloatField,
    dx: FloatField,
    dy: FloatField,
    sin_sg1: FloatField,
    sin_sg2: FloatField,
    sin_sg3: FloatField,
    sin_sg4: FloatField,
    xfx: FloatField,
    yfx: FloatField,
):
    with computation(PARALLEL), interval(...):
        xfx = flux_x(cx, dxa, dy, sin_sg3, sin_sg1, xfx)
        yfx = flux_y(cy, dya, dx, sin_sg4, sin_sg2, yfx)


@gtstencil()
def cmax_multiply_by_frac(
    cxd: FloatField,
    xfx: FloatField,
    mfxd: FloatField,
    cyd: FloatField,
    yfx: FloatField,
    mfyd: FloatField,
    frac: float,
):
    """multiply all other inputs in-place by frac."""
    with computation(PARALLEL), interval(...):
        cxd = cxd * frac
        xfx = xfx * frac
        mfxd = mfxd * frac
        cyd = cyd * frac
        yfx = yfx * frac
        mfyd = mfyd * frac


@gtstencil()
def cmax_stencil1(cx: FloatField, cy: FloatField, cmax: FloatField):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy))


@gtstencil()
def cmax_stencil2(
    cx: FloatField, cy: FloatField, sin_sg5: FloatField, cmax: FloatField
):
    with computation(PARALLEL), interval(...):
        cmax = max(abs(cx), abs(cy)) + 1.0 - sin_sg5


@gtstencil()
def dp_fluxadjustment(
    dp1: FloatField,
    mfx: FloatField,
    mfy: FloatField,
    rarea: FloatField,
    dp2: FloatField,
):
    with computation(PARALLEL), interval(...):
        dp2 = dp1 + (mfx - mfx[1, 0, 0] + mfy - mfy[0, 1, 0]) * rarea


@gtscript.function
def adjustment(q, dp1, fx, fy, rarea, dp2):
    return (q * dp1 + (fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) * rarea) / dp2


@gtstencil()
def q_adjust(
    q: FloatField,
    dp1: FloatField,
    fx: FloatField,
    fy: FloatField,
    rarea: FloatField,
    dp2: FloatField,
):
    with computation(PARALLEL), interval(...):
        q = adjustment(q, dp1, fx, fy, rarea, dp2)


@gtstencil()
def q_adjustments(
    q: FloatField,
    qset: FloatField,
    dp1: FloatField,
    fx: FloatField,
    fy: FloatField,
    rarea: FloatField,
    dp2: FloatField,
    it: int,
    nsplt: int,
):
    with computation(PARALLEL), interval(...):
        if it < nsplt - 1:
            q = adjustment(q, dp1, fx, fy, rarea, dp2)
        else:
            qset = adjustment(q, dp1, fx, fy, rarea, dp2)


def compute(comm, tracers, dp1, mfxd, mfyd, cxd, cyd, mdt, nq):
    grid = spec.grid
    shape = mfxd.data.shape
    # start HALO update on q (in dyn_core in fortran -- just has started when
    # this function is called...)
    xfx = utils.make_storage_from_shape(
        shape, origin=grid.compute_origin(add=(0, -grid.halo, 0))
    )
    yfx = utils.make_storage_from_shape(
        shape, origin=grid.compute_origin(add=(-grid.halo, 0, 0))
    )
    fx = utils.make_storage_from_shape(shape, origin=grid.compute_origin())
    fy = utils.make_storage_from_shape(shape, origin=grid.compute_origin())
    ra_x = utils.make_storage_from_shape(
        shape, origin=grid.compute_origin(add=(0, -grid.halo, 0))
    )
    ra_y = utils.make_storage_from_shape(
        shape, origin=grid.compute_origin(add=(-grid.halo, 0, 0))
    )
    cmax = utils.make_storage_from_shape(shape, origin=grid.compute_origin())
    dp2 = utils.make_storage_from_shape(shape, origin=grid.compute_origin())

    flux_compute(
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
        xfx,
        yfx,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(),
    )
    # {
    # # TODO for if we end up using the Allreduce and compute cmax globally
    # (or locally). For now, hardcoded.
    # split = int(grid.npz / 6)
    # cmax_stencil1(
    #     cxd, cyd, cmax, origin=grid.compute_origin(),
    #     domain=(grid.nic, grid.njc, split)
    # )
    # cmax_stencil2(
    #     cxd,
    #     cyd,
    #     grid.sin_sg5,
    #     cmax,
    #     origin=(grid.is_, grid.js, split),
    #     domain=(grid.nic, grid.njc, grid.npz - split + 1),
    # )
    # cmax_flat = np.amax(cmax, axis=(0, 1))
    # # cmax_flat is a gt4py storage still, but of dimension [npz+1]...

    # cmax_max_all_ranks = cmax_flat.data
    # # TODO mpi allreduce.... can we not?
    # # comm.Allreduce(cmax_flat, cmax_max_all_ranks, op=MPI.MAX)
    # }
    cmax_max_all_ranks = 2.0
    nsplt = math.floor(1.0 + cmax_max_all_ranks)
    # NOTE: cmax is not usually a single value, it varies with k, if return to
    # that, make nsplt a column as well and compute frac inside cmax_split_vars.

    # nsplt3d = utils.make_storage_from_shape(cyd.shape, origin=grid.compute_origin())
    # nsplt3d[:] = nsplt
    frac = 1.0
    if nsplt > 1.0:
        frac = 1.0 / nsplt
        cmax_multiply_by_frac(
            cxd,
            xfx,
            mfxd,
            cyd,
            yfx,
            mfyd,
            frac,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(add=(1, 1, 0)),
        )

    # complete HALO update on q
    if global_config.get_do_halo_exchange():
        for qname in utils.tracer_variables[0:nq]:
            q = tracers[qname + "_quantity"]
            comm.halo_update(q, n_points=utils.halo)

    ra_stencil_update(
        grid.area,
        xfx,
        ra_x,
        yfx,
        ra_y,
        origin=grid.full_origin(),
        domain=grid.domain_shape_full(),
    )

    # TODO: Revisit: the loops over q and nsplt have two inefficient options
    # duplicating storages/stencil calls, return to this, maybe you have more
    # options now, or maybe the one chosen here is the worse one.

    dp1_orig = copy(dp1, origin=grid.full_origin(), domain=grid.domain_shape_full())
    for qname in utils.tracer_variables[0:nq]:
        q = tracers[qname + "_quantity"]
        # handling the q and it loop switching
        copy_stencil(
            dp1_orig,
            dp1,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(),
        )
        qn2 = grid.quantity_wrap(
            copy(
                q.storage,
                origin=grid.full_origin(),
                domain=grid.domain_shape_full(),
            ),
            units="kg/m^2",
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

                q_adjustments(
                    qn2.storage,
                    q.storage,
                    dp1,
                    fx,
                    fy,
                    grid.rarea,
                    dp2,
                    it,
                    nsplt,
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
                if global_config.get_do_halo_exchange():
                    comm.halo_update(qn2, n_points=utils.halo)
