import gt4py.gtscript as gtscript
from gt4py.gtscript import PARALLEL, computation, horizontal, interval, region

import fv3core._config as spec
import fv3core.stencils.delnflux as delnflux
import fv3core.stencils.xppm as xppm
import fv3core.stencils.yppm as yppm
import fv3core.utils.corners as corners
import fv3core.utils.global_config as global_config
import fv3core.utils.gt4py_utils as utils
from fv3core.utils.typing import FloatField, FloatFieldIJ


def q_i_stencil(
    q: FloatField,
    area: FloatFieldIJ,
    yfx: FloatField,
    fy2: FloatField,
    ra_y: FloatField,
    q_i: FloatField,
):
    with computation(PARALLEL), interval(...):
        fyy = yfx * fy2
        q_i[0, 0, 0] = (q * area + fyy - fyy[0, 1, 0]) / ra_y


def q_j_stencil(
    q: FloatField,
    area: FloatFieldIJ,
    xfx: FloatField,
    fx2: FloatField,
    ra_x: FloatField,
    q_j: FloatField,
):
    with computation(PARALLEL), interval(...):
        fx1 = xfx * fx2
        q_j[0, 0, 0] = (q * area + fx1 - fx1[1, 0, 0]) / ra_x


@gtscript.function
def transport_flux(f, f2, mf):
    return 0.5 * (f + f2) * mf


def transport_flux_xy(
    fx: FloatField,
    fx2: FloatField,
    fy: FloatField,
    fy2: FloatField,
    mfx: FloatField,
    mfy: FloatField,
):
    with computation(PARALLEL), interval(...):
        with horizontal(region[:, :-1]):
            fx = transport_flux(fx, fx2, mfx)
        with horizontal(region[:-1, :]):
            fy = transport_flux(fy, fy2, mfy)


class FvTp2d:
    """
    ONLY USE_SG=False compiler flag implements
    """

    def __init__(self, namelist, hord):
        self.grid = spec.grid
        shape = self.grid.domain_shape_full(add=(1, 1, 1))
        origin = self.grid.compute_origin()
        self._tmp_q_i = utils.make_storage_from_shape(shape, origin)
        self._tmp_q_j = utils.make_storage_from_shape(shape, origin)
        self._tmp_fx2 = utils.make_storage_from_shape(shape, origin)
        self._tmp_fy2 = utils.make_storage_from_shape(shape, origin)
        ord_ou = hord
        ord_in = 8 if hord == 10 else hord
        stencil_kwargs = {
            "backend": global_config.get_backend(),
            "rebuild": global_config.get_rebuild(),
        }
        stencil_wrapper = gtscript.stencil(**stencil_kwargs)
        self.stencil_q_i = stencil_wrapper(q_i_stencil)
        self.stencil_q_j = stencil_wrapper(q_j_stencil)
        self.stencil_transport_flux = stencil_wrapper(transport_flux_xy)
        self.xppm_object_in = xppm.XPPM(spec.namelist, ord_in)
        self.yppm_object_in = yppm.YPPM(spec.namelist, ord_in)
        self.xppm_object_ou = xppm.XPPM(spec.namelist, ord_ou)
        self.yppm_object_ou = yppm.YPPM(spec.namelist, ord_ou)

    def __call__(
        self,
        q,
        crx,
        cry,
        xfx,
        yfx,
        ra_x,
        ra_y,
        fx,
        fy,
        kstart=0,
        nk=None,
        nord=None,
        damp_c=None,
        mass=None,
        mfx=None,
        mfy=None,
    ):
        grid = self.grid
        if nk is None:
            nk = grid.npz - kstart
        kslice = slice(kstart, kstart + nk)
        compute_origin = (grid.is_, grid.js, kstart)
        corners.copy_corners_y_stencil(
            q, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
        )
        self.yppm_object_in(
            q, cry, self._tmp_fy2, grid.isd, grid.ied, kstart=kstart, nk=nk
        )
        self.stencil_q_i(
            q,
            grid.area,
            yfx,
            self._tmp_fy2,
            ra_y,
            self._tmp_q_i,
            origin=(grid.isd, grid.js, kstart),
            domain=(grid.nid, grid.njc + 1, nk),
        )
        self.xppm_object_ou(
            self._tmp_q_i, crx, fx, grid.js, grid.je, kstart=kstart, nk=nk
        )

        corners.copy_corners_x_stencil(
            q, origin=(grid.isd, grid.jsd, kstart), domain=(grid.nid, grid.njd, nk)
        )
        self.xppm_object_in(
            q, crx, self._tmp_fx2, grid.jsd, grid.jed, kstart=kstart, nk=nk
        )
        self.stencil_q_j(
            q,
            grid.area,
            xfx,
            self._tmp_fx2,
            ra_x,
            self._tmp_q_j,
            origin=(grid.is_, grid.jsd, kstart),
            domain=(grid.nic + 1, grid.njd, nk),
        )
        self.yppm_object_ou(
            self._tmp_q_j, cry, fy, grid.is_, grid.ie, kstart=kstart, nk=nk
        )
        if mfx is not None and mfy is not None:
            self.stencil_transport_flux(
                fx,
                self._tmp_fx2,
                fy,
                self._tmp_fy2,
                mfx,
                mfy,
                origin=compute_origin,
                domain=(grid.nic + 1, grid.njc + 1, nk),
            )
            if (mass is not None) and (nord is not None) and (damp_c is not None):
                delnflux.compute_delnflux_no_sg(
                    q, fx, fy, nord, damp_c, kstart, nk, mass=mass
                )
        else:

            self.stencil_transport_flux(
                fx,
                self._tmp_fx2,
                fy,
                self._tmp_fy2,
                xfx,
                yfx,
                origin=compute_origin,
                domain=(grid.nic + 1, grid.njc + 1, nk),
            )
            if (nord is not None) and (damp_c is not None):
                delnflux.compute_delnflux_no_sg(q, fx, fy, nord, damp_c, kstart, nk)
