import gt4py.gtscript as gtscript
from gt4py.gtscript import BACKWARD, PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.utils.global_constants as constants
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy
from fv3core.utils.corners import fill_4corners


sd = utils.sd
origin = (1, 1, 0)
DZ_MIN = constants.DZ_MIN


# def copy(q_in):
#    q_out = utils.make_storage_from_shape(q_in.shape, origin)
#    copy_stencil(q_in, q_out)#, origin=(0,0,0), domain=grid.domain_shape_buffer_1cell())
#    return q_out


# call update_dz_c(is, ie, js, je, npz, ng, dt2, dp_ref, zs, gridstruct%area, ut, vt, gz, ws3, &
#              npx, npy, gridstruct%sw_corner, gridstruct%se_corner, &
#              gridstruct%ne_corner, gridstruct%nw_corner, bd, gridstruct%grid_type)

# subroutine update_dz_c(is, ie, js, je, km, ng, dt, dp0, zs, area, ut, vt, gz, ws, &
#        npx, npy, sw_corner, se_corner, ne_corner, nw_corner, bd, grid_type)
# ! !INPUT PARAMETERS:
#   type(fv_grid_bounds_type), intent(IN) :: bd
#   integer, intent(in):: is, ie, js, je, ng, km, npx, npy, grid_type
#   logical, intent(IN):: sw_corner, se_corner, ne_corner, nw_corner
#   real, intent(in):: dt
#   real, intent(in):: dp0(km)
#   real, intent(in), dimension(is-ng:ie+ng,js-ng:je+ng,km):: ut, vt
#   real, intent(in), dimension(is-ng:ie+ng,js-ng:je+ng):: area
#   real, intent(inout):: gz(is-ng:ie+ng,js-ng:je+ng,km+1)
#   real, intent(in   ):: zs(is-ng:ie+ng, js-ng:je+ng)
#   real, intent(  out):: ws(is-ng:ie+ng, js-ng:je+ng)
# ! Local Work array:
#   real:: gz2(is-ng:ie+ng,js-ng:je+ng)
#   real, dimension(is-1:ie+2,js-1:je+1):: xfx, fx
#   real, dimension(is-1:ie+1,js-1:je+2):: yfx, fy
#   real, parameter:: r14 = 1./14.
#   integer  i, j, k
#   integer:: is1, ie1, js1, je1
#   integer:: ie2, je2
#   real:: rdt, top_ratio, bot_ratio, int_ratio


#   rdt = 1. / dt


#   is1 = is - 1
#   js1 = js - 1

#   ie1 = ie + 1
#   je1 = je + 1

#   ie2 = ie + 2
#   je2 = je + 2

#   do k = 1, km+1

#   top_ratio = dp0(1 ) / (dp0(   1)+dp0(2 ))
#
#      if ( k==1 ) then   ! top
#         do j=js1, je1
#            do i=is1, ie2
#               xfx(i,j) = ut(i,j,1) + (ut(i,j,1)-ut(i,j,2))*top_ratio
#            enddo
#         enddo
#         do j=js1, je2
#            do i=is1, ie1
#               yfx(i,j) = vt(i,j,1) + (vt(i,j,1)-vt(i,j,2))*top_ratio
#            enddo
#         enddo


@gtscript.function
def p_weighted_average_top(vel, dp0):
    # TODO: ratio is a constant, where should this be placed?
    ratio = dp0 / (dp0 + dp0[0, 0, 1])
    # return (1. + ratio) * vel - ratio * vel[0, 0, 1]
    return vel + (vel - vel[0, 0, 1]) * ratio


#   bot_ratio = dp0(km) / (dp0(km-1)+dp0(km))
#
#      elseif ( k==km+1 ) then  ! bottom
#         do j=js1, je1
#            do i=is1, ie2
#               xfx(i,j) = ut(i,j,km) + (ut(i,j,km)-ut(i,j,km-1))*bot_ratio
#            enddo
#         enddo
#         do j=js1, je2
#            do i=is1, ie1
#               yfx(i,j) = vt(i,j,km) + (vt(i,j,km)-vt(i,j,km-1))*bot_ratio
#            enddo
#         enddo


@gtscript.function
def p_weighted_average_bottom(vel, dp0):
    ratio = dp0[0, 0, -1] / (dp0[0, 0, -2] + dp0[0, 0, -1])
    # return (1. + ratio ) * vel[0, 0, -1] - ratio * vel[0, 0, -2]
    return vel[0, 0, -1] + (vel[0, 0, -1] - vel[0, 0, -2]) * ratio


#      else     ! compute domain
#         int_ratio = 1./(dp0(k-1)+dp0(k))
#         do j=js1, je1
#            do i=is1, ie2
#               xfx(i,j) = (dp0(k)*ut(i,j,k-1)+dp0(k-1)*ut(i,j,k))*int_ratio
#            enddo
#         enddo
#         do j=js1, je2
#            do i=is1, ie1
#               yfx(i,j) = (dp0(k)*vt(i,j,k-1)+dp0(k-1)*vt(i,j,k))*int_ratio
#            enddo
#         enddo
#      endif


@gtscript.function
def p_weighted_average_domain(vel, dp0):
    # ratio = dp0 / ( dp0[0, 0, -1] + dp0 )
    # return ratio * vel[0, 0, -1] + (1. - ratio) * vel
    int_ratio = 1.0 / (dp0[0, 0, -1] + dp0)
    return (dp0 * vel[0, 0, -1] + dp0[0, 0, -1] * vel) * int_ratio


#      do j=js-ng, je+ng
#         do i=is-ng, ie+ng
#            gz2(i,j) = gz(i,j,k)
#         enddo
#      enddo

#      if (grid_type < 3) call fill_4corners(gz2, 1, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)

#      do j=js1, je1
#         do i=is1, ie2
#            if( xfx(i,j) > 0.0 ) then
#                fx(i,j) = gz2(i-1,j)
#            else
#                fx(i,j) = gz2(i  ,j)
#            endif
#            fx(i,j) = xfx(i,j)*fx(i,j)
#         enddo
#      enddo

#      if (grid_type < 3) call fill_4corners(gz2, 2, bd, npx, npy, sw_corner, se_corner, ne_corner, nw_corner)

#      do j=js1,je2
#         do i=is1,ie1
#            if( yfx(i,j) > 0.0 ) then
#                fy(i,j) = gz2(i,j-1)
#            else
#                fy(i,j) = gz2(i,j)
#            endif
#            fy(i,j) = yfx(i,j)*fy(i,j)
#         enddo
#      enddo


@gtscript.function
def xy_flux(gz_x, gz_y, xfx, yfx):
    fx = xfx * (gz_x[-1, 0, 0] if xfx > 0.0 else gz_x)
    fy = yfx * (gz_y[0, -1, 0] if yfx > 0.0 else gz_y)
    return fx, fy


#      do j=js1, je1
#         do i=is1,ie1
#            gz(i,j,k) = (gz2(i,j)*area(i,j) +  fx(i,j)- fx(i+1,j)+ fy(i,j)- fy(i,j+1)) &
#                      / (         area(i,j) + xfx(i,j)-xfx(i+1,j)+yfx(i,j)-yfx(i,j+1))
#         enddo
#      enddo

#   end do

# ! Enforce monotonicity of height to prevent blowup
#   do i=is1, ie1
#      do j=js1, je1
#         ws(i,j) = ( zs(i,j) - gz(i,j,km+1) ) * rdt
#      enddo
#   end do

#   do k = km, 1, -1
#      do j=js1, je1
#         do i=is1, ie1
#            gz(i,j,k) = max( gz(i,j,k), gz(i,j,k+1) + dz_min )
#         enddo
#      enddo
#   enddo


@gtstencil()
def update_dz_c(
    dp_ref: sd,
    zs: sd,
    area: sd,
    ut: sd,
    vt: sd,
    gz: sd,
    gz_x: sd,
    gz_y: sd,
    ws3: sd,
    *,
    dt: float,
):
    with computation(PARALLEL):
        with interval(0, 1):
            xfx = p_weighted_average_top(ut, dp_ref)
            yfx = p_weighted_average_top(vt, dp_ref)
        with interval(1, -1):
            xfx = p_weighted_average_domain(ut, dp_ref)
            yfx = p_weighted_average_domain(vt, dp_ref)
        with interval(-1, None):
            xfx = p_weighted_average_bottom(ut, dp_ref)
            yfx = p_weighted_average_bottom(vt, dp_ref)
    with computation(PARALLEL), interval(...):
        fx, fy = xy_flux(gz_x, gz_y, xfx, yfx)
        # TODO: check if below gz is ok, or if we need gz_y to pass this
        gz = (gz_y * area + fx - fx[1, 0, 0] + fy - fy[0, 1, 0]) / (
            area + xfx - xfx[1, 0, 0] + yfx - yfx[0, 1, 0]
        )
    with computation(PARALLEL), interval(-1, None):
        rdt = 1.0 / dt
        ws3 = (zs - gz) * rdt
    with computation(BACKWARD), interval(0, -1):
        gz_kp1 = gz[0, 0, 1] + DZ_MIN
        gz = gz if gz > gz_kp1 else gz_kp1


def compute(dp_ref, zs, ut, vt, gz_in, ws3, dt2):
    # TODO: once we have a concept for corners, the following 4 lines should be refactored
    grid = spec.grid
    gz = copy(gz_in, origin=origin)
    gz_x = copy(gz, origin=origin)
    ws = copy(ws3, domain=grid.domain_shape_buffer_1cell())
    fill_4corners(gz_x, "x", grid)
    gz_y = copy(gz_x, origin=origin)
    fill_4corners(gz_y, "y", grid)
    update_dz_c(
        dp_ref,
        zs,
        grid.area,
        ut,
        vt,
        gz,
        gz_x,
        gz_y,
        ws3,
        dt=dt2,
        origin=origin,
        domain=(grid.nic + 3, grid.njc + 3, grid.npz + 1),
    )
    grid.overwrite_edges(gz, gz_in, 2, 2)
    grid.overwrite_edges(ws3, ws, 2, 2)
    return gz, ws3
