Fortran Changelog
=================

This document outlines changes to the source code from the patterns or conventions used in the original Fortran. This information is meant to help map between the updated Python source code and the Fortran equivalent.

d_sw:
- in flux_adjust, "w" is renamed to "q" since the variable refers to any scalar, not just vertical wind
- gx/gy for pt fluxes, fx/fy for delp fluxes have been renamed to pt_x_flux, delp_x_flux and similar for y fluxes
- converted one of the usages of ut to u and dx (ut = u * dx) and similar for vt to v and dy
- ubke and vbke calculations in Fortran use dt4 which is 0.25 * dt and dt5 which is 0.2 * dt in the new code
- in ubke/vbke calculations, renamed ub to ub_contra since it is contravariant wind (similarly for vb)
- in ubke/vbke calculations, renamed ut to uc_contra since it is contravariant wind (similarly for vt)
- renamed one of the usages of ub/vb to vort_x_delta and vort_y_delta, where they hold x and y finite differences of the vort variable
- renamed first usage of ptc to u_contra_dyc and first usage of vort to v_contra_dxc
- in xppm/yppm routines, separated "courant" which was b-wind * dt into dt (timestep) and ub_contra/vb_contra
- a2b_ord4 uses lon for the Fortran code's `grid(:, :, 1)` and lat for `grid(:, :, 2)`, and similarly lon_agrid/lat_agrid for the components of the Fortran code's agrid variable.
