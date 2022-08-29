#!/bin/bash
#
# Script to convert xppm.py/xtp_u.py into yppm/ytp_v.py. Can be deleted once we
# have a way to use the same codebase for x-direction and y-direction advection.
#

set -e -x

cp pace/fv3core/stencils/xppm.py pace/fv3core/stencils/yppm.py
cp pace/fv3core/stencils/xtp_u.py pace/fv3core/stencils/ytp_v.py

for fname in pace/fv3core/stencils/yppm.py pace/fv3core/stencils/ytp_v.py
do
    gsed -i 's/ub/vb/g' $fname
    gsed -i 's/dx/dy/g' $fname
    gsed -i 's/xt/yt/g' $fname
    gsed -i 's/eyternals/externals/g' $fname
    gsed -i 's/xflux/yflux/g' $fname
    gsed -i 's/_x/_y/g' $fname
    gsed -i 's/_u/_v/g' $fname
    gsed -i 's/u_/v_/g' $fname
    gsed -i 's/u,/v,/g' $fname
    gsed -i 's/u:/v:/g' $fname
    gsed -i 's/u\[/v\[/g' $fname
    gsed -i 's/u)/v)/g' $fname
    gsed -i 's/iord/jord/g' $fname
    gsed -i 's/\[-1, 0/\[0, -1/g' $fname
    gsed -i 's/\[-2, 0/\[0, -2/g' $fname
    gsed -i 's/\[1, 0/\[0, 1/g' $fname
    gsed -i 's/\[2, 0/\[0, 2/g' $fname
    gsed -i 's/ u / v /g' $fname
    gsed -i 's/x-/y-/g' $fname
    gsed -i 's/i_start/j_start/g' $fname
    gsed -i 's/i_end/j_end/g' $fname
    gsed -i 's/\[j_start - 1, :/\[:, j_start - 1/g' $fname
    gsed -i 's/\[j_start, :/\[:, j_start/g' $fname
    gsed -i 's/\[j_start + 1, :/\[:, j_start + 1/g' $fname
    gsed -i 's/\[j_end - 2, :/\[:, j_end - 2/g' $fname
    gsed -i 's/\[j_end - 1, :/\[:, j_end - 1/g' $fname
    gsed -i 's/\[j_end, :/\[:, j_end/g' $fname
    gsed -i 's/\[j_end + 1, :/\[:, j_end + 1/g' $fname
    gsed -i 's/\[j_end + 2, :/\[:, j_end + 2/g' $fname
done

gsed -i 's/i_start/j_start/g' pace/fv3core/stencils/yppm.py
gsed -i 's/i_end/j_end/g' pace/fv3core/stencils/yppm.py
gsed -i 's/XPiecewise/YPiecewise/g' pace/fv3core/stencils/yppm.py
gsed -i 's/u\*/v\*/g' pace/fv3core/stencils/yppm.py

gsed -i 's/j_start - 1 : j_start + 1, j_start/i_start, j_start - 1 : j_start + 1/g' pace/fv3core/stencils/ytp_v.py
gsed -i 's/j_start - 1 : j_start + 1, j_end + 1/i_end + 1, j_start - 1 : j_start + 1/g' pace/fv3core/stencils/ytp_v.py
gsed -i 's/j_end : j_end + 2, j_start/i_start, j_end : j_end + 2/g' pace/fv3core/stencils/ytp_v.py
gsed -i 's/j_end : j_end + 2, j_end + 1/i_end + 1, j_end : j_end + 2/g' pace/fv3core/stencils/ytp_v.py
gsed -i 's/j_end, j_start, jord, j_end, j_start/j_end, j_start, jord, i_end, i_start/g' pace/fv3core/stencils/ytp_v.py
gsed -i 's/xppm/yppm/g' pace/fv3core/stencils/ytp_v.py

gsed -i 's/region\[j_start - 1 : j_start + 2, :\], region\[j_end - 1 : j_end + 2, :\]/region\[:, j_start - 1 : j_start + 2\], region\[:, j_end - 1 : j_end + 2\]/g' pace/fv3core/stencils/yppm.py
