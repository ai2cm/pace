import math as math

import gt4py.gtscript as gtscript
import numpy as np
from gt4py.gtscript import PARALLEL, computation, interval

import fv3core._config as spec
import fv3core.stencils.remap_profile as remap_profile
import fv3core.utils.gt4py_utils as utils
from fv3core.decorators import gtstencil
from fv3core.stencils.basic_operations import copy


sd = utils.sd
r3 = 1.0 / 3.0
r23 = 2.0 / 3.0


def grid():
    return spec.grid


@gtstencil()
def set_dp(dp1: sd, pe1: sd):
    with computation(PARALLEL), interval(...):
        dp1 = pe1[0, 0, 1] - pe1


@gtstencil()
def lagrangian_contributions(
    pe1: sd,
    ptop: sd,
    pbot: sd,
    q4_1: sd,
    q4_2: sd,
    q4_3: sd,
    q4_4: sd,
    dp1: sd,
    q2_adds: sd,
    r3: float,
    r23: float,
):
    with computation(PARALLEL), interval(...):
        pl = pe1
        pr = pe1
        dp = pe1
        esl = pe1
        if pe1 < pbot and pe1[0, 0, 1] > ptop:
            # We are in the right pressure range to contribute to the Eulerian cell
            if pe1 <= ptop:
                # we are in the first Lagrangian level that conributes
                pl = (ptop - pe1) / dp1
                if pbot <= pe1[0, 0, 1]:
                    # eulerian grid element is contained in the Lagrangian element
                    pr = (pbot - pe1) / dp1
                    q2_adds = (
                        q4_2
                        + 0.5 * (q4_4 + q4_3 - q4_2) * (pr + pl)
                        - q4_4 * r3 * (pr * (pr + pl) + pl ** 2)
                    )
                else:
                    # Eulerian element encompasses multiple Lagrangian elements and this is just the first one
                    q2_adds = (
                        (pe1[0, 0, 1] - ptop)
                        * (
                            q4_2
                            + 0.5 * (q4_4 + q4_3 - q4_2) * (1.0 + pl)
                            - q4_4 * r3 * (1.0 + pl * (1.0 + pl))
                        )
                        / (pbot - ptop)
                    )
            else:
                # we are in a farther-down level
                if pbot > pe1[0, 0, 1]:
                    # add the whole level to the Eulerian cell
                    q2_adds = dp1 * q4_1 / (pbot - ptop)
                else:
                    # this is the bottom layer that contributes
                    dp = pbot - pe1
                    esl = dp / dp1
                    q2_adds = (
                        dp
                        * (q4_2 + 0.5 * esl * (q4_3 - q4_2 + q4_4 * (1.0 - r23 * esl)))
                        / (pbot - ptop)
                    )
        else:
            q2_adds = 0


def region_mode(j_2d, i1, i_extent, grid):
    if j_2d is None:
        jslice = slice(grid.js, grid.je + 1)
    else:
        jslice = slice(j_2d, j_2d + 1)
    origin = (i1, jslice.start, 0)
    domain = (i_extent, jslice.stop - jslice.start, grid.npz)
    return origin, domain, jslice


def compute(
    q1,
    pe1,
    pe2,
    qs,
    mode,
    i1,
    i2,
    kord,
    qmin=0.0,
    j_2d=None,
    j_interface=False,
    version="stencil",
):
    iv = mode
    dp1, q4_1, q4_2, q4_3, q4_4, origin, domain, jslice, i_extent = setup_data(
        q1, pe1, i1, i2, j_2d, j_interface
    )
    q4_1, q4_2, q4_3, q4_4 = remap_profile.compute(
        qs, q4_1, q4_2, q4_3, q4_4, dp1, spec.grid.npz, i1, i2, iv, kord, jslice, qmin
    )
    do_lagrangian_contributions(
        q1,
        pe1,
        pe2,
        q4_1,
        q4_2,
        q4_3,
        q4_4,
        dp1,
        i1,
        i2,
        kord,
        jslice,
        origin,
        domain,
        version,
    )
    return q1


def do_lagrangian_contributions(
    q1,
    pe1,
    pe2,
    q4_1,
    q4_2,
    q4_3,
    q4_4,
    dp1,
    i1,
    i2,
    kord,
    jslice,
    origin,
    domain,
    version,
):
    if version == "transliterated":
        lagrangian_contributions_transliterated(
            q1, pe1, pe2, q4_1, q4_2, q4_3, q4_4, dp1, i1, i2, kord, jslice
        )
    elif version == "stencil":
        lagrangian_contributions_stencil(
            q1,
            pe1,
            pe2,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            dp1,
            i1,
            i2,
            kord,
            jslice,
            origin,
            domain,
        )
    else:
        raise Exception(version + " is not an implemented remapping version")


def setup_data(q1, pe1, i1, i2, j_2d=None, j_interface=False):
    grid = spec.grid
    i_extent = i2 - i1 + 1
    origin, domain, jslice = region_mode(j_2d, i1, i_extent, grid)
    if j_interface:
        jslice = slice(jslice.start, jslice.stop + 1)
        domain = (domain[0], jslice.stop - jslice.start, domain[2])

    dp1 = utils.make_storage_from_shape(q1.shape, origin=origin)
    q4_1 = copy(q1, origin=(0, 0, 0), domain=grid.domain_shape_standard())
    q4_2 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    q4_3 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    q4_4 = utils.make_storage_from_shape(q4_1.shape, origin=(grid.is_, 0, 0))
    set_dp(dp1, pe1, origin=origin, domain=domain)
    return dp1, q4_1, q4_2, q4_3, q4_4, origin, domain, jslice, i_extent


def lagrangian_contributions_stencil(
    q1, pe1, pe2, q4_1, q4_2, q4_3, q4_4, dp1, i1, i2, kord, jslice, origin, domain
):
    # A stencil with a loop over k2:
    km = spec.grid.npz
    klevs = np.arange(km)
    orig = spec.grid.default_origin()
    q2_adds = utils.make_storage_from_shape(q4_1.shape, origin=orig)
    for k_eul in klevs:
        eulerian_top_pressure = pe2[:, :, k_eul]
        eulerian_bottom_pressure = pe2[:, :, k_eul + 1]
        top_p = utils.repeat(eulerian_top_pressure[:, :, np.newaxis], km + 1, axis=2)
        bot_p = utils.repeat(eulerian_bottom_pressure[:, :, np.newaxis], km + 1, axis=2)
        ptop = utils.make_storage_data(top_p, q4_1.shape)
        pbot = utils.make_storage_data(bot_p, q4_1.shape)

        lagrangian_contributions(
            pe1,
            ptop,
            pbot,
            q4_1,
            q4_2,
            q4_3,
            q4_4,
            dp1,
            q2_adds,
            r3,
            r23,
            origin=origin,
            domain=domain,
        )

        q1[i1 : i2 + 1, jslice, k_eul] = utils.sum(
            q2_adds[i1 : i2 + 1, jslice, :], axis=2
        )


def lagrangian_contributions_transliterated(
    q1, pe1, pe2, q4_1, q4_2, q4_3, q4_4, dp1, i1, i2, kord, jslice
):
    grid = spec.grid
    i_vals = np.arange(i1, i2 + 1)
    kn = grid.npz
    km = grid.npz
    for j in range(jslice.start, jslice.stop):
        for ii in i_vals:
            k0 = 0
            for k2 in np.arange(kn):  # loop over new, remapped ks]
                for k1 in np.arange(k0, km):  # loop over old ks
                    # find the top edge of new grid: pe2[ii, k2]
                    if (
                        pe2[ii, j, k2] >= pe1[ii, j, k1]
                        and pe2[ii, j, k2] <= pe1[ii, j, k1 + 1]
                    ):
                        pl = (pe2[ii, j, k2] - pe1[ii, j, k1]) / dp1[ii, j, k1]
                        if (
                            pe2[ii, j, k2 + 1] <= pe1[ii, j, k1 + 1]
                        ):  # then the new grid layer is entirely within the old one
                            pr = (pe2[ii, j, k2 + 1] - pe1[ii, j, k1]) / dp1[ii, j, k1]
                            q1[ii, j, k2] = (
                                q4_2[ii, j, k1]
                                + 0.5
                                * (q4_4[ii, j, k1] + q4_3[ii, j, k1] - q4_2[ii, j, k1])
                                * (pr + pl)
                                - q4_4[ii, j, k1] * r3 * (pr * (pr + pl) + pl ** 2)
                            )
                            k0 = k1
                            break
                        else:  # new grid layer extends into more old grid layers
                            qsum = (pe1[ii, j, k1 + 1] - pe2[ii, j, k2]) * (
                                q4_2[ii, j, k1]
                                + 0.5
                                * (q4_4[ii, j, k1] + q4_3[ii, j, k1] - q4_2[ii, j, k1])
                                * (1.0 + pl)
                                - q4_4[ii, j, k1] * (r3 * (1.0 + pl * (1.0 + pl)))
                            )

                            for mm in np.arange(k1 + 1, km):  # find the bottom edge
                                if (
                                    pe2[ii, j, k2 + 1] > pe1[ii, j, mm + 1]
                                ):  # Not there yet; add the whole layer
                                    qsum = qsum + dp1[ii, j, mm] * q4_1[ii, j, mm]
                                else:
                                    dp = pe2[ii, j, k2 + 1] - pe1[ii, j, mm]
                                    esl = dp / dp1[ii, j, mm]
                                    qsum = qsum + dp * (
                                        q4_2[ii, j, mm]
                                        + 0.5
                                        * esl
                                        * (
                                            q4_3[ii, j, mm]
                                            - q4_2[ii, j, mm]
                                            + q4_4[ii, j, mm] * (1.0 - r23 * esl)
                                        )
                                    )
                                    k0 = mm
                                    flag = 1
                                    break
                            # Add everything up and divide by the pressure difference
                            q1[ii, j, k2] = qsum / (pe2[ii, j, k2 + 1] - pe2[ii, j, k2])
                            break

    """
    # #Pythonized
    # kn = grid.npz
    # i_vals = np.arange(i1, i2 + 1)
    # klevs = np.arange(km+1)
    # for ii in i_vals:
    #     for k2 in np.arange(kn):  # loop over new, remapped ks]
    #         top1 = pe2[ii, 0, k2] >= pe1[ii, 0,:]
    #         k1 = klevs[top1][-1]
    #         pl = (pe2[ii, 0, k2] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
    #         if pe2[ii, 0, k2+1] <= pe1[ii, 0, k1+1]:
    #             #The new grid is contained within the old one
    #             pr = (pe2[ii, 0, k2 + 1] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
    #             q2[ii, j_2d, k2] = (
    #                 q4_2[ii, 0, k1]
    #                 + 0.5
    #                 * (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1])
    #                 * (pr + pl)
    #                 - q4_4[ii, 0, k1] * r3 * (pr * (pr + pl) + pl ** 2)
    #             )
    #             # continue
    #         else:
    #             # new grid layer extends into more old grid layers
    #             qsum = (pe1[ii, 0, k1 + 1] - pe2[ii, 0, k2]) * (
    #                         q4_2[ii, 0, k1]
    #                         + 0.5
    #                         * (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1])
    #                         * (1.0 + pl)
    #                         - q4_4[ii, 0, k1] * (r3 * (1.0 + pl * (1.0 + pl)))
    #             )
    #             bottom_2 = pe2[ii, 0, k2+1] > pe1[ii, 0, k1+1:]
    #             mm = klevs[k1+1:][bottom_2][-1]
    #             qsum = qsum + np.sum(dp1[ii, 0, k1+1:mm] * q4_1[ii, 0, k1+1:mm])
    #             if not bottom_2.all():
    #                 dp = pe2[ii, 0, k2 + 1] - pe1[ii, 0, mm]
    #                 esl = dp / dp1[ii, 0, mm]
    #                 qsum = qsum + dp * (
    #                     q4_2[ii,0,mm]
    #                     + 0.5
    #                     * esl
    #                     * (
    #                         q4_3[ii, 0, mm]
    #                         - q4_2[ii, 0, mm]
    #                         + q4_4[ii, 0, mm] * (1.0 - r23 * esl)
    #                     )
    #                 )
    #             q2[ii, j_2d, k2] = qsum / (pe2[ii, 0, k2 + 1] - pe2[ii, 0, k2])

    # # transliterated fortran
    # i_vals = np.arange(i1, i2 + 1)
    # kn = grid.npz
    # elems = np.ones((i_extent,kn))
    # for ii in i_vals:
    #     k0 = 0
    #     for k2 in np.arange(kn):  # loop over new, remapped ks]
    #         for k1 in np.arange(k0, km):  # loop over old ks
    #             # find the top edge of new grid: pe2[ii, k2]
    #             if pe2[ii, 0, k2] >= pe1[ii, 0, k1] and pe2[ii, 0, k2] <= pe1[ii, 0, k1 + 1]:
    #                 pl = (pe2[ii, 0, k2] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
    #                 if (
    #                     pe2[ii, 0, k2 + 1] <= pe1[ii, 0, k1 + 1]
    #                 ):  # then the new grid layer is entirely within the old one
    #                     pr = (pe2[ii, 0, k2 + 1] - pe1[ii, 0, k1]) / dp1[ii, 0, k1]
    #                     q2[ii, j_2d, k2] = (
    #                         q4_2[ii, 0, k1]
    #                         + 0.5
    #                         * (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1])
    #                         * (pr + pl)
    #                         - q4_4[ii, 0, k1] * r3 * (pr * (pr + pl) + pl ** 2)
    #                     )
    #                     k0 = k1
    #                     elems[ii-i1,k2]=0
    #                     break
    #                 else:  # new grid layer extends into more old grid layers
    #                     qsum = (pe1[ii, 0, k1 + 1] - pe2[ii, 0, k2]) * (
    #                         q4_2[ii, 0, k1]
    #                         + 0.5
    #                         * (q4_4[ii, 0, k1] + q4_3[ii, 0, k1] - q4_2[ii, 0, k1])
    #                         * (1.0 + pl)
    #                         - q4_4[ii, 0, k1] * (r3 * (1.0 + pl * (1.0 + pl)))
    #                     )

    #                     for mm in np.arange(k1 + 1, km):  # find the bottom edge
    #                         if pe2[ii, 0, k2 + 1] > pe1[ii, 0, mm + 1]:  #Not there yet; add the whole layer
    #                             qsum = qsum + dp1[ii, 0, mm] * q4_1[ii, 0, mm]
    #                         else:
    #                             dp = pe2[ii, 0, k2 + 1] - pe1[ii, 0, mm]
    #                             esl = dp / dp1[ii, 0, mm]
    #                             qsum = qsum + dp * (
    #                                 q4_2[ii, 0, mm]
    #                                 + 0.5
    #                                 * esl
    #                                 * (
    #                                     q4_3[ii, 0, mm]
    #                                     - q4_2[ii, 0, mm]
    #                                     + q4_4[ii, 0, mm] * (1.0 - r23 * esl)
    #                                 )
    #                             )
    #                             k0 = mm
    #                             flag = 1
    #                             elems[ii-i1,k2]=0
    #                             break
    #                     #Add everything up and divide by the pressure difference
    #                     q2[ii, j_2d, k2] = qsum / (pe2[ii, 0, k2 + 1] - pe2[ii, 0, k2])
    #                     break
    """
