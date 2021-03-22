import fv3core._config as spec
import fv3core.stencils.remapping_part1 as remap_part1
import fv3core.stencils.remapping_part2 as remap_part2
import fv3core.utils.gt4py_utils as utils


sd = utils.sd


def compute(
    tracers,
    pt,
    delp,
    delz,
    peln,
    u,
    v,
    w,
    ua,
    va,
    cappa,
    q_con,
    pkz,
    pk,
    pe,
    hs,
    te0_2d,
    ps,
    wsd,
    omga,
    ak,
    bk,
    pfull,
    dp1,
    ptop,
    akap,
    zvir,
    last_step,
    consv_te,
    mdt,
    bdt,
    kord_tracer,
    do_adiabatic_init,
    nq,
):
    """
    Remap the deformed Lagrangian surfaces onto the reference, or "Eulerian",
    coordinate levels.
    """
    grid = spec.grid

    gz = utils.make_storage_from_shape(pt.shape, grid.compute_origin())
    cvm = utils.make_storage_from_shape(pt.shape, grid.compute_origin())
    te_2d = utils.make_storage_from_shape(pt.shape, grid.compute_origin())
    zsum1 = utils.make_storage_from_shape(pt.shape, grid.compute_origin())
    remap_part1.compute(
        tracers,
        pt,
        delp,
        delz,
        peln,
        u,
        v,
        w,
        ua,
        cappa,
        q_con,
        pkz,
        pk,
        pe,
        hs,
        dp1,
        ps,
        wsd,
        omga,
        ak,
        bk,
        gz,
        cvm,
        ptop,
        akap,
        zvir,
        nq,
    )
    remap_part2.compute(
        tracers["qvapor"],
        tracers["qliquid"],
        tracers["qice"],
        tracers["qrain"],
        tracers["qsnow"],
        tracers["qgraupel"],
        tracers["qcld"],
        pt,
        delp,
        delz,
        peln,
        u,
        v,
        w,
        ua,
        cappa,
        q_con,
        gz,
        pkz,
        pk,
        pe,
        hs,
        te_2d,
        te0_2d,
        dp1,
        cvm,
        zsum1,
        pfull,
        ptop,
        akap,
        zvir,
        last_step,
        bdt,
        mdt,
        consv_te,
        do_adiabatic_init,
    )
