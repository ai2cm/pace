import os

import f90nml
import numpy as np
import pytest  # noqa

import pace.util
from pace import fv3core
from pace.util.null_comm import NullComm


def test_geos_wrapper():
    namelist = pace.util.Namelist.from_f90nml(
        f90nml.read(os.path.join(data_path, "input.nml"))
    )

    comm = NullComm(0, 0)
    backend = "numpy"

    wrapper = fv3core.GeosDycoreWrapper(namelist, comm, backend)
    nhalo = 3
    shape = (
        wrapper.dycore_config.npx + 2 * nhalo + 1,
        wrapper.dycore_config.npy + 2 * nhalo + 1,
        wrapper.dycore_config.npz + 2 * nhalo + 1,
    )
    assert isinstance(wrapper, fv3core.GeosDycoreWrapper)
    assert isinstance(wrapper.dynamical_core, fv3core.DynamicalCore)

    u = np.ones(shape)
    v = np.ones(shape)
    w = np.ones(shape)
    delz = np.ones(shape)
    pt = np.ones(shape)
    delp = np.ones(shape)
    q = np.ones(shape)
    ps = np.ones(shape)
    pe = np.ones(shape)
    pk = np.ones(shape)
    peln = np.ones(shape)
    pkz = np.ones(shape)
    phis = np.ones(shape)
    q_con = np.ones(shape)
    omga = np.ones(shape)
    ua = np.ones(shape)
    va = np.ones(shape)
    uc = np.ones(shape)
    vc = np.ones(shape)
    mfxd = np.ones(shape)
    mfyd = np.ones(shape)
    cxd = np.ones(shape)
    cyd = np.ones(shape)
    diss_estd = np.ones(shape)

    wrapper(
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
