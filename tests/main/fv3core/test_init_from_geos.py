import os

import f90nml
import numpy as np
import pytest  # noqa

import pace.util
from pace import fv3core


def test_geos_wrapper_init(data_path: str):
    namelist = pace.util.Namelist.from_f90nml(
        f90nml.read(os.path.join(data_path, "input.nml"))
    )

    comm = None
    backend = "numpy"

    wrapper = fv3core.GeosDycoreWrapper(namelist, comm, backend)
    assert isinstance(wrapper, fv3core.GeosDycoreWrapper)
    assert isinstance(wrapper.dynamical_core, fv3core.DynamicalCore)


def test_dycore_state_from_geos(data_path: str):
    namelist = pace.util.Namelist.from_f90nml(
        f90nml.read(os.path.join(data_path, "input.nml"))
    )
    data_inputs = np.loadtxt(os.path.join(data_path, "data.txt"))

    comm = None
    backend = "numpy"

    wrapper = fv3core.GeosDycoreWrapper(namelist, comm, backend)
    wrapper._put_fortran_data_in_dycore(data_inputs)
    assert isinstance(wrapper.dycore_state, fv3core.DycoreState)
