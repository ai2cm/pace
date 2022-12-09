import unittest.mock

import numpy as np
import pytest

from pace.driver.safety_checks import SafetyChecker
from pace.util import Quantity


def test_register_variable():
    SafetyChecker.clear_all_checks()
    SafetyChecker.register_variable("u", minimum_value=10)
    assert len(SafetyChecker.checks) == 1


def test_double_register():
    SafetyChecker.clear_all_checks()
    SafetyChecker.register_variable("u", minimum_value=10)
    with pytest.raises(NotImplementedError):
        SafetyChecker.register_variable("u", maximum_value=20)


def test_check_state():
    SafetyChecker.clear_all_checks()
    SafetyChecker.register_variable("u", minimum_value=10, maximum_value=20)
    checker = SafetyChecker()
    u = unittest.mock.MagicMock()
    u.data.min.return_value = 11
    u.data.max.return_value = 19
    dycore_state = unittest.mock.MagicMock(u=u)
    checker.check_state(dycore_state)


def test_check_state_failing_min():
    SafetyChecker.clear_all_checks()
    SafetyChecker.register_variable("u", minimum_value=10)
    checker = SafetyChecker()
    u = unittest.mock.MagicMock()
    u.data.min.return_value = 9
    dycore_state = unittest.mock.MagicMock(u=u)
    with pytest.raises(RuntimeError):
        checker.check_state(dycore_state)


def test_check_state_failing_max():
    SafetyChecker.clear_all_checks()
    SafetyChecker.register_variable("u", maximum_value=10)
    checker = SafetyChecker()
    u = unittest.mock.MagicMock()
    u.data.max.return_value = 11
    dycore_state = unittest.mock.MagicMock(u=u)
    with pytest.raises(RuntimeError):
        checker.check_state(dycore_state)


def test_check_state_domain_only():
    SafetyChecker.clear_all_checks()
    SafetyChecker.register_variable("u", maximum_value=10, compute_domain_only=True)
    checker = SafetyChecker()
    u_data = np.ones((4, 4, 2))
    u_data[0:1, 0:1, :] = 100
    u_quantity = Quantity(
        u_data,
        ("x", "y", "z"),
        "unknown",
        origin=(1, 1, 0),
        extent=(3, 3, 2),
        gt4py_backend="numpy",
    )
    dycore_state = unittest.mock.MagicMock(u=u_quantity)
    checker.check_state(dycore_state)


def test_check_nan_value():
    SafetyChecker.clear_all_checks()
    SafetyChecker.register_variable("u", maximum_value=10, compute_domain_only=True)
    checker = SafetyChecker()
    u_data = np.ones((4, 4, 2))
    u_data[2, 2, 1] = np.nan
    u_quantity = Quantity(
        u_data,
        ("x", "y", "z"),
        "unknown",
        origin=(0, 0, 0),
        extent=(4, 4, 2),
        gt4py_backend="numpy",
    )
    dycore_state = unittest.mock.MagicMock(u=u_quantity)
    with pytest.raises(RuntimeError):
        checker.check_state(dycore_state)


def test_variable_not_present():
    SafetyChecker.clear_all_checks()
    SafetyChecker.register_variable("v", maximum_value=10)
    checker = SafetyChecker()
    u = unittest.mock.MagicMock()
    u.data.max.return_value = 11
    dycore_state = unittest.mock.MagicMock(u=u)
    with pytest.raises(NotImplementedError):
        checker.check_state(dycore_state)
