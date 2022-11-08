import unittest.mock

import pytest

from pace.driver.safety_checks import SafetyChecker


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


def test_variable_not_present():
    SafetyChecker.clear_all_checks()
    SafetyChecker.register_variable("v", maximum_value=10)
    checker = SafetyChecker()
    u = unittest.mock.MagicMock()
    u.data.max.return_value = 11
    dycore_state = unittest.mock.MagicMock(u=u)
    with pytest.raises(NotImplementedError):
        checker.check_state(dycore_state)
