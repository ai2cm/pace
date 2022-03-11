import contextlib
import unittest.mock

import pace.dsl
import pace.dsl.dace.dace_config


"""
Tests that the global dace configuration pace.dsl.dace.dace_config.dace_config
determines whether we use dace to run wrapped functions.

These tests can be refactored or removed if we change the global dace configuration
functionality.
"""


@contextlib.contextmanager
def use_dace(use: bool):
    original_setting = pace.dsl.dace.dace_config.dace_config.orchestrate
    try:
        pace.dsl.dace.dace_config.dace_config.orchestrate = use
        yield
    finally:
        pace.dsl.dace.dace_config.dace_config.orchestrate = original_setting


def test_computepath_function_calls_dace():
    def foo():
        pass

    with use_dace(True):
        wrapped = pace.dsl.computepath_function(foo)
        with unittest.mock.patch(
            "pace.dsl.dace.orchestrate.call_sdfg"
        ) as mock_call_sdfg:
            wrapped()
        assert mock_call_sdfg.called
        assert mock_call_sdfg.call_args.args[0].f == foo


def test_computepath_function_does_not_call_dace():
    def foo():
        pass

    with use_dace(False):
        wrapped = pace.dsl.computepath_function(foo)
        with unittest.mock.patch(
            "pace.dsl.dace.orchestrate.call_sdfg"
        ) as mock_call_sdfg:
            wrapped()
        assert not mock_call_sdfg.called


def test_computepath_method_calls_dace():

    with use_dace(True):

        class A:
            @pace.dsl.computepath_method
            def foo(self):
                pass

        with unittest.mock.patch(
            "pace.dsl.dace.orchestrate.call_sdfg"
        ) as mock_call_sdfg:
            a = A()
            a.foo()
        assert mock_call_sdfg.called


def test_computepath_method_does_not_call_dace():

    with use_dace(False):

        class A:
            @pace.dsl.computepath_method
            def foo(self):
                pass

        with unittest.mock.patch(
            "pace.dsl.dace.orchestrate.call_sdfg"
        ) as mock_call_sdfg:
            a = A()
            a.foo()
        assert not mock_call_sdfg.called
