import unittest.mock

from pace.dsl.dace.dace_config import DaceConfig
from pace.dsl.dace.orchestration import (
    DaCeOrchestration,
    orchestrate,
    orchestrate_function,
)


"""
Tests that the dace configuration pace.dsl.dace.dace_config
which determines whether we use dace to run wrapped functions.
"""


def test_orchestrate_function_calls_dace():
    def foo():
        pass

    dace_config = DaceConfig(
        communicator=None,
        backend="gtc:dace",
        orchestration=DaCeOrchestration.BuildAndRun,
    )
    wrapped = orchestrate_function(config=dace_config)(foo)
    with unittest.mock.patch(
        "pace.dsl.dace.orchestration._call_sdfg"
    ) as mock_call_sdfg:
        wrapped()
    assert mock_call_sdfg.called
    assert mock_call_sdfg.call_args.args[0].f == foo


def test_orchestrate_function_does_not_call_dace():
    def foo():
        pass

    dace_config = DaceConfig(
        communicator=None,
        backend="gtc:dace",
        orchestration=DaCeOrchestration.Python,
    )
    wrapped = orchestrate_function(config=dace_config)(foo)
    with unittest.mock.patch(
        "pace.dsl.dace.orchestration._call_sdfg"
    ) as mock_call_sdfg:
        wrapped()
    assert not mock_call_sdfg.called


def test_orchestrate_calls_dace():
    dace_config = DaceConfig(
        communicator=None,
        backend="gtc:dace",
        orchestration=DaCeOrchestration.BuildAndRun,
    )

    class A:
        def __init__(self):
            orchestrate(obj=self, config=dace_config, method_to_orchestrate="foo")

        def foo(self):
            pass

    with unittest.mock.patch(
        "pace.dsl.dace.orchestration._call_sdfg"
    ) as mock_call_sdfg:
        a = A()
        a.foo()
    assert mock_call_sdfg.called


def test_orchestrate_does_not_call_dace():
    dace_config = DaceConfig(
        communicator=None,
        backend="gtc:dace",
        orchestration=DaCeOrchestration.Python,
    )

    class A:
        def __init__(self):
            orchestrate(obj=self, config=dace_config, method_to_orchestrate="foo")

        def foo(self):
            pass

    with unittest.mock.patch(
        "pace.dsl.dace.orchestration._call_sdfg"
    ) as mock_call_sdfg:
        a = A()
        a.foo()
    assert not mock_call_sdfg.called
