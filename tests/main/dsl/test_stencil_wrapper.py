import contextlib
import unittest.mock

import gt4py.cartesian.gtscript
import numpy as np
import pytest
from gt4py.cartesian.gtscript import PARALLEL, computation, interval

import pace.util
from pace.dsl.dace.dace_config import DaceConfig, DaCeOrchestration
from pace.dsl.gt4py_utils import make_storage_from_shape
from pace.dsl.stencil import FrozenStencil, _convert_quantities_to_storage
from pace.dsl.stencil_config import CompilationConfig, StencilConfig
from pace.dsl.typing import FloatField


def get_stencil_config(
    *,
    backend: str,
    orchestration: DaCeOrchestration = DaCeOrchestration.Python,
    **kwargs,
):
    dace_config = DaceConfig(None, backend=backend, orchestration=orchestration)
    config = StencilConfig(
        compilation_config=CompilationConfig(
            backend=backend,
            **kwargs,
        ),
        dace_config=dace_config,
    )
    return config


@contextlib.contextmanager
def mock_gtscript_stencil(mock):
    original_stencil = gt4py.cartesian.gtscript.stencil
    try:
        gt4py.cartesian.gtscript.stencil = mock
        yield
    finally:
        gt4py.cartesian.gtscript.stencil = original_stencil


class MockFieldInfo:
    def __init__(self, axes):
        self.axes = axes


@pytest.mark.parametrize(
    "field_info, origin, field_origins",
    [
        pytest.param(
            {"a": MockFieldInfo(["I"])},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (1,)},
            id="single_field_I",
        ),
        pytest.param(
            {"a": MockFieldInfo(["J"])},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (2,)},
            id="single_field_J",
        ),
        pytest.param(
            {"a": MockFieldInfo(["K"])},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (3,)},
            id="single_field_K",
        ),
        pytest.param(
            {"a": MockFieldInfo(["I", "J"])},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (1, 2)},
            id="single_field_IJ",
        ),
        pytest.param(
            {"a": MockFieldInfo(["I", "J", "K"])},
            {"_all_": (1, 2, 3), "a": (1, 2, 3)},
            {"_all_": (1, 2, 3), "a": (1, 2, 3)},
            id="single_field_origin_mapping",
        ),
        pytest.param(
            {"a": MockFieldInfo(["I", "J", "K"]), "b": MockFieldInfo(["I"])},
            {"_all_": (1, 2, 3), "a": (1, 2, 3)},
            {"_all_": (1, 2, 3), "a": (1, 2, 3), "b": (1,)},
            id="two_fields_update_origin_mapping",
        ),
        pytest.param(
            {"a": None},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (1, 2, 3)},
            id="single_field_None",
        ),
        pytest.param(
            {"a": MockFieldInfo(["I", "J"]), "b": MockFieldInfo(["I", "J", "K"])},
            (1, 2, 3),
            {"_all_": (1, 2, 3), "a": (1, 2), "b": (1, 2, 3)},
            id="two_fields",
        ),
    ],
)
def test_compute_field_origins(field_info, origin, field_origins):
    result = FrozenStencil._compute_field_origins(field_info, origin)
    assert result == field_origins


def copy_stencil(q_in: FloatField, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = q_in


@pytest.mark.parametrize("validate_args", [True, False])
@pytest.mark.parametrize("device_sync", [False])
@pytest.mark.parametrize("rebuild", [False])
@pytest.mark.parametrize("format_source", [False])
def test_copy_frozen_stencil(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = get_stencil_config(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )
    q_in = make_storage_from_shape((3, 3, 3), backend=backend)
    q_in[:] = 1.0
    q_out = make_storage_from_shape((3, 3, 3), backend=backend)
    q_out[:] = 2.0
    stencil(q_in, q_out)
    np.testing.assert_array_equal(q_in, q_out)


@pytest.mark.parametrize("device_sync", [False])
@pytest.mark.parametrize("rebuild", [False])
@pytest.mark.parametrize("format_source", [False])
def test_frozen_stencil_raises_if_given_origin(
    backend: str,
    rebuild: bool,
    format_source: bool,
    device_sync: bool,
):
    # only guaranteed when validating args
    config = get_stencil_config(
        backend=backend,
        rebuild=rebuild,
        validate_args=True,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )
    q_in = make_storage_from_shape((3, 3, 3), backend=backend)
    q_out = make_storage_from_shape((3, 3, 3), backend=backend)
    with pytest.raises(TypeError, match="origin"):
        stencil(q_in, q_out, origin=(0, 0, 0))


@pytest.mark.parametrize("device_sync", [False])
@pytest.mark.parametrize("rebuild", [False])
@pytest.mark.parametrize("format_source", [False])
def test_frozen_stencil_raises_if_given_domain(
    backend: str,
    rebuild: bool,
    format_source: bool,
    device_sync: bool,
):
    # only guaranteed when validating args
    config = get_stencil_config(
        backend=backend,
        rebuild=rebuild,
        validate_args=True,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )
    q_in = make_storage_from_shape((3, 3, 3), backend=backend)
    q_out = make_storage_from_shape((3, 3, 3), backend=backend)
    with pytest.raises(TypeError, match="domain"):
        stencil(q_in, q_out, domain=(3, 3, 3))


@pytest.mark.parametrize(
    "rebuild, validate_args, format_source, device_sync",
    [[False, False, False, False], [True, False, False, False]],
)
def test_frozen_stencil_kwargs_passed_to_init(
    backend: str,
    rebuild: bool,
    validate_args: bool,
    format_source: bool,
    device_sync: bool,
):
    config = get_stencil_config(
        backend=backend,
        rebuild=rebuild,
        validate_args=validate_args,
        format_source=format_source,
        device_sync=device_sync,
    )
    stencil_object = FrozenStencil(
        copy_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    ).stencil_object
    mock_stencil = unittest.mock.MagicMock(return_value=stencil_object)
    with mock_gtscript_stencil(mock_stencil):
        FrozenStencil(
            copy_stencil,
            origin=(0, 0, 0),
            domain=(3, 3, 3),
            stencil_config=config,
            externals={},
        )
    mock_stencil.assert_called_once_with(
        definition=copy_stencil,
        externals={},
        **config.stencil_kwargs(func=copy_stencil),
        build_info={},
    )


def field_after_parameter_stencil(q_in: FloatField, param: float, q_out: FloatField):
    with computation(PARALLEL), interval(...):
        q_out = param * q_in


def test_frozen_field_after_parameter(backend):
    config = get_stencil_config(
        backend=backend,
        rebuild=False,
        validate_args=False,
        format_source=False,
        device_sync=False,
    )
    FrozenStencil(
        field_after_parameter_stencil,
        origin=(0, 0, 0),
        domain=(3, 3, 3),
        stencil_config=config,
        externals={},
    )


@pytest.mark.parametrize("backend", ("numpy", "cuda"))
@pytest.mark.parametrize("rebuild", [True])
@pytest.mark.parametrize("validate_args", [True])
def test_backend_options(
    backend: str,
    rebuild: bool,
    validate_args: bool,
):
    expected_options = {
        "numpy": {
            "backend": "numpy",
            "rebuild": True,
            "format_source": False,
            "name": "test_stencil_wrapper.copy_stencil",
        },
        "cuda": {
            "backend": "cuda",
            "rebuild": True,
            "device_sync": False,
            "format_source": False,
            "name": "test_stencil_wrapper.copy_stencil",
        },
    }

    actual = get_stencil_config(
        backend=backend, rebuild=rebuild, validate_args=validate_args
    ).stencil_kwargs(func=copy_stencil)
    expected = expected_options[backend]
    assert actual == expected


def get_mock_quantity():
    return unittest.mock.MagicMock(spec=pace.util.Quantity)


def test_convert_quantities_to_storage_no_args():
    args = []
    kwargs = {}
    _convert_quantities_to_storage(args, kwargs)
    assert len(args) == 0
    assert len(kwargs) == 0


def test_convert_quantities_to_storage_one_arg_quantity():
    quantity = get_mock_quantity()
    args = [quantity]
    kwargs = {}
    _convert_quantities_to_storage(args, kwargs)
    assert len(args) == 1
    assert args[0] == quantity.data
    assert len(kwargs) == 0


def test_convert_quantities_to_storage_one_kwarg_quantity():
    quantity = get_mock_quantity()
    args = []
    kwargs = {"val": quantity}
    _convert_quantities_to_storage(args, kwargs)
    assert len(args) == 0
    assert len(kwargs) == 1
    assert kwargs["val"] == quantity.data


def test_convert_quantities_to_storage_one_arg_nonquantity():
    non_quantity = unittest.mock.MagicMock(spec=tuple)
    args = [non_quantity]
    kwargs = {}
    _convert_quantities_to_storage(args, kwargs)
    assert len(args) == 1
    assert args[0] == non_quantity
    assert len(kwargs) == 0


def test_convert_quantities_to_storage_one_kwarg_non_quantity():
    non_quantity = unittest.mock.MagicMock(spec=tuple)
    args = []
    kwargs = {"val": non_quantity}
    _convert_quantities_to_storage(args, kwargs)
    assert len(args) == 0
    assert len(kwargs) == 1
    assert kwargs["val"] == non_quantity
