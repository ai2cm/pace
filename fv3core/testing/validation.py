import inspect
from typing import Callable, Mapping, Tuple

import numpy as np

import fv3core.stencils.updatedzd
from fv3gfs.util.quantity import Quantity


def get_selective_class(
    cls: type,
    selective_arg_names,
    origin_domain_func: Callable[..., Tuple[Tuple[int, ...], Tuple[int, ...]]],
):
    """
    Convert a model class into one that sets nans on non-validated outputs,
    and gives a helper function to retrieve the output subset we want to validate.

    Using this ensures that if these non-validated values are ever used, a test
    will fail because an output will have NaN.
    """

    class SelectivelyValidated:
        """
        Wrapper class that sets non-validated outputs to nan, and gives a helper
        function to retrieve the output subset to be validated.
        """

        def __init__(self, *args, **kwargs):
            self.wrapped = cls(*args, **kwargs)
            origin, domain = origin_domain_func(self.wrapped)
            self._validation_slice = tuple(
                slice(start, start + n) for start, n in zip(origin, domain)
            )
            self._all_argument_names = tuple(
                inspect.getfullargspec(self.wrapped).args[1:]
            )
            assert "self" not in self._all_argument_names
            self._selective_argument_names = selective_arg_names

        def __call__(self, *args, **kwargs):
            kwargs.update(self._args_to_kwargs(args))
            self.wrapped(**kwargs)
            self._set_nans(kwargs)

        def _args_to_kwargs(self, args):
            return dict(zip(self._all_argument_names, args))

        def subset_output(self, varname: str, output: np.ndarray) -> np.ndarray:
            """
            Given an output array, return the slice of the array which we'd
            like to validate against reference data
            """
            if varname in self._selective_argument_names:
                output = output[self._validation_slice]
            return output

        def _set_nans(self, kwargs):
            for name in set(kwargs.keys()).intersection(self._selective_argument_names):
                array = kwargs[name]
                validation_data = np.copy(array[self._validation_slice])
                array[:] = np.nan
                array[self._validation_slice] = validation_data

    return SelectivelyValidated


def get_selective_tracer_advection(
    cls: type,
    origin_domain_func: Callable[..., Tuple[Tuple[int, ...], Tuple[int, ...]]],
):
    class SelectivelyValidatedTracerAdvection:
        """
        We have to treat tracers separately because they are a dictionary,
        not a storage.
        """

        def __init__(self, *args, **kwargs):
            self.wrapped = cls(*args, **kwargs)
            origin, domain = origin_domain_func(self.wrapped)
            self._validation_slice = tuple(
                slice(start, start + n) for start, n in zip(origin, domain)
            )
            self._all_argument_names = tuple(
                inspect.getfullargspec(self.wrapped).args[1:]
            )
            assert "self" not in self._all_argument_names

        def __call__(self, *args, **kwargs):
            kwargs.update(self._args_to_kwargs(args))
            self.wrapped(**kwargs)
            self._set_nans(kwargs["tracers"])

        def _args_to_kwargs(self, args):
            return dict(zip(self._all_argument_names, args))

        def subset_output(self, varname: str, output: np.ndarray) -> np.ndarray:
            """
            Given an output array, return the slice of the array which we'd
            like to validate against reference data
            """
            if varname == "tracers":
                # tracers are still an array for this routine
                output = output[self._validation_slice]
            return output

        def _set_nans(self, tracers: Mapping[str, Quantity]):
            # tracers is a dict of Quantity for this routine
            for quantity in tracers.values():
                validation_data = np.copy(quantity.data[self._validation_slice])
                quantity.data[:] = np.nan
                quantity.data[self._validation_slice] = validation_data

    return SelectivelyValidatedTracerAdvection


def get_compute_domain(
    instance,
) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    origin = instance.grid.compute_origin()
    domain = instance.grid.domain_shape_compute(add=(0, 0, 1))
    return origin, domain


def enable_selective_validation():
    """
    Replaces certain function-classes with wrapped versions that set data we aren't
    validating to NaN, and have an attribute function `subset_output` that
    takes in a string variable name and an output array and returns the
    subset of that array which should be validated.

    This wrapping removes any attributes of the wrapped module.
    """
    # to enable selective validation for a new class, add a new monkeypatch
    # this should require only a new function for (origin, domain)
    # note we have not implemented disabling selective validation once enabled
    fv3core.stencils.updatedzd.UpdateHeightOnDGrid = get_selective_class(
        fv3core.stencils.updatedzd.UpdateHeightOnDGrid,
        ["height", "zh"],  # must include both function and savepoint names
        get_compute_domain,
    )
    # make absolutely sure you don't write just the savepoint name, this would
    # selecively validate without making sure it's safe to do so

    fv3core.stencils.tracer_2d_1l.TracerAdvection = get_selective_tracer_advection(
        fv3core.stencils.tracer_2d_1l.TracerAdvection,
        get_compute_domain,
    )
