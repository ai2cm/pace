import logging
from typing import ClassVar, Dict, Optional

import numpy as np

from pace.fv3core.initialization.dycore_state import DycoreState
from pace.util.quantity import Quantity


logger = logging.getLogger(__name__)


class VariableBounds:
    def __init__(
        self,
        minimum_value: Optional[float] = None,
        maximum_value: Optional[float] = None,
        compute_domain_only: bool = False,
    ) -> None:
        self.minimum_value = minimum_value
        self.maximum_value = maximum_value
        self.compute_domain_only = compute_domain_only


class SafetyChecker:
    """Safety-Checker that checks the state for sanity of variables

    Raises:
        NotImplementedError: Doubly-registered variables
        NotImplementedError: Variables not in the state
        RuntimeError: Variables outside the specified bounds
    """

    checks: ClassVar[Dict[str, VariableBounds]] = {}

    @classmethod
    def register_variable(
        cls,
        name: str,
        minimum_value: Optional[float] = None,
        maximum_value: Optional[float] = None,
        compute_domain_only: bool = False,
    ):
        """Register a variable in the checker

        Args:
            name (str): name of the variable in the dycore state
            minimum_value (Optional[float], optional): Minimum value if specified.
                Defaults to None.
            maximum_value (Optional[float], optional): Maximum value if specified.
                Defaults to None.
            compute_domain_only (bool, optional): If evaluation should only happen
                on the compute or the entire domain. Defaults to False.

        Raises:
            NotImplementedError: If variables are doubly-registered
        """
        if name in cls.checks:
            raise NotImplementedError("Can only register variables once")
        else:
            cls.checks[name] = VariableBounds(
                minimum_value, maximum_value, compute_domain_only
            )

    @classmethod
    def clear_all_checks(cls):
        """Clear all the registered checks"""
        cls.checks.clear()

    def check_state(self, state: DycoreState):
        """check the given dycore state with all the registered constraints

        Args:
            state (DycoreState): State to check

        Raises:
            NotImplementedError: If one of the registered variables are not in the state
            RuntimeError: If one of the variables exceeds its specified bounds
        """
        for variable, variable_bounds in self.checks.items():
            try:
                var: Quantity = state.__getattribute__(variable)
            except AttributeError:
                raise NotImplementedError("Variable is not in the state")
            if variable_bounds.compute_domain_only:

                min_value = var.view[:].min()
                max_value = var.view[:].max()
            else:
                min_value = var.data.min()
                max_value = var.data.max()

            if (
                variable_bounds.minimum_value
                and min_value < variable_bounds.minimum_value
            ):
                raise RuntimeError(
                    f"Variable {variable} is outside of its specified bounds: \
                    {variable_bounds.minimum_value} specified, {min_value} found"
                )
            if (
                variable_bounds.maximum_value
                and max_value > variable_bounds.maximum_value
            ):
                raise RuntimeError(
                    f"Variable {variable} is outside of its specified bounds: \
                    {variable_bounds.maximum_value} specified, {max_value} found"
                )
            if np.isnan(var.view[:]).any():
                raise RuntimeError(f"Variable {variable} contains a NaN value")
