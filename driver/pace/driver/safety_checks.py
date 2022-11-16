from typing import ClassVar, Dict, Optional, Tuple

from pace.fv3core.initialization.dycore_state import DycoreState
from pace.util.quantity import Quantity
import logging

logger = logging.getLogger(__name__)


class SafetyChecker:
    """Safety-Checker that checks the state for sanity of variables

    Raises:
        NotImplementedError: Doubly-registered variables
        NotImplementedError: Variables not in the state
        RuntimeError: Variables outside the specified bounds
    """

    checks: ClassVar[Dict[str, Tuple[Optional[int], Optional[int], bool]]] = {}

    @classmethod
    def register_variable(
        cls,
        name: str,
        minimum_value: Optional[int] = None,
        maximum_value: Optional[int] = None,
        compute_domain_only: bool = False,
    ):
        """Register a variable in the checker

        Args:
            name (str): name of the variable in the dycore state
            minimum_value (Optional[int], optional): Minimum value if specified.
                Defaults to None.
            maximum_value (Optional[int], optional): Maximum value if specified.
                Defaults to None.
            compute_domain_only (bool, optional): If evaluation should only happen
                on the compute or the entire domain. Defaults to False.

        Raises:
            NotImplementedError: If variables are doubly-registered
        """
        if name in cls.checks:
            raise NotImplementedError("Can only register variables once")
        else:
            cls.checks[name] = (minimum_value, maximum_value, compute_domain_only)

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
        for variable, bounds in self.checks.items():
            try:
                var: Quantity = state.__getattribute__(variable)
            except AttributeError:
                raise NotImplementedError("Variable is not in the state")
            if bounds[2]:
                min_value = var.data[
                    var.origin[0] : var.origin[0] + var.extent[0],
                    var.origin[1] : var.origin[1] + var.extent[1],
                    var.origin[2] : var.origin[2] + var.extent[2],
                ].min()
                max_value = var.data[
                    var.origin[0] : var.origin[0] + var.extent[0],
                    var.origin[1] : var.origin[1] + var.extent[1],
                    var.origin[2] : var.origin[2] + var.extent[2],
                ].max()
            else:
                min_value = var.data.min()
                max_value = var.data.max()

            if bounds[0] and min_value < bounds[0]:
                logger.info(
                    f"Variable {variable} is outside of its specified bounds: \
                    {bounds[0]} specified, {min_value} found"
                )
            if bounds[1] and max_value > bounds[1]:
                logger.info(
                    f"Variable {variable} is outside of its specified bounds: \
                    {bounds[1]} specified, {max_value} found"
                )
