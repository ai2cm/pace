from typing import ClassVar, Dict, Optional, Tuple

from pace.fv3core.initialization.dycore_state import DycoreState


class SafetyChecker:
    checks: ClassVar[Dict[str, Tuple[Optional[int], Optional[int]]]] = {}

    def __init__(self) -> None:
        pass

    @classmethod
    def register_variable(cls, name, minimum_value=None, maximum_value=None):
        if name in cls.checks:
            raise NotImplementedError("Can only register variables once")
        else:
            cls.checks[name] = (minimum_value, maximum_value)

    @classmethod
    def clear_all_checks(cls):
        cls.checks.clear()

    def check_state(self, state: DycoreState):
        for variable, bounds in self.checks.items():
            try:
                var = state.__getattribute__(variable)
            except AttributeError:
                raise NotImplementedError("Variable is not in the state")
            if bounds[0] and var.data.min() < bounds[0]:
                raise RuntimeError(
                    f"Variable {variable} is outside of its specified bounds: {bounds[0]} specified, {var.data.min()} found"
                )
            if bounds[1] and var.data.max() > bounds[1]:
                raise RuntimeError(
                    f"Variable {variable} is outside of its specified bounds: {bounds[1]} specified, {var.data.max()} found"
                )
