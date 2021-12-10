import collections
import collections.abc
import functools
import types


ArgSpec = collections.namedtuple(
    "ArgSpec", ["arg_name", "standard_name", "units", "intent"]
)
VALID_INTENTS = ["in", "out", "inout", "unknown"]


def state_inputs(*arg_specs: ArgSpec):
    for sp in arg_specs:
        if sp.intent not in VALID_INTENTS:
            raise ValueError(
                f"intent for {sp.arg_name} is {sp.intent}, "
                "must be one of {VALID_INTENTS}"
            )

    def decorator(func):
        @functools.wraps(func)
        def wrapped(state, *args, **kwargs):
            namespace = get_namespace(arg_specs, state)
            func(namespace, *args, **kwargs)

        return wrapped

    return decorator


def get_namespace(arg_specs, state):
    namespace_kwargs = {}
    for sp in arg_specs:
        arg_name, standard_name, units, intent = sp
        if standard_name not in state:
            raise ValueError(f"{standard_name} not present in state")
        elif units != state[standard_name].units:
            raise ValueError(
                f"{standard_name} has units "
                f"{state[standard_name].units} when {units} is required"
            )
        elif intent not in VALID_INTENTS:
            raise ValueError(
                f"expected intent to be one of {VALID_INTENTS}, got {intent}"
            )
        else:
            namespace_kwargs[arg_name] = state[standard_name].storage
            namespace_kwargs[arg_name + "_quantity"] = state[standard_name]
    return types.SimpleNamespace(**namespace_kwargs)
