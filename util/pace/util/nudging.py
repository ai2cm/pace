from datetime import timedelta
from typing import Mapping

from .quantity import Quantity


def apply_nudging(
    state,
    reference_state,
    nudging_timescales: Mapping[str, timedelta],
    timestep: timedelta,
):
    """
    Nudge the given state towards the reference state according to the provided
    nudging timescales.

    Nudging is applied to the state in-place.

    Args:
        state (dict): A state dictionary.
        reference_state (dict): A reference state dictionary.
        nudging_timescales (dict): A dictionary whose keys are standard names and
            values are timedelta objects indicating the relaxation timescale for that
            variable.
        timestep (timedelta): length of the timestep

    Returns:
        nudging_tendencies (dict): A dictionary whose keys are standard names
            and values are Quantity objects indicating the nudging tendency
            of that standard name.
    """
    tendencies = get_nudging_tendencies(state, reference_state, nudging_timescales)
    _apply_tendencies(state, tendencies, timestep)
    return tendencies


def _apply_tendencies(state, tendencies, timestep: timedelta):
    """Apply a dictionary of tendencies to a state, in-place. Assumes the tendencies
    are in units of <state units> per second.
    """
    for name, tendency in tendencies.items():
        if name not in state:
            raise ValueError(f"no state variable to apply tendency for {name}")
        state[name].view[:] += tendency.view[:] * timestep.total_seconds()


def get_nudging_tendencies(
    state, reference_state, nudging_timescales: Mapping[str, timedelta]
):
    """
    Return the nudging tendency of the given state towards the reference state
    according to the provided nudging timescales.

    Args:
        state (dict): A state dictionary.
        reference_state (dict): A reference state dictionary.
        nudging_timescales (dict): A dictionary whose keys are standard names and
            values are timedelta objects indicating the relaxation timescale for that
            variable.

    Returns:
        nudging_tendencies (dict): A dictionary whose keys are standard names
            and values are Quantity objects indicating the nudging tendency
            of that standard name.
    """
    return_dict = {}
    for name, timescale in nudging_timescales.items():
        quantity = state[name]
        reference = reference_state[name]
        return_data = (reference.view[:] - quantity.view[:]) / timescale.total_seconds()
        return_dict[name] = Quantity(
            return_data, dims=quantity.dims, units=quantity.units + " s^-1"
        )
    return return_dict
