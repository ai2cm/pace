import json
import os
import copy

__all__ = [
    'PHYSICS_PROPERTIES', 'DYNAMICS_PROPERTIES', 'properties_by_std_name',
    'get_restart_standard_names'
]

dirname = os.path.dirname(os.path.realpath(__file__))
PHYSICS_PROPERTIES = json.load(open(os.path.join(dirname, 'physics_properties.json'), 'r'))
DYNAMICS_PROPERTIES = json.load(open(os.path.join(dirname, 'dynamics_properties.json'), 'r'))
tracer_properties = None

# these variables are found not to be needed for smooth restarts
# later we could represent this as a key in the dynamics/physics_PROPERTIES
RESTART_EXCLUDE_NAMES = [
    'convective_cloud_fraction',
    'convective_cloud_top_pressure',
    'convective_cloud_bottom_pressure',
]


properties_by_std_name = {}
for entry in PHYSICS_PROPERTIES + DYNAMICS_PROPERTIES:
    properties_by_std_name[entry['name']] = entry

_basic_properties_by_std_name = copy.deepcopy(properties_by_std_name)


def set_tracer_properties(properties):
    global tracer_properties
    global properties_by_std_name
    tracer_properties = properties
    properties_by_std_name = copy.deepcopy(_basic_properties_by_std_name)
    for entry in tracer_properties:
        properties_by_std_name[entry['name']] = entry


def get_restart_standard_names():
    """Return a list of variable names needed for a smooth restart."""
    return_dict = {}
    for std_name, properties in properties_by_std_name.items():
        restart_name = properties.get('restart_name', properties['fortran_name'])
        return_dict[restart_name] = std_name
    for name in RESTART_EXCLUDE_NAMES:
        if name in return_dict:
            return_dict.pop(name)
    return return_dict
