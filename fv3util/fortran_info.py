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
    return_dict = {}
    for std_name, properties in properties_by_std_name.items():
        restart_name = properties.get('restart_name', properties['fortran_name'])
        return_dict[restart_name] = std_name
    return return_dict
