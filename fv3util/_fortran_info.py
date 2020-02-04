import json
import os

__all__ = [
    'PHYSICS_PROPERTIES', 'DYNAMICS_PROPERTIES', 'PROPERTIES_BY_STD_NAME',
    'get_restart_standard_names'
]

dirname = os.path.dirname(os.path.realpath(__file__))
PHYSICS_PROPERTIES = json.load(open(os.path.join(dirname, 'physics_properties.json'), 'r'))
DYNAMICS_PROPERTIES = json.load(open(os.path.join(dirname, 'dynamics_properties.json'), 'r'))

PROPERTIES_BY_STD_NAME = {}
for entry in PHYSICS_PROPERTIES + DYNAMICS_PROPERTIES:
    PROPERTIES_BY_STD_NAME[entry['name']] = entry


def get_restart_standard_names():
    return_dict = {}
    for var_data in PHYSICS_PROPERTIES + DYNAMICS_PROPERTIES:
        restart_name = var_data.get('restart_name', var_data['fortran_name'])
        return_dict[restart_name] = var_data['name']
    return return_dict
