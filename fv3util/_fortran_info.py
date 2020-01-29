import json
import os

__all__ = ['physics_properties', 'dynamics_properties']

dirname = os.path.dirname(os.path.realpath(__file__))
physics_properties = json.load(open(os.path.join(dirname, 'physics_properties.json'), 'r'))
dynamics_properties = json.load(open(os.path.join(dirname, 'dynamics_properties.json'), 'r'))
