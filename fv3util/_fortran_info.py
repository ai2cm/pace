import json
import os

__all__ = ['PHYSICS_PROPERTIES', 'DYNAMICS_PROPERTIES']

dirname = os.path.dirname(os.path.realpath(__file__))
PHYSICS_PROPERTIES = json.load(open(os.path.join(dirname, 'physics_properties.json'), 'r'))
DYNAMICS_PROPERTIES = json.load(open(os.path.join(dirname, 'dynamics_properties.json'), 'r'))
