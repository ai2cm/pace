from typing import Dict, Tuple

from pace.fv3core.stencils.map_single import MapSingle


class MapSingleFactory:
    _object_pool: Dict[Tuple[int, ...], MapSingle] = {}
    """Pool of MapSingle objects."""

    def __init__(self, stencil_factory):
        self.stencil_factory = stencil_factory

    def __call__(
        self,
        kord: int,
        mode: int,
        i1: int,
        i2: int,
        j1: int,
        j2: int,
        *args,
        **kwargs,
    ):
        key_tuple = (kord, mode, i1, i2, j1, j2)
        if key_tuple not in self._object_pool:
            self._object_pool[key_tuple] = MapSingle(self.stencil_factory, *key_tuple)
        return self._object_pool[key_tuple](*args, **kwargs)
