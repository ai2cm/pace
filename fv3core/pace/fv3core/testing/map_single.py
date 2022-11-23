from typing import Dict, Tuple

import pace.dsl
import pace.util
from pace.fv3core.stencils.map_single import MapSingle
from pace.util import X_DIM, Y_DIM, Z_DIM


class MapSingleFactory:
    _object_pool: Dict[Tuple[int, int, Tuple[str, ...]], MapSingle] = {}
    """Pool of MapSingle objects."""

    def __init__(
        self,
        stencil_factory: pace.dsl.StencilFactory,
        quantity_factory: pace.util.QuantityFactory,
    ):
        self.stencil_factory = stencil_factory
        self.quantity_factory = quantity_factory

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
        key_tuple = (kord, mode, (X_DIM, Y_DIM, Z_DIM))
        if key_tuple not in self._object_pool:
            # stencil_factory: StencilFactory,
            # quantity_factory: pace.util.QuantityFactory,
            # kord: int,
            # mode: int,
            # dims: List[str],
            self._object_pool[key_tuple] = MapSingle(
                self.stencil_factory, self.quantity_factory, *key_tuple
            )
        return self._object_pool[key_tuple](*args, **kwargs)
