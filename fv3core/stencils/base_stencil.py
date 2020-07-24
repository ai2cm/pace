import fv3core.utils.gt4py_utils as utils
import gt4py.gtscript as gtscript

# from fv3core._config import grid


class BaseStencil:
    def __init__(self, inputs_data, main_origin=None, main_domain=None):
        for k, v in inputs_data.items():
            setattr(self, k, v)
        self.main_origin = main_origin
        self.main_domain = main_domain
        self.main_stencil = gtscript.stencil(
            definition=self.main_defs,
            name=self.__class__.__name__,
            backend=utils.backend,
        )

    def args(self):
        return (self.main_origin, self.main_domain)

    def compute(self, args):
        self.main_stencil(*args, origin=self.main_origin, domain=self.main_domain)
        # for edge in self.compute_edges:
        #    method = getattr(BaseStencil, edge)

    def x_edge(self, args):
        raise NotImplementedError()
        # if grid.south_edge:
        #     self.x_edge_stencil(
        #         *args, origin=(0, grid.js, 0), domain=(maxshape[0], 1, maxshape[2])
        #     )

        # if grid.north_edge:
        #     x_edge_stencil(
        #         *args, origin=(0, grid.je, 0), domain=(maxshape[0], 1, maxshape[2])
        #     )

    @staticmethod
    def x_edge_stencil(args):
        pass
