import fv3core._config as spec
import fv3core.stencils.dyn_core as dyn_core
import fv3gfs.util as fv3util
from fv3core.decorators import StencilWrapper
from fv3core.testing import ParallelTranslate2PyState, TranslateFortranData2Py
from fv3core.utils.grid import axis_offsets


class TranslateDynCore(ParallelTranslate2PyState):
    inputs = {
        "q_con": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "cappa": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "delp": {
            "dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "default",
        },
        "pt": {"dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "K"},
        "u": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "v": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "uc": {
            "dims": [fv3util.X_INTERFACE_DIM, fv3util.Y_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "vc": {
            "dims": [fv3util.X_DIM, fv3util.Y_INTERFACE_DIM, fv3util.Z_DIM],
            "units": "m/s",
        },
        "w": {"dims": [fv3util.X_DIM, fv3util.Y_DIM, fv3util.Z_DIM], "units": "m/s"},
    }

    def __init__(self, grids):
        super().__init__(grids)
        grid = grids[0]
        self._base.in_vars["data_vars"] = {
            "cappa": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "w": {},
            "delz": {},
            "delp": {},
            "pt": {},
            "pe": {
                "istart": grid.is_ - 1,
                "iend": grid.ie + 1,
                "jstart": grid.js - 1,
                "jend": grid.je + 1,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "pk": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
            },
            "phis": {"kstart": 0, "kend": 0},
            "wsd": grid.compute_dict(),
            "omga": {},
            "ua": {},
            "va": {},
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "mfxd": grid.x3d_compute_dict(),
            "mfyd": grid.y3d_compute_dict(),
            "cxd": grid.x3d_compute_domain_y_dict(),
            "cyd": grid.y3d_compute_domain_x_dict(),
            "pkz": grid.compute_dict(),
            "peln": {
                "istart": grid.is_,
                "iend": grid.ie,
                "jstart": grid.js,
                "jend": grid.je,
                "kend": grid.npz + 1,
                "kaxis": 1,
            },
            "q_con": {},
            "ak": {},
            "bk": {},
            "diss_estd": {},
            "pfull": {},
        }
        self._base.in_vars["data_vars"]["wsd"]["kstart"] = grid.npz
        self._base.in_vars["data_vars"]["wsd"]["kend"] = None

        self._base.in_vars["parameters"] = [
            "mdt",
            "akap",
            "ptop",
            "n_map",
            "ks",
        ]

        self._base.out_vars = {}
        for v, d in self._base.in_vars["data_vars"].items():
            self._base.out_vars[v] = d

        del self._base.out_vars["ak"]
        del self._base.out_vars["bk"]
        del self._base.out_vars["pfull"]
        del self._base.out_vars["phis"]

        # TODO: Fix edge_interpolate4 in d2a2c_vect to match closer and the
        # variables here should as well.
        self.max_error = 2e-6

    def compute_parallel(self, inputs, communicator):

        self._base.compute_func = dyn_core.AcousticDynamics(
            communicator,
            spec.namelist,
            inputs["ak"],
            inputs["bk"],
            inputs["phis"],
        )
        return super().compute_parallel(inputs, communicator)


class TranslatePGradC(TranslateFortranData2Py):
    def __init__(self, grid):
        super().__init__(grid)
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "delpc": {},
            "pkc": grid.default_buffer_k_dict(),
            "gz": grid.default_buffer_k_dict(),
        }
        self.in_vars["parameters"] = ["dt2"]
        self.out_vars = {"uc": grid.x3d_domain_dict(), "vc": grid.y3d_domain_dict()}

    def compute_from_storage(self, inputs):
        origin = self.grid.compute_origin()
        domain = self.grid.domain_shape_compute(add=(1, 1, 0))
        ax_offsets = axis_offsets(self.grid, origin, domain)
        pgradc = StencilWrapper(
            dyn_core.p_grad_c_stencil,
            externals={
                "hydrostatic": spec.namelist.hydrostatic,
                **ax_offsets,
            },
            origin=origin,
            domain=domain,
        )
        pgradc(
            rdxc=self.grid.rdxc,
            rdyc=self.grid.rdyc,
            **inputs,
        )
        return inputs
