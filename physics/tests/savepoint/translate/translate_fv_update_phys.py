import dataclasses

import numpy as np

import pace.dsl
import pace.dsl.gt4py_utils as utils
import pace.util
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.stencils.fv_update_phys import ApplyPhysicsToDycore
from pace.stencils.testing.translate_physics import (
    ParallelPhysicsTranslate2Py,
    transform_dwind_serialized_data,
)
from pace.util.utils import safe_assign_array


try:
    import cupy as cp
except ImportError:
    cp = None


try:
    import cupy as cp
except ImportError:
    cp = None


@dataclasses.dataclass()
class DycoreState:
    u: FloatField
    v: FloatField
    delp: FloatField
    ps: FloatFieldIJ
    pe: FloatField
    pt: FloatField
    peln: FloatField
    pk: FloatField
    qvapor: FloatField
    qliquid: FloatField
    qice: FloatField
    qrain: FloatField
    qsnow: FloatField
    qgraupel: FloatField
    ua: FloatField
    va: FloatField

    def __getitem__(self, item):
        return getattr(self, item)


class TranslateFVUpdatePhys(ParallelPhysicsTranslate2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.stencil_factory = stencil_factory
        self.grid_indexing = self.stencil_factory.grid_indexing
        self._base.in_vars["data_vars"] = {
            "u_dt": {},
            "v_dt": {},
            "t_dt": {},
            "ua": {},
            "va": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "peln": {
                "istart": self.grid_indexing.isc,
                "iend": self.grid_indexing.iec,
                "jstart": self.grid_indexing.jsc,
                "jend": self.grid_indexing.jec,
                "kend": namelist.npz,
                "kaxis": 1,
            },
            "delp": {},
            "pt": {},
            "ps": {},
            "pe": {
                "istart": self.grid_indexing.isc - 1,
                "iend": self.grid_indexing.iec + 1,
                "jstart": self.grid_indexing.jsc - 1,
                "jend": self.grid_indexing.jec + 1,
                "kend": namelist.npz + 1,
                "kaxis": 1,
            },
            "pk": grid.compute_buffer_k_dict(),
        }
        self._base.out_vars = {
            "qvapor": {},
            "qliquid": {},
            "qice": {},
            "qrain": {},
            "qsnow": {},
            "qgraupel": {},
            "pt": {},
            "u": {},
            "v": {},
            "ua": {},
            "va": {},
        }
        self.namelist = namelist

    def transform_dwind_serialized_data(self, data):
        return transform_dwind_serialized_data(
            data, self.stencil_factory.grid_indexing, self.stencil_factory.backend
        )

    def storage_vars(self):
        return self._base.in_vars["data_vars"]

    def make_storage_data_input_vars(self, inputs, storage_vars=None):
        for varname in [*self._base.in_vars["data_vars"]]:
            info = self._base.in_vars["data_vars"][varname]
            dwind_format = info["dwind"] if "dwind" in info else False
            if "serialname" in info:
                serialname = info["serialname"]
            else:
                serialname = varname
            if dwind_format:
                inputs[serialname] = self.transform_dwind_serialized_data(
                    inputs[serialname]
                )

        if storage_vars is None:
            storage_vars = self.storage_vars()
        for p in self._base.in_vars["parameters"]:
            if type(inputs[p]) in [np.int64, np.int32]:
                inputs[p] = int(inputs[p])
        for d, info in storage_vars.items():
            serialname = info["serialname"] if "serialname" in info else d
            self._base.update_info(info, inputs)
            if "kaxis" in info:
                inputs[serialname] = np.moveaxis(inputs[serialname], info["kaxis"], 2)
            istart, jstart, kstart = self._base.collect_start_indices(
                inputs[serialname].shape, info
            )

            names_4d = None
            if len(inputs[serialname].shape) == 4:
                names_4d = info.get("names_4d", utils.tracer_variables)

            dummy_axes = info.get("dummy_axes", None)
            axis = info.get("axis", 2)
            inputs[d] = self._base.make_storage_data(
                np.squeeze(inputs[serialname]),
                istart=istart,
                jstart=jstart,
                kstart=kstart,
                dummy_axes=dummy_axes,
                axis=axis,
                names_4d=names_4d,
                read_only=d not in self._base.write_vars,
                full_shape="full_shape" in storage_vars[d],
            )
            if d != serialname:
                del inputs[serialname]
        remove_names = set(inputs.keys()).difference(self._base.in_vars["data_vars"])
        for name in remove_names:
            inputs.pop(name)

    def compute_parallel(self, inputs, communicator):
        self.make_storage_data_input_vars(inputs)

        tendencies = {}
        for key in ["u_dt", "v_dt", "t_dt"]:
            storage = inputs.pop(key)
            tendencies[key] = pace.util.Quantity(
                storage,
                dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
                units="test",
                origin=(0, 0, 0),
                extent=storage.shape,
            )
        state = DycoreState(**inputs)
        self._base.compute_func = ApplyPhysicsToDycore(
            self.stencil_factory,
            self.grid.quantity_factory,
            self.grid.grid_data,
            self.namelist,
            communicator,
            self.grid.driver_grid_data,
            state,
            tendencies["u_dt"],
            tendencies["v_dt"],
        )
        dims_u = [pace.util.X_DIM, pace.util.Y_INTERFACE_DIM, pace.util.Z_DIM]
        u_quantity = self.grid.make_quantity(
            state.u,
            dims=dims_u,
            origin=self.grid.sizer.get_origin(dims_u),
            extent=self.grid.sizer.get_extent(dims_u),
        )
        dims_v = [pace.util.X_INTERFACE_DIM, pace.util.Y_DIM, pace.util.Z_DIM]
        v_quantity = self.grid.make_quantity(
            state.v,
            dims=dims_v,
            origin=self.grid.sizer.get_origin(dims_v),
            extent=self.grid.sizer.get_extent(dims_v),
        )
        state.u = u_quantity
        state.v = v_quantity
        self._base.compute_func(
            state,
            tendencies["u_dt"],
            tendencies["v_dt"],
            tendencies["t_dt"],
            dt=float(self.namelist.dt_atmos),
        )
        out = {}
        ds = self.grid.default_domain_dict()
        out["qvapor"] = state.qvapor[self.grid.slice_dict(ds)]
        out["qliquid"] = state.qliquid[self.grid.slice_dict(ds)]
        out["qice"] = state.qice[self.grid.slice_dict(ds)]
        out["qrain"] = state.qrain[self.grid.slice_dict(ds)]
        out["qsnow"] = state.qsnow[self.grid.slice_dict(ds)]
        out["qgraupel"] = state.qgraupel[self.grid.slice_dict(ds)]
        out["pt"] = state.pt[self.grid.slice_dict(ds)]
        utils.device_sync(backend=self.stencil_factory.backend)
        # This alloc then copy pattern is requried to deal transparently with
        # arrays on different device
        out["u"] = np.empty_like(inputs["u"][self.grid.y3d_domain_interface()])
        out["v"] = np.empty_like(inputs["v"][self.grid.x3d_domain_interface()])
        safe_assign_array(out["u"], inputs["u"][self.grid.y3d_domain_interface()])
        safe_assign_array(out["v"], inputs["v"][self.grid.x3d_domain_interface()])
        out["ua"] = state.ua[self.grid.slice_dict(ds)]
        out["va"] = state.va[self.grid.slice_dict(ds)]
        return out
