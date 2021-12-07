import dataclasses
import functools
from typing import Any, Iterable, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from gt4py import gtscript

import fv3core.utils.global_config as global_config
import pace.util
from fv3core.grid import MetricTerms
from pace.dsl import gt4py_utils as utils
from pace.dsl.stencil import GridIndexing, StencilConfig, StencilFactory
from pace.dsl.typing import FloatFieldI, FloatFieldIJ
from pace.util.halo_data_transformer import QuantityHaloSpec


TRACER_DIM = "tracers"


class Grid:
    # indices = ["is_", "ie", "isd", "ied", "js", "je", "jsd", "jed"]
    index_pairs = [("is_", "js"), ("ie", "je"), ("isd", "jsd"), ("ied", "jed")]
    shape_params = ["npz", "npx", "npy"]
    # npx -- number of grid corners on one tile of the domain
    # grid.ie == npx - 1identified east edge in fortran
    # But we need to add the halo - 1 to change this check to 0 based python arrays
    # grid.ie == npx + halo - 2

    def __init__(
        self, indices, shape_params, rank, layout, data_fields={}, local_indices=False
    ):
        self.rank = rank
        self.partitioner = pace.util.TilePartitioner(layout)
        self.subtile_index = self.partitioner.subtile_index(self.rank)
        self.layout = layout
        for s in self.shape_params:
            setattr(self, s, int(shape_params[s]))
        self.subtile_width_x = int((self.npx - 1) / self.layout[0])
        self.subtile_width_y = int((self.npy - 1) / self.layout[1])
        for ivar, jvar in self.index_pairs:
            local_i, local_j = int(indices[ivar]), int(indices[jvar])
            if not local_indices:
                local_i, local_j = self.global_to_local_indices(local_i, local_j)
            setattr(self, ivar, local_i)
            setattr(self, jvar, local_j)
        self.nid = int(self.ied - self.isd + 1)
        self.njd = int(self.jed - self.jsd + 1)
        self.nic = int(self.ie - self.is_ + 1)
        self.njc = int(self.je - self.js + 1)
        self.halo = utils.halo
        self.global_is, self.global_js = self.local_to_global_indices(self.is_, self.js)
        self.global_ie, self.global_je = self.local_to_global_indices(self.ie, self.je)
        self.global_isd, self.global_jsd = self.local_to_global_indices(
            self.isd, self.jsd
        )
        self.global_ied, self.global_jed = self.local_to_global_indices(
            self.ied, self.jed
        )
        self.west_edge = self.global_is == self.halo
        self.east_edge = self.global_ie == self.npx + self.halo - 2
        self.south_edge = self.global_js == self.halo
        self.north_edge = self.global_je == self.npy + self.halo - 2

        self.j_offset = self.js - self.jsd - 1
        self.i_offset = self.is_ - self.isd - 1
        self.sw_corner = self.west_edge and self.south_edge
        self.se_corner = self.east_edge and self.south_edge
        self.nw_corner = self.west_edge and self.north_edge
        self.ne_corner = self.east_edge and self.north_edge
        self.data_fields = {}
        self.add_data(data_fields)
        self._sizer = None
        self._quantity_factory = None
        self._grid_data = None
        self._damping_coefficients = None

    @property
    def sizer(self):
        if self._sizer is None:
            # in the future this should use from_namelist, when we have a non-flattened
            # namelist
            self._sizer = pace.util.SubtileGridSizer.from_tile_params(
                nx_tile=self.npx - 1,
                ny_tile=self.npy - 1,
                nz=self.npz,
                n_halo=self.halo,
                extra_dim_lengths={
                    MetricTerms.LON_OR_LAT_DIM: 2,
                    MetricTerms.TILE_DIM: 6,
                    MetricTerms.CARTESIAN_DIM: 3,
                    TRACER_DIM: len(utils.tracer_variables),
                },
                layout=self.layout,
            )
        return self._sizer

    @property
    def quantity_factory(self):
        if self._quantity_factory is None:
            self._quantity_factory = pace.util.QuantityFactory.from_backend(
                self.sizer, backend=global_config.get_backend()
            )
        return self._quantity_factory

    def make_quantity(
        self,
        array,
        dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
        units="Unknown",
        origin=None,
        extent=None,
    ):
        if origin is None:
            origin = self.compute_origin()
        if extent is None:
            extent = self.domain_shape_compute()
        return pace.util.Quantity(
            array, dims=dims, units=units, origin=origin, extent=extent
        )

    def quantity_dict_update(
        self,
        data_dict,
        varname,
        dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
        units="Unknown",
    ):
        data_dict[varname + "_quantity"] = self.quantity_wrap(
            data_dict[varname], dims=dims, units=units
        )

    def quantity_wrap(
        self,
        data,
        dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
        units="Unknown",
    ):
        origin = self.sizer.get_origin(dims)
        extent = self.sizer.get_extent(dims)
        return pace.util.Quantity(
            data, dims=dims, units=units, origin=origin, extent=extent
        )

    def global_to_local_1d(self, global_value, subtile_index, subtile_length):
        return global_value - subtile_index * subtile_length

    def global_to_local_x(self, i_global):
        return self.global_to_local_1d(
            i_global, self.subtile_index[1], self.subtile_width_x
        )

    def global_to_local_y(self, j_global):
        return self.global_to_local_1d(
            j_global, self.subtile_index[0], self.subtile_width_y
        )

    def global_to_local_indices(self, i_global, j_global):
        i_local = self.global_to_local_x(i_global)
        j_local = self.global_to_local_y(j_global)
        return i_local, j_local

    def local_to_global_1d(self, local_value, subtile_index, subtile_length):
        return local_value + subtile_index * subtile_length

    def local_to_global_indices(self, i_local, j_local):
        i_global = self.local_to_global_1d(
            i_local, self.subtile_index[1], self.subtile_width_x
        )
        j_global = self.local_to_global_1d(
            j_local, self.subtile_index[0], self.subtile_width_y
        )
        return i_global, j_global

    def add_data(self, data_dict):
        self.data_fields.update(data_dict)
        for k, v in self.data_fields.items():
            setattr(self, k, v)

    def irange_compute(self):
        return range(self.is_, self.ie + 1)

    def irange_compute_x(self):
        return range(self.is_, self.ie + 2)

    def jrange_compute(self):
        return range(self.js, self.je + 1)

    def jrange_compute_y(self):
        return range(self.js, self.je + 2)

    def irange_domain(self):
        return range(self.isd, self.ied + 1)

    def jrange_domain(self):
        return range(self.jsd, self.jed + 1)

    def krange(self):
        return range(0, self.npz)

    def compute_interface(self):
        return self.slice_dict(self.compute_dict())

    def x3d_interface(self):
        return self.slice_dict(self.x3d_compute_dict())

    def y3d_interface(self):
        return self.slice_dict(self.y3d_compute_dict())

    def x3d_domain_interface(self):
        return self.slice_dict(self.x3d_domain_dict())

    def y3d_domain_interface(self):
        return self.slice_dict(self.y3d_domain_dict())

    def add_one(self, num):
        if num is None:
            return None
        return num + 1

    def slice_dict(self, d, ndim: int = 3):
        iters: str = "ijk" if ndim > 1 else "k"
        return tuple(
            [
                slice(d[f"{iters[i]}start"], self.add_one(d[f"{iters[i]}end"]))
                for i in range(ndim)
            ]
        )

    def default_domain_dict(self):
        return {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.jsd,
            "jend": self.jed,
            "kstart": 0,
            "kend": self.npz - 1,
        }

    def default_dict_buffer_2d(self):
        mydict = self.default_domain_dict()
        mydict["iend"] += 1
        mydict["jend"] += 1
        return mydict

    def compute_dict(self):
        return {
            "istart": self.is_,
            "iend": self.ie,
            "jstart": self.js,
            "jend": self.je,
            "kstart": 0,
            "kend": self.npz - 1,
        }

    def compute_dict_buffer_2d(self):
        mydict = self.compute_dict()
        mydict["iend"] += 1
        mydict["jend"] += 1
        return mydict

    def default_buffer_k_dict(self):
        mydict = self.default_domain_dict()
        mydict["kend"] = self.npz
        return mydict

    def compute_buffer_k_dict(self):
        mydict = self.compute_dict()
        mydict["kend"] = self.npz
        return mydict

    def x3d_domain_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied + 1,
            "jstart": self.jsd,
            "jend": self.jed,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_domain_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.jsd,
            "jend": self.jed + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def x3d_compute_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie + 1,
            "jstart": self.js,
            "jend": self.je,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_compute_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie,
            "jstart": self.js,
            "jend": self.je + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def x3d_compute_domain_y_dict(self):
        horizontal_dict = {
            "istart": self.is_,
            "iend": self.ie + 1,
            "jstart": self.jsd,
            "jend": self.jed,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def y3d_compute_domain_x_dict(self):
        horizontal_dict = {
            "istart": self.isd,
            "iend": self.ied,
            "jstart": self.js,
            "jend": self.je + 1,
        }
        return {**self.default_domain_dict(), **horizontal_dict}

    def domain_shape_full(self, *, add: Tuple[int, int, int] = (0, 0, 0)):
        """Domain shape for the full array including halo points."""
        return (self.nid + add[0], self.njd + add[1], self.npz + add[2])

    def domain_shape_compute(self, *, add: Tuple[int, int, int] = (0, 0, 0)):
        """Compute domain shape excluding halo points."""
        return (self.nic + add[0], self.njc + add[1], self.npz + add[2])

    def copy_right_edge(self, var, i_index, j_index):
        return np.copy(var[i_index:, :, :]), np.copy(var[:, j_index:, :])

    def insert_left_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        if len(var.shape) < 3:
            var[:i_index, :] = edge_data_i
            var[:, :j_index] = edge_data_j
        else:
            var[:i_index, :, :] = edge_data_i
            var[:, :j_index, :] = edge_data_j

    def insert_right_edge(self, var, edge_data_i, i_index, edge_data_j, j_index):
        if len(var.shape) < 3:
            var[i_index:, :] = edge_data_i
            var[:, j_index:] = edge_data_j
        else:
            var[i_index:, :, :] = edge_data_i
            var[:, j_index:, :] = edge_data_j

    def uvar_edge_halo(self, var):
        return self.copy_right_edge(var, self.ie + 2, self.je + 1)

    def vvar_edge_halo(self, var):
        return self.copy_right_edge(var, self.ie + 1, self.je + 2)

    def compute_origin(self, add: Tuple[int, int, int] = (0, 0, 0)):
        """Start of the compute domain (e.g. (halo, halo, 0))"""
        return (self.is_ + add[0], self.js + add[1], add[2])

    def full_origin(self, add: Tuple[int, int, int] = (0, 0, 0)):
        """Start of the full array including halo points (e.g. (0, 0, 0))"""
        return (self.isd + add[0], self.jsd + add[1], add[2])

    def horizontal_starts_from_shape(self, shape):
        if shape[0:2] in [
            self.domain_shape_compute()[0:2],
            self.domain_shape_compute(add=(1, 0, 0))[0:2],
            self.domain_shape_compute(add=(0, 1, 0))[0:2],
            self.domain_shape_compute(add=(1, 1, 0))[0:2],
        ]:
            return self.is_, self.js
        elif shape[0:2] == (self.nic + 2, self.njc + 2):
            return self.is_ - 1, self.js - 1
        else:
            return 0, 0

    def get_halo_update_spec(
        self,
        shape,
        origin,
        halo_points,
        dims=[pace.util.X_DIM, pace.util.Y_DIM, pace.util.Z_DIM],
    ) -> QuantityHaloSpec:
        """Build memory specifications for the halo update."""
        return self.grid_indexing.get_quantity_halo_spec(
            shape,
            origin,
            dims=dims,
            n_halo=halo_points,
            backend=global_config.get_backend(),
        )

    @property
    def grid_indexing(self) -> "GridIndexing":
        return GridIndexing(
            domain=self.domain_shape_compute(),
            n_halo=self.halo,
            south_edge=self.south_edge,
            north_edge=self.north_edge,
            west_edge=self.west_edge,
            east_edge=self.east_edge,
        )

    @property
    def stencil_factory(self) -> "StencilFactory":
        return StencilFactory(
            config=StencilConfig(
                backend=global_config.get_backend(),
                rebuild=global_config.get_rebuild(),
                validate_args=global_config.get_validate_args(),
            ),
            grid_indexing=self.grid_indexing,
        )

    @property
    def damping_coefficients(self) -> "DampingCoefficients":
        if self._damping_coefficients is not None:
            return self._damping_coefficients
        self._damping_coefficients = DampingCoefficients(
            divg_u=self.divg_u,
            divg_v=self.divg_v,
            del6_u=self.del6_u,
            del6_v=self.del6_v,
            da_min=self.da_min,
            da_min_c=self.da_min_c,
        )
        return self._damping_coefficients

    def set_damping_coefficients(self, damping_coefficients: "DampingCoefficients"):
        self._damping_coefficients = damping_coefficients

    @property
    def grid_data(self) -> "GridData":
        if self._grid_data is not None:
            return self._grid_data
        horizontal = HorizontalGridData(
            lon=self.bgrid1,
            lat=self.bgrid2,
            lon_agrid=self.agrid1,
            lat_agrid=self.agrid2,
            area=self.area,
            area_64=self.area_64,
            rarea=self.rarea,
            rarea_c=self.rarea_c,
            dx=self.dx,
            dy=self.dy,
            dxc=self.dxc,
            dyc=self.dyc,
            dxa=self.dxa,
            dya=self.dya,
            rdx=self.rdx,
            rdy=self.rdy,
            rdxc=self.rdxc,
            rdyc=self.rdyc,
            rdxa=self.rdxa,
            rdya=self.rdya,
            a11=self.a11,
            a12=self.a12,
            a21=self.a21,
            a22=self.a22,
            edge_w=self.edge_w,
            edge_e=self.edge_e,
            edge_s=self.edge_s,
            edge_n=self.edge_n,
        )
        vertical = VerticalGridData(ptop=-1.0e7, ks=-1)
        contravariant = ContravariantGridData(
            self.cosa,
            self.cosa_u,
            self.cosa_v,
            self.cosa_s,
            self.sina_u,
            self.sina_v,
            self.rsina,
            self.rsin_u,
            self.rsin_v,
            self.rsin2,
        )
        angle = AngleGridData(
            self.sin_sg1,
            self.sin_sg2,
            self.sin_sg3,
            self.sin_sg4,
            self.cos_sg1,
            self.cos_sg2,
            self.cos_sg3,
            self.cos_sg4,
        )
        self._grid_data = GridData(
            horizontal_data=horizontal,
            vertical_data=vertical,
            contravariant_data=contravariant,
            angle_data=angle,
        )
        return self._grid_data

    def set_grid_data(self, grid_data: "GridData"):
        self._grid_data = grid_data

    def make_grid_data(self, npx, npy, npz, communicator, backend):
        metric_terms = MetricTerms.from_tile_sizing(
            npx=npx, npy=npy, npz=npz, communicator=communicator, backend=backend
        )
        self.set_grid_data(GridData.new_from_metric_terms(metric_terms))
        self.set_damping_coefficients(
            DampingCoefficients.new_from_metric_terms(metric_terms)
        )


@dataclasses.dataclass(frozen=True)
class HorizontalGridData:
    """
    Terms defining the horizontal grid.
    """

    lon: FloatFieldIJ
    lat: FloatFieldIJ
    lon_agrid: FloatFieldIJ
    lat_agrid: FloatFieldIJ
    area: FloatFieldIJ
    area_64: FloatFieldIJ
    rarea: FloatFieldIJ
    # TODO: refactor this to "area_c" and invert where used
    rarea_c: FloatFieldIJ
    dx: FloatFieldIJ
    dy: FloatFieldIJ
    dxc: FloatFieldIJ
    dyc: FloatFieldIJ
    dxa: FloatFieldIJ
    dya: FloatFieldIJ
    # TODO: refactor usages to invert "normal" versions instead
    rdx: FloatFieldIJ
    rdy: FloatFieldIJ
    rdxc: FloatFieldIJ
    rdyc: FloatFieldIJ
    rdxa: FloatFieldIJ
    rdya: FloatFieldIJ
    a11: FloatFieldIJ
    a12: FloatFieldIJ
    a21: FloatFieldIJ
    a22: FloatFieldIJ
    edge_w: FloatFieldIJ
    edge_e: FloatFieldIJ
    edge_s: FloatFieldI
    edge_n: FloatFieldI


@dataclasses.dataclass
class VerticalGridData:
    """
    Terms defining the vertical grid.

    Eulerian vertical grid is defined by p = ak + bk * p_ref
    """

    # TODO: make these non-optional, make FloatFieldK a true type and use it
    ptop: float
    ks: int
    ak: Optional[Any] = None
    bk: Optional[Any] = None
    p_ref: Optional[Any] = None
    """
    reference pressure (Pa) used to define pressure at vertical interfaces,
    where p = ak + bk * p_ref
    ptop is the top of the atmosphere and ks is the lowest index (highest layer) for
    which rayleigh friction

    """


@dataclasses.dataclass(frozen=True)
class ContravariantGridData:
    """
    Grid variables used for converting vectors from covariant to
    contravariant components.
    """

    cosa: FloatFieldIJ
    cosa_u: FloatFieldIJ
    cosa_v: FloatFieldIJ
    cosa_s: FloatFieldIJ
    sina_u: FloatFieldIJ
    sina_v: FloatFieldIJ
    rsina: FloatFieldIJ
    rsin_u: FloatFieldIJ
    rsin_v: FloatFieldIJ
    rsin2: FloatFieldIJ


@dataclasses.dataclass(frozen=True)
class AngleGridData:
    """
    sin and cos of certain angles used in metric calculations.

    Corresponds in the fortran code to sin_sg and cos_sg.
    """

    sin_sg1: FloatFieldIJ
    sin_sg2: FloatFieldIJ
    sin_sg3: FloatFieldIJ
    sin_sg4: FloatFieldIJ
    cos_sg1: FloatFieldIJ
    cos_sg2: FloatFieldIJ
    cos_sg3: FloatFieldIJ
    cos_sg4: FloatFieldIJ


@dataclasses.dataclass(frozen=True)
class DampingCoefficients:
    """
    Terms used to compute damping coefficients.
    """

    divg_u: FloatFieldIJ
    divg_v: FloatFieldIJ
    del6_u: FloatFieldIJ
    del6_v: FloatFieldIJ
    da_min: float
    da_min_c: float

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms):
        return cls(
            divg_u=metric_terms.divg_u.storage,
            divg_v=metric_terms.divg_v.storage,
            del6_u=metric_terms.del6_u.storage,
            del6_v=metric_terms.del6_v.storage,
            da_min=metric_terms.da_min,
            da_min_c=metric_terms.da_min_c,
        )


class GridData:
    # TODO: add docstrings to remaining properties

    def __init__(
        self,
        horizontal_data: HorizontalGridData,
        vertical_data: VerticalGridData,
        contravariant_data: ContravariantGridData,
        angle_data: AngleGridData,
    ):
        self._horizontal_data = horizontal_data
        self._vertical_data = vertical_data
        self._contravariant_data = contravariant_data
        self._angle_data = angle_data

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms):
        # TODO fix <Quantity>.storage mask for FieldI
        shape = metric_terms.lon.data.shape
        edge_n = utils.make_storage_data(
            metric_terms.edge_n.data,
            (shape[0],),
            axis=0,
            backend=metric_terms.edge_n.gt4py_backend,
        )
        edge_s = utils.make_storage_data(
            metric_terms.edge_s.data,
            (shape[0],),
            axis=0,
            backend=metric_terms.edge_s.gt4py_backend,
        )
        edge_e = utils.make_storage_data(
            metric_terms.edge_e.data,
            (1, shape[1]),
            axis=1,
            backend=metric_terms.edge_e.gt4py_backend,
        )
        edge_w = utils.make_storage_data(
            metric_terms.edge_w.data,
            (1, shape[1]),
            axis=1,
            backend=metric_terms.edge_w.gt4py_backend,
        )

        horizontal_data = HorizontalGridData(
            lon=metric_terms.lon.storage,
            lat=metric_terms.lat.storage,
            lon_agrid=metric_terms.lon_agrid.storage,
            lat_agrid=metric_terms.lat_agrid.storage,
            area=metric_terms.area.storage,
            area_64=metric_terms.area.storage,
            rarea=metric_terms.rarea.storage,
            rarea_c=metric_terms.rarea_c.storage,
            dx=metric_terms.dx.storage,
            dy=metric_terms.dy.storage,
            dxc=metric_terms.dxc.storage,
            dyc=metric_terms.dyc.storage,
            dxa=metric_terms.dxa.storage,
            dya=metric_terms.dya.storage,
            rdx=metric_terms.rdx.storage,
            rdy=metric_terms.rdy.storage,
            rdxc=metric_terms.rdxc.storage,
            rdyc=metric_terms.rdyc.storage,
            rdxa=metric_terms.rdxa.storage,
            rdya=metric_terms.rdya.storage,
            a11=metric_terms.a11.storage,
            a12=metric_terms.a12.storage,
            a21=metric_terms.a21.storage,
            a22=metric_terms.a22.storage,
            edge_w=edge_w,
            edge_e=edge_e,
            edge_s=edge_s,
            edge_n=edge_n,
        )
        ak = metric_terms.ak.data
        bk = metric_terms.bk.data
        # TODO fix <Quantity>.storage mask for FieldK
        ak = utils.make_storage_data(
            ak, ak.shape, len(ak.shape) * (0,), backend=metric_terms.ak.gt4py_backend
        )
        bk = utils.make_storage_data(
            bk, bk.shape, len(bk.shape) * (0,), backend=metric_terms.ak.gt4py_backend
        )
        vertical_data = VerticalGridData(
            ak=ak,
            bk=bk,
            ptop=metric_terms.ptop,
            ks=metric_terms.ks,
        )
        contravariant_data = ContravariantGridData(
            cosa=metric_terms.cosa.storage,
            cosa_u=metric_terms.cosa_u.storage,
            cosa_v=metric_terms.cosa_v.storage,
            cosa_s=metric_terms.cosa_s.storage,
            sina_u=metric_terms.sina_u.storage,
            sina_v=metric_terms.sina_v.storage,
            rsina=metric_terms.rsina.storage,
            rsin_u=metric_terms.rsin_u.storage,
            rsin_v=metric_terms.rsin_v.storage,
            rsin2=metric_terms.rsin2.storage,
        )
        angle_data = AngleGridData(
            sin_sg1=metric_terms.sin_sg1.storage,
            sin_sg2=metric_terms.sin_sg2.storage,
            sin_sg3=metric_terms.sin_sg3.storage,
            sin_sg4=metric_terms.sin_sg4.storage,
            cos_sg1=metric_terms.cos_sg1.storage,
            cos_sg2=metric_terms.cos_sg2.storage,
            cos_sg3=metric_terms.cos_sg3.storage,
            cos_sg4=metric_terms.cos_sg4.storage,
        )
        return cls(horizontal_data, vertical_data, contravariant_data, angle_data)

    @property
    def lon(self):
        """longitude of cell corners"""
        return self._horizontal_data.lon

    @property
    def lat(self):
        """latitude of cell corners"""
        return self._horizontal_data.lat

    @property
    def lon_agrid(self):
        """longitude on the A-grid (cell centers)"""
        return self._horizontal_data.lon_agrid

    @property
    def lat_agrid(self):
        """latitude on the A-grid (cell centers)"""
        return self._horizontal_data.lat_agrid

    @property
    def area(self):
        """Gridcell area"""
        return self._horizontal_data.area

    @property
    def area_64(self):
        """Gridcell area (64-bit)"""
        return self._horizontal_data.area_64

    @property
    def rarea(self):
        """1 / area"""
        return self._horizontal_data.rarea

    @property
    def rarea_c(self):
        return self._horizontal_data.rarea_c

    @property
    def dx(self):
        """distance between cell corners in x-direction"""
        return self._horizontal_data.dx

    @property
    def dy(self):
        """distance between cell corners in y-direction"""
        return self._horizontal_data.dy

    @property
    def dxc(self):
        """distance between gridcell centers in x-direction"""
        return self._horizontal_data.dxc

    @property
    def dyc(self):
        """distance between gridcell centers in y-direction"""
        return self._horizontal_data.dyc

    @property
    def dxa(self):
        """distance between centers of west and east edges of gridcell"""
        return self._horizontal_data.dxa

    @property
    def dya(self):
        """distance between centers of north and south edges of gridcell"""
        return self._horizontal_data.dya

    @property
    def rdx(self):
        """1 / dx"""
        return self._horizontal_data.rdx

    @property
    def rdy(self):
        """1 / dy"""
        return self._horizontal_data.rdy

    @property
    def rdxc(self):
        """1 / dxc"""
        return self._horizontal_data.rdxc

    @property
    def rdyc(self):
        """1 / dyc"""
        return self._horizontal_data.rdyc

    @property
    def rdxa(self):
        """1 / dxa"""
        return self._horizontal_data.rdxa

    @property
    def rdya(self):
        """1 / dya"""
        return self._horizontal_data.rdya

    @property
    def a11(self):
        return self._horizontal_data.a11

    @property
    def a12(self):
        return self._horizontal_data.a12

    @property
    def a21(self):
        return self._horizontal_data.a21

    @property
    def a22(self):
        return self._horizontal_data.a22

    @property
    def edge_w(self):
        return self._horizontal_data.edge_w

    @property
    def edge_e(self):
        return self._horizontal_data.edge_e

    @property
    def edge_s(self):
        return self._horizontal_data.edge_s

    @property
    def edge_n(self):
        return self._horizontal_data.edge_n

    @property
    def p_ref(self) -> float:
        """
        reference pressure (Pa) used to define pressure at vertical interfaces,
        where p = ak + bk * p_ref
        """
        return self._vertical_data.p_ref

    @p_ref.setter
    def p_ref(self, value):
        self._vertical_data.p_ref = value

    @property
    def ak(self):
        """
        constant used to define pressure at vertical interfaces,
        where p = ak + bk * p_ref
        """
        return self._vertical_data.ak

    @ak.setter
    def ak(self, value):
        self._vertical_data.ak = value

    @property
    def bk(self):
        """
        constant used to define pressure at vertical interfaces,
        where p = ak + bk * p_ref
        """
        return self._vertical_data.bk

    @bk.setter
    def bk(self, value):
        self._vertical_data.bk = value

    @property
    def ks(self):
        return self._vertical_data.ks

    @ks.setter
    def ks(self, value):
        self._vertical_data.ks = value

    @property
    def ptop(self):
        """pressure at top of atmosphere (Pa)"""
        return self._vertical_data.ptop

    @ptop.setter
    def ptop(self, value):
        self._vertical_data.ptop = value

    @property
    def cosa(self):
        return self._contravariant_data.cosa

    @property
    def cosa_u(self):
        return self._contravariant_data.cosa_u

    @property
    def cosa_v(self):
        return self._contravariant_data.cosa_v

    @property
    def cosa_s(self):
        return self._contravariant_data.cosa_s

    @property
    def sina_u(self):
        return self._contravariant_data.sina_u

    @property
    def sina_v(self):
        return self._contravariant_data.sina_v

    @property
    def rsina(self):
        return self._contravariant_data.rsina

    @property
    def rsin_u(self):
        return self._contravariant_data.rsin_u

    @property
    def rsin_v(self):
        return self._contravariant_data.rsin_v

    @property
    def rsin2(self):
        return self._contravariant_data.rsin2

    @property
    def sin_sg1(self):
        return self._angle_data.sin_sg1

    @property
    def sin_sg2(self):
        return self._angle_data.sin_sg2

    @property
    def sin_sg3(self):
        return self._angle_data.sin_sg3

    @property
    def sin_sg4(self):
        return self._angle_data.sin_sg4

    @property
    def cos_sg1(self):
        return self._angle_data.cos_sg1

    @property
    def cos_sg2(self):
        return self._angle_data.cos_sg2

    @property
    def cos_sg3(self):
        return self._angle_data.cos_sg3

    @property
    def cos_sg4(self):
        return self._angle_data.cos_sg4


def quantity_wrap(storage, dims: Sequence[str], grid_indexing: GridIndexing):
    origin, extent = grid_indexing.get_origin_domain(dims)
    return pace.util.Quantity(
        storage,
        dims=dims,
        units="unknown",
        origin=origin,
        extent=extent,
    )


# TODO: delete this routine in favor of grid_indexing.axis_offsets
def axis_offsets(
    grid: Union[Grid, GridIndexing],
    origin: Iterable[int],
    domain: Iterable[int],
) -> Mapping[str, gtscript.AxisIndex]:
    """Return the axis offsets relative to stencil compute domain.

    Args:
        grid: indexing data
        origin: origin of a stencil's computation
        domain: shape over which computation is being performed

    Returns:
        axis_offsets: Mapping from offset name to value. i_start, i_end, j_start, and
            j_end indicate the offset to the edges of the tile face in each direction.
            local_is, local_ie, local_js, and local_je indicate the offset to the
            edges of the cell-centered compute domain in each direction.
    """
    origin = tuple(origin)
    domain = tuple(domain)
    if isinstance(grid, Grid):
        return _old_grid_axis_offsets(grid, origin, domain)
    else:
        return _grid_indexing_axis_offsets(grid, origin, domain)


@functools.lru_cache(maxsize=None)
def _old_grid_axis_offsets(
    grid: Grid,
    origin: Tuple[int, ...],
    domain: Tuple[int, ...],
) -> Mapping[str, gtscript.AxisIndex]:
    if grid.west_edge:
        proc_offset = grid.is_ - grid.global_is
        origin_offset = grid.is_ - origin[0]
        i_start = gtscript.I[0] + proc_offset + origin_offset
    else:
        i_start = gtscript.I[0] - np.iinfo(np.int16).max

    if grid.east_edge:
        proc_offset = grid.npx + grid.halo - 2 - grid.global_is
        endpt_offset = (grid.is_ - origin[0]) - domain[0] + 1
        i_end = gtscript.I[-1] + proc_offset + endpt_offset
    else:
        i_end = gtscript.I[-1] + np.iinfo(np.int16).max

    if grid.south_edge:
        proc_offset = grid.js - grid.global_js
        origin_offset = grid.js - origin[1]
        j_start = gtscript.J[0] + proc_offset + origin_offset
    else:
        j_start = gtscript.J[0] - np.iinfo(np.int16).max

    if grid.north_edge:
        proc_offset = grid.npy + grid.halo - 2 - grid.global_js
        endpt_offset = (grid.js - origin[1]) - domain[1] + 1
        j_end = gtscript.J[-1] + proc_offset + endpt_offset
    else:
        j_end = gtscript.J[-1] + np.iinfo(np.int16).max

    return {
        "i_start": i_start,
        "local_is": gtscript.I[0] + grid.is_ - origin[0],
        "i_end": i_end,
        "local_ie": gtscript.I[-1] + grid.ie - origin[0] - domain[0] + 1,
        "j_start": j_start,
        "local_js": gtscript.J[0] + grid.js - origin[1],
        "j_end": j_end,
        "local_je": gtscript.J[-1] + grid.je - origin[1] - domain[1] + 1,
    }


@functools.lru_cache(maxsize=None)
def _grid_indexing_axis_offsets(
    grid: GridIndexing,
    origin: Tuple[int, ...],
    domain: Tuple[int, ...],
) -> Mapping[str, gtscript.AxisIndex]:
    return grid.axis_offsets(origin=origin, domain=domain)
