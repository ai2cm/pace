import dataclasses
import pathlib

import xarray as xr

import pace.util


# TODO: if we can remove translate tests in favor of checkpointer tests,
# we can remove this "disallowed" import (pace.util does not depend on pace.dsl)
try:
    from pace.dsl.gt4py_utils import split_cartesian_into_storages
except ImportError:
    split_cartesian_into_storages = None
from pace.util import Z_INTERFACE_DIM, get_fs

from .generation import MetricTerms


@dataclasses.dataclass(frozen=True)
class DampingCoefficients:
    """
    Terms used to compute damping coefficients.
    """

    divg_u: pace.util.Quantity
    divg_v: pace.util.Quantity
    del6_u: pace.util.Quantity
    del6_v: pace.util.Quantity
    da_min: float
    da_min_c: float

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms):
        return cls(
            divg_u=metric_terms.divg_u,
            divg_v=metric_terms.divg_v,
            del6_u=metric_terms.del6_u,
            del6_v=metric_terms.del6_v,
            da_min=metric_terms.da_min,
            da_min_c=metric_terms.da_min_c,
        )


@dataclasses.dataclass(frozen=True)
class HorizontalGridData:
    """
    Terms defining the horizontal grid.
    """

    lon: pace.util.Quantity
    lat: pace.util.Quantity
    lon_agrid: pace.util.Quantity
    lat_agrid: pace.util.Quantity
    area: pace.util.Quantity
    area_64: pace.util.Quantity
    rarea: pace.util.Quantity
    # TODO: refactor this to "area_c" and invert where used
    rarea_c: pace.util.Quantity
    dx: pace.util.Quantity
    dy: pace.util.Quantity
    dxc: pace.util.Quantity
    dyc: pace.util.Quantity
    dxa: pace.util.Quantity
    dya: pace.util.Quantity
    # TODO: refactor usages to invert "normal" versions instead
    rdx: pace.util.Quantity
    rdy: pace.util.Quantity
    rdxc: pace.util.Quantity
    rdyc: pace.util.Quantity
    rdxa: pace.util.Quantity
    rdya: pace.util.Quantity
    ee1: pace.util.Quantity
    ee2: pace.util.Quantity
    es1: pace.util.Quantity
    ew2: pace.util.Quantity
    a11: pace.util.Quantity
    a12: pace.util.Quantity
    a21: pace.util.Quantity
    a22: pace.util.Quantity
    edge_w: pace.util.Quantity
    edge_e: pace.util.Quantity
    edge_s: pace.util.Quantity
    edge_n: pace.util.Quantity

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms) -> "HorizontalGridData":
        return cls(
            lon=metric_terms.lon,
            lat=metric_terms.lat,
            lon_agrid=metric_terms.lon_agrid,
            lat_agrid=metric_terms.lat_agrid,
            area=metric_terms.area,
            area_64=metric_terms.area,
            rarea=metric_terms.rarea,
            rarea_c=metric_terms.rarea_c,
            dx=metric_terms.dx,
            dy=metric_terms.dy,
            dxc=metric_terms.dxc,
            dyc=metric_terms.dyc,
            dxa=metric_terms.dxa,
            dya=metric_terms.dya,
            rdx=metric_terms.rdx,
            rdy=metric_terms.rdy,
            rdxc=metric_terms.rdxc,
            rdyc=metric_terms.rdyc,
            rdxa=metric_terms.rdxa,
            rdya=metric_terms.rdya,
            ee1=metric_terms.ee1,
            ee2=metric_terms.ee2,
            es1=metric_terms.es1,
            ew2=metric_terms.ew2,
            a11=metric_terms.a11,
            a12=metric_terms.a12,
            a21=metric_terms.a21,
            a22=metric_terms.a22,
            edge_w=metric_terms.edge_w,
            edge_e=metric_terms.edge_e,
            edge_s=metric_terms.edge_s,
            edge_n=metric_terms.edge_n,
        )


@dataclasses.dataclass
class VerticalGridData:
    """
    Terms defining the vertical grid.

    Eulerian vertical grid is defined by p = ak + bk * p_ref
    """

    # TODO: make these non-optional, make FloatFieldK a true type and use it
    ak: pace.util.Quantity
    bk: pace.util.Quantity
    """
    reference pressure (Pa) used to define pressure at vertical interfaces,
    where p = ak + bk * p_ref
    """

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms) -> "VerticalGridData":
        return cls(
            ak=metric_terms.ak,
            bk=metric_terms.bk,
        )

    @classmethod
    def from_restart(
        cls, restart_path: str, quantity_factory: pace.util.QuantityFactory
    ):
        fs = get_fs(restart_path)
        restart_files = fs.ls(restart_path)
        data_file = restart_files[
            [fname.endswith("fv_core.res.nc") for fname in restart_files].index(True)
        ]

        ak_bk_data_file = pathlib.Path(restart_path) / data_file
        if not fs.isfile(ak_bk_data_file):
            raise ValueError(
                """vertical_grid_from_restart is true,
                but no fv_core.res.nc in restart data file."""
            )

        ak = quantity_factory.empty([Z_INTERFACE_DIM], units="Pa")
        bk = quantity_factory.empty([Z_INTERFACE_DIM], units="")
        with fs.open(ak_bk_data_file, "rb") as f:
            ds = xr.open_dataset(f).isel(Time=0).drop_vars("Time")
            ak.view[:] = ds["ak"].values
            bk.view[:] = ds["bk"].values

        return cls(ak=ak, bk=bk)

    @property
    def ptop(self) -> float:
        """
        top of atmosphere pressure (Pa)
        """
        if self.bk.view[0] != 0:
            raise ValueError("ptop is not well-defined when top-of-atmosphere bk != 0")
        return float(self.ak.view[0])


@dataclasses.dataclass(frozen=True)
class ContravariantGridData:
    """
    Grid variables used for converting vectors from covariant to
    contravariant components.
    """

    cosa: pace.util.Quantity
    cosa_u: pace.util.Quantity
    cosa_v: pace.util.Quantity
    cosa_s: pace.util.Quantity
    sina_u: pace.util.Quantity
    sina_v: pace.util.Quantity
    rsina: pace.util.Quantity
    rsin_u: pace.util.Quantity
    rsin_v: pace.util.Quantity
    rsin2: pace.util.Quantity

    @classmethod
    def new_from_metric_terms(
        cls, metric_terms: MetricTerms
    ) -> "ContravariantGridData":
        return cls(
            cosa=metric_terms.cosa,
            cosa_u=metric_terms.cosa_u,
            cosa_v=metric_terms.cosa_v,
            cosa_s=metric_terms.cosa_s,
            sina_u=metric_terms.sina_u,
            sina_v=metric_terms.sina_v,
            rsina=metric_terms.rsina,
            rsin_u=metric_terms.rsin_u,
            rsin_v=metric_terms.rsin_v,
            rsin2=metric_terms.rsin2,
        )


@dataclasses.dataclass(frozen=True)
class AngleGridData:
    """
    sin and cos of certain angles used in metric calculations.

    Corresponds in the fortran code to sin_sg and cos_sg.
    """

    sin_sg1: pace.util.Quantity
    sin_sg2: pace.util.Quantity
    sin_sg3: pace.util.Quantity
    sin_sg4: pace.util.Quantity
    cos_sg1: pace.util.Quantity
    cos_sg2: pace.util.Quantity
    cos_sg3: pace.util.Quantity
    cos_sg4: pace.util.Quantity

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms) -> "AngleGridData":
        return cls(
            sin_sg1=metric_terms.sin_sg1,
            sin_sg2=metric_terms.sin_sg2,
            sin_sg3=metric_terms.sin_sg3,
            sin_sg4=metric_terms.sin_sg4,
            cos_sg1=metric_terms.cos_sg1,
            cos_sg2=metric_terms.cos_sg2,
            cos_sg3=metric_terms.cos_sg3,
            cos_sg4=metric_terms.cos_sg4,
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

        horizontal_data = HorizontalGridData.new_from_metric_terms(metric_terms)
        vertical_data = VerticalGridData.new_from_metric_terms(metric_terms)
        contravariant_data = ContravariantGridData.new_from_metric_terms(metric_terms)
        angle_data = AngleGridData.new_from_metric_terms(metric_terms)
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
    def ee1(self) -> pace.util.Quantity:
        return self._horizontal_data.ee1

    @property
    def ee2(self) -> pace.util.Quantity:
        return self._horizontal_data.ee2

    @property
    def es1(self) -> pace.util.Quantity:
        return self._horizontal_data.es1

    @property
    def ew2(self) -> pace.util.Quantity:
        return self._horizontal_data.ew2

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

    # Ajda
    # @property
    # def p_ref(self) -> float:
    #     """
    #     reference pressure (Pa) used to define pressure at vertical interfaces,
    #     where p = ak + bk * p_ref
    #     """
    #     return self._vertical_data.p_ref

    # @p_ref.setter
    # def p_ref(self, value):
    #     self._vertical_data.p_ref = value

    @property
    def ak(self) -> pace.util.Quantity:
        """
        constant used to define pressure at vertical interfaces,
        where p = ak + bk * p_ref
        """
        return self._vertical_data.ak

    @ak.setter
    def ak(self, value: pace.util.Quantity):
        self._vertical_data.ak = value

    @property
    def bk(self) -> pace.util.Quantity:
        """
        constant used to define pressure at vertical interfaces,
        where p = ak + bk * p_ref
        """
        return self._vertical_data.bk

    @bk.setter
    def bk(self, value: pace.util.Quantity):
        self._vertical_data.bk = value

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


@dataclasses.dataclass(frozen=True)
class DriverGridData:
    """
    Terms used to Apply Physics changes to the Dycore.
    Attributes:
      vlon1: x-component of unit lon vector in eastward longitude direction
      vlon2: y-component of unit lon vector in eastward longitude direction
      vlon3: z-component of unit lon vector in eastward longitude direction
      vlat1: x-component of unit lat vector in northward latitude direction
      vlat2: y-component of unit lat vector in northward latitude direction
      vlat3: z-component of unit lat vector in northward latitude direction
      edge_vect_w: factor to interpolate A to C grids at the western grid edge
      edge_vect_e: factor to interpolate A to C grids at the easter grid edge
      edge_vect_s: factor to interpolate A to C grids at the southern grid edge
      edge_vect_n: factor to interpolate A to C grids at the northern grid edge
      es1_1: x-component of grid local unit vector in x-direction at cell edge
      es1_2: y-component of grid local unit vector in x-direction at cell edge
      es1_3: z-component of grid local unit vector in x-direction at cell edge
      ew2_1: x-component of grid local unit vector in y-direction at cell edge
      ew2_2: y-component of grid local unit vector in y-direction at cell edge
      ew2_3: z-component of grid local unit vector in y-direction at cell edge
    """

    vlon1: pace.util.Quantity
    vlon2: pace.util.Quantity
    vlon3: pace.util.Quantity
    vlat1: pace.util.Quantity
    vlat2: pace.util.Quantity
    vlat3: pace.util.Quantity
    edge_vect_w: pace.util.Quantity
    edge_vect_e: pace.util.Quantity
    edge_vect_s: pace.util.Quantity
    edge_vect_n: pace.util.Quantity
    es1_1: pace.util.Quantity
    es1_2: pace.util.Quantity
    es1_3: pace.util.Quantity
    ew2_1: pace.util.Quantity
    ew2_2: pace.util.Quantity
    ew2_3: pace.util.Quantity

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms) -> "DriverGridData":
        return cls.new_from_grid_variables(
            vlon=metric_terms.vlon,
            vlat=metric_terms.vlon,
            edge_vect_n=metric_terms.edge_vect_n,
            edge_vect_s=metric_terms.edge_vect_s,
            edge_vect_e=metric_terms.edge_vect_e,
            edge_vect_w=metric_terms.edge_vect_w,
            es1=metric_terms.es1,
            ew2=metric_terms.ew2,
        )

    @classmethod
    def new_from_grid_variables(
        cls,
        vlon: pace.util.Quantity,
        vlat: pace.util.Quantity,
        edge_vect_n: pace.util.Quantity,
        edge_vect_s: pace.util.Quantity,
        edge_vect_e: pace.util.Quantity,
        edge_vect_w: pace.util.Quantity,
        es1: pace.util.Quantity,
        ew2: pace.util.Quantity,
    ) -> "DriverGridData":

        try:
            vlon1, vlon2, vlon3 = split_quantity_along_last_dim(vlon)
            vlat1, vlat2, vlat3 = split_quantity_along_last_dim(vlat)
            es1_1, es1_2, es1_3 = split_quantity_along_last_dim(es1)
            ew2_1, ew2_2, ew2_3 = split_quantity_along_last_dim(ew2)
        except (AttributeError, TypeError):
            vlon1, vlon2, vlon3 = split_cartesian_into_storages(vlon)
            vlat1, vlat2, vlat3 = split_cartesian_into_storages(vlat)
            es1_1, es1_2, es1_3 = split_cartesian_into_storages(es1)
            ew2_1, ew2_2, ew2_3 = split_cartesian_into_storages(ew2)

        return cls(
            vlon1=vlon1,
            vlon2=vlon2,
            vlon3=vlon3,
            vlat1=vlat1,
            vlat2=vlat2,
            vlat3=vlat3,
            es1_1=es1_1,
            es1_2=es1_2,
            es1_3=es1_3,
            ew2_1=ew2_1,
            ew2_2=ew2_2,
            ew2_3=ew2_3,
            edge_vect_w=edge_vect_w,
            edge_vect_e=edge_vect_e,
            edge_vect_s=edge_vect_s,
            edge_vect_n=edge_vect_n,
        )


def split_quantity_along_last_dim(quantity):
    """Split a quantity along the last dimension into a list of quantities.

    Args:
        quantity: Quantity to split.

    Returns:
        List of quantities.
    """
    return_list = []
    for i in range(quantity.data.shape[-1]):
        return_list.append(
            pace.util.Quantity(
                data=quantity.data[..., i],
                dims=quantity.dims[:-1],
                units=quantity.units,
                origin=quantity.origin[:-1],
                extent=quantity.extent[:-1],
                gt4py_backend=quantity.gt4py_backend,
            )
        )
    return return_list
