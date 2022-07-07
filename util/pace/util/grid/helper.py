import dataclasses
from typing import Any, Optional

from pace.dsl.gt4py_utils import split_cartesian_into_storages
from pace.dsl.typing import FloatField, FloatFieldI, FloatFieldIJ

from .generation import MetricTerms


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
        edge_n = metric_terms.edge_n.storage
        edge_s = metric_terms.edge_s.storage
        edge_e = metric_terms.edge_e.storage
        edge_w = metric_terms.edge_w.storage

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
        ak = metric_terms.ak.storage
        bk = metric_terms.bk.storage
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

    vlon1: FloatFieldIJ
    vlon2: FloatFieldIJ
    vlon3: FloatFieldIJ
    vlat1: FloatFieldIJ
    vlat2: FloatFieldIJ
    vlat3: FloatFieldIJ
    edge_vect_w: FloatFieldI
    edge_vect_e: FloatFieldI
    edge_vect_s: FloatFieldIJ
    edge_vect_n: FloatFieldIJ
    es1_1: FloatFieldIJ
    es1_2: FloatFieldIJ
    es1_3: FloatFieldIJ
    ew2_1: FloatFieldIJ
    ew2_2: FloatFieldIJ
    ew2_3: FloatFieldIJ

    @classmethod
    def new_from_metric_terms(cls, metric_terms: MetricTerms) -> "DriverGridData":
        return cls.new_from_grid_variables(
            vlon=metric_terms.vlon.storage,
            vlat=metric_terms.vlon.storage,
            edge_vect_n=metric_terms.edge_vect_n.storage,
            edge_vect_s=metric_terms.edge_vect_s.storage,
            edge_vect_e=metric_terms.edge_vect_e_2d.storage,
            edge_vect_w=metric_terms.edge_vect_w_2d.storage,
            es1=metric_terms.es1.storage,
            ew2=metric_terms.ew2.storage,
        )

    @classmethod
    def new_from_grid_variables(
        cls,
        vlon: FloatField,
        vlat: FloatField,
        edge_vect_n: FloatFieldI,
        edge_vect_s: FloatFieldI,
        edge_vect_e: FloatFieldIJ,
        edge_vect_w: FloatFieldIJ,
        es1: FloatField,
        ew2: FloatField,
    ) -> "DriverGridData":

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
