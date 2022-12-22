from gt4py.cartesian.gtscript import PARALLEL, computation, interval

import pace.dsl
import pace.fv3core.stencils.d_sw as d_sw
import pace.util
from pace import fv3core
from pace.dsl.typing import FloatField, FloatFieldIJ
from pace.fv3core.testing import TranslateDycoreFortranData2Py


class TranslateD_SW(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.max_error = 3.2e-10
        self.stencil_factory = stencil_factory
        dycore_config = fv3core.DynamicalCoreConfig.from_namelist(namelist)
        column_namelist = d_sw.get_column_namelist(
            config=dycore_config.acoustic_dynamics.d_grid_shallow_water,
            quantity_factory=self.grid.quantity_factory,
        )
        self.compute_func = d_sw.DGridShallowWaterLagrangianDynamics(  # type: ignore
            stencil_factory=self.stencil_factory,
            quantity_factory=self.grid.quantity_factory,
            grid_data=self.grid.grid_data,
            damping_coefficients=self.grid.damping_coefficients,
            column_namelist=column_namelist,
            nested=self.grid.nested,
            stretched_grid=self.grid.stretched_grid,
            config=dycore_config.d_grid_shallow_water,
        )
        self.in_vars["data_vars"] = {
            "uc": grid.x3d_domain_dict(),
            "vc": grid.y3d_domain_dict(),
            "w": {},
            "delpc": {},
            "delp": {},
            "u": grid.y3d_domain_dict(),
            "v": grid.x3d_domain_dict(),
            "xfx": grid.x3d_compute_domain_y_dict(),
            "crx": grid.x3d_compute_domain_y_dict(),
            "yfx": grid.y3d_compute_domain_x_dict(),
            "cry": grid.y3d_compute_domain_x_dict(),
            "mfx": grid.x3d_compute_dict(),
            "mfy": grid.y3d_compute_dict(),
            "cx": grid.x3d_compute_domain_y_dict(),
            "cy": grid.y3d_compute_domain_x_dict(),
            "heat_source": {},
            "diss_est": {},
            "q_con": {},
            "pt": {},
            "ua": {},
            "va": {},
            "zh": {},
            "divgd": grid.default_dict_buffer_2d(),
        }
        for name, info in self.in_vars["data_vars"].items():
            info["serialname"] = name + "d"
        self.in_vars["parameters"] = ["dt"]
        self.out_vars = self.in_vars["data_vars"].copy()
        del self.out_vars["zh"]


def ubke(
    uc: FloatField,
    vc: FloatField,
    cosa: FloatFieldIJ,
    rsina: FloatFieldIJ,
    ut: FloatField,
    ub: FloatField,
    dt4: float,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        dt = 2.0 * dt5
        ub, _ = d_sw.interpolate_uc_vc_to_cell_corners(uc, vc, cosa, rsina, ut, ut)
        ub = ub * dt


class TranslateUbKE(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "uc": {},
            "vc": {},
            "ut": {},
            "ub": grid.compute_dict_buffer_2d(),
        }
        self.in_vars["parameters"] = ["dt5", "dt4"]
        self.out_vars = {"ub": grid.compute_dict_buffer_2d()}
        origin = self.grid.compute_origin()
        domain = self.grid.domain_shape_compute(add=(1, 1, 0))
        self.stencil_factory = stencil_factory
        ax_offsets = self.stencil_factory.grid_indexing.axis_offsets(origin, domain)
        self.compute_func = self.stencil_factory.from_origin_domain(  # type: ignore
            ubke, externals=ax_offsets, origin=origin, domain=domain
        )

    def compute_from_storage(self, inputs):
        inputs["cosa"] = self.grid.cosa
        inputs["rsina"] = self.grid.rsina
        self.compute_func(**inputs)
        return inputs


def vbke(
    vc: FloatField,
    uc: FloatField,
    cosa: FloatFieldIJ,
    rsina: FloatFieldIJ,
    vt: FloatField,
    vb: FloatField,
    dt4: float,
    dt5: float,
):
    with computation(PARALLEL), interval(...):
        dt = 2.0 * dt5
        _, vb = d_sw.interpolate_uc_vc_to_cell_corners(uc, vc, cosa, rsina, vt, vt)
        vb = vb * dt


class TranslateVbKE(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "vc": {},
            "uc": {},
            "vt": {},
            "vb": grid.compute_dict_buffer_2d(),
        }
        self.in_vars["parameters"] = ["dt5", "dt4"]
        self.out_vars = {"vb": grid.compute_dict_buffer_2d()}
        origin = self.grid.compute_origin()
        domain = self.grid.domain_shape_compute(add=(1, 1, 0))
        self.stencil_factory = stencil_factory
        ax_offsets = self.stencil_factory.grid_indexing.axis_offsets(origin, domain)
        self.compute_func = self.stencil_factory.from_origin_domain(  # type: ignore
            vbke, externals=ax_offsets, origin=origin, domain=domain
        )

    def compute_from_storage(self, inputs):
        inputs["cosa"] = self.grid.cosa
        inputs["rsina"] = self.grid.rsina
        self.compute_func(**inputs)
        return inputs


class TranslateFluxCapacitor(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "cx": grid.x3d_compute_domain_y_dict(),
            "cy": grid.y3d_compute_domain_x_dict(),
            "xflux": grid.x3d_compute_dict(),
            "yflux": grid.y3d_compute_dict(),
            "crx_adv": grid.x3d_compute_domain_y_dict(),
            "cry_adv": grid.y3d_compute_domain_x_dict(),
            "fx": grid.x3d_compute_dict(),
            "fy": grid.y3d_compute_dict(),
        }
        self.out_vars = {}
        for outvar in ["cx", "cy", "xflux", "yflux"]:
            self.out_vars[outvar] = self.in_vars["data_vars"][outvar]
        self.stencil_factory = stencil_factory
        self.compute_func = self.stencil_factory.from_origin_domain(  # type: ignore
            d_sw.flux_capacitor,
            origin=grid.full_origin(),
            domain=grid.domain_shape_full(),
        )


class TranslateHeatDiss(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "fx2": {},
            "fy2": {},
            "w": {},
            "dw": {},
            "heat_source": {},
            "diss_est": {},
        }
        self.out_vars = {
            "heat_source": grid.compute_dict(),
            "diss_est": grid.compute_dict(),
            "dw": grid.compute_dict(),
        }
        self.namelist = namelist  # type: ignore
        self.stencil_factory = stencil_factory

    def compute_from_storage(self, inputs):
        column_namelist = d_sw.get_column_namelist(
            config=self.namelist, quantity_factory=self.grid.quantity_factory
        )
        # TODO add these to the serialized data or remove the test
        inputs["damp_w"] = column_namelist["damp_w"]
        inputs["ke_bg"] = column_namelist["ke_bg"]
        inputs["dt"] = (
            self.namelist.dt_atmos / self.namelist.k_split / self.namelist.n_split
        )
        inputs["rarea"] = self.grid.rarea
        heat_diss_stencil = self.stencil_factory.from_origin_domain(
            d_sw.heat_diss,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )
        heat_diss_stencil(**inputs)
        return inputs


class TranslateWdivergence(TranslateDycoreFortranData2Py):
    def __init__(
        self,
        grid,
        namelist: pace.util.Namelist,
        stencil_factory: pace.dsl.StencilFactory,
    ):
        super().__init__(grid, namelist, stencil_factory)
        self.in_vars["data_vars"] = {
            "q": {"serialname": "w"},
            "delp": {},
            "gx": {},
            "gy": {},
        }
        self.out_vars = {"q": {"serialname": "w"}}
        self.stencil_factory = stencil_factory
        self.compute_func = self.stencil_factory.from_origin_domain(  # type: ignore
            d_sw.apply_fluxes,
            origin=self.grid.compute_origin(),
            domain=self.grid.domain_shape_compute(),
        )

    def compute_from_storage(self, inputs):
        inputs["rarea"] = self.grid.rarea
        self.compute_func(**inputs)
        return inputs
