from abc import ABCMeta, abstractmethod
from cmath import inf
from enum import Enum
from typing import Callable, Tuple

import numpy as np
from numpy import pi
import cartopy.crs as ccrs
from matplotlib import pyplot as plt

from ..utils import find1dpeak, find2dpeak
from ..dsp.DSPLine import DSPLine
from ..dsp.DSPSpectrogram import DSPSpectrogram
from .B3DPlotter import B3DPlotter
from . import getUnitAbbrev


class AxeProjection(Enum):
    RECTILINEAR = 0
    POLAR = 1
    NORTH_POLAR = 2
    PLATECARREE = 3
    DIM3D = 4


class FigureProjection(Enum):
    MPL = 0
    EARTH3D = 1


class GridElementFactory(object):

    __slots__ = ["gs", "coord"]

    @classmethod
    def create(cls, gs: "GridSpecFactory", coord: slice) -> "GridElementFactory":
        ret = cls()
        ret.gs = gs
        ret.coord = coord
        return ret

    def get_gridspec(self) -> "GridSpecFactory":
        return self.gs


class GridSpecFactory(object):

    __slots__ = ["figure", "nrow", "ncol"]

    @classmethod
    def create(cls, figure: "FigureFactory", nrow: int, ncol: int) -> "GridSpecFactory":
        ret = cls()
        ret.figure = figure
        ret.nrow = nrow
        ret.ncol = ncol
        return ret

    def __getitem__(self, ind) -> GridElementFactory:
        ge = GridElementFactory.create(gs=self, coord=ind)
        return ge


class APlottable(metaclass=ABCMeta):

    __slots__ = ["plottable", "kwargs"]

    def __init__(self, plottable, kwargs: dict) -> None:
        self.plottable = plottable
        self.kwargs = kwargs

    @abstractmethod
    def _make_mline(self) -> Tuple["array", "array", str, str, str, str]:
        pass

    def render(self, axe: "AxeFactory") -> dict:
        maxe = axe.mpl_axe
        (
            xd,
            yd,
            name_of_x_var,
            unit_of_x_var,
            name_of_y_var,
            unit_of_y_var,
        ) = self._make_mline()

        fill = ""
        plot_mth = maxe.plot
        if axe.projection == AxeProjection.PLATECARREE:
            args = (xd * 180 / pi, yd * 180 / pi)
        else:
            args = (xd, yd)

        kwargs = self.kwargs.copy()
        nb_peaks = kwargs.pop("find_peaks", 0)
        _ = kwargs.pop("transform", None)
        lpeaks = find1dpeak(
            nb_peaks=nb_peaks,
            xd=xd,
            yd=yd,
            name_of_x_var=name_of_x_var,
            unit_of_x_var=unit_of_x_var,
        )

        info = {
            "plottable": self,
            "plot_method": plot_mth,
            "fill": fill,
            "args": args,
            "mpl_kwargs": kwargs,
            "peaks": lpeaks,
            "name_of_x_var": name_of_x_var,
            "unit_of_x_var": unit_of_x_var,
            "name_of_y_var": name_of_y_var,
            "unit_of_y_var": unit_of_y_var,
            "xmin": np.min(xd),
            "xmax": np.max(xd),
            "ymin": np.min(yd),
            "ymax": np.max(yd),
        }
        return info

    def scaleAndLabelData(self, info, amp_x, amp_y):
        xd, yd = info["args"]

        _, x_mult, x_lbl, x_unit = getUnitAbbrev(amp_x, unit=info["unit_of_x_var"])
        _, y_mult, y_lbl, y_unit = getUnitAbbrev(amp_y, unit=info["unit_of_y_var"])

        args = (xd / x_mult, yd / y_mult)
        x_label = "%s (%s%s)" % (info["name_of_x_var"], x_lbl, x_unit)
        y_label = "%s (%s%s)" % (info["name_of_y_var"], y_lbl, y_unit)

        info["scaled_args"] = args

        scaled_peaks = []
        for p in info["peaks"]:
            (xp,) = p.coord
            yp = p.value
            txt = "(%.1f %s%s,%.1f)" % (xp / x_mult, x_lbl, x_unit, p.value / y_mult)
            scaled_peaks.append((xp, yp, txt))
        info["scaled_peaks"] = scaled_peaks

        return info, x_label, y_label


class PlottableDSPLine(APlottable):

    __slots__ = []

    def _make_mline(self) -> Tuple["array", "array", str, str, str, str]:
        transform = self.kwargs.get("transform", self.plottable.default_transform)
        xd = self.plottable.generateXSerie()
        yd = transform(self.plottable.y_serie)
        name_of_x_var = self.plottable.name_of_x_var
        unit_of_x_var = self.plottable.unit_of_x_var
        name_of_y_var = ""
        unit_of_y_var = ""

        return xd, yd, name_of_x_var, unit_of_x_var, name_of_y_var, unit_of_y_var


class PlottableArray(APlottable):

    __slots__ = []

    def _make_mline(self) -> Tuple["array", "array", str, str, str, str]:
        transform = self.kwargs.get("transform", lambda x: x)
        yd = transform(self.plottable)
        ns = len(yd)
        xd = np.arange(ns)
        name_of_x_var = ""
        unit_of_x_var = ""
        name_of_y_var = ""
        unit_of_y_var = ""

        return xd, yd, name_of_x_var, unit_of_x_var, name_of_y_var, unit_of_y_var


class PlottableTuple(APlottable):

    __slots__ = []

    def _make_mline(self) -> Tuple["array", "array", str, str, str, str]:
        transform = self.kwargs.get("transform", lambda x: x)
        xd, yd = self.plottable
        yd = transform(yd)
        name_of_x_var = ""
        unit_of_x_var = ""
        name_of_y_var = ""
        unit_of_y_var = ""

        return xd, yd, name_of_x_var, unit_of_x_var, name_of_y_var, unit_of_y_var


class PlottableDSPSpectrogram(APlottable):

    __slots__ = []

    def _make_mline(self) -> Tuple["array", "array", str, str, str, str]:
        pass

    def render(self, axe: "AxeFactory") -> dict:
        mline = self.plottable
        maxe = axe.mpl_axe

        kwargs = self.kwargs.copy()
        fill = kwargs.pop("fill", "pcolormesh")
        transform = kwargs.pop("transform", mline.default_transform)
        find_peaks = kwargs.pop("find_peaks", 0)

        x_samp = mline.generateXSerie()
        name_of_x_var = mline.name_of_x_var
        x_unit = mline.unit_of_x_var

        y_samp = mline.generateYSerie()
        name_of_y_var = mline.name_of_y_var
        y_unit = mline.unit_of_y_var

        plot_mth = getattr(maxe, fill)
        if fill == "plot_surface":
            kwargs.pop("levels", None)
        elif fill == "pcolormesh":
            kwargs.pop("levels", None)
        elif fill == "contourf":
            pass
        elif fill == "contour":
            pass
        Z = transform(mline.img)
        if fill == "plot_surface" and mline.projection == "polar":
            P, R = np.meshgrid(x_samp, y_samp)
            X, Y = R * np.cos(P), R * np.sin(P)
        else:
            X, Y = np.meshgrid(x_samp, y_samp)

        args = (X, Y, Z)

        lpeaks = mline.findPeaksWithTransform(transform=transform, nb_peaks=find_peaks)

        info = {
            "plottable": self,
            "plot_method": plot_mth,
            "fill": fill,
            "args": args,
            "mpl_kwargs": kwargs,
            "peaks": lpeaks,
            "name_of_x_var": name_of_x_var,
            "unit_of_x_var": x_unit,
            "name_of_y_var": name_of_y_var,
            "unit_of_y_var": y_unit,
            "xmin": np.min(x_samp),
            "xmax": np.max(x_samp),
            "ymin": np.min(y_samp),
            "ymax": np.max(y_samp),
        }
        return info

    def scaleAndLabelData(self, info, amp_x, amp_y):
        xd, yd, zd = info["args"]

        _, x_mult, x_lbl, x_unit = getUnitAbbrev(amp_x, unit=info["unit_of_x_var"])
        _, y_mult, y_lbl, y_unit = getUnitAbbrev(amp_y, unit=info["unit_of_y_var"])

        args = (xd / x_mult, yd / y_mult, zd)
        x_label = "%s (%s%s)" % (info["name_of_x_var"], x_lbl, x_unit)
        y_label = "%s (%s%s)" % (info["name_of_y_var"], y_lbl, y_unit)

        info["scaled_args"] = args

        scaled_peaks = []
        for p in info["peaks"]:
            xp, yp = p.coord
            txt = "(%.1f,%.1f,%.1f)" % (xp / x_mult, yp / y_mult, p.value)
            scaled_peaks.append((xp, yp, txt))
        info["scaled_peaks"] = scaled_peaks

        return info, x_label, y_label


class PlottableFactory(object):
    @staticmethod
    def create(mline, kwargs: dict) -> APlottable:
        if isinstance(mline, DSPLine):
            ret = PlottableDSPLine(mline, kwargs)

        elif isinstance(mline, DSPSpectrogram):
            ret = PlottableDSPSpectrogram(mline, kwargs)

        elif isinstance(mline, tuple):
            ret = PlottableTuple(mline, kwargs)

        else:
            ret = PlottableArray(mline, kwargs)

        return ret


class AxeFactory(object):

    __slots__ = [
        "figure",
        "title",
        "spec",
        "projection",
        "kwargs",
        "plottable_factories",
        "mpl_axe",
    ]

    @classmethod
    def create(
        cls,
        title: str,
        spec: GridElementFactory,
        projection: AxeProjection = AxeProjection.RECTILINEAR,
        **kwargs,
    ) -> "AxeFactory":
        """
        Args:
            title: Title of the axe
            spec: Position in the GridSpec
            projection: Projection to use

        """
        gs = spec.get_gridspec()
        if gs.figure.projection == FigureProjection.EARTH3D and (
            gs.ncol != 1 or gs.nrow != 1
        ):
            raise AssertionError(
                f"Cannot use GridSpec different from (1,1) with figure projection 'earth3d'. Got ({gs.nrow},{gs.ncol})"
            )

        res = cls()
        res.figure = gs.figure
        res.title = title
        res.spec = spec
        res.projection = projection
        res.kwargs = kwargs
        res.mpl_axe = None

        gs.figure.registerAxeFactory(res)

        return res

    def __init__(self) -> None:
        self.plottable_factories = []

    def registerPlottableFactory(self, line_factory: APlottable):
        self.plottable_factories.append(line_factory)

    def plot(self, plottable, **kwargs) -> APlottable:
        fill = kwargs.get("fill", "")
        if self.projection == AxeProjection.PLATECARREE and not isinstance(
            plottable, tuple
        ):
            raise AssertionError(
                f"With '{self.projection}' axe projection, only (lon,lat) data is accepted. Got {plottable}"
            )
        elif self.projection == AxeProjection.DIM3D and not (
            isinstance(plottable, DSPSpectrogram) and fill == "plot_surface"
        ):
            raise AssertionError(
                f"With '{self.projection}' axe projection, only DSPSpectrogram is accepted. Got {plottable}"
            )

        res = PlottableFactory.create(plottable, kwargs)
        self.registerPlottableFactory(res)
        return res

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "AxesSubplot":
        if not self.mpl_axe is None:
            return self.mpl_axe

        ge = self.spec
        if self.projection == AxeProjection.PLATECARREE:
            proj = ccrs.PlateCarree()
        elif self.projection == AxeProjection.POLAR:
            proj = "polar"
        elif self.projection == AxeProjection.NORTH_POLAR:
            proj = "polar"
        elif self.projection == AxeProjection.DIM3D:
            proj = "3d"
        else:
            proj = "rectilinear"

        sharex = self.kwargs.pop("sharex", None)
        sharey = self.kwargs.pop("sharey", None)

        if not sharex is None:
            sharex = sharex.createMplAxe(mfig, mgs)
        if not sharey is None:
            sharey = sharey.createMplAxe(mfig, mgs)

        maxe = mfig.add_subplot(
            mgs[ge.coord],
            projection=proj,
            sharex=sharex,
            sharey=sharey,
            **self.kwargs,
        )
        self.mpl_axe = maxe

        maxe.set_title(self.title)

        if self.projection == AxeProjection.PLATECARREE:
            maxe.stock_img()
            maxe.gridlines(crs=proj, draw_labels=True)
        elif self.projection == AxeProjection.NORTH_POLAR:
            maxe.set_theta_zero_location("N")
            maxe.set_theta_direction(-1)
            maxe.grid(True)
        else:
            maxe.grid(True)

        self.mpl_axe = maxe

        return self.mpl_axe


class FigureFactory(object):

    __slots__ = ["title", "grid_spec", "projection", "axe_factories"]

    @classmethod
    def create(
        cls, title: str, projection: FigureProjection = FigureProjection.MPL
    ) -> "FigureFactory":
        """
        Args:
            title: Title of the figure
            projection: Projection to use

        """
        res = cls()
        res.title = title
        res.grid_spec = None
        res.projection = projection

        return res

    def __init__(self) -> None:
        self.axe_factories = []

    def add_gridspec(self, nrow: int, ncol: int) -> GridSpecFactory:
        res = GridSpecFactory.create(self, nrow, ncol)
        self.grid_spec = res
        return res

    def registerAxeFactory(self, axe_factory: AxeFactory):
        self.axe_factories.append(axe_factory)


def _render_earth3d(fig: FigureFactory) -> "B3DPlotter":
    app = B3DPlotter()
    app.plotEarth()

    axe = fig.axe_factories[0]

    for plottable in axe.plottable_factories:
        traj = plottable.plottable
        app.plotTrajectory(traj)

    return app


def _render_mpl(fig: FigureFactory) -> "Figure":
    mfig = plt.figure()
    mfig.suptitle(fig.title)

    gs = fig.grid_spec
    mgs = mfig.add_gridspec(gs.nrow, gs.ncol)

    for axe in fig.axe_factories:
        maxe = axe.createMplAxe(mfig, mgs)

        if len(axe.plottable_factories) == 0:
            continue

        rendered_plottables = []

        global_xmin = np.nan
        global_xmax = np.nan
        global_ymin = np.nan
        global_ymax = np.nan

        unit_of_x_var = None
        unit_of_y_var = None

        for plottable in axe.plottable_factories:
            info = plottable.render(axe)

            if info["unit_of_x_var"] != "":
                if unit_of_x_var is None:
                    unit_of_x_var = info["unit_of_x_var"]
                else:
                    if unit_of_x_var != info["unit_of_x_var"]:
                        raise AssertionError(
                            "Inconsistent X unit between lines in a same axe"
                        )

            if info["unit_of_y_var"] != "":
                if unit_of_y_var is None:
                    unit_of_y_var = info["unit_of_y_var"]
                else:
                    if unit_of_y_var != info["unit_of_y_var"]:
                        raise AssertionError(
                            "Inconsistent Y unit between lines in a same axe"
                        )

            rendered_plottables.append(info)

            if info["xmin"] < global_xmin or np.isnan(global_xmin):
                global_xmin = info["xmin"]
            if info["xmax"] > global_xmax or np.isnan(global_xmax):
                global_xmax = info["xmax"]
            if info["ymin"] < global_ymin or np.isnan(global_ymin):
                global_ymin = info["ymin"]
            if info["ymax"] > global_ymax or np.isnan(global_ymax):
                global_ymax = info["ymax"]

        amp_x = global_xmax - global_xmin
        amp_y = global_ymax - global_ymin

        for info in rendered_plottables:
            info, x_label, y_label = info["plottable"].scaleAndLabelData(
                info, amp_x, amp_y
            )
            args = info["scaled_args"]
            kwargs = info["mpl_kwargs"]
            ret = info["plot_method"](*args, **kwargs)
            for xp, yp, txt in info["scaled_peaks"]:
                maxe.plot([xp], [yp], marker="o", color="red", linestyle="")
                maxe.annotate(txt, xy=(xp, yp), fontsize="x-small")

            if info["fill"] == "plot_surface":
                pass
            elif info["fill"] == "pcolormesh":
                maxe.figure.colorbar(ret, ax=maxe)
            elif info["fill"] == "contourf":
                maxe.figure.colorbar(ret, ax=maxe)
            elif info["fill"] == "contour":
                maxe.clabel(ret, inline=True, fontsize=10)

        if x_label != "":
            maxe.set_xlabel(x_label)
        if y_label != "":
            maxe.set_ylabel(y_label)

    mfig.tight_layout()

    return mfig


def render(fig: FigureFactory):
    if fig.projection == FigureProjection.MPL:
        fig = _render_mpl(fig)
    else:
        fig = _render_earth3d(fig)

    return fig


def showFigures():
    plt.show()
