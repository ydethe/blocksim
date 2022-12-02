from abc import ABCMeta, abstractmethod
from ast import Assert
from asyncio.log import logger
from typing import Tuple
from pathlib import Path
from datetime import datetime, timedelta

from pandas import Timedelta, Timestamp
from parse import parse
import networkx as nx
from networkx.classes.graph import Graph
import numpy as np
from numpy import pi
import matplotlib.image as mpimg
import pandas as pd

from .GPlottable import GPlottable, GVariable
from ..utils import find1dpeak
from ..dsp.DSPLine import (
    DSPRectilinearLine,
    DSPPolarLine,
    DSPNorthPolarLine,
    DSPHistogram,
)
from .. import logger
from ..dsp.DSPFilter import DSPBodeDiagram
from ..loggers.Logger import Logger
from ..dsp.DSPMap import DSPRectilinearMap, DSPPolarMap, DSPNorthPolarMap
from .GraphicSpec import AxeProjection, FigureProjection
from ..satellite.Trajectory import Trajectory
from ..control.Earth6DDLPosition import Earth6DDLPosition
from . import getUnitAbbrev


class APlottable(metaclass=ABCMeta):
    """This base abstract class describes all the entities able to be plotted:

    Daughter classes shall make sure that the class attribute *compatible_baxe* is up to date.

    * `blocksim.control.Earth6DDLPosition.Earth6DDLPosition` instances. See `PlottableCube`
    * networkx graphs. See `PlottableGraph`
    * `blocksim.dsp.DSPLine.ADSPLine`. See `PlottableDSPLine`
    * simple arrays. See `PlottableArray`
    * `blocksim.satellite.Trajectory.Trajectory` instances. See `PlottableTrajectory`
    * `blocksim.dsp.DSPMap.ADSPMap`. See `PlottableDSPSpectrogram`
    * tuple of arrays or dictionaries, see `PlottableGeneric`. The dictionaries keys are:

        * data
        * name
        * unit

    Args:
        plottable: one of the instance above
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = ["plottable", "kwargs", "twinx", "twiny"]

    compatible_baxe = []

    def __init__(self, plottable, kwargs: dict) -> None:
        self.plottable = plottable
        self.twinx = kwargs.pop("twinx", None)
        self.twiny = kwargs.pop("twiny", None)
        self.kwargs = kwargs

    @abstractmethod
    def _make_mline(
        self, axe: "blocksim.graphics.BAxe.ABaxe"
    ) -> Tuple["array", "array", str, str, str, str]:
        """This makes the job of turning a generic plotable into a tuple of useful values

        Returns:
            An numpy array of X coordinates
            An numpy array of Y coordinates
            The name of the X coordinate
            The physical unit of the X variable
            The name of the Y coordinate
            The physical unit of the Y variable

        """
        pass

    def preprocess(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        """Makes the final preparation before plotting with matplotlib

        Args:
            axe: The axe associated with the plottable

        Returns:
            A dictionary with:

            * plot_method: the callable plot method to use to plot the data
            * fill: for 3D plots, the fill method
            * args: the data to be plotted with matplotlib. 2 or 3 elements tuple with numpy arrays
            * mpl_kwargs: the plotting options useable with matplotlib
            * peaks: the peaks found in the data
            * annotations: list of Annotation instances
            * name_of_x_var: name of the X variable
            * unit_of_x_var: unit of the X variable
            * name_of_y_var: name of the Y variable
            * unit_of_y_var: unit of the Y variable
            * xmin: smallest X value
            * xmax: largest X value
            * ymin: smallest Y value, taking into account the X bounds given by `BAxe.set_xlim`
            * ymax: largest Y value, taking into account the X bounds given by `BAxe.set_xlim`

        """
        (
            xd,
            yd,
            name_of_x_var,
            unit_of_x_var,
            name_of_y_var,
            unit_of_y_var,
        ) = self._make_mline(axe)
        fill = ""

        args = (xd, yd)

        kwargs = self.kwargs.copy()
        nb_peaks = kwargs.pop("find_peaks", 0)
        _ = kwargs.pop("transform", None)
        annotations = kwargs.pop("annotations", [])
        plt_mth = "plot"
        lpeaks = find1dpeak(
            nb_peaks=nb_peaks,
            xd=xd,
            yd=yd,
            name_of_x_var=name_of_x_var,
            unit_of_x_var=unit_of_x_var,
        )
        for p in lpeaks:
            logger.info(f"Found peak : {p}")

        # X bounds shall be determined be examining all shared axes
        xmin, xmax = axe.xbounds
        ns = len(xd)
        iok = list(range(ns - 1))
        if not xmin is None:
            iok = np.intersect1d(np.where(xd > xmin)[0], iok)

        if not xmax is None:
            iok = np.intersect1d(np.where(xd <= xmax)[0], iok)

        if len(iok) == 0 or np.all(np.isnan(yd[iok])):
            ymin = np.nan
            ymax = np.nan
        else:
            ymin = np.nanmin(yd[iok])
            ymax = np.nanmax(yd[iok])

        if (
            not isinstance(xd[0], datetime)
            and not isinstance(xd[0], Timestamp)
            and not isinstance(xd[0], np.datetime64)
            and np.all(np.isnan(xd))
        ):
            xmin = np.nan
            xmax = np.nan
        else:
            xmin = np.nanmin(xd)
            xmax = np.nanmax(xd)

        info = {
            # "axe": axe,
            "plottable": self,
            "plot_method": plt_mth,
            "fill": fill,
            "args": args,
            "mpl_kwargs": kwargs,
            "peaks": lpeaks,
            "annotations": annotations,
            "name_of_x_var": name_of_x_var,
            "unit_of_x_var": unit_of_x_var,
            "name_of_y_var": name_of_y_var,
            "unit_of_y_var": unit_of_y_var,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        }
        return info

    def render(
        self,
        axe: "blocksim.graphics.BAxe.ABAxe",
        maxe: "Axes",
        info: dict,
        amp_x: float,
        amp_y: float,
    ) -> dict:
        """This method, knowing the maximum aplitude of data in the axe,
        scales the data by a power of 3.n so that the prefux k (kilo), M (mega), etc can be used

        Args:
            axe: Axe  defining the plot region
            maxe: Matplotlib instance of axe
            info: The data computed by APlottable.render
            amp_x: The amplitude of X coordinates
            amp_y: The amplitude of Y coordinates

        Returns:
            A dictionary with:

                * plottable: this instance
                * plot_method: the callable plot method to use to plot the data
                * fill: for 3D plots, the fill method
                * args: the data to be plotted with matplotlib. 2 or 3 elements tuple with numpy arrays
                * mpl_kwargs: the plotting options useable with matplotlib
                * peaks: the peaks found in the data
                * name_of_x_var: name of the X variable
                * unit_of_x_var: unit of the X variable
                * name_of_y_var: name of the Y variable
                * unit_of_y_var: unit of the Y variable
                * xmin: smallest X value
                * xmax: largest X value
                * ymin: smallest Y value
                * ymax: largest Y value
                * scaled_args: the scaled data to be plotted with matplotlib. 2 or 3 elements tuple with numpy arrays
                * scaled_peaks: the peaks found in the data with scaled coordinates
                * scaled_annotations: the peaks found in the data with scaled coordinates
            The label to use for X axis
            The label to use for Y axis

        """
        args = info["args"]

        x_mult = 1
        x_lbl = ""
        x_unit = info["unit_of_x_var"]
        y_mult = 1
        y_lbl = ""
        y_unit = info["unit_of_y_var"]
        if axe.projection == AxeProjection.LOGX:
            scaled_args = args
        elif axe.projection == AxeProjection.LOGY:
            scaled_args = args
        elif axe.projection == AxeProjection.LOGXY:
            scaled_args = args
        elif axe.projection == AxeProjection.GRAPH:
            scaled_args = args
        elif hasattr(info["args"][0], "__getitem__") and (
            isinstance(info["args"][0][0], datetime)
            or isinstance(info["args"][0][0], Timestamp)
            or isinstance(info["args"][0][0], np.datetime64)
        ):
            scaled_args = args
        elif len(info["args"]) == 2:
            (xd, yd) = info["args"]
            if not np.isnan(amp_x):
                _, x_mult, x_lbl, x_unit = getUnitAbbrev(amp_x, unit=info["unit_of_x_var"])
            if not np.isnan(amp_y):
                _, y_mult, y_lbl, y_unit = getUnitAbbrev(amp_y, unit=info["unit_of_y_var"])
            scaled_args = (xd / x_mult, yd / y_mult)
        else:
            scaled_args = args

        x_label = "%s (%s%s)" % (info["name_of_x_var"], x_lbl, x_unit)
        y_label = "%s (%s%s)" % (info["name_of_y_var"], y_lbl, y_unit)

        for p in info["peaks"]:
            (xp,) = p.coord
            xp /= x_mult
            yp = p.value / y_mult
            txt = "(%.1f %s%s,%.1f)" % (xp, x_lbl, x_unit, yp)
            maxe.plot([xp], [yp], marker="o", color="red", linestyle="")
            maxe.annotate(txt, xy=(xp, yp), fontsize="x-small")

        # scaled_annotations = []
        for a in info["annotations"]:
            (xa, ya) = a.coord
            maxe.annotate(a.text, xy=(xa / x_mult, ya / y_mult), fontsize="x-small")

        info["x_mult"] = x_mult
        info["y_mult"] = y_mult
        info["x_label"] = x_label
        info["y_label"] = y_label

        kwargs = info["mpl_kwargs"]
        plt_mth = info["plot_method"]
        mth = getattr(maxe, plt_mth)

        # ax = axs[2]
        # ax.set_title('Manual DateFormatter', loc='left', y=0.85, x=0.02, fontsize='medium')
        # # Text in the x axis will be displayed in 'YYYY-mm' format.
        # ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%b'))
        # # Rotates and right-aligns the x labels so they don't crowd each other.
        # for label in ax.get_xticklabels(which='major'):
        #     label.set(rotation=30, horizontalalignment='right')

        ret = mth(*scaled_args, **kwargs)
        if isinstance(ret, list):
            info["mline"] = ret[0]
        else:
            info["mline"] = ret

        if info["fill"] == "plot_surface":
            pass
        elif info["fill"] == "pcolormesh":
            maxe.figure.colorbar(ret, ax=maxe)
        elif info["fill"] == "contourf":
            maxe.figure.colorbar(ret, ax=maxe)
        elif info["fill"] == "contour":
            maxe.clabel(ret, inline=True, fontsize=10)

        return info


class Plottable6DDLPosition(APlottable):
    """Allows plotting a `blocksim.control.Earth6DDLPosition.Earth6DDLPosition`
    Only possible in `blocksim.graphics.GraphicSpec.FigureProjection.EARTH3D`.
    Available plotting options:

    * color

    Args:
        plottable: a `blocksim.control.Earth6DDLPosition.Earth6DDLPosition` instance
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = []

    compatible_baxe = [AxeProjection.PANDA3D]

    def _make_mline(
        self, axe: "blocksim.graphics.BAxe.ABaxe"
    ) -> Tuple["array", "array", str, str, str, str]:
        pass

    def preprocess(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        if not axe.projection in self.__class__.compatible_baxe:
            raise AssertionError(f"{axe.projection} is not in {self.__class__.compatible_baxe}")

        mpl_kwargs = self.kwargs.copy()
        if not "size" in mpl_kwargs:
            raise AssertionError(f"Missing 'size' argument when plotting a Earth6DDLPosition")

        _ = mpl_kwargs.pop("annotations", None)
        info = {
            # "axe": axe,
            "plottable": self,
            "plot_method": "plotCube",
            "fill": "",
            "args": (self.plottable.position,),
            "mpl_kwargs": mpl_kwargs,
            "peaks": [],
            "annotations": [],
            "name_of_x_var": "x",  # UNUSED
            "unit_of_x_var": "m",  # UNUSED
            "name_of_y_var": "y",  # UNUSED
            "unit_of_y_var": "m",  # UNUSED
            "xmin": 0.0,  # UNUSED
            "xmax": 0.0,  # UNUSED
            "ymin": 0.0,  # UNUSED
            "ymax": 0.0,  # UNUSED
        }

        return info

    # def render(
    #     self,
    #     axe: "blocksim.graphics.BAxe.ABAxe",
    #     maxe: "Axes",
    #     info: dict,
    #     amp_x: float,
    #     amp_y: float,
    # ) -> dict:
    #     info["scaled_args"] = info["args"]
    #     info["scaled_peaks"] = info["peaks"]
    #     info["scaled_annotations"] = info["annotations"]
    #     info["x_mult"] = 1.0
    #     info["y_mult"] = 1.0
    #     info["x_label"] = 1.0
    #     info["y_label"] = 1.0
    #     info["mline"] = None

    #     return info


class PlottableGraph(APlottable):
    """Allows plotting a networkx MultiDiGraph
    Only possible in `blocksim.graphics.GraphicSpec.FigureProjection.MPL`.
    Available plotting options:
    see https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html

    Args:
        plottable: a networkx MultiDiGraph instance
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = []

    compatible_baxe = [AxeProjection.GRAPH]

    def _make_mline(self, axe: "blocksim.graphics.BAxe.ABaxe"):
        pass

    def preprocess(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        """Makes the final preparation before plotting with matplotlib

        Args:
            axe: The axe associated with the plottable

        Returns:
            A dictionary with:

            * plottable: this instance
            * plot_method: the callable plot method to use to plot the data
            * fill: for 3D plots, the fill method
            * args: the data to be plotted with matplotlib. 2 or 3 elements tuple with numpy arrays
            * mpl_kwargs: the plotting options useable with matplotlib
            * peaks: the peaks found in the data
            * name_of_x_var: name of the X variable
            * unit_of_x_var: unit of the X variable
            * name_of_y_var: name of the Y variable
            * unit_of_y_var: unit of the Y variable
            * xmin: smallest X value
            * xmax: largest X value
            * ymin: smallest Y value
            * ymax: largest Y value

        """
        if not axe.projection in self.__class__.compatible_baxe:
            raise AssertionError(f"{axe.projection} is not in {self.__class__.compatible_baxe}")

        kwds = self.kwargs.copy()
        if not "node_size" in kwds.keys():
            kwds["node_size"] = 1000
        _ = kwds.pop("annotations", None)

        info = {
            # "axe": axe,
            "plottable": self,
            "plot_method": "plotGraph",
            "fill": "",
            "args": (self.plottable,),
            "mpl_kwargs": kwds,
            "peaks": [],
            "annotations": [],
            "name_of_x_var": "",
            "unit_of_x_var": "-",
            "name_of_y_var": "",
            "unit_of_y_var": "-",
            "xmin": np.nan,
            "xmax": np.nan,
            "ymin": np.nan,
            "ymax": np.nan,
        }
        return info

    def render(
        self,
        axe: "blocksim.graphics.BAxe.ABAxe",
        maxe: "Axes",
        info: dict,
        amp_x: float,
        amp_y: float,
    ) -> dict:
        info = super().render(axe, maxe, info, amp_x=amp_x, amp_y=amp_y)
        info["x_label"] = ""
        info["y_label"] = ""

        return info


class PlottableDSPRectilinearLine(APlottable):
    """Specialisation of `APlottable` for `blocksim.dsp.DSPLine.ADSPLine`

    Args:
        plottable: a `blocksim.dsp.DSPLine.ADSPLine` instance
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = []

    compatible_baxe = [
        AxeProjection.RECTILINEAR,
        AxeProjection.LOGX,
        AxeProjection.LOGY,
        AxeProjection.LOGXY,
        # AxeProjection.PLATECARREE,
    ]

    def _make_mline(
        self, axe: "blocksim.graphics.BAxe.ABaxe"
    ) -> Tuple["array", "array", str, str, str, str]:
        transform = self.kwargs.get("transform", self.plottable.default_transform)
        xd = self.plottable.generateXSerie()
        yd = transform(np.array(self.plottable.y_serie))
        name_of_x_var = self.plottable.name_of_x_var
        if name_of_x_var == "":
            name_of_x_var = "-"

        unit_of_x_var = getattr(self.plottable, "unit_of_x_var", "-")
        name_of_y_var = ""
        unit_of_y_var = "-"

        return xd, yd, name_of_x_var, unit_of_x_var, name_of_y_var, unit_of_y_var


class PlottableBodeDiagram(PlottableDSPRectilinearLine):

    __slots__ = []

    compatible_baxe = [
        AxeProjection.RECTILINEAR,
        AxeProjection.LOGY,
    ]


class PlottableDSPPolarLine(PlottableDSPRectilinearLine):

    __slots__ = []

    compatible_baxe = [
        AxeProjection.POLAR,
    ]


class PlottableDSPNorthPolarLine(PlottableDSPRectilinearLine):

    __slots__ = []

    compatible_baxe = [
        AxeProjection.NORTH_POLAR,
    ]


class PlottableDSPHistogram(PlottableDSPRectilinearLine):

    __slots__ = []

    compatible_baxe = [
        AxeProjection.RECTILINEAR,
        AxeProjection.LOGY,
    ]

    def preprocess(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        info = super().preprocess(axe)
        plottable = info["plottable"]
        kwargs = info["mpl_kwargs"]
        kwargs["width"] = plottable.plottable.samplingPeriod
        info["plot_method"] = "bar"
        return info


class PlottableGeneric(APlottable):

    __slots__ = []

    compatible_baxe = [
        AxeProjection.RECTILINEAR,
        AxeProjection.LOGX,
        AxeProjection.LOGY,
        AxeProjection.LOGXY,
        AxeProjection.NORTH_POLAR,
        AxeProjection.PLATECARREE,
        AxeProjection.POLAR,
    ]

    def __init__(self, plottable, kwargs: dict) -> None:
        super().__init__(plottable, kwargs)

    def _make_mline(
        self, axe: "blocksim.graphics.BAxe.ABaxe"
    ) -> Tuple["array", "array", str, str, str, str]:
        transform = self.kwargs.get("transform", lambda x: x)

        (
            xd,
            yd,
            name_of_x_var,
            unit_of_x_var,
            name_of_y_var,
            unit_of_y_var,
        ) = self.plottable.make_line(transform=transform)

        if axe.projection == AxeProjection.PLATECARREE:
            xd *= 180 / pi
            yd *= 180 / pi
            unit_of_x_var = "deg"
            unit_of_y_var = "deg"

        return xd, yd, name_of_x_var, unit_of_x_var, name_of_y_var, unit_of_y_var


class PlottableTrajectory(APlottable):
    """Specialisation of `APlottable` to handle `blocksim.satellite.Trajectory.Trajectory`

    Args:
        plottable: a `blocksim.satellite.Trajectory.Trajectory` instance
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = []

    compatible_baxe = [
        AxeProjection.PANDA3D,
        AxeProjection.DIM3D,
        AxeProjection.PLATECARREE,
        AxeProjection.RECTILINEAR,
    ]

    def _make_mline(
        self, axe: "blocksim.graphics.BAxe.ABaxe"
    ) -> Tuple["array", "array", str, str, str, str]:
        # Only used if axe.figure.projection==FigureProjection.MPL
        lon, lat = self.plottable.getGroundTrack()

        dlon = np.abs(np.diff(lon))
        list_ilmax = np.where(dlon > pi)[0]
        ns = len(lon)

        decal = 0
        for k in range(len(list_ilmax)):
            ilmax = list_ilmax[k] + decal
            if ilmax > 0 and ilmax < decal + ns - 1:
                new_lat = (lat[ilmax] + lat[ilmax + 1]) / 2
                lon = np.insert(lon, ilmax + 1, [pi, np.nan, -pi])
                lat = np.insert(lat, ilmax + 1, [new_lat, np.nan, new_lat])
                decal += 3

        if axe.projection == AxeProjection.PLATECARREE:
            lon *= 180 / pi
            lat *= 180 / pi
            sunit = "deg"
        else:
            sunit = "rad"

        name_of_x_var = "Longitude"
        unit_of_x_var = sunit
        name_of_y_var = "Latitude"
        unit_of_y_var = sunit

        return lon, lat, name_of_x_var, unit_of_x_var, name_of_y_var, unit_of_y_var

    def preprocess(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        if not axe.projection in self.__class__.compatible_baxe:
            raise AssertionError(f"{axe.projection} is not in {self.__class__.compatible_baxe}")

        if axe.figure.projection == FigureProjection.EARTH3D:
            app = axe.figure.mpl_fig
            kwargs = self.kwargs.copy()
            _ = kwargs.pop("transform", None)
            info = {
                # "axe": axe,
                "plottable": self,
                "plot_method": "plotTrajectory",
                "fill": "",
                "args": (self.plottable,),
                "mpl_kwargs": kwargs,
                "peaks": [],
                "annotations": [],
                "name_of_x_var": "x",  # UNUSED
                "unit_of_x_var": "m",  # UNUSED
                "name_of_y_var": "y",  # UNUSED
                "unit_of_y_var": "m",  # UNUSED
                "xmin": 0.0,  # UNUSED
                "xmax": 0.0,  # UNUSED
                "ymin": 0.0,  # UNUSED
                "ymax": 0.0,  # UNUSED
            }
        elif axe.figure.projection == FigureProjection.MPL:
            info = super().preprocess(axe)

        return info


class APlottableDSPMap(APlottable):
    """Specialisation of `APlottable` for `blocksim.dsp.DSPMap.ADSPMap`

    Args:
        plottable: a `blocksim.dsp.DSPMap.ADSPMap` instance
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = []

    def preprocess(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        if not axe.projection in self.__class__.compatible_baxe:
            raise AssertionError(f"{axe.projection} is not in {self.__class__.compatible_baxe}")

        mline = self.plottable

        kwargs = self.kwargs.copy()
        fill = kwargs.pop("fill", "pcolormesh")
        transform = kwargs.pop("transform", mline.default_transform)
        find_peaks = kwargs.pop("find_peaks", 0)
        annotations = kwargs.pop("annotations", [])

        if axe.projection == AxeProjection.DIM3D:
            plot_mth = "plot_surface"
            kwargs.pop("levels", None)
        else:
            plot_mth = fill

        if fill == "pcolormesh":
            kwargs.pop("levels", None)
        elif fill == "contourf":
            pass
        elif fill == "contour":
            pass
        (
            X,
            Y,
            name_of_x_var,
            unit_of_x_var,
            name_of_y_var,
            unit_of_y_var,
        ) = self._make_mline(axe)

        Z = transform(mline.img)
        args = (X, Y, Z)

        lpeaks = mline.findPeaksWithTransform(transform=transform, nb_peaks=find_peaks)
        for p in lpeaks:
            logger.info(f"Found peak : {p}")

        if np.all(np.isnan(Y)):
            ymin = np.nan
            ymax = np.nan
        else:
            ymin = np.nanmin(Y)
            ymax = np.nanmax(Y)

        if np.all(np.isnan(X)):
            xmin = np.nan
            xmax = np.nan
        else:
            xmin = np.nanmin(X)
            xmax = np.nanmax(X)

        info = {
            # "axe": axe,
            "plottable": self,
            "plot_method": plot_mth,
            "fill": fill,
            "args": args,
            "mpl_kwargs": kwargs,
            "peaks": lpeaks,
            "annotations": annotations,
            "name_of_x_var": name_of_x_var,
            "unit_of_x_var": unit_of_x_var,
            "name_of_y_var": name_of_y_var,
            "unit_of_y_var": unit_of_y_var,
            "xmin": xmin,
            "xmax": xmax,
            "ymin": ymin,
            "ymax": ymax,
        }
        return info

    def render(
        self,
        axe: "blocksim.graphics.BAxe.ABAxe",
        maxe: "Axes",
        info: dict,
        amp_x: float,
        amp_y: float,
    ) -> dict:
        args = info["args"]

        x_mult = 1
        x_lbl = ""
        x_unit = info["unit_of_x_var"]
        y_mult = 1
        y_lbl = ""
        y_unit = info["unit_of_y_var"]
        if len(args) == 1:
            scaled_args = args
        elif len(args) == 3:
            xd, yd, zd = args

            _, x_mult, x_lbl, x_unit = getUnitAbbrev(amp_x, unit=info["unit_of_x_var"])
            _, y_mult, y_lbl, y_unit = getUnitAbbrev(amp_y, unit=info["unit_of_y_var"])

            scaled_args = (xd / x_mult, yd / y_mult, zd)

        x_label = "%s (%s%s)" % (info["name_of_x_var"], x_lbl, x_unit)
        y_label = "%s (%s%s)" % (info["name_of_y_var"], y_lbl, y_unit)

        for p in info["peaks"]:
            xp, yp = p.coord
            xp /= x_mult
            yp /= y_mult
            txt = "(%.1f,%.1f,%.1f)" % (xp, yp, p.value)
            maxe.plot([xp], [yp], marker="o", color="red", linestyle="")
            maxe.annotate(txt, xy=(xp, yp), fontsize="x-small")

        for a in info["annotations"]:
            (xa, ya) = a.coord
            maxe.annotate(a.text, xy=(xa / x_mult, ya / y_mult), fontsize="x-small")

        info["x_mult"] = x_mult
        info["y_mult"] = y_mult
        info["x_label"] = x_label
        info["y_label"] = y_label

        kwargs = info["mpl_kwargs"]
        plt_mth = info["plot_method"]
        mth = getattr(maxe, plt_mth)
        ret = mth(*scaled_args, **kwargs)
        info["mline"] = ret

        if info["fill"] == "plot_surface":
            pass
        elif info["fill"] == "pcolormesh":
            maxe.figure.colorbar(ret, ax=maxe)
        elif info["fill"] == "contourf":
            maxe.figure.colorbar(ret, ax=maxe)
        elif info["fill"] == "contour":
            maxe.clabel(ret, inline=True, fontsize=10)

        if axe.projection == AxeProjection.PLATECARREE and info["fill"] in [
            "contourf",
            "pcolormesh",
        ]:
            maxe.coastlines()

        return info


class PlottableImage(APlottableDSPMap):
    """Specialisation of `APlottable` for plotting images

    Args:
        plottable: a Path instance
        kwargs: The dictionary of options for plotting

    """

    __slots__ = []

    compatible_baxe = [
        AxeProjection.RECTILINEAR,
    ]

    def _make_mline(
        self, axe: "blocksim.graphics.BAxe.ABaxe"
    ) -> Tuple["array", "array", str, str, str, str]:
        pass

    def preprocess(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        img = mpimg.imread(str(self.plottable))

        kwargs = self.kwargs.copy()
        _ = kwargs.pop("transform", None)
        _ = kwargs.pop("annotations", None)

        info = {
            # "axe": axe,
            "plottable": self,
            "plot_method": "imshow",
            "fill": "",
            "args": (img,),
            "mpl_kwargs": kwargs,
            "peaks": [],
            "annotations": [],
            "name_of_x_var": "",
            "unit_of_x_var": "-",
            "name_of_y_var": "",
            "unit_of_y_var": "-",
            "xmin": np.nan,
            "xmax": np.nan,
            "ymin": np.nan,
            "ymax": np.nan,
        }
        return info


class PlottableDSPRectilinearMap(APlottableDSPMap):
    """Specialisation of `APlottable` for `blocksim.dsp.DSPMap.ADSPMap`

    Args:
        plottable: a `blocksim.dsp.DSPMap.ADSPMap` instance
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = []

    compatible_baxe = [
        AxeProjection.RECTILINEAR,
        AxeProjection.LOGX,
        AxeProjection.LOGY,
        AxeProjection.LOGXY,
        AxeProjection.PLATECARREE,
        AxeProjection.DIM3D,
    ]

    def _make_mline(
        self, axe: "blocksim.graphics.BAxe.ABaxe"
    ) -> Tuple["array", "array", str, str, str, str]:
        mline = self.plottable

        x_samp = mline.generateXSerie()
        y_samp = mline.generateYSerie()

        name_of_x_var = mline.name_of_x_var
        unit_of_x_var = getattr(mline, "unit_of_x_var", "-")

        name_of_y_var = mline.name_of_y_var
        unit_of_y_var = getattr(mline, "unit_of_y_var", "-")

        X, Y = np.meshgrid(x_samp, y_samp)
        if axe.projection == AxeProjection.PLATECARREE:
            X *= 180 / pi
            Y *= 180 / pi

        return X, Y, name_of_x_var, unit_of_x_var, name_of_y_var, unit_of_y_var


class PlottableDSPPolarMap(PlottableDSPRectilinearMap):
    __slots__ = []

    compatible_baxe = [
        AxeProjection.POLAR,
        AxeProjection.DIM3D,
    ]

    def _make_mline(
        self, axe: "blocksim.graphics.BAxe.ABaxe"
    ) -> Tuple["array", "array", str, str, str, str]:
        mline = self.plottable

        x_samp = mline.generateXSerie()
        y_samp = mline.generateYSerie()

        name_of_x_var = mline.name_of_x_var
        unit_of_x_var = getattr(mline, "unit_of_x_var", "-")

        name_of_y_var = mline.name_of_y_var
        unit_of_y_var = getattr(mline, "unit_of_y_var", "-")

        P, R = np.meshgrid(x_samp, y_samp)

        return P, R, name_of_x_var, unit_of_x_var, name_of_y_var, unit_of_y_var


class PlottableDSPNorthPolarMap(PlottableDSPPolarMap):
    __slots__ = []

    compatible_baxe = [
        AxeProjection.NORTH_POLAR,
        AxeProjection.DIM3D,
    ]


class PlottableFactory(object):
    """Factory class that instanciates the adapted daughter class of `APlottable` to handle the object to plot"""

    __slots__ = []

    @classmethod
    def create(cls, mline, kwargs: dict) -> APlottable:
        """Creates the adapted daughter class of `APlottable` to handle the object to plot

        Args:
            mline: Object to plot. Can be:

            * a `blocksim.dsp.DSPLine.ADSPLine`
            * a `blocksim.dsp.DSPMap.ADSPMap`
            * a 2 elements tuple of numpy arrays
            * a simple numpy arrays
            * a networkx DiGraph
            * a 2 elements tuple of dictionaries, with keys:

                * data
                * name
                * unit

            kwargs: The plotting options for the object

        Returns:
            The APlottable instance suited to the object

        """
        if isinstance(mline, Graph):
            ret = PlottableGraph(mline, kwargs)

        elif isinstance(mline, Trajectory):
            ret = PlottableTrajectory(mline, kwargs)

        elif isinstance(mline, Earth6DDLPosition):
            ret = Plottable6DDLPosition(mline, kwargs)

        elif isinstance(mline, DSPRectilinearLine):
            ret = PlottableDSPRectilinearLine(mline, kwargs)

        elif isinstance(mline, DSPPolarLine):
            ret = PlottableDSPPolarLine(mline, kwargs)

        elif isinstance(mline, DSPNorthPolarLine):
            ret = PlottableDSPNorthPolarLine(mline, kwargs)

        elif isinstance(mline, DSPHistogram):
            ret = PlottableDSPHistogram(mline, kwargs)

        elif isinstance(mline, DSPRectilinearMap):
            ret = PlottableDSPRectilinearMap(mline, kwargs)

        elif isinstance(mline, DSPBodeDiagram):
            ret = PlottableBodeDiagram(mline, kwargs)

        elif isinstance(mline, DSPPolarMap):
            ret = PlottableDSPPolarMap(mline, kwargs)

        elif isinstance(mline, DSPNorthPolarMap):
            ret = PlottableDSPNorthPolarMap(mline, kwargs)

        elif isinstance(mline, tuple):
            if len(mline) == 0:
                raise AssertionError(f"Plottable reduced to an empty tuple")

            if np.isscalar(mline[0]):
                gp = GPlottable.from_serie(sy=mline)
            else:
                gp = GPlottable.from_tuple(mline)
            ret = PlottableGeneric(gp, kwargs)

        elif isinstance(mline, (pd.Series, GPlottable, np.ndarray, list)):
            gp = GPlottable.from_serie(sy=mline)
            ret = PlottableGeneric(gp, kwargs)

        elif isinstance(mline, Path):
            ret = PlottableImage(mline, kwargs)

        return ret
