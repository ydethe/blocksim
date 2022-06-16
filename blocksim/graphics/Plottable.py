from abc import ABCMeta, abstractmethod
from typing import Tuple
from pathlib import Path

import networkx as nx
from networkx.classes.graph import Graph
import numpy as np
from numpy import pi
import matplotlib.image as mpimg

from ..utils import find1dpeak
from ..dsp.DSPLine import (
    DSPRectilinearLine,
    DSPPolarLine,
    DSPNorthPolarLine,
    DSPHistogram,
)
from ..dsp.DSPMap import DSPRectilinearMap, DSPPolarMap, DSPNorthPolarMap
from .GraphicSpec import AxeProjection, FigureProjection, Annotation
from ..satellite.Trajectory import Trajectory, Cube
from . import getUnitAbbrev


class APlottable(metaclass=ABCMeta):
    """This base abstract class describes all the entities able to be plotted:

    Daughter classes shall make sure that the class attribute *compatible_baxe* is up to date.

    * `blocksim.satellite.Trajectory.Cube` instances. See `PlottableCube`
    * networkx graphs. See `PlottableGraph`
    * `blocksim.dsp.DSPLine.DSPLine`. See `PlottableDSPLine`
    * simple arrays. See `PlottableArray`
    * `blocksim.satellite.Trajectory.Trajectory` instances. See `PlottableTrajectory`
    * `blocksim.dsp.DSPMap.DSPMap`. See `PlottableDSPSpectrogram`
    * tuple of arrays or dictionaries, see `PlottableTuple`. The dictionaries keys are:

        * data
        * name
        * unit

    Args:
        plottable: one of the instance above
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = ["plottable", "kwargs"]

    compatible_baxe = []

    def __init__(self, plottable, kwargs: dict) -> None:
        self.plottable = plottable
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

    def render(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
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
        maxe = axe.mpl_axe
        (
            xd,
            yd,
            name_of_x_var,
            unit_of_x_var,
            name_of_y_var,
            unit_of_y_var,
        ) = self._make_mline(axe)

        fill = ""
        plot_mth = maxe.plot

        args = (xd, yd)

        kwargs = self.kwargs.copy()
        nb_peaks = kwargs.pop("find_peaks", 0)
        _ = kwargs.pop("transform", None)
        annotations = kwargs.pop("annotations", [])
        lpeaks = find1dpeak(
            nb_peaks=nb_peaks,
            xd=xd,
            yd=yd,
            name_of_x_var=name_of_x_var,
            unit_of_x_var=unit_of_x_var,
        )

        # X bounds shall be determined be examining all shared axes
        xmin, xmax = axe.xbounds
        ns = len(xd)
        iok = list(range(ns))
        if not xmin is None:
            iok = np.intersect1d(np.where(xd > xmin)[0], iok)

        if not xmax is None:
            iok = np.intersect1d(np.where(xd <= xmax)[0], iok)

        info = {
            "axe": axe,
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
            "xmin": np.min(xd),
            "xmax": np.max(xd),
            "ymin": np.min(yd[iok]),
            "ymax": np.max(yd[iok]),
        }
        return info

    def scaleAndLabelData(self, info: dict, amp_x: float, amp_y: float) -> dict:
        """This method, knowing the maximum aplitude of data in the axe,
        scales the data by a power of 3.n so that the prefux k (kilo), M (mega), etc can be used

        Args:
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
        axe = info["axe"]
        xd, yd = info["args"]

        x_mult = 1
        x_lbl = ""
        x_unit = info["unit_of_x_var"]
        y_mult = 1
        y_lbl = ""
        y_unit = info["unit_of_y_var"]
        if axe.projection == AxeProjection.LOGX:
            args = (xd, yd)
        elif axe.projection == AxeProjection.LOGY:
            args = (xd, yd)
        elif axe.projection == AxeProjection.LOGXY:
            args = (xd, yd)
        else:
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
            scaled_peaks.append((xp / x_mult, yp, txt))

        scaled_annotations = []
        for a in info["annotations"]:
            (xa, ya) = a.coord
            an = Annotation(coord=(xa / x_mult, ya / y_mult), text=a.text)
            scaled_annotations.append(an)

        info["scaled_peaks"] = scaled_peaks
        info["scaled_annotations"] = scaled_annotations
        info["x_mult"] = x_mult
        info["y_mult"] = y_mult
        info["x_label"] = x_label
        info["y_label"] = y_label

        return info


class PlottableCube(APlottable):
    """Allows plotting a `blocksim.satellite.Trajectory.Cube`
    Only possible in `blocksim.graphics.enums.FigureProjection.EARTH3D`.
    Available plotting options:

    * color

    See `blocksim.graphics.B3DPlotter.B3DPlotter.plotCube`

    Args:
        plottable: a `blocksim.satellite.Trajectory.Cube` instance
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = []

    compatible_baxe = [AxeProjection.DIM3D]

    def _make_mline(
        self, axe: "blocksim.graphics.BAxe.ABaxe"
    ) -> Tuple["array", "array", str, str, str, str]:
        pass

    def render(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        if not axe.projection in self.__class__.compatible_baxe:
            raise AssertionError(
                f"{axe.projection} is not in {self.__class__.compatible_baxe}"
            )

        app = axe.figure.mpl_fig
        mpl_kwargs = self.kwargs.copy()
        mpl_kwargs["size"] = self.plottable.size
        _ = mpl_kwargs.pop("annotations", None)
        info = {
            "axe": axe,
            "plottable": self,
            "plot_method": app.plotCube,
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

    def scaleAndLabelData(self, info: dict, amp_x: float, amp_y: float) -> dict:
        info["scaled_args"] = info["args"]
        info["scaled_peaks"] = info["peaks"]
        info["scaled_annotations"] = info["annotations"]
        info["x_mult"] = 1.0
        info["y_mult"] = 1.0
        info["x_label"] = 1.0
        info["y_label"] = 1.0

        return info


class PlottableGraph(APlottable):
    """Allows plotting a networkx MultiDiGraph
    Only possible in `blocksim.graphics.enums.FigureProjection.MPL`.
    Available plotting options:
    see https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw_networkx.html

    Args:
        plottable: a networkx MultiDiGraph instance
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = []

    compatible_baxe = [AxeProjection.RECTILINEAR]

    def __init__(self, plottable, kwargs: dict) -> None:
        self.plottable = plottable
        self.kwargs = kwargs

    def _make_mline(self, axe: "blocksim.graphics.BAxe.ABaxe"):
        pass

    def render(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
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
            raise AssertionError(
                f"{axe.projection} is not in {self.__class__.compatible_baxe}"
            )

        maxe = axe.mpl_axe

        kwds = self.kwargs.copy()
        if not "node_size" in kwds.keys():
            kwds["node_size"] = 1000
        annotat_ions = kwds.pop("annotations", None)

        info = {
            "axe": axe,
            "plottable": self,
            "plot_method": nx.draw_networkx,
            "fill": "",
            "args": (self.plottable,),
            "mpl_kwargs": kwds,
            "peaks": 0,
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

    def scaleAndLabelData(self, info: dict, amp_x: float, amp_y: float) -> dict:
        """This method, knowing the maximum aplitude of data in the axe,
        scales the data by a power of 3.n so that the prefux k (kilo), M (mega), etc can be used

        Args:
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
            The label to use for X axis
            The label to use for Y axis

        """
        info["scaled_args"] = info["args"]
        info["scaled_peaks"] = []
        info["scaled_annotations"] = []
        info["x_mult"] = 1.0
        info["y_mult"] = 1.0
        info["x_label"] = ""
        info["y_label"] = ""

        return info


class PlottableDSPRectilinearLine(APlottable):
    """Specialisation of `APlottable` for `blocksim.dsp.DSPLine.DSPLine`

    Args:
        plottable: a `blocksim.dsp.DSPLine.DSPLine` instance
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

    def render(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        info = super().render(axe)
        axe = info["axe"]
        plottable = info["plottable"]
        kwargs = info["mpl_kwargs"]
        kwargs["width"] = plottable.plottable.samplingPeriod
        maxe = axe.mpl_axe
        info["plot_method"] = maxe.bar
        return info


class PlottableArray(APlottable):
    """Specialisation of `APlottable` to handle simple numpy arrays,

    Args:
        plottable: a numpy array
        kwargs: The dictionary of options for plotting (color, width,etc)

    Examples:
        >>> _ = PlottableArray(plottable=np.arange(10), kwargs={})

    """

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

    def _make_mline(
        self, axe: "blocksim.graphics.BAxe.ABaxe"
    ) -> Tuple["array", "array", str, str, str, str]:
        transform = self.kwargs.get("transform", lambda x: x)
        yd = transform(np.array(self.plottable))
        ns = len(yd)
        xd = np.arange(ns)
        name_of_x_var = ""
        unit_of_x_var = "-"
        name_of_y_var = ""
        unit_of_y_var = "-"

        if axe.projection == AxeProjection.PLATECARREE:
            xd *= 180 / pi
            yd *= 180 / pi
            unit_of_x_var = "deg"
            unit_of_y_var = "deg"

        return xd, yd, name_of_x_var, unit_of_x_var, name_of_y_var, unit_of_y_var


class PlottableTuple(APlottable):
    """Specialisation of `APlottable` to handle a 2 elements tuple of dictionaries
    Each dictionary contains the following keys:

    * data (mandatory)
    * name (optional)
    * unit (optional)

    Args:
        plottable: a 2 elements tuple of dictionaries
        kwargs: The dictionary of options for plotting (color, width,etc)

    Examples:
        >>> xdesc = {"data": np.arange(10), "name": "Time", "unit": "s"}
        >>> ydesc = {"data": np.arange(10), "name": "Time", "unit": "s"}
        >>> _ = PlottableTuple(plottable=(xdesc, ydesc), kwargs={})

    """

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

    def _extractDataFromDesc(self, desc) -> Tuple["array", str, str]:
        if isinstance(desc, dict):
            dat = np.array(desc.get("data"))
            name_of_var = desc.get("name", "")
            unit_of_var = desc.get("unit", "")
        else:
            dat = desc
            name_of_var = ""
            unit_of_var = ""

        if unit_of_var == "":
            unit_of_var = "-"

        return np.array(dat), name_of_var, unit_of_var

    def _make_mline(
        self, axe: "blocksim.graphics.BAxe.ABaxe"
    ) -> Tuple["array", "array", str, str, str, str]:
        transform = self.kwargs.get("transform", lambda x: x)
        xdesc, ydesc = self.plottable

        xd, name_of_x_var, unit_of_x_var = self._extractDataFromDesc(xdesc)
        yd, name_of_y_var, unit_of_y_var = self._extractDataFromDesc(ydesc)

        yd = transform(yd)

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

    def render(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        if not axe.projection in self.__class__.compatible_baxe:
            raise AssertionError(
                f"{axe.projection} is not in {self.__class__.compatible_baxe}"
            )

        if axe.figure.projection == FigureProjection.EARTH3D:
            app = axe.figure.mpl_fig
            kwargs = self.kwargs.copy()
            _ = kwargs.pop("transform", None)
            info = {
                "axe": axe,
                "plottable": self,
                "plot_method": app.plotTrajectory,
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
            info = super().render(axe)

        return info

    def scaleAndLabelData(self, info, amp_x, amp_y) -> dict:
        info["scaled_args"] = info["args"]
        info["scaled_peaks"] = info["peaks"]
        info["scaled_annotations"] = info["annotations"]
        info["x_mult"] = 1.0
        info["y_mult"] = 1.0
        info["x_label"] = 1.0
        info["y_label"] = 1.0

        return info


class APlottableDSPMap(APlottable):
    """Specialisation of `APlottable` for `blocksim.dsp.DSPMap.DSPMap`

    Args:
        plottable: a `blocksim.dsp.DSPMap.DSPMap` instance
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = []

    def render(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        if not axe.projection in self.__class__.compatible_baxe:
            raise AssertionError(
                f"{axe.projection} is not in {self.__class__.compatible_baxe}"
            )

        mline = self.plottable
        maxe = axe.mpl_axe

        kwargs = self.kwargs.copy()
        fill = kwargs.pop("fill", "pcolormesh")
        transform = kwargs.pop("transform", mline.default_transform)
        find_peaks = kwargs.pop("find_peaks", 0)
        annotations = kwargs.pop("annotations", [])

        if axe.projection == AxeProjection.DIM3D:
            plot_mth = maxe.plot_surface
            kwargs.pop("levels", None)
        else:
            plot_mth = getattr(maxe, fill)

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

        info = {
            "axe": axe,
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
            "xmin": np.min(X),
            "xmax": np.max(X),
            "ymin": np.min(Y),
            "ymax": np.max(Y),
        }
        return info

    def scaleAndLabelData(self, info, amp_x, amp_y) -> dict:
        xd, yd, zd = info["args"]

        _, x_mult, x_lbl, x_unit = getUnitAbbrev(amp_x, unit=info["unit_of_x_var"])
        _, y_mult, y_lbl, y_unit = getUnitAbbrev(amp_y, unit=info["unit_of_y_var"])

        args = (xd / x_mult, yd / y_mult, zd)
        x_label = "%s (%s%s)" % (info["name_of_x_var"], x_lbl, x_unit)
        y_label = "%s (%s%s)" % (info["name_of_y_var"], y_lbl, y_unit)

        scaled_peaks = []
        for p in info["peaks"]:
            xp, yp = p.coord
            txt = "(%.1f,%.1f,%.1f)" % (xp / x_mult, yp / y_mult, p.value)
            scaled_peaks.append((xp / x_mult, yp / y_mult, txt))

        scaled_annotations = []
        for a in info["annotations"]:
            xa, ya = a.coord
            an = Annotation(coord=(xa / x_mult, ya / y_mult), text=a.text)
            scaled_annotations.append(an)

        info["scaled_args"] = args
        info["scaled_peaks"] = scaled_peaks
        info["scaled_annotations"] = scaled_annotations
        info["x_mult"] = x_mult
        info["y_mult"] = y_mult
        info["x_label"] = x_label
        info["y_label"] = y_label

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

    def render(self, axe: "blocksim.graphics.BAxe.ABaxe") -> dict:
        maxe = axe.mpl_axe
        maxe.grid(False)
        img = mpimg.imread(str(self.plottable))

        kwargs = self.kwargs.copy()
        _ = kwargs.pop("transform", None)
        _ = kwargs.pop("annotations", None)

        info = {
            "axe": axe,
            "plottable": self,
            "plot_method": maxe.imshow,
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

    def scaleAndLabelData(self, info: dict, amp_x: float, amp_y: float) -> dict:
        info["scaled_peaks"] = []
        info["scaled_args"] = info["args"]
        info["scaled_annotations"] = []
        info["x_mult"] = 1
        info["y_mult"] = 1
        info["x_label"] = ""
        info["y_label"] = ""

        return info


class PlottableDSPRectilinearMap(APlottableDSPMap):
    """Specialisation of `APlottable` for `blocksim.dsp.DSPMap.DSPMap`

    Args:
        plottable: a `blocksim.dsp.DSPMap.DSPMap` instance
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

            * a `blocksim.dsp.DSPLine.DSPLine`
            * a `blocksim.dsp.DSPMap.DSPMap`
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

        elif isinstance(mline, Cube):
            ret = PlottableCube(mline, kwargs)

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

        elif isinstance(mline, DSPPolarMap):
            ret = PlottableDSPPolarMap(mline, kwargs)

        elif isinstance(mline, DSPNorthPolarMap):
            ret = PlottableDSPNorthPolarMap(mline, kwargs)

        elif isinstance(mline, tuple):
            ret = PlottableTuple(mline, kwargs)

        elif isinstance(mline, Path):
            ret = PlottableImage(mline, kwargs)

        else:
            ret = PlottableArray(mline, kwargs)

        return ret
