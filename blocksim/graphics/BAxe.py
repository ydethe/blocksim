from abc import ABCMeta, abstractmethod
import types
from functools import lru_cache

import numpy as np
from numpy import pi, sqrt, cos, sin
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic

from .. import logger
from .B3DPlotter import B3DPlotter
from ..constants import Req
from .GraphicSpec import FigureProjection, AxeProjection
from .BLayout import BGridElement
from .Plottable import *


class ABaxe(metaclass=ABCMeta):
    """Class that describes the axe. Not yet a matplotlib axe

    Args:
        figure: Parent figure containing the BAxe
        title: Title of the axe
        spec: Position in the BGridSpec
        sharex: ABaxe instance to share X limits with
        sharey: ABaxe instance to share Y limits with

    """

    __slots__ = [
        "figure",
        "title",
        "spec",
        "parent_sharex",
        "children_sharex",
        "parent_sharey",
        "children_sharey",
        "kwargs",
        "plottable_factories",
        "xbounds",
        "ybounds",
        "mpl_axe",
    ]

    projection = None

    @abstractmethod
    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "Axes":
        pass

    def render(self, maxe: "Axes") -> "AxesSubplot":
        """Creates a matplotlib axe according to the BAxe.plot commands

        Args:
            maxe: Matplotlib axes

        Returns:
            The created matplotlib axe

        """
        if self.title != "":
            maxe.set_title(self.title)

        if not self.projection == AxeProjection.PANDA3D:
            maxe.grid(True)

        if len(self.plottable_factories) == 0:
            return maxe

        display_legend = False
        rendered_plottables = []

        global_xmin = None
        global_xmax = None
        global_ymin = None
        global_ymax = None

        for plottable in self.plottable_factories:
            info = plottable.preprocess(self)

            rendered_plottables.append(info)

            if global_xmin is None:
                global_xmin = info["xmin"]
            if global_xmax is None:
                global_xmax = info["xmin"]
            if global_ymin is None:
                global_ymin = info["ymin"]
            if global_ymax is None:
                global_ymax = info["ymin"]

            if info["xmin"] < global_xmin:
                global_xmin = info["xmin"]
            if info["xmax"] > global_xmax:
                global_xmax = info["xmax"]
            if info["ymin"] < global_ymin:
                global_ymin = info["ymin"]
            if info["ymax"] > global_ymax:
                global_ymax = info["ymax"]

        amp_x = global_xmax - global_xmin
        amp_y = global_ymax - global_ymin

        x_label = y_label = ""
        x_unit = y_unit = "-"
        for info in rendered_plottables:
            plottable = info["plottable"]
            if plottable.twinx is None:
                info = plottable.render(self, maxe, info, amp_x, amp_y)
            else:
                maxe_tx = maxe.twinx()
                info = plottable.render(self, maxe_tx, info, amp_x, amp_y)
                line = info["mline"]
                maxe_tx.tick_params(axis="y", labelcolor=line.get_color())

            # Handling unit and label of X axis
            if x_unit == "-":
                x_label = info["x_label"]
                x_unit = info["unit_of_x_var"]
            if info["unit_of_x_var"] != "-":
                if x_unit != info["unit_of_x_var"]:
                    raise AssertionError(
                        f"Inconsistent X units: {x_label} with {info['x_label']}"
                    )
                x_label = info["x_label"]

            # Handling unit and label of Y axis
            y_labelp = info["y_label"]
            y_unitp = info["unit_of_y_var"]
            if y_unit == "-":
                y_label = y_labelp
                y_unit = y_unitp
            if y_unitp != "-":
                if y_unit != y_unitp:
                    raise AssertionError(
                        f"Inconsistent Y units: {y_label} with {y_labelp}"
                    )
                y_label = y_labelp

            # Check if we shall call maxe.legend (at the end)
            if "label" in info["mpl_kwargs"].keys():
                display_legend = True

        if display_legend:
            maxe.legend(loc="best")

        if (
            self.projection == AxeProjection.RECTILINEAR
            or self.projection == AxeProjection.LOGX
            or self.projection == AxeProjection.LOGY
            or self.projection == AxeProjection.LOGXY
            or self.projection == AxeProjection.DIM3D
        ):
            if x_label != "":
                maxe.set_xlabel(x_label)
            if y_label != "":
                maxe.set_ylabel(y_label)

                xmin, xmax = self.xbounds
                maxe.set_xlim(xmin, xmax)

        if (
            self.projection == AxeProjection.RECTILINEAR
            or self.projection == AxeProjection.LOGX
            or self.projection == AxeProjection.LOGY
            or self.projection == AxeProjection.LOGXY
            or self.projection == AxeProjection.DIM3D
        ):
            ymin, ymax = self.ybounds
            if ymin is None and not np.isnan(global_ymin):
                ymin = global_ymin / info["y_mult"]
            if ymax is None and not np.isnan(global_ymax):
                ymax = global_ymax / info["y_mult"]
            maxe.set_ylim(ymin, ymax)

        if (
            self.projection == AxeProjection.POLAR
            or self.projection == AxeProjection.NORTH_POLAR
        ):
            ymin, ymax = self.ybounds
            if ymin is None and not np.isnan(global_ymin):
                ymin = global_ymin / info["y_mult"]
            if ymax is None and not np.isnan(global_ymax):
                ymax = global_ymax / info["y_mult"]
            maxe.set_rlim(ymin, ymax)

        return maxe

    def _solveSharedBAxes(self, mfig: "Figure", mgs: "SubplotSpec", maxe: "Axes"):
        if not self.parent_sharex is None:
            sharex = self.parent_sharex.createMplAxe(mfig, mgs)
            maxe.get_shared_x_axes().join(maxe, sharex)

        if not self.parent_sharey is None:
            sharey = self.parent_sharey.createMplAxe(mfig, mgs)
            maxe.get_shared_y_axes().join(maxe, sharey)

    def __init__(
        self,
        figure: "blocksim.graphics.BFigure.ABFigure",
        title: str,
        spec: BGridElement,
        sharex: "blocksim.graphics.BAxe.ABaxe" = None,
        sharey: "blocksim.graphics.BAxe.ABaxe" = None,
        kwargs={},
    ):
        self.figure = figure
        self.title = title
        self.spec = spec
        self.parent_sharex = sharex
        self.children_sharex = []
        self.parent_sharey = sharey
        self.children_sharey = []
        self.kwargs = kwargs

        if sharex is None:
            self.xbounds = None, None
        else:
            self.xbounds = sharex.xbounds
            sharex._addChildSharex(self)

        if sharey is None:
            self.ybounds = None, None
        else:
            self.ybounds = sharey.xbounds
            sharey._addChildSharey(self)

        self.plottable_factories = []
        self.mpl_axe = None

    def _addChildSharex(self, sharex: "blocksim.graphics.BAxe.ABaxe"):
        self.children_sharex.append(sharex)

    def _addChildSharey(self, sharey: "blocksim.graphics.BAxe.ABaxe"):
        self.children_sharey.append(sharey)

    def _findRootSharex(self) -> "blocksim.graphics.BAxe.ABaxe":
        if self.parent_sharex is None:
            return self
        else:
            return self.parent_sharex._findRootSharex()

    def _findRootSharey(self) -> "blocksim.graphics.BAxe.ABaxe":
        if self.parent_sharey is None:
            return self
        else:
            return self.parent_sharey._findRootSharey()

    def registerPlottable(self, plottable: APlottable):
        """Registers the APlottable in the list of objects handled by the axe

        Args:
            plottable: APlottable object

        """

        if not self.projection in plottable.compatible_baxe:
            raise AssertionError(
                f"{self.projection} not in {plottable.compatible_baxe}"
            )

        self.plottable_factories.append(plottable)

    def set_xlim(self, xmin: float = None, xmax: float = None, _from_root: bool = True):
        """Set X limits
        The values are given in S.I. units (without scaling)

        Args:
            xmin: The left xlim in data coordinates. Passing None leaves the limit unchanged.
            xmax: The right xlim in data coordinates. Passing None leaves the limit unchanged.

        """
        if _from_root:
            rootx = self._findRootSharex()
        else:
            rootx = self

        rootx.xbounds = xmin, xmax
        for axe in rootx.children_sharex:
            axe.set_xlim(xmin, xmax, _from_root=False)

    def set_ylim(self, ymin: float = None, ymax: float = None, _from_root: bool = True):
        """Set Y limits
        The values are given in S.I. units (without scaling)

        Args:
            ymin: The left ylim in data coordinates. Passing None leaves the limit unchanged.
            ymax: The right ylim in data coordinates. Passing None leaves the limit unchanged.

        """
        if _from_root:
            rooty = self._findRootSharey()
        else:
            rooty = self

        rooty.ybounds = ymin, ymax
        for axe in rooty.children_sharey:
            axe.set_ylim(ymin, ymax, _from_root=False)

    def plot(self, plottable, **kwargs) -> APlottable:
        """Records the plot command (without executing it) and does some checks

        Args:
            plottable: Object to plot. Can be:

            * a `blocksim.dsp.DSPLine.ADSPLine`
            * a `blocksim.dsp.DSPMap.ADSPMap`
            * a 2 elements tuple of numpy arrays
            * a simple numpy arrays
            kwargs: The plotting options for the object

        """
        if plottable is None:
            return
        res = PlottableFactory.create(plottable, kwargs=kwargs)
        self.registerPlottable(res)
        return res

    def scatter(self, plottable, **kwargs) -> APlottable:
        """Records the scatter command (without executing it) and does some checks

        Args:
            plottable: Object to plot. Can be:

            * a `blocksim.dsp.DSPLine.ADSPLine`
            * a `blocksim.dsp.DSPMap.ADSPMap`
            * a 2 elements tuple of numpy arrays
            * a simple numpy arrays
            kwargs: The plotting options for the object

        """
        if plottable is None:
            return
        if not "marker" in kwargs:
            kwargs["marker"] = "+"
        kwargs["linestyle"] = ""
        return self.plot(plottable=plottable, **kwargs)


class BAxeGraph(ABaxe):

    __slots__ = []

    projection = AxeProjection.GRAPH

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "Axes":
        if not self.mpl_axe is None:
            return self.mpl_axe

        ge = self.spec

        maxe = mfig.add_subplot(
            mgs[ge.coord],
            **self.kwargs,
        )

        self.mpl_axe = maxe

        return maxe

    def render(self, maxe) -> "AxesSubplot":
        def _plotGraph(self, G, pos=None, arrows=None, with_labels=True, **kwds):
            nx.draw_networkx(
                G, pos=pos, arrows=arrows, with_labels=with_labels, ax=self, **kwds
            )

        maxe.plotGraph = types.MethodType(_plotGraph, maxe)

        super().render(maxe)

        maxe.grid(False)

        return maxe


class BAxeRectilinear(ABaxe):

    __slots__ = []

    projection = AxeProjection.RECTILINEAR

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "Axes":
        if not self.mpl_axe is None:
            return self.mpl_axe

        ge = self.spec

        maxe = mfig.add_subplot(
            mgs[ge.coord],
            **self.kwargs,
        )

        self.mpl_axe = maxe

        self._solveSharedBAxes(mfig, mgs, maxe)

        return maxe


class BAxeSemiLogX(BAxeRectilinear):

    __slots__ = []

    projection = AxeProjection.LOGX

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "Axes":
        maxe = super().createMplAxe(mfig, mgs)
        maxe.set_xscale("log", nonpositive="mask")

        return maxe


class BAxeSemiLogY(BAxeRectilinear):

    __slots__ = []

    projection = AxeProjection.LOGY

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "Axes":
        maxe = super().createMplAxe(mfig, mgs)
        maxe.set_yscale("log", nonpositive="mask")

        return maxe


class BAxeSemiLogXY(BAxeRectilinear):

    __slots__ = []

    projection = AxeProjection.LOGXY

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "Axes":
        maxe = super().createMplAxe(mfig, mgs)
        maxe.set_xscale("log", nonpositive="mask")
        maxe.set_yscale("log", nonpositive="mask")

        return maxe


class BAxePolar(ABaxe):

    __slots__ = []

    projection = AxeProjection.POLAR

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "Axes":
        if not self.mpl_axe is None:
            return self.mpl_axe

        ge = self.spec

        maxe = mfig.add_subplot(
            mgs[ge.coord],
            projection="polar",
            **self.kwargs,
        )

        self.mpl_axe = maxe

        self._solveSharedBAxes(mfig, mgs, maxe)

        return maxe


class BAxeNorthPolar(ABaxe):

    __slots__ = []

    projection = AxeProjection.NORTH_POLAR

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "Axes":
        if not self.mpl_axe is None:
            return self.mpl_axe

        ge = self.spec

        maxe = mfig.add_subplot(
            mgs[ge.coord],
            projection="polar",
            **self.kwargs,
        )
        maxe.set_theta_zero_location("N")
        maxe.set_theta_direction(-1)

        self.mpl_axe = maxe

        self._solveSharedBAxes(mfig, mgs, maxe)

        return maxe


class BAxePlateCarree(ABaxe):

    __slots__ = []

    projection = AxeProjection.PLATECARREE

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "Axes":
        if not self.mpl_axe is None:
            return self.mpl_axe

        ge = self.spec
        proj = ccrs.PlateCarree()

        maxe = mfig.add_subplot(
            mgs[ge.coord],
            projection=proj,
            **self.kwargs,
        )

        maxe.stock_img()
        maxe.gridlines(crs=proj, draw_labels=True)

        self.mpl_axe = maxe

        self._solveSharedBAxes(mfig, mgs, maxe)

        return maxe

    def plotDeviceReach(
        self, coord: tuple, elev_min: float, sat_alt: float, **kwargs
    ) -> APlottable:
        """Plots a line that represents the device reach

        See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html for the possible values in kwargs

        Args:
            coord: The position of the point, in longitude/latitude (rad)
            elev_min: Minimum elevation angle (rad)
            sat_alt: Satellite altitude, **assuming circular orbit** (m)
            kwargs: The plotting options for the object

        Returns:
            The created APlottable

        """

        g_lon, g_lat = coord

        # https://scitools.org.uk/cartopy/docs/v0.17/cartopy/geodesic.html#cartopy.geodesic.Geodesic.circle
        r = Req + sat_alt
        d_lim = sqrt(r**2 - Req**2 * cos(elev_min) ** 2) - Req * sin(elev_min)
        alpha_lim = np.arccos((Req**2 + r**2 - d_lim**2) / (2 * r * Req))
        rad = alpha_lim * Req

        g = Geodesic()
        val = g.circle(g_lon * 180 / pi, g_lat * 180 / pi, radius=rad)
        c_lon = np.array(val[:, 0])
        c_lat = np.array(val[:, 1])

        return self.plot(plottable=(c_lon * pi / 180, c_lat * pi / 180), **kwargs)


class BAxeDim3D(ABaxe):

    __slots__ = []

    projection = AxeProjection.DIM3D

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "Axes":
        if not self.mpl_axe is None:
            return self.mpl_axe

        ge = self.spec

        maxe = mfig.add_subplot(
            mgs[ge.coord],
            projection="3d",
            **self.kwargs,
        )

        self.mpl_axe = maxe

        self._solveSharedBAxes(mfig, mgs, maxe)

        return maxe


class BAxePanda3D(ABaxe):

    __slots__ = []

    projection = AxeProjection.PANDA3D

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "Axes":
        maxe = B3DPlotter()
        maxe.plotEarth()

        return maxe

    # def render(self, maxe: "Axes") -> "AxesSubplot":
    #     pass


class BAxeFactory(object):
    """Factory class that instanciates the adapted daughter class of `APlottable` to handle the object to plot"""

    __slots__ = []

    @classmethod
    def create(
        cls,
        figure: "blocksim.graphics.BFigure.ABFigure",
        title: str,
        spec: BGridElement,
        projection: AxeProjection,
        sharex: "blocksim.graphics.BAxe.ABaxe" = None,
        sharey: "blocksim.graphics.BAxe.ABaxe" = None,
        kwargs={},
    ) -> ABaxe:
        """Creates the adapted daughter class of `ABaxe`

        Args:
            figure: parent BFigure
            title: title of the BAxe
            spec : coordinates of the BAxe in the BFigure's layout
            projection: projection of the BAxe. Used to determine which subclass of ABaxe to use
            sharex: ABaxe to share X limits with
            sharey: ABaxe to share Y limits with
            kwargs: The plotting options for the object

        Returns:
            The ABaxe instance suited to the projection

        """

        if projection == AxeProjection.DIM3D:
            baxe = BAxeDim3D(
                figure,
                title,
                spec,
                sharex,
                sharey,
                kwargs,
            )
        elif projection == AxeProjection.NORTH_POLAR:
            baxe = BAxeNorthPolar(
                figure,
                title,
                spec,
                sharex,
                sharey,
                kwargs,
            )
        elif projection == AxeProjection.PLATECARREE:
            baxe = BAxePlateCarree(
                figure,
                title,
                spec,
                sharex,
                sharey,
                kwargs,
            )
        elif projection == AxeProjection.POLAR:
            baxe = BAxePolar(
                figure,
                title,
                spec,
                sharex,
                sharey,
                kwargs,
            )
        elif projection == AxeProjection.RECTILINEAR:
            baxe = BAxeRectilinear(
                figure,
                title,
                spec,
                sharex,
                sharey,
                kwargs,
            )
        elif projection == AxeProjection.LOGX:
            baxe = BAxeSemiLogX(
                figure,
                title,
                spec,
                sharex,
                sharey,
                kwargs,
            )
        elif projection == AxeProjection.LOGY:
            baxe = BAxeSemiLogY(
                figure,
                title,
                spec,
                sharex,
                sharey,
                kwargs,
            )
        elif projection == AxeProjection.LOGXY:
            baxe = BAxeSemiLogXY(
                figure,
                title,
                spec,
                sharex,
                sharey,
                kwargs,
            )
        elif projection == AxeProjection.GRAPH:
            baxe = BAxeGraph(
                figure,
                title,
                spec,
                sharex,
                sharey,
                kwargs,
            )
        elif projection == AxeProjection.PANDA3D:
            baxe = BAxePanda3D(
                figure,
                title,
                spec,
                sharex,
                sharey,
                kwargs,
            )

        return baxe
