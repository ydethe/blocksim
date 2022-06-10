from typing import Any, List

from nptyping import NDArray
import numpy as np
from numpy import pi, sqrt, cos, sin
import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic

from ..dsp.DSPLine import DSPLine
from ..dsp.DSPSpectrogram import DSPSpectrogram
from ..constants import Req
from ..satellite.Trajectory import Trajectory
from .GraphicSpec import AxeProjection, FigureProjection
from .BLayout import BGridElement
from .Plottable import *


class BAxe(object):
    """Class that describes the axe. Not yet a matplotlib axe

    Args:
        title: Title of the axe
        spec: Position in the BGridSpec
        projection: Projection to use

    """

    __slots__ = [
        "figure",
        "title",
        "spec",
        "projection",
        "parent_sharex",
        "children_sharex",
        "parent_sharey",
        "children_sharey",
        "kwargs",
        "plottable_factories",
        "mpl_axe",
        "xbounds",
        "ybounds",
    ]

    def __init__(
        self,
        figure: "BFigure",
        title: str,
        spec: BGridElement,
        projection: AxeProjection = AxeProjection.RECTILINEAR,
        sharex: "BAxe" = None,
        sharey: "BAxe" = None,
        **kwargs,
    ):
        gs = spec.get_gridspec()
        if figure.projection == FigureProjection.EARTH3D and (
            gs.ncol != 1 or gs.nrow != 1
        ):
            raise AssertionError(
                f"Cannot use BGridSpec different from (1,1) with figure projection 'earth3d'. Got ({gs.nrow},{gs.ncol})"
            )

        self.figure = figure
        self.title = title
        self.spec = spec
        self.projection = projection
        self.parent_sharex = sharex
        self.children_sharex = []
        self.parent_sharey = sharey
        self.children_sharey = []
        self.kwargs = kwargs
        self.mpl_axe = None

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

    def _addChildSharex(self, sharex: "Baxe"):
        self.children_sharex.append(sharex)

    def _addChildSharey(self, sharey: "Baxe"):
        self.children_sharey.append(sharey)

    def _findRootSharex(self) -> "BAxe":
        if self.parent_sharex is None:
            return self
        else:
            return self.parent_sharex._findRootSharex()

    def _findRootSharey(self) -> "BAxe":
        if self.parent_sharey is None:
            return self
        else:
            return self.parent_sharey._findRootSharey()

    def registerPlottableFactory(self, plottable: APlottable):
        """Registers the APlottable in the list of objects handled by the axe

        Args:
            plottable: APlottable object

        """
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

            * a `blocksim.dsp.DSPLine.DSPLine`
            * a `blocksim.dsp.DSPSpectrogram.DSPSpectrogram`
            * a 2 elements tuple of numpy arrays
            * a simple numpy arrays
            kwargs: The plotting options for the object

        """
        if self.projection == AxeProjection.PLATECARREE:
            if (
                not isinstance(plottable, tuple)
                and not isinstance(plottable, DSPSpectrogram)
                and not isinstance(plottable, Trajectory)
            ):
                raise AssertionError(
                    f"With '{self.projection}' axe projection, only (lon,lat), Trajectory or rectilinear DSPSpectrogram data are accepted. Got {plottable}"
                )
            elif isinstance(plottable, DSPSpectrogram):
                if not plottable.projection == "rectilinear":
                    raise AssertionError(
                        f"With '{self.projection}' axe projection and DSPSpectrogram data, only the rectilinear projection is accepted. Got {plottable.projection}"
                    )

        if (
            self.projection == AxeProjection.POLAR
            or self.projection == AxeProjection.NORTH_POLAR
        ) and (isinstance(plottable, DSPLine) or isinstance(plottable, DSPSpectrogram)):
            if plottable.projection != "polar":
                raise AssertionError(
                    f"With '{self.projection}' axe projection, only polar projecions are allowed. Got {plottable.projection}"
                )
        if self.projection == AxeProjection.DIM3D and not isinstance(
            plottable, DSPSpectrogram
        ):
            raise AssertionError(
                f"With '{self.projection}' axe projection, only DSPSpectrogram is accepted. Got {plottable}"
            )

        res = PlottableFactory.create(plottable, kwargs)
        self.registerPlottableFactory(res)
        return res

    def plotDeviceReach(
        self, coord: tuple, elev_min: float, sat_alt: float, **kwargs
    ) -> PlottableTuple:
        """Plots a line that represents the device reach

        See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html for the possible values in kwargs

        Args:
            coord: The position of the point, in longitude/latitude (rad)
            elev_min: Minimum elevation angle (rad)
            sat_alt: Satellite altitude, **assuming circular orbit** (m)
            kwargs: The plotting options for the object

        Returns:
            The created PlottableTuple

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

    def render(self, mfig: "Figure", mgs: "SubplotSpec") -> "AxesSubplot":
        """Creates a matplotlib axe according to the BAxe.plot commands

        Args:
            mfig: Matplotlib figure
            mgs: Matplotlib grid_spec

        Returns:
            The created matplotlib axe

        """
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

        if not self.parent_sharex is None:
            sharex = self.parent_sharex.render(mfig, mgs)
        else:
            sharex = None

        if not self.parent_sharey is None:
            sharey = self.parent_sharey.render(mfig, mgs)
        else:
            sharey = None

        maxe = mfig.add_subplot(
            mgs[ge.coord],
            projection=proj,
            sharex=sharex,
            sharey=sharey,
            **self.kwargs,
        )

        if self.title != "":
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