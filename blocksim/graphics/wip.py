from abc import ABCMeta, abstractmethod
from enum import Enum
from typing import Callable, Tuple

from singleton3 import Singleton
import numpy as np
from numpy import pi
import cartopy.crs as ccrs
from matplotlib import pyplot as plt

from ..utils import find1dpeak
from ..dsp.DSPLine import DSPLine
from ..dsp.DSPSpectrogram import DSPSpectrogram
from .B3DPlotter import B3DPlotter
from . import getUnitAbbrev


class AxeProjection(Enum):
    #: For rectilinear plots (the most frequent use case)
    RECTILINEAR = 0
    #: For trigonometric polar plots
    POLAR = 1
    #: For north azimuthal plots
    NORTH_POLAR = 2
    #: For Mercator cartography
    PLATECARREE = 3
    #: For 3D plots
    DIM3D = 4


class FigureProjection(Enum):
    #: For matplotlib plots
    MPL = 0
    #: For panda3d 3d plots
    EARTH3D = 1


class BGridElement(object):
    """This class stores the position of an axe in a grid

    Args:
        gs: Grid that gives axes' positions in a BFigure
        coord: Positoin of the BGridElement in the grid

    """

    __slots__ = ["gs", "coord"]

    def __init__(self, gs: "BGridSpec", coord: slice):
        self.gs = gs
        self.coord = coord

    def get_gridspec(self) -> "BGridSpec":
        """Returns the BGridSpec associated with the BGridElement

        Returns:
            The BGridSpec associated with the BGridElement

        """
        return self.gs


class BGridSpec(object):
    """This class stores the layout of the axes in a BFigure

    Args:
        figure: The BFigure associated with the BGridSpec
        nrow: Number of rows of the layout
        ncol: Number of columns of the layout

    """

    __slots__ = ["figure", "nrow", "ncol"]

    def __init__(self, figure: "BFigure", nrow: int, ncol: int):
        self.figure = figure
        self.nrow = nrow
        self.ncol = ncol

    def __getitem__(self, ind) -> BGridElement:
        ge = BGridElement(gs=self, coord=ind)
        return ge


class APlottable(metaclass=ABCMeta):
    """This base abstract class describes all the entity able to be plotted:

    * DSPLine
    * DSPSpectrogram
    * tuple of arrays
    * simple arrays

    Args:
        plottable: one of the instance above
        kwargs: The dictionary of options for plotting (color, width,etc)

    """

    __slots__ = ["plottable", "kwargs"]

    def __init__(self, plottable, kwargs: dict) -> None:
        self.plottable = plottable
        self.kwargs = kwargs

    @abstractmethod
    def _make_mline(self) -> Tuple["array", "array", str, str, str, str]:
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

    def render(self, axe: "BAxe") -> dict:
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

    def scaleAndLabelData(
        self, info: dict, amp_x: float, amp_y: float
    ) -> Tuple[dict, str, str]:
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
    """Specialisation of `APlottable` for `blocksim.dsp.DSPLine.DSPLine`"""

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
    """Specialisation of `APlottable` to handle simple numpy arrays"""

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
    """Specialisation of `APlottable` to handle a 2 elements tuple of numpy arrays"""

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
    """Specialisation of `APlottable` for `blocksim.dsp.DSPSpectrogram.DSPSpectrogram`"""

    __slots__ = []

    def _make_mline(self) -> Tuple["array", "array", str, str, str, str]:
        pass

    def render(self, axe: "BAxe") -> dict:
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
        Z = transform(mline.img)
        if fill == "plot_surface" and mline.projection == "polar":
            P, R = np.meshgrid(x_samp, y_samp)
            X, Y = R * np.cos(P), R * np.sin(P)
        else:
            X, Y = np.meshgrid(x_samp, y_samp)

        if axe.projection == AxeProjection.PLATECARREE:
            args = (X * 180 / pi, Y * 180 / pi, Z)
        else:
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
    """Factory class that instanciates the adapted daughter class of `APlottable` to handle the object to plot"""

    @staticmethod
    def create(mline, kwargs: dict) -> APlottable:
        """Creates the adapted daughter class of `APlottable` to handle the object to plot

        Args:
            mline: Object to plot. Can be:

            * a `blocksim.dsp.DSPLine.DSPLine`
            * a `blocksim.dsp.DSPSpectrogram.DSPSpectrogram`
            * a 2 elements tuple of numpy arrays
            * a simple numpy arrays
            kwargs: The plotting options for the object

        Returns:
            The APlottable instance suited to the object

        """
        if isinstance(mline, DSPLine):
            ret = PlottableDSPLine(mline, kwargs)

        elif isinstance(mline, DSPSpectrogram):
            ret = PlottableDSPSpectrogram(mline, kwargs)

        elif isinstance(mline, tuple):
            ret = PlottableTuple(mline, kwargs)

        else:
            ret = PlottableArray(mline, kwargs)

        return ret


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
        "kwargs",
        "plottable_factories",
        "mpl_axe",
    ]

    def __init__(
        self,
        title: str,
        spec: BGridElement,
        projection: AxeProjection = AxeProjection.RECTILINEAR,
        **kwargs,
    ):
        gs = spec.get_gridspec()
        if gs.figure.projection == FigureProjection.EARTH3D and (
            gs.ncol != 1 or gs.nrow != 1
        ):
            raise AssertionError(
                f"Cannot use BGridSpec different from (1,1) with figure projection 'earth3d'. Got ({gs.nrow},{gs.ncol})"
            )

        self.figure = gs.figure
        self.title = title
        self.spec = spec
        self.projection = projection
        self.kwargs = kwargs
        self.mpl_axe = None

        gs.figure.registerAxeFactory(self)

        self.plottable_factories = []

    def registerPlottableFactory(self, plottable: APlottable):
        """Registers the APlottable in the list of objects handled by the axe

        Args:
            plottable: APlottable object

        """
        self.plottable_factories.append(plottable)

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
            if not isinstance(plottable, tuple) and not isinstance(
                plottable, DSPSpectrogram
            ):
                raise AssertionError(
                    f"With '{self.projection}' axe projection, only (lon,lat) or rectilinear DSPSpectrogram data are accepted. Got {plottable}"
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

    def createMplAxe(self, mfig: "Figure", mgs: "SubplotSpec") -> "AxesSubplot":
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


class BFigure(object):
    """Class that describes the figure. Not yet a matplotlib figure

    Args:
        title: Title of the figure
        projection: Projection to use

    """

    __slots__ = ["title", "grid_spec", "projection", "axe_factories"]

    def __init__(self, title: str, projection: FigureProjection):
        self.title = title
        self.grid_spec = None
        self.projection = projection
        self.axe_factories = []

    def add_gridspec(self, nrow: int, ncol: int) -> BGridSpec:
        """Defines the axes layout in the figure

        Args:
            nrow: Number of rows
            ncol: Nimber of columns

        Returns:
            The layout

        """
        res = BGridSpec(self, nrow, ncol)
        self.grid_spec = res
        return res

    def registerAxeFactory(self, baxe: BAxe):
        """Registers a new BAxe in the list of related BAxe

        Args:
            baxe: The BAxe to add

        """
        self.axe_factories.append(baxe)

    def render(self, tight_layout: bool = False) -> "Figure":
        """Actually renders the figure with matplotlib

        Returns:
            The matplotlib figure

        """
        if self.projection == FigureProjection.MPL:
            fig = _render_mpl(self, tight_layout=tight_layout)
        else:
            fig = _render_earth3d(self)

        return fig


class FigureFactory(object, metaclass=Singleton):  # type: ignore
    """Class to instanciate BFigures and keep track of all created figures."""

    __slots__ = ["figures"]

    def __init__(self):
        self.figures = []

    @classmethod
    def create(
        cls, title: str, projection: FigureProjection = FigureProjection.MPL
    ) -> BFigure:
        """Creates a BFigure, and record it in the list of BFigures

        Args:
            title: Title of the figure
            projection: Projection to use

        Returns:
            The created BFigure

        """
        factory = cls()

        res = BFigure(title=title, projection=projection)

        factory.figures.append(res)

        return res


def _render_earth3d(fig: BFigure) -> "B3DPlotter":
    app = B3DPlotter()
    app.plotEarth()

    axe = fig.axe_factories[0]

    for plottable in axe.plottable_factories:
        traj = plottable.plottable
        app.plotTrajectory(traj)

    return app


def _render_mpl(fig: BFigure, tight_layout: bool = False) -> "Figure":
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

        if (
            axe.projection == AxeProjection.RECTILINEAR
            or axe.projection == AxeProjection.DIM3D
        ):
            if x_label != "":
                maxe.set_xlabel(x_label)
            if y_label != "":
                maxe.set_ylabel(y_label)

    if tight_layout:
        mfig.tight_layout()

    return mfig


def showFigures(tight_layout: bool = False):
    """Renders and shows all BFigure"""
    factory = FigureFactory()
    for f in factory.figures:
        f.render(tight_layout=tight_layout)

    plt.show()
