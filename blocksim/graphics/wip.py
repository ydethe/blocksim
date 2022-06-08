from enum import Enum

import numpy as np
from numpy import pi
import cartopy.crs as ccrs
from matplotlib import pyplot as plt

from blocksim.dsp.DSPLine import DSPLine
from blocksim.graphics.B3DPlotter import B3DPlotter


class AxeProjection(Enum):
    RECTILINEAR = 0
    POLAR = 1
    NORTH_POLAR = 2
    PLATECARREE = 3


class FigureProjection(Enum):
    MPL = 0
    EARTH3D = 1


class GridElementFactory(object):

    __slots__ = ["gs", "coord"]

    def __init__(self, gs: "GridSpecFactory", coord: slice):
        self.gs = gs
        self.coord = coord

    def get_gridspec(self) -> "GridSpecFactory":
        return self.gs


class GridSpecFactory(object):

    __slots__ = ["fig", "nrow", "ncol"]

    def __init__(self, fig: "FigureFactory", nrow: int, ncol: int):
        self.fig = fig
        self.nrow = nrow
        self.ncol = ncol

    def __getitem__(self, ind) -> GridElementFactory:
        ge = GridElementFactory(gs=self, coord=ind)
        return ge


class PlottableFactory(object):

    __slots__ = ["plottable", "kwargs"]

    def __init__(self, plottable, kwargs: dict) -> None:
        self.plottable = plottable
        self.kwargs = kwargs


class AxeFactory(object):

    __slots__ = ["title", "spec", "projection", "kwargs", "plottable_factories"]

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
        if gs.fig.projection == FigureProjection.EARTH3D and (
            gs.ncol != 1 or gs.nrow != 1
        ):
            raise AssertionError(
                f"Cannot use GridSpec different from (1,1) with figure projection 'earth3d'. Got ({gs.nrow},{gs.ncol})"
            )

        res = cls()
        res.title = title
        res.spec = spec
        res.projection = projection
        res.kwargs = kwargs

        gs.fig.registerAxeFactory(res)

        return res

    def __init__(self) -> None:
        self.plottable_factories = []

    def registerPlottableFactory(self, line_factory: PlottableFactory):
        self.plottable_factories.append(line_factory)

    def plot(self, plottable, **kwargs) -> PlottableFactory:
        if self.projection == AxeProjection.PLATECARREE and (
            not isinstance(plottable, tuple) or isinstance(plottable, DSPLine)
        ):
            raise AssertionError(
                f"With '{self.projection}' axe projection, only (lon,lat) data is accepted. Got {plottable}"
            )

        res = PlottableFactory(plottable, kwargs)
        self.registerPlottableFactory(res)
        return res


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
        res = GridSpecFactory(self, nrow, ncol)
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
        maxe = None
        for plottable in axe.plottable_factories:
            if maxe is None:
                ge = axe.spec
                if axe.projection == AxeProjection.PLATECARREE:
                    proj = ccrs.PlateCarree()
                elif axe.projection == AxeProjection.POLAR:
                    proj = "polar"
                elif axe.projection == AxeProjection.NORTH_POLAR:
                    proj = "polar"
                else:
                    proj = "rectilinear"

                maxe = mfig.add_subplot(mgs[ge.coord], projection=proj, **axe.kwargs)
                maxe.set_title(axe.title)

                if axe.projection == AxeProjection.PLATECARREE:
                    maxe.stock_img()
                    maxe.gridlines(crs=proj, draw_labels=True)
                elif axe.projection == AxeProjection.NORTH_POLAR:
                    maxe.set_theta_zero_location("N")
                    maxe.set_theta_direction(-1)
                    maxe.grid(True)
                else:
                    maxe.grid(True)

            if axe.projection == AxeProjection.PLATECARREE:
                xd, yd = plottable.plottable
                xd = np.array(xd) * 180 / pi
                yd = np.array(yd) * 180 / pi
            else:
                mline = plottable.plottable
                if isinstance(mline, DSPLine):
                    xd = mline.generateXSerie()
                    yd = mline.default_transform(mline.y_serie)
                elif isinstance(mline, tuple):
                    xd, yd = mline
                else:
                    yd = np.array(mline)
                    ns = len(yd)
                    xd = np.arange(ns)

            maxe.plot(xd, yd, **plottable.kwargs)

    return mfig


def render(fig: FigureFactory):
    if fig.projection == FigureProjection.MPL:
        fig = _render_mpl(fig)
    else:
        fig = _render_earth3d(fig)

    return fig


def showFigures():
    plt.show()
