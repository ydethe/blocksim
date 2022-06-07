import numpy as np
from numpy import pi
import cartopy.crs as ccrs
from matplotlib import pyplot as plt

from blocksim.dsp.DSPLine import DSPLine
from blocksim.graphics.B3DPlotter import B3DPlotter


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


class LineFactory(object):

    __slots__ = ["line", "kwargs"]

    def __init__(self, line, kwargs: dict) -> None:
        self.line = line
        self.kwargs = kwargs


class AxeFactory(object):

    __slots__ = ["title", "spec", "projection", "kwargs", "line_factories"]

    @classmethod
    def create(
        cls,
        title: str,
        spec: GridElementFactory,
        projection: str = "rectilinear",
        **kwargs,
    ) -> "AxeFactory":
        """
        Args:
            title: Title of the axe
            spec: Position in the GridSpec
            projection: Can be 'polar', 'north_polar', 'rectilinear', or 'PlateCarree'

        """
        gs = spec.get_gridspec()
        if gs.fig.projection == "earth3d" and (gs.ncol != 1 or gs.nrow != 1):
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
        self.line_factories = []

    def registerLineFactory(self, line_factory: LineFactory):
        self.line_factories.append(line_factory)

    def plot(self, line, **kwargs) -> LineFactory:
        if self.projection == "PlateCarree" and (
            not isinstance(line, tuple) or isinstance(line, DSPLine)
        ):
            raise AssertionError(
                f"With '{self.projection}' axe projection, only (lon,lat) data is accepted. Got {line}"
            )

        res = LineFactory(line, kwargs)
        self.registerLineFactory(res)
        return res


class FigureFactory(object):

    __slots__ = ["title", "grid_spec", "projection", "axe_factories"]

    @classmethod
    def create(cls, title: str, projection: str = "mpl") -> "FigureFactory":
        """
        Args:
            title: Title of the figure
            projection: Projection to use. Can be 'mpl' or 'earth3d'

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

    for line in axe.line_factories:
        traj = line.line
        app.plotTrajectory(traj)

    return app


def _render_mpl(fig: FigureFactory) -> "Figure":
    mfig = plt.figure()
    mfig.suptitle(fig.title)

    gs = fig.grid_spec
    mgs = mfig.add_gridspec(gs.nrow, gs.ncol)

    for axe in fig.axe_factories:
        maxe = None
        for line in axe.line_factories:
            if maxe is None:
                ge = axe.spec
                if axe.projection == "PlateCarree":
                    proj = ccrs.PlateCarree()
                elif "polar" in axe.projection:
                    proj = "polar"
                else:
                    proj = axe.projection

                maxe = mfig.add_subplot(mgs[ge.coord], projection=proj, **axe.kwargs)
                maxe.set_title(axe.title)

                if axe.projection == "PlateCarree":
                    maxe.stock_img()
                    maxe.gridlines(crs=proj, draw_labels=True)
                elif axe.projection == "north_polar":
                    maxe.set_theta_zero_location("N")
                    maxe.set_theta_direction(-1)
                    maxe.grid(True)
                else:
                    maxe.grid(True)

            if axe.projection == "PlateCarree":
                xd, yd = line.line
                xd = np.array(xd) * 180 / pi
                yd = np.array(yd) * 180 / pi
            else:
                mline = line.line
                if isinstance(mline, DSPLine):
                    xd = mline.generateXSerie()
                    yd = mline.default_transform(mline.y_serie)
                elif isinstance(mline, tuple):
                    xd, yd = mline
                else:
                    yd = np.array(mline)
                    ns = len(yd)
                    xd = np.arange(ns)

            maxe.plot(xd, yd, **line.kwargs)

    return mfig


def render(fig: FigureFactory):
    if fig.projection == "mpl":
        fig = _render_mpl(fig)
    else:
        app = _render_earth3d(fig)


def showFigures():
    plt.show()
