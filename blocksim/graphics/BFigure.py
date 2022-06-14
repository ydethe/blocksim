from matplotlib.figure import Figure
from singleton3 import Singleton
import numpy as np
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from .B3DPlotter import B3DPlotter
from .GraphicSpec import AxeProjection, FigureProjection
from .BLayout import BGridSpec, BGridElement
from .BAxe import BAxeFactory, ABaxe


class ABFigure(metaclass=ABCMeta):
    """Class that describes the figure. Not yet a matplotlib figure

    Args:
        title: Title of the figure
        projection: The projection to use

    """

    __slots__ = ["title", "grid_spec", "axe_factories", "projection", "mpl_fig"]

    def __init__(self, title: str, projection: FigureProjection = FigureProjection.MPL):
        self.title = title
        self.grid_spec = None
        self.axe_factories = []
        self.mpl_fig = None
        self.projection = projection

    def add_baxe(
        self,
        title: str,
        spec: BGridElement,
        projection: AxeProjection = AxeProjection.RECTILINEAR,
        sharex: ABaxe = None,
        sharey: ABaxe = None,
        **kwargs,
    ) -> ABaxe:
        if spec.axe is None:
            """Creates a ABaxe"""
            axe = BAxeFactory.create(
                figure=self,
                title=title,
                spec=spec,
                projection=projection,
                sharex=sharex,
                sharey=sharey,
                kwargs=kwargs,
            )
            self.registerAxeFactory(axe)
            spec.axe = axe
        else:
            axe = spec.axe

        return axe

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

    def registerAxeFactory(self, baxe: ABaxe):
        """Registers a new ABaxe in the list of related ABaxe

        Args:
            baxe: The ABaxe to add

        """
        self.axe_factories.append(baxe)

    @abstractmethod
    def render(self, tight_layout: bool = False) -> "Figure":
        """Actually renders the figure with matplotlib

        Args:
            tight_layout: To activate tight_layout in matplotlib

        Returns:
            The matplotlib figure

        """
        pass


class MplFigure(ABFigure):
    def render(self, tight_layout: bool = False) -> "Figure":
        if not self.mpl_fig is None:
            return self.mpl_fig

        mfig = plt.figure()
        mfig.suptitle(self.title)

        gs = self.grid_spec
        mgs = mfig.add_gridspec(gs.nrow, gs.ncol)

        for axe in self.axe_factories:
            display_legend = False
            maxe = axe.render(mfig, mgs)

            if len(axe.plottable_factories) == 0:
                continue

            rendered_plottables = []

            global_xmin = np.nan
            global_xmax = np.nan
            global_ymin = np.nan
            global_ymax = np.nan

            for plottable in axe.plottable_factories:
                info = plottable.render(axe)

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

            x_label = y_label = ""
            x_unit = y_unit = "-"
            for info in rendered_plottables:
                info = info["plottable"].scaleAndLabelData(info, amp_x, amp_y)

                x_labelp = info["x_label"]
                x_unitp = info["unit_of_x_var"]
                if x_unit == "-":
                    x_label = x_labelp
                    x_unit = x_unitp
                if x_unitp != "-":
                    if x_unit != x_unitp:
                        raise AssertionError(
                            f"Inconsistent X units: {x_label} with {x_labelp}"
                        )
                    x_label = x_labelp

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

                args = info["scaled_args"]
                kwargs = info["mpl_kwargs"]
                ret = info["plot_method"](*args, **kwargs)
                if "label" in kwargs.keys():
                    display_legend = True
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

                xmin, xmax = axe.xbounds
                maxe.set_xlim(xmin, xmax)

                ymin, ymax = axe.ybounds
                if ymin is None and not np.isnan(global_ymin):
                    ymin = global_ymin / info["y_mult"]
                if ymax is None and not np.isnan(global_ymax):
                    ymax = global_ymax / info["y_mult"]
                maxe.set_ylim(ymin, ymax)

                if display_legend:
                    maxe.legend(loc="best")

        if tight_layout:
            mfig.tight_layout()

        self.mpl_fig = mfig

        return mfig


class B3DFigure(ABFigure):
    def render(self, tight_layout: bool = False) -> "Figure":
        if not self.mpl_fig is None:
            return self.mpl_fig

        if self.grid_spec.ncol != 1 or self.grid_spec.nrow != 1:
            raise AssertionError(
                f"With {FigureProjection.EARTH3D}, only (1,1) GridPsec are allowed. Got ({self.grid_spec.nrow},{self.grid_spec.ncol})"
            )

        app = B3DPlotter()
        app.plotEarth()

        self.mpl_fig = app
        axe = self.axe_factories[0]

        for plottable in axe.plottable_factories:
            info = plottable.render(axe)
            info = plottable.scaleAndLabelData(info, 1, 1)

            args = info["scaled_args"]
            kwargs = info["mpl_kwargs"]
            ret = info["plot_method"](*args, **kwargs)

        self.mpl_fig = app

        return app


class FigureFactory(object, metaclass=Singleton):  # type: ignore
    """Class to instanciate BFigures and keep track of all created figures."""

    __slots__ = ["figures"]

    def __init__(self):
        self.figures = []

    @classmethod
    def create(
        cls, title: str = "", projection: FigureProjection = FigureProjection.MPL
    ) -> ABFigure:
        """Creates a BFigure, and record it in the list of BFigures

        Args:
            title: Title of the figure
            projection: Projection to use

        Returns:
            The created BFigure

        """
        factory = cls()

        if projection == FigureProjection.MPL:
            res = MplFigure(title=title, projection=projection)
        elif projection == FigureProjection.EARTH3D:
            res = B3DFigure(title=title, projection=projection)

        factory.figures.append(res)

        return res


def showFigures(tight_layout: bool = False, show: bool = True):
    """Renders and shows all BFigure"""
    factory = FigureFactory()
    mfigs = []
    for f in factory.figures:
        mfig = f.render(tight_layout=tight_layout)
        mfigs.append(mfig)

    if show:
        plt.show()
    else:
        return mfigs
