from matplotlib.figure import Figure
from singleton3 import Singleton
import numpy as np
from matplotlib import pyplot as plt
from abc import ABCMeta, abstractmethod

from .GraphicSpec import AxeProjection, FigureProjection
from .BLayout import BGridSpec, BGridElement
from .BAxe import BAxeFactory, ABaxe


class ABFigure(metaclass=ABCMeta):
    """Class that describes the figure. Not yet a matplotlib figure

    Args:
        title: Title of the figure
        projection: The projection to use

    """

    __slots__ = ["title", "grid_spec", "axe_factories", "mpl_fig"]

    projection = None

    def __init__(self, title: str):
        self.title = title
        self.grid_spec = None
        self.axe_factories = []
        self.mpl_fig = None

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

    projection = FigureProjection.MPL

    def render(self, tight_layout: bool = False) -> "Figure":
        if not self.mpl_fig is None:
            return self.mpl_fig

        mfig = plt.figure()
        self.mpl_fig = mfig
        mfig.suptitle(self.title)

        gs = self.grid_spec
        mgs = mfig.add_gridspec(gs.nrow, gs.ncol)

        for axe in self.axe_factories:
            axe.render(self, mgs)

        if tight_layout:
            mfig.tight_layout()

        return mfig


class B3DFigure(ABFigure):

    projection = FigureProjection.EARTH3D

    def add_gridspec(self, nrow: int, ncol: int) -> BGridSpec:
        if ncol != 1 or nrow != 1:
            raise AssertionError(
                f"With {FigureProjection.EARTH3D}, only (1,1) GridPsec are allowed. Got ({nrow},{ncol})"
            )
        return super().add_gridspec(nrow=nrow, ncol=ncol)

    def registerAxeFactory(self, baxe: ABaxe):
        if len(self.axe_factories) == 1:
            raise AssertionError(f"Only one axe allowed in {FigureProjection.EARTH3D}")
        self.axe_factories.append(baxe)

    def render(self, tight_layout: bool = False) -> "Figure":
        if not self.mpl_fig is None:
            return self.mpl_fig

        # mfig = B3DPlotter()
        # mfig.plotEarth()
        mfig = None

        self.mpl_fig = mfig

        for axe in self.axe_factories:
            maxe = axe.render(mfig, self.grid_spec)

        # for plottable in axe.plottable_factories:
        #     info = plottable.render(axe)
        #     info = plottable.scaleAndLabelData(info, 1, 1)

        #     args = info["scaled_args"]
        #     kwargs = info["mpl_kwargs"]
        #     ret = info["plot_method"](*args, **kwargs)

        return mfig


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
            res = MplFigure(title=title)
        elif projection == FigureProjection.EARTH3D:
            res = B3DFigure(title=title)

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
