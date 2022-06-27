"""Advanced plotting functions.
Allows plotting from a `blocksim.loggers.Logger.Logger`, or from `blocksim.dsp.DSPLine.DSPRectilinearLine`
3D plot around Earth are also possible

"""
import os
from typing import Tuple

import numpy as np
from numpy import log10

from matplotlib import pyplot as plt

from .. import logger
from ..loggers.Logger import Logger
from ..dsp import phase_unfold


def getUnitAbbrev(
    samp: float, unit: str, force_mult: int = None
) -> Tuple[float, float, str, str]:
    """Given a scale factor, gives the prefix for the unit to display

    Args:
        samp: Sample
        unit: Physical unit
        force_mult: Multiplier to use

    Returns:
        scaled_samp: Scaled sample
        mult: Division coefficient of samp
        lbl: Scale factor label
        unit: Unit to display

    Example:
        >>> getUnitAbbrev(0.1, 's')
        (100.0, 0.001, 'm', 's')
        >>> getUnitAbbrev(13.6, 's')
        (13.6, 1, '', 's')
        >>> getUnitAbbrev(76, 's') # doctest: +ELLIPSIS
        (1.266..., 60, '', 'min')
        >>> getUnitAbbrev(1.5e-3, 'm')
        (1.5, 0.001, 'm', 'm')
        >>> getUnitAbbrev(1.5e-3, 's')
        (1.5, 0.001, 'm', 's')
        >>> getUnitAbbrev(90, 's')
        (1.5, 60, '', 'min')

    """
    d = {
        1: "",
        1000: "k",
        1e6: "M",
        1e9: "G",
        1e12: "T",
        1e-3: "m",
        1e-6: "µ",
        1e-9: "n",
        1e-12: "p",
        1e-15: "f",
        1e-18: "a",
        1e-21: "z",
        1e-24: "y",
    }
    d_time = {
        1: "s",
        60: "min",
        3600: "h",
        86400: "day",
        86400 * 30: "month",
        860400 * 30 * 12: "yr",
    }
    if unit == "s" and samp >= 1:
        if force_mult is None:
            for mult in reversed(d_time.keys()):
                if samp / mult >= 1:
                    break
        else:
            mult = force_mult
        unit = d_time[mult]
        lbl = ""
    else:
        if force_mult is None:
            if samp == 0:
                samp = 1
            xm = np.abs(samp)
            pm = (int(log10(xm)) // 3) * 3
            mult = 10**pm
        else:
            mult = force_mult
        lbl = d[mult]

    if unit == "":
        unit = "-"

    return samp / mult, mult, lbl, unit


def format_parameter(samp: float, unit: str) -> str:
    """Given a scalar value and a unit, returns the txt to display
    with appropriate unit and muyliplier

    Args:
        samp: The scalar value
        unit: The associated unit

    Returns:
        str: The text to display

    Examples:
        >>> format_parameter(1.5e-3, 'm')
        '1.5 mm'
        >>> format_parameter(1.5e-3, 's')
        '1.5 ms'
        >>> format_parameter(90, 's')
        '1.5 min'

    """
    scaled_samp, mult, lbl, unit = getUnitAbbrev(samp, unit)
    txt = "%.3g %s%s" % (scaled_samp, lbl, unit)
    return txt


def createFigureFromSpec(
    spec: "blocksim.graphics.GraphicSpec.FigureSpec",
    log: Logger,
    fig: "blocksim.graphics.BFigure.MplFigure" = None,
) -> "blocksim.graphics.BFigure.MplFigure":
    """Parses a FigureSpec to build a matplotlib figure, and returns it

    Args:
        spec: A FigureSpec instance
        log: The Logger to read into
        fig: A ABFigure. If None, the function creates ones

    Returns:
        The matplotlib figure

    """
    from .GraphicSpec import AxeProjection
    from .BFigure import FigureFactory

    n = len(spec.axes)

    if fig is None:
        title = spec.props.get("title", "")
        fig = FigureFactory.create(title=title)

    nrow = spec.props.get("nrow", 1)
    ncol = spec.props.get("ncol", 1)

    gs = fig.add_gridspec(nrow, ncol)

    l_axes = []

    for k in range(n):
        coord = spec.axes[k].props["coord"]
        shx = spec.axes[k].props["sharex"]
        if shx is None:
            sharex = None
        else:
            sharex = l_axes[shx]
        title = spec.axes[k].props["title"]
        sproj = spec.axes[k].props.pop("projection", "rectilinear")
        if sproj == "rectilinear":
            proj = AxeProjection.RECTILINEAR
        elif sproj == "logx":
            proj = AxeProjection.LOGX
        elif sproj == "logy":
            proj = AxeProjection.LOGY
        elif sproj == "logxy":
            proj = AxeProjection.LOGXY
        elif sproj == "polar":
            proj = AxeProjection.POLAR
        elif sproj == "north_polar":
            proj = AxeProjection.NORTH_POLAR
        elif sproj == "map":
            proj = AxeProjection.PLATECARREE
        else:
            raise AssertionError(f"Projection not supported '{sproj}'")

        axe = fig.add_baxe(title=title, spec=gs[coord], projection=proj, sharex=sharex)
        l_axes.append(axe)

        for d in spec.axes[k].lines:
            lp = d.copy()
            lp.pop("varx", None)
            lp.pop("vary", None)

            varx = d["varx"]
            vary = d["vary"]
            if log is None:
                plottable = (varx, vary)
            else:
                plottable = (log, varx, vary)
            if "label" in d.keys():
                line = axe.plot(plottable, **lp)
            elif type(vary) == type(""):
                line = axe.plot(plottable, label=vary, **lp)
            else:
                line = axe.plot(plottable, **lp)

    return fig


def plotVerif(
    log: Logger, fig_title: str, *axes
) -> "blocksim.graphics.BFigure.MplFigure":
    """Plots a set of axes and curves on a single figure

    Args:
        log: Logger which contains the simulated values
        fig_title: Title of the figure
        axes: List of lists of dicts
            Each list is the description of an axe, each dict the description of a line.
            Each dict has a key "var", which is the name of a variable contained in *log*.
            The other keys are keyword arguments for the plot method of matplotlib

    Returns:
        The resulting figure

    """
    from .GraphicSpec import AxeSpec, FigureSpec

    l_aspec = []
    for ind, l_lines in enumerate(axes):
        aProp = dict()

        aProp["title"] = "Axe %i" % (ind + 1)
        aProp["coord"] = (ind, 0)
        aProp["sharex"] = ind - 1 if ind > 0 else None

        lSpec = []
        for l in l_lines:
            if "title" in l.keys():
                aProp["title"] = l.pop("title", "Axe %i" % (ind + 1))
                aProp["sharex"] = l.pop("sharex", None)
                aProp["coord"] = l["coord"], 0
            else:
                l["vary"] = l.pop("var")
                l["varx"] = "t"
                lSpec.append(l)

        aSpec = AxeSpec(aProp, lSpec)

        l_aspec.append(aSpec)

    spec = FigureSpec({"title": fig_title, "nrow": len(axes), "ncol": 1}, axes=l_aspec)
    fig = createFigureFromSpec(spec, log)

    return fig


def quickPlot(*args, **kwargs) -> "blocksim.graphics.BFigure.MplFigure":
    """Quickly plots data

    Args:
        args: List of plottables, handled by `blocksim.graphics.Plottable.PlottableFactory`
        kwargs: Plotting options

    Returns:
        The axe used to plot

    """
    from .GraphicSpec import AxeProjection
    from .BFigure import FigureFactory

    axe = kwargs.pop("axe", None)
    title = kwargs.pop("title", "")
    if axe is None:
        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        proj = kwargs.pop("projection", AxeProjection.RECTILINEAR)
        axe = fig.add_baxe(title=title, spec=gs[0, 0], projection=proj)

    for a in args:
        axe.plot(a, **kwargs)

    return axe


def showFigures(
    save_fig: bool = False,
    tight_layout: bool = False,
    one_by_one: bool = False,
    show: bool = True,
    save_dir: str = "",
):
    from .BFigure import FigureFactory

    """Renders and shows all BFigure"""
    factory = FigureFactory()
    mfig = None
    for f in factory.figures:
        mfig = f.render(tight_layout=tight_layout)
        if save_fig:
            # PRN_G18,_T0:_2022-06-20_00:00:00.104813+00:00_UTC
            fic = f.title.replace(" ", "_").replace(",", "").replace(":", "_") + ".png"
            mfig.savefig(os.path.join(save_dir, fic))
        if one_by_one and show:
            plt.show()

    if not one_by_one and show:
        plt.show()

    return mfig
