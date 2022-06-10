"""Advanced plotting functions.
Allows plotting from a `blocksim.loggers.Logger.Logger`, or from `blocksim.dsp.DSPLine.DSPLine`
3D plot around Earth are also possible

"""

from typing import Tuple

import numpy as np
from numpy import log10

from matplotlib import pyplot as plt


from .. import logger
from ..loggers.Logger import Logger
from ..dsp.DSPFilter import ADSPFilter
from ..dsp.DSPLine import DSPLine
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


def plotFromLogger(
    log: Logger, id_x: str, id_y: str, axe: "BAxe", **kwargs
) -> "APlottable":
    """Plots a value on a matplotlib axe

    Args:
        log: The Logger to read into
        id_x: Name or expression for the X axis
        id_y: Name or expression for the Y axis
        axe: The BAxe to draw on. Obtained by fig.add_baxe
        kwargs: matplotlib plotting options for the 'plot' method.

    Returns:
        The created APlottable

    """
    if type(id_x) == type(""):
        val_x = log.getValue(id_x)
        name_x = id_x
        if id_x in log.getParametersName():
            p = log.getParameter(id_x)
            unit_x = p.unit
        else:
            unit_x = ""
    elif hasattr(id_x, "__iter__"):
        val_x = id_x
        name_x = ""
        unit_x = ""
    else:
        raise SystemError("[ERROR]Unacceptable argument for id_x : %s" % (str(id_x)))

    if type(id_y) == type(""):
        val_y = log.getValue(id_y)
        name_y = id_y
        if id_y in log.getParametersName():
            p = log.getParameter(id_y)
            unit_y = p.unit
        else:
            unit_y = ""
    elif hasattr(id_y, "__iter__"):
        val_y = id_y
        name_y = ""
        unit_y = ""
    else:
        raise SystemError("[ERROR]Unacceptable argument for id_y : %s" % (str(id_y)))

    if not "label" in kwargs.keys():
        kwargs["label"] = id_y

    line = axe.plot(
        (
            {"data": val_x, "name": name_x, "unit": unit_x},
            {"data": val_y, "name": name_y, "unit": unit_y},
        ),
        **kwargs,
    )

    return line


def createFigureFromSpec(spec: "FigureSpec", log: Logger, fig=None) -> "BFigure":
    """Parses a FigureSpec to build a matplotlib figure, and returns it

    Args:
        spec: A FigureSpec instance
        log: The Logger to read into
        fig: A BFigure. If None, the function creates ones

    Returns:
        The matplotlib figure

    """
    from .enums import AxeProjection
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

        # spec.axes[k].props["_axe"] = axe

        for d in spec.axes[k].lines:
            lp = d.copy()
            lp.pop("varx", None)
            lp.pop("vary", None)
            # lp.pop("_line", None)
            # lp.pop("_xdata", None)
            # lp.pop("_ydata", None)

            varx = d["varx"]
            vary = d["vary"]
            if "label" in d.keys():
                line = plotFromLogger(log, varx, vary, axe=axe, **lp)
            elif type(vary) == type(""):
                line = plotFromLogger(log, varx, vary, axe=axe, label=vary, **lp)
            else:
                line = plotFromLogger(log, varx, vary, axe=axe, **lp)

            # line = axe.plottable_factories[-1]
            # d["_line"] = line
            # xdata, ydata = line.get_data()
            # d["_xdata"] = xdata
            # d["_ydata"] = ydata

    return fig


def plotBode(
    filt: ADSPFilter,
    axe_amp: "BAxe",
    axe_pha: "BAxe",
    fpoints: int = 200,
    pow_lim: float = -100.0,
    **kwargs,
) -> Tuple["PlottableDictTuple", "PlottableDictTuple"]:
    """Plots the bode diagram of a filter

    Args:
        filt: Filter to analyse
        axe_amp: The BAxe amplitude axis to draw on. Obtained by fig.add_baxe
        axe_pha: The BAxe phase axis to draw on. Obtained by fig.add_baxe
        fpoints: If int, number of frequency samples to use for the plot
            If iterable, list of frequency samples to use for the plot
        kwargs: Plotting options.

    Examples:
        >>> from blocksim.graphics.BFigure import FigureFactory
        >>> from blocksim.dsp.DSPFilter import ArbitraryDSPFilter
        >>> f = ArbitraryDSPFilter(name="MTI", samplingPeriod=1e6, num=[1, -1])
        >>> fig = fig = FigureFactory.create()
        >>> gs = fig.add_gridspec(2, 1)
        >>> axe_amp=fig.add_baxe('Amplitude', spec=gs[0,0])
        >>> axe_pha=fig.add_baxe('Amplitude', spec=gs[1,0])
        >>> _ = plotBode(f, axe_amp=axe_amp,axe_pha=axe_pha)

    """
    from scipy.signal import TransferFunction, freqz

    fs = 1 / filt.samplingPeriod

    b, a = filt.generateCoefficients()

    if hasattr(fpoints, "__iter__"):
        freq = fpoints
    else:
        freq = np.arange(0, fs / 2, fs / 2 / fpoints)

    num, den = TransferFunction._z_to_zinv(b, a)
    _, y = freqz(num, den, worN=freq, fs=fs)

    line_amp = axe_amp.plot(
        plottable=(
            {"data": freq, "unit": "Hz", "name": "Frequency"},
            {
                "data": DSPLine.to_db(y, lim_db=pow_lim),
                "name": "Amplitude",
                "unit": "dB",
            },
        ),
        **kwargs,
    )

    pha = phase_unfold(y)
    line_pha = axe_pha.plot(
        plottable=(
            {"data": freq, "unit": "Hz", "name": "Frequency"},
            {
                "data": 180 / np.pi * pha,
                "name": "Phase",
                "unit": "deg",
            },
        )
    )

    return line_amp, line_pha


def plotVerif(log: Logger, fig_title: str, *axes) -> "BFigure":
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


def quickPlot(*args, **kwargs) -> "BFigure":
    """Quickly plots data

    Args:
        args: List of plottables, handled by `blocksim.graphics.Plottable.PlottableFactory`
        kwargs: Plotting options

    Returns:
        The axe used to plot

    """
    from .enums import AxeProjection
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


def showFigures(tight_layout: bool = False):
    from .BFigure import FigureFactory

    """Renders and shows all BFigure"""
    factory = FigureFactory()
    for f in factory.figures:
        f.render(tight_layout=tight_layout)

    plt.show()
