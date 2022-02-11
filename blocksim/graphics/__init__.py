from typing import Iterable, Any

from parse import compile
import numpy as np
from scipy.special import erfc
from scipy.interpolate import interp2d
from numpy import sqrt, log10
from numpy.polynomial import Polynomial
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage
from matplotlib.backend_bases import Event

from .. import logger
from ..Logger import Logger
from ..dsp.DSPFilter import ADSPFilter
from ..dsp.DSPLine import DSPLine
from ..dsp.DSPSpectrogram import DSPSpectrogram
from ..dsp import phase_unfold
from .AxeSpec import AxeSpec
from .FigureSpec import FigureSpec


def getUnitAbbrev(mult: float) -> str:
    """Given a scale factor, gives the prefix for the unit to display

    Args:
      mult
        Scale factor

    Returns:
      Prefix

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
    return d[mult]


def format_parameter(samp: float, unit: str) -> str:
    xm = np.abs(samp)
    pm = (int(log10(xm)) // 3) * 3
    x_unit_mult = 10**pm
    x_unit_lbl = getUnitAbbrev(x_unit_mult)
    txt = "%.3g %s%s" % (samp / x_unit_mult, x_unit_lbl, unit)
    return txt


def plotFromLogger(
    log: Logger, id_x: str, id_y: str, axe: "AxesSubplot", **kwargs
) -> "Line2D":
    """Plots a value on a matplotlib axe

    Args:
      log
        :class:`blocksim.Logger.Logger` instance
      id_x
        Name or expression for the X axis
      id_y
        Name or expression for the Y axis
      axe
        The axis to draw on
      kwargs
        matplotlib plotting options for the 'plot' method

    Returns:
      The lines drawn by matplotlib

    """
    if type(id_x) == type(""):
        val_x = log.getValue(id_x)
    elif hasattr(id_x, "__iter__"):
        val_x = id_x
    else:
        raise SystemError("[ERROR]Unacceptable argument for id_x : %s" % (str(id_x)))

    if type(id_y) == type(""):
        val_y = log.getValue(id_y)
    elif hasattr(id_y, "__iter__"):
        val_y = id_y
    else:
        raise SystemError("[ERROR]Unacceptable argument for id_y : %s" % (str(id_y)))

    (line,) = axe.plot(val_x, val_y, **kwargs)

    return line


def createFigureFromSpec(spec: FigureSpec, log: Logger, fig=None) -> "Figure":
    """Parses a :class:`FigureSpec` to build a matplotlib figure, and returns it

    Args:
      spec
        A :class:`FigureSpec` instance
      log
        A :class:`blocksim.Logger.Logger` to read data from
      fig
        A matplotlib figure. If None, the function creates ones

    Returns:
      The matplotlib figure

    """
    n = len(spec.axes)

    if fig is None:
        fig = plt.figure()

    fig.suptitle = spec.props["title"]
    l_axes = []

    for k in range(1, n + 1):
        nrow = spec.axes[k - 1].props["nrow"]
        ncol = spec.axes[k - 1].props["ncol"]
        ind = spec.axes[k - 1].props["ind"]
        shx = spec.axes[k - 1].props["sharex"]
        title = spec.axes[k - 1].props["title"]

        if shx is None:
            axe = fig.add_subplot(nrow, ncol, ind)
            axe.grid(True)
        else:
            axe = fig.add_subplot(nrow, ncol, ind, sharex=l_axes[shx - 1])
            axe.grid(True)
        l_axes.append(axe)

        spec.axes[k - 1].props["_axe"] = axe

        if not title is None:
            axe.set_title(title)

        disp_leg = False
        for d in spec.axes[k - 1].lines:
            lp = d.copy()
            lp.pop("varx", None)
            lp.pop("vary", None)
            lp.pop("_line", None)
            lp.pop("_xdata", None)
            lp.pop("_ydata", None)

            varx = d["varx"]
            vary = d["vary"]
            if "label" in d.keys():
                line = plotFromLogger(log, varx, vary, axe, **lp)
                disp_leg = True
            elif type(vary) == type(""):
                line = plotFromLogger(log, varx, vary, axe, label=vary, **lp)
                disp_leg = True
            else:
                line = plotFromLogger(log, varx, vary, axe, **lp)

            d["_line"] = line
            xdata, ydata = line.get_data()
            d["_xdata"] = xdata
            d["_ydata"] = ydata

        if disp_leg:
            axe.legend()

    fig.tight_layout()

    return fig


def plot3DSpectrogram(spg: DSPSpectrogram, axe: "AxesSubplot", **kwargs) -> AxesImage:
    """Plots a 3D surface with the following refinements :

    * a callable *transform* is applied to all samples
    * the label of the plot is the name given at instanciation

    Args:
      spg
        Spectrogram to plot
      axe
        Matplotlib axe to draw on
      kwargs
        Plotting options. The following extra keys are allowed:
        * transform for a different transform from the one given at instanciation
        * find_peaks to search peaks
        * x_unit_mult to have a more readable unit prefix

    Returns:
      The matplotlib image generated

    """
    axe.grid(True)
    transform = kwargs.pop("transform", spg.default_transform)
    if "x_unit_mult" in kwargs.keys():
        x_unit_mult = kwargs.pop("x_unit_mult")
    else:
        x_samp = spg.generateXSerie()
        xm = np.max(np.abs(x_samp))
        pm = (int(log10(xm)) // 3) * 3
        x_unit_mult = 10**pm
    x_unit_lbl = getUnitAbbrev(x_unit_mult)

    if "y_unit_mult" in kwargs.keys():
        y_unit_mult = kwargs.pop("y_unit_mult")
    else:
        y_samp = spg.generateYSerie()
        ym = np.max(np.abs(y_samp))
        pm = (int(log10(ym)) // 3) * 3
        y_unit_mult = 10**pm
    y_unit_lbl = getUnitAbbrev(y_unit_mult)
    lbl = kwargs.pop("label", spg.name)

    X, Y = np.meshgrid(
        spg.generateXSerie() / x_unit_mult, spg.generateYSerie() / y_unit_mult
    )
    Z = transform(spg.img)

    ret = axe.plot_surface(X, Y, Z, label=lbl, **kwargs)
    axe.set_xlabel("%s (%s%s)" % (spg.name_of_x_var, x_unit_lbl, spg.unit_of_x_var))
    axe.set_ylabel("%s (%s%s)" % (spg.name_of_y_var, y_unit_lbl, spg.unit_of_y_var))

    axe.figure.colorbar(ret, ax=axe)

    return ret


def plotSpectrogram(spg: DSPSpectrogram, axe: "AxesSubplot", **kwargs) -> AxesImage:
    """Plots a line with the following refinements :

    * a callable *transform* is applied to all samples
    * the label of the plot is the name given at instanciation

    Args:
      spg
        Spectrogram to plot
      axe
        Matplotlib axe to draw on
      kwargs
        Plotting options. The following extra keys are allowed:
        * transform for a different transform from the one given at instanciation
        * find_peaks to search peaks
        * x_unit_mult to have a more readable unit prefix
        * search_fig (bool) to generate a search figure

    Returns:
      The matplotlib image generated

    """
    axe.grid(True)
    transform = kwargs.pop("transform", spg.default_transform)
    search_fig = kwargs.pop("search_fig", False)
    find_peaks = kwargs.pop("find_peaks", 0)

    if "x_unit_mult" in kwargs.keys():
        x_unit_mult = kwargs.pop("x_unit_mult")
    else:
        x_samp = spg.generateXSerie()
        xm = np.max(np.abs(x_samp))
        pm = (int(log10(xm)) // 3) * 3
        x_unit_mult = 10**pm
    x_unit_lbl = getUnitAbbrev(x_unit_mult)

    if "y_unit_mult" in kwargs.keys():
        y_unit_mult = kwargs.pop("y_unit_mult")
    else:
        y_samp = spg.generateYSerie()
        ym = np.max(np.abs(y_samp))
        pm = (int(log10(ym)) // 3) * 3
        y_unit_mult = 10**pm
    y_unit_lbl = getUnitAbbrev(y_unit_mult)
    lbl = kwargs.pop("label", spg.name)

    Z = transform(spg.img)

    ret = axe.imshow(
        Z,
        aspect="auto",
        extent=(
            spg.generateXSerie(0) / x_unit_mult,
            spg.generateXSerie(-1) / x_unit_mult,
            spg.generateYSerie(0) / y_unit_mult,
            spg.generateYSerie(-1) / y_unit_mult,
        ),
        origin="lower",
        label=lbl,
        **kwargs
    )
    # colorbar
    axe.figure.colorbar(ret, ax=axe)

    axe.set_xlabel("%s (%s%s)" % (spg.name_of_x_var, x_unit_lbl, spg.unit_of_x_var))
    axe.set_ylabel("%s (%s%s)" % (spg.name_of_y_var, y_unit_lbl, spg.unit_of_y_var))

    if find_peaks > 0:
        lpeaks = spg.findPeaksWithTransform(transform=transform, nb_peaks=find_peaks)
        for p in lpeaks:
            x, y = p.coord
            axe.plot(
                [x / x_unit_mult],
                [y / y_unit_mult],
                marker="o",
                color="red",
                linestyle="",
            )
            axe.annotate(
                "(%.1f,%.1f,%.1f)" % (x / x_unit_mult, y / y_unit_mult, p.value),
                xy=(x / x_unit_mult, y / y_unit_mult),
                fontsize="x-small",
            )

    def on_click(event: Event) -> Any:
        # event.name (str): the event name
        # event.canvas (FigureCanvasBase): the FigureCanvas instance generating the event
        # event.guiEvent: the GUI event that triggered the Matplotlib event
        # event.x, event.y (int): mouse x and y position in pixels from left and bottom of canvas
        # event.inaxes (Axes or None): the Axes instance over which the mouse is, if any; else None
        # event.xdata, event.ydata (float or None): mouse x and y position in data coordinates, if the mouse is over an axes
        # event.button (None or MouseButton or {'up', 'down'}): the button pressed: None, MouseButton, 'up', or 'down' (up and down are used for scroll events)
        # event.key (None or str): the key pressed: None, any character, 'shift', 'win', or 'control'
        # event.step (float): The number of scroll steps (positive for 'up', negative for 'down'). This applies only to 'scroll_event' and defaults to 0 otherwise.
        # event.dblclick (bool): Whether the event is a double-click. This applies only to 'button_press_event' and is False otherwise. In particular, it's not used in 'button_release_event'.
        if event.inaxes != axe:
            return
        axe.axe_d.lines = []
        axe.axe_v.lines = []

        dval = spg.generateXSerie() / x_unit_mult
        vval = spg.generateYSerie() / y_unit_mult

        itp = interp2d(x=dval, y=vval, z=Z, kind="cubic", copy=False, bounds_error=True)
        axe.axe_d.set_title(
            "%s = %.3f %s%s, %s = %.3f %s%s"
            % (
                spg.name_of_x_var,
                event.xdata,
                x_unit_lbl,
                spg.unit_of_x_var,
                spg.name_of_y_var,
                event.ydata,
                y_unit_lbl,
                spg.unit_of_y_var,
            )
        )

        new_d = itp(dval, event.ydata)
        axe.axe_d.plot(dval, new_d)
        axe.axe_d.set_xlim((dval.min(), dval.max()))
        axe.axe_d.set_ylim((new_d.min(), new_d.max()))

        new_v = itp(event.xdata, vval)
        axe.axe_v.plot(vval, new_v)
        axe.axe_v.set_xlim((vval.min(), vval.max()))
        axe.axe_v.set_ylim((new_v.min(), new_v.max()))

        axe.mkr.set_data([event.xdata], [event.ydata])

        axe.figure.canvas.draw()
        axe.click_fig.canvas.draw()

    if search_fig:
        axe.figure.colorbar(ret, ax=axe)
        axe.click_fig = plt.figure()
        axe.axe_d = axe.click_fig.add_subplot(211)
        axe.axe_v = axe.click_fig.add_subplot(212)
        axe.axe_d.grid(True)
        axe.axe_v.grid(True)
        axe.axe_d.set_xlabel(
            "%s (%s%s)"
            % (
                spg.name_of_x_var,
                x_unit_lbl,
                spg.unit_of_x_var,
            )
        )
        axe.axe_v.set_xlabel(
            "%s (%s%s)"
            % (
                spg.name_of_y_var,
                y_unit_lbl,
                spg.unit_of_y_var,
            )
        )

        cid = axe.figure.canvas.mpl_connect("button_press_event", on_click)

        vm = np.max(Z)
        kd, kv = np.where(Z == vm)

        class TmpEvent:
            pass

        evt = TmpEvent()
        evt.inaxes = axe
        evt.xdata = spg.generateXSerie(kv[0]) / x_unit_mult
        evt.ydata = spg.generateYSerie(kd[0]) / y_unit_mult

        (axe.mkr,) = axe.plot([], [], marker="x", color="black", linestyle="")

        on_click(evt)

    return ret


def plotBode(filt: ADSPFilter, axe_amp: "AxesSubplot", axe_pha: "AxesSubplot"):
    """Plots the bode diagram of a filter

    Args:
      filt
        Filter to analyse
      axe_amp
        Matplotlib axe to draw the ampltiude on
      axe_pha
        Matplotlib axe to draw the unfolded phase on

    """
    fs = 1 / filt.samplingPeriod

    n = 200
    b = filt.generateCoefficients()

    freq = np.linspace(0, fs / 2, n)

    p = Polynomial(b)
    z = np.exp(-1j * 2 * np.pi * freq / fs)
    y = p(z)

    axe_amp.plot(freq, DSPLine.to_db(y))
    axe_amp.grid(True)
    axe_amp.set_ylabel("Ampliude (dB)")

    pha = phase_unfold(y)

    axe_pha.plot(freq, 180 / np.pi * pha)
    axe_pha.grid(True)
    axe_pha.set_xlabel("Frequency (Hz)")
    axe_pha.set_ylabel("Phase (deg)")


def plotDSPLine(line: DSPLine, axe: "AxesSubplot", **kwargs) -> "Line2D":
    """Plots a DSPLine with the following refinements :

    * a callable *transform* is applied to all samples
    * the X and Y axes are labeled according to *x_unit_mult*
    * the X axe is labeled according to the class attributes name_of_x_var and unit_of_x_var
    * the *find_peaks* highest peaks are displayed (default : 0)
    * the label of the plot is the name given at instanciation

    Args:
      line
        Line to be plotted
      axe
        Matplotlib axe to draw on. Pass None to let the function create on axe
      kwargs
        Plotting options. The following extra keys are allowed:
        * transform for a different transform from the one given at instanciation
        * find_peaks to search peaks
        * x_unit_mult to have a more readable unit prefix

    """
    if axe is None:
        fig = plt.figure()
        axe = fig.add_subplot(111)
    axe.grid(True)
    x_samp = line.generateXSerie()
    transform = kwargs.pop("transform", line.default_transform)
    find_peaks = kwargs.pop("find_peaks", 0)

    if "x_unit_mult" in kwargs.keys():
        x_unit_mult = kwargs.pop("x_unit_mult")
    else:
        xm = np.max(np.abs(x_samp))
        lxm = log10(xm)
        ilxm = int(np.round(lxm, 0))
        ilxm3 = ilxm // 3
        pm = ilxm3 * 3
        x_unit_mult = 10**pm
    axe.x_unitilxm3_mult = x_unit_mult
    x_unit_lbl = getUnitAbbrev(x_unit_mult)
    lbl = kwargs.pop("label", line.name)

    (ret,) = axe.plot(
        x_samp / x_unit_mult, transform(line.y_serie), label=lbl, **kwargs
    )
    axe.set_xlabel(
        "%s (%s%s)"
        % (line.__class__.name_of_x_var, x_unit_lbl, line.__class__.unit_of_x_var)
    )

    if find_peaks > 0:
        lpeaks = line.findPeaksWithTransform(transform=transform, nb_peaks=find_peaks)
        for p in lpeaks:
            (x,) = p.coord
            axe.plot(
                [x / x_unit_mult], [p.value], linestyle="", marker="o", color="red"
            )
            axe.annotate(
                "(%.1f %s%s,%.1f)"
                % (x / x_unit_mult, x_unit_lbl, line.__class__.unit_of_x_var, p.value),
                xy=(x / x_unit_mult, p.value),
                fontsize="x-small",
            )

    return ret


def plotVerif(log: Logger, fig_title: str, *axes) -> "Figure":
    """Plots a set of axes and curves on a single figure

    Args:
      log
        Logger which contains the simulated values
      fig_title
        Title of the figure
      axes:
        List of lists of dicts
        Each list is the description of an axe, each dict the description of a line.
        Each dict has a key "var", which is the name of a variable contained in *log*.
        The other keys are keyword arguments for the plot method of matplotlib

    Returns:
      The resulting figure

    """
    l_aspec = []
    for ind, l_lines in enumerate(axes):
        aProp = dict()

        aProp["title"] = "Axe %i" % (ind + 1)
        aProp["nrow"] = len(axes)
        aProp["ncol"] = 1
        aProp["ind"] = ind + 1
        aProp["sharex"] = ind if ind > 0 else None

        lSpec = []
        for l in l_lines:
            if "title" in l.keys():
                aProp["title"] = l.pop("title", "Axe %i" % (ind + 1))
                aProp["sharex"] = l.pop("sharex", None)
                aProp["nrow"] = l["nrow"]
                aProp["ncol"] = l["ncol"]
                aProp["ind"] = l["ind"]
            else:
                l["vary"] = l.pop("var")
                l["varx"] = "t"
                lSpec.append(l)

        aSpec = AxeSpec(aProp, lSpec)

        l_aspec.append(aSpec)

    spec = FigureSpec({"title": fig_title}, axes=l_aspec)
    fig = createFigureFromSpec(spec, log)

    return fig
