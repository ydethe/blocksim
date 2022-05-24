"""Advanced plotting functions.
Allows plotting from a `blocksim.loggers.Logger.Logger`, or from `blocksim.dsp.DSPLine.DSPLine`
3D plot around Earth are also possible

"""

from typing import Any, Tuple, Iterable

from parse import compile
from nptyping import NDArray, Shape
import numpy as np
from scipy.interpolate import interp2d
from numpy import log10
from matplotlib import pyplot as plt
from matplotlib.backend_bases import Event
import networkx as nx

from .. import logger
from ..loggers.Logger import Logger
from ..dsp.DSPFilter import ADSPFilter
from ..dsp.DSPLine import DSPLine
from ..dsp.DSPSpectrogram import DSPSpectrogram
from ..dsp import phase_unfold
from .AxeSpec import AxeSpec
from .FigureSpec import FigureSpec
from ..satellite.Trajectory import Trajectory


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
            xm = np.abs(samp)
            pm = (int(log10(xm)) // 3) * 3
            mult = 10**pm
        else:
            mult = force_mult
        lbl = d[mult]
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


def createAxeFromSpec(spec: "SubplotSpec" = None, **kwargs) -> "AxesSubplot":
    if spec is None:
        fig = plt.figure()
        gs = fig.add_gridspec(1, 1)
        spec = gs[0, 0]

    gs = spec.get_gridspec()
    fig = gs.figure
    axe = fig.add_subplot(spec, **kwargs)
    axe.grid(True)
    return axe


def plotFromLogger(
    log: Logger, id_x: str, id_y: str, spec: "SubplotSpec" = None, **kwargs
) -> "AxesSubplot":
    """Plots a value on a matplotlib axe

    Args:
        log: The Logger to read into
        id_x: Name or expression for the X axis
        id_y: Name or expression for the Y axis
        spec: The matplotlib SubplotSpec that defines the axis to draw on. Obtained by fig.add_gridspec and slicing
        kwargs: matplotlib plotting options for the 'plot' method

    Returns:
        The Axes used by matplotlib

    """
    sharex = kwargs.pop("sharex", None)
    sharey = kwargs.pop("sharey", None)
    axe_opt = {"sharex": sharex, "sharey": sharey}
    axe = createAxeFromSpec(spec, **axe_opt)

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

    if not "label" in kwargs.keys():
        kwargs["label"] = id_y

    (line,) = axe.plot(val_x, val_y, **kwargs)

    line.x_unit_mult = 1.0

    return axe


def createFigureFromSpec(spec: FigureSpec, log: Logger, fig=None) -> "Figure":
    """Parses a FigureSpec to build a matplotlib figure, and returns it

    Args:
        spec: A FigureSpec instance
        log: The Logger to read into
        fig: A matplotlib figure. If None, the function creates ones

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
        proj = spec.axes[k - 1].props.pop("projection", "rectilinear")

        if shx is None:
            axe = fig.add_subplot(nrow, ncol, ind, projection=proj)
            axe.grid(True)
        else:
            axe = fig.add_subplot(
                nrow, ncol, ind, projection=proj, sharex=l_axes[shx - 1]
            )
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
                axe = plotFromLogger(log, varx, vary, axe, **lp)
                disp_leg = True
            elif type(vary) == type(""):
                axe = plotFromLogger(log, varx, vary, axe, label=vary, **lp)
                disp_leg = True
            else:
                axe = plotFromLogger(log, varx, vary, axe, **lp)

            line = axe.get_lines()[-1]
            d["_line"] = line
            xdata, ydata = line.get_data()
            d["_xdata"] = xdata
            d["_ydata"] = ydata

        if disp_leg:
            axe.legend()

    fig.tight_layout()

    return fig


def plotSpectrogram(
    spg: DSPSpectrogram, spec: "SubplotSpec" = None, fill: str = "pcolormesh", **kwargs
) -> "AxesSubplot":
    """Plots a line with the following refinements :

    * a callable *transform* is applied to all samples
    * the label of the plot is the name given at instanciation

    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html for the possible values in kwargs
    Args:
        spg: DSPSpectrogram to plot
        spec: The matplotlib SubplotSpec that defines the axis to draw on. Obtained by fig.add_gridspec and slicing
        fill: Method to plot the DSPSpectrogram. Can be 'plot_surface', 'pcolormesh', 'contour' or 'contourf'
        kwargs: Plotting options. The following extra keys are allowed:

            * transform for a different transform from the one given at instanciation
            * find_peaks to search peaks
            * x_unit_mult to have a more readable unit prefix

    Returns:
        The matplotlib image generated

    """
    if fill == "plot_surface":
        proj = "3d"
    else:
        proj = spg.projection

    sharex = kwargs.pop("sharex", None)
    sharey = kwargs.pop("sharey", None)
    axe_opt = {"sharex": sharex, "sharey": sharey, "projection": proj}
    axe = createAxeFromSpec(spec, **axe_opt)

    transform = kwargs.pop("transform", spg.default_transform)
    find_peaks = kwargs.pop("find_peaks", 0)

    x_samp = spg.generateXSerie()
    xm = np.max(np.abs(x_samp))
    x_unit_mult = kwargs.pop("x_unit_mult", None)
    _, x_unit_mult, x_unit_lbl, x_unit = getUnitAbbrev(
        xm, spg.unit_of_x_var, force_mult=x_unit_mult
    )

    y_samp = spg.generateYSerie()
    ym = np.max(np.abs(y_samp))
    y_unit_mult = kwargs.pop("y_unit_mult", None)
    _, y_unit_mult, y_unit_lbl, y_unit = getUnitAbbrev(
        ym, spg.unit_of_y_var, force_mult=y_unit_mult
    )
    # lbl = kwargs.pop("label", spg.name)

    Z = transform(spg.img)
    if fill == "plot_surface" and spg.projection == "polar":
        P, R = np.meshgrid(
            spg.generateXSerie() / x_unit_mult, spg.generateYSerie() / y_unit_mult
        )
        X, Y = R * np.cos(P), R * np.sin(P)
    else:
        X, Y = np.meshgrid(
            spg.generateXSerie() / x_unit_mult, spg.generateYSerie() / y_unit_mult
        )

    if fill == "plot_surface":
        kwargs.pop("levels", None)
        ret = axe.plot_surface(X, Y, Z, **kwargs)
        axe.figure.colorbar(ret, ax=axe)
    elif fill == "pcolormesh":
        kwargs.pop("levels", None)
        ret = axe.pcolormesh(X, Y, Z, **kwargs)
        axe.figure.colorbar(ret, ax=axe)
    elif fill == "contourf":
        ret = axe.contourf(X, Y, Z, **kwargs)
        axe.figure.colorbar(ret, ax=axe)
    elif fill == "contour":
        ret = axe.contour(X, Y, Z, **kwargs)
        axe.clabel(ret, inline=True, fontsize=10)

    if spg.name_of_x_var != "":
        axe.set_xlabel(
            "%s (%s%s)" % (spg.name_of_x_var, x_unit_lbl, x_unit_lbl + x_unit)
        )
    if spg.name_of_y_var != "":
        axe.set_ylabel(
            "%s (%s%s)" % (spg.name_of_y_var, y_unit_lbl, y_unit_lbl + y_unit)
        )

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

    def on_click(event: Event) -> Any:  # pragma: no cover
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
        zdata = itp(event.xdata, event.ydata)
        print(zdata)

    return axe


def plotBode(
    filt: ADSPFilter,
    spec_amp: "SubplotSpec",
    spec_pha: "SubplotSpec",
    fpoints: int = 200,
    pow_lim: float = -100.0,
    **kwargs
) -> Tuple["AxesSubplot", "AxesSubplot"]:
    """Plots the bode diagram of a filter

    Args:
        filt: Filter to analyse
        spec_amp: The matplotlib SubplotSpec that defines the amplitude axis to draw on. Obtained by fig.add_gridspec and slicing
        spec_pha: The matplotlib SubplotSpec that defines the phase axis to draw on. Obtained by fig.add_gridspec and slicing
        fpoints: If int, number of frequency samples to use for the plot
            If iterable, list of frequency samples to use for the plot
        kwargs: Plotting options. The following extra keys are allowed:

            * x_unit_mult to have a more readable unit prefix

    Examples:
        >>> from blocksim.dsp.DSPFilter import ArbitraryDSPFilter
        >>> f = ArbitraryDSPFilter(name="MTI", samplingPeriod=1e6, num=[1, -1])
        >>> fig = plt.figure()
        >>> gs = fig.add_gridspec(2, 1)
        >>> _ = plotBode(f, spec_amp=gs[0, 0], spec_pha=gs[1, 0])

    """
    from scipy.signal import TransferFunction, freqz

    axe_amp = createAxeFromSpec(spec_amp)
    axe_pha = createAxeFromSpec(spec_pha)

    fs = 1 / filt.samplingPeriod

    b, a = filt.generateCoefficients()

    if hasattr(fpoints, "__iter__"):
        freq = fpoints
    else:
        freq = np.arange(0, fs / 2, fs / 2 / fpoints)

    num, den = TransferFunction._z_to_zinv(b, a)
    _, y = freqz(num, den, worN=freq, fs=fs)

    x_unit_mult = kwargs.get("x_unit_mult", None)
    xm = np.max(np.abs(freq))
    scaled_samp, x_unit_mult, x_unit_lbl, x_unit = getUnitAbbrev(
        xm, "Hz", force_mult=x_unit_mult
    )

    axe_amp.plot(freq / x_unit_mult, DSPLine.to_db(y, lim_db=pow_lim))
    axe_amp.grid(True)
    axe_amp.set_ylabel("Amplitude (dB)")

    pha = phase_unfold(y)

    axe_pha.plot(freq / x_unit_mult, 180 / np.pi * pha)
    axe_pha.grid(True)
    axe_pha.set_xlabel("Frequency (%s%s)" % (x_unit_lbl, x_unit))
    axe_pha.set_ylabel("Phase (deg)")

    return axe_amp, axe_pha


def plotDSPLine(line: DSPLine, spec: "SubplotSpec" = None, **kwargs) -> "AxesSubplot":
    """Plots a DSPLine with the following refinements :

    * a callable *transform* is applied to all samples
    * the X and Y axes are labeled according to *x_unit_mult*
    * the X axe is labeled according to the class attributes name_of_x_var and unit_of_x_var
    * the *find_peaks* highest peaks are displayed (default : 0)
    * the label of the plot is the name given at instanciation

    See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html for the possible values in kwargs

    Args:
        line: DSPLine to be plotted
        spec: The matplotlib SubplotSpec that defines the axis to draw on. Obtained by fig.add_gridspec and slicing
        kwargs: Plotting options. The following extra keys are allowed:

            * transform for a different transform from the one given at instanciation
            * find_peaks to search peaks
            * x_unit_mult to have a more readable unit prefix

    """
    sharex = kwargs.pop("sharex", None)
    sharey = kwargs.pop("sharey", None)
    axe_opt = {"sharex": sharex, "sharey": sharey}
    axe = createAxeFromSpec(spec, **axe_opt)

    x_samp = line.generateXSerie()
    transform = kwargs.pop("transform", line.default_transform)
    find_peaks = kwargs.pop("find_peaks", 0)

    x_unit_mult = kwargs.pop("x_unit_mult", None)
    xm = np.max(np.abs(x_samp))
    scaled_samp, x_unit_mult, x_unit_lbl, x_unit = getUnitAbbrev(
        xm, line.unit_of_x_var, force_mult=x_unit_mult
    )
    lbl = kwargs.pop("label", line.name)

    (ret,) = axe.plot(
        x_samp / x_unit_mult, transform(line.y_serie), label=lbl, **kwargs
    )
    axe.set_xlabel("%s (%s%s)" % (line.name_of_x_var, x_unit_lbl, x_unit))

    if find_peaks > 0:
        lpeaks = line.findPeaksWithTransform(transform=transform, nb_peaks=find_peaks)
        for p in lpeaks:
            (x,) = p.coord
            axe.plot(
                [x / x_unit_mult], [p.value], linestyle="", marker="o", color="red"
            )
            axe.annotate(
                "(%.1f %s%s,%.1f)" % (x / x_unit_mult, x_unit_lbl, x_unit, p.value),
                xy=(x / x_unit_mult, p.value),
                fontsize="x-small",
            )

    ret.x_unit_mult = x_unit_mult

    return axe


def plotVerif(log: Logger, fig_title: str, *axes) -> "Figure":
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


def plotGraph(G, spec=None, **kwds) -> "Axes":
    """See https://networkx.org/documentation/stable/reference/generated/networkx.drawing.nx_pylab.draw.html#networkx.drawing.nx_pylab.draw

    Args:
        G: graph to draw
        spec: The matplotlib SubplotSpec that defines the axis to draw on. Obtained by fig.add_gridspec and slicing
        kwds: See https://networkx.org/documentation/stable/reference/generated/networkx.drawing.layout.kamada_kawai_layout.html#networkx.drawing.layout.kamada_kawai_layout

    Returns
        The actual axe used for plotting

    """
    axe = createAxeFromSpec(spec)
    axe.grid(False)
    axe.set_aspect("equal")

    if not "node_size" in kwds.keys():
        kwds["node_size"] = 1000
    pos = nx.kamada_kawai_layout(G)
    nx.draw_networkx(G, pos=pos, ax=axe, **kwds)

    return axe


def plot3DEarth(trajectories: Iterable[Trajectory]) -> "B3DPlotter":
    """Shows a 3D tracetory around a 3D Earth

    Args:
        trajectories: list of Trajectory objects to plot

    Returns:
        A B3DPlotter instance. Call app.run() to show the window

    """
    from .B3DPlotter import B3DPlotter

    app = B3DPlotter()

    app.buildEarth()

    for traj in trajectories:
        app.plotTrajectory(traj)

    return app


def plotBER(fic, spec=None, **kwds) -> "Axes":
    """Helper function that plots a BER curve from a log file where the lines are :

    "[{level}] - SNR = {snr} dB, it={it}, Bits Received = {bit_rx}, Bit errors = {bit_err}, BER = {ber}"

    Args:
        fic: ASCII file to read
        axe_spec: The matplotlib SubplotSpec that defines the axis to draw on. Obtained by fig.add_gridspec and slicing
        kwds: plotting options

    Returns:
        The actual axe used for plotting

    """
    p = compile(
        "[{level}] - SNR = {snr} dB, it={it}, Bits Received = {bit_rx}, Bit errors = {bit_err}, BER = {ber}"
    )

    f = open(fic, "r")
    snr = []
    ber = []
    for line in f:
        dat = p.parse(line)
        snr.append(float(dat["snr"]))
        ber.append(float(dat["ber"]))

    sharex = kwds.pop("sharex", None)
    sharey = kwds.pop("sharey", None)
    axe_opt = {"sharex": sharex, "sharey": sharey}
    axe = createAxeFromSpec(spec, **axe_opt)

    axe.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    axe.semilogy(snr, ber, label="Simu BER", **kwds)
    axe.legend()
    axe.set_xlabel("$SNR$ (dB)")
    axe.set_ylabel("BER")

    return axe
