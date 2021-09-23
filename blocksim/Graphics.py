from typing import Iterable

from parse import compile
import numpy as np
from scipy.special import erfc
from numpy import sqrt, log10
from numpy.polynomial import Polynomial
from matplotlib import pyplot as plt
from matplotlib.image import AxesImage

from .Logger import Logger
from .dsp.DSPFilter import DSPFilter
from .dsp.DSPLine import DSPLine
from .dsp.DSPSpectrogram import DSPSpectrogram
from .dsp.utils import phase_unfold
from .B3DPlotter import B3DPlotter
from .source.Trajectory import Trajectory


class AxeSpec(object):
    """Class that provides a description of an axe, without data.
    It the lines to be drawn
    (with the name of the variables instead of a concrete set of data)

    Args:
      props
        A dictionary. Supported keys :

        * nrow for the number of rows subdivisions
        * ncol for the number of columns subdivisions
        * ind for the number of axe (1 is the first one in the layout)
        * sharex is the numer of an axe whose X axe will be shared with the instance of :class:`AxeSpec`
        * title for the title of the axe
      lines
        List of dict to specify the lines' spec. Supported keys :

        * the matplotlib keyword arguments of the funcion *plot*
        * varx for the name of the X variable
        * vary for the name of the y variable

    """

    def __init__(self, props, lines):
        self.props = props
        self.lines = lines

    def __repr__(self, ntabs=0):
        st = " " * ntabs
        s = ""
        s += st + 10 * "=" + " Axe '%s' " % self.props["title"] + 10 * "=" + "\n"
        kys = list(self.props.keys())
        kys.sort()
        for k in kys:
            if k == "title":
                continue
            s += st + "%s:\t'%s'\n" % (k, self.props[k])

        for k, l in enumerate(self.lines):
            s += st + 10 * "-" + " Line #%i " % (k + 1) + 10 * "-" + "\n"
            kys = list(l.keys())
            kys.sort()
            for k in kys:
                s += 2 * st + "%s:\t'%s'\n" % (k, l[k])

        return s


class FigureSpec(object):
    """Class that provides a description of a figure, without data.
    It handles the axes layout, and the lines to be drawn
    (with the name of the variables instead of a concrete set of data)

    Args:
      props
        A dictionary. Only key supported : title for the figure title
      axes
        List of :class:`AxeSpec` to specify the axes' spec

    Examples:
      >>> fs = FigureSpec.specForOneAxeMultiLines([{'var':'th_mes','linestyle':'', 'marker':'+'}])

    """

    def __init__(self, props: dict, axes: AxeSpec):
        self.props = props
        self.axes = axes

    def __repr__(self):
        """Representation of a FigureSpec

        Examples:
          >>> fs = FigureSpec.specForOneAxeMultiLines([{'var':'th_mes','linestyle':'', 'marker':'+'}])
          >>> _ = str(fs)

        """
        s = "FigureSec instance :\n"
        s += 10 * "=" + " Figure '%s' " % self.props["title"] + 10 * "=" + "\n"
        for aSpec in self.axes:
            s += aSpec.__repr__(ntabs=2)

        return s.strip()

    def __str__(self):
        return self.__repr__()

    @classmethod
    def specForOneAxeMultiLines(cls, line_list: Iterable[dict]) -> "FigureSpec":
        """Returns a :class:`FigureSpec` to draw all the given variables on one same axe

        Args:
          line_list
            List of dictionary, whose keys are :

            * the matplotlib keyword arguments of the funcion *plot*
            * varx for the name of the X variable. If not specified, varx will be assumed to be the time variable 't'
            * var or vary for the name of the y variable

        Returns
          A :class:`FigureSpec` that describes the layout

        """
        lines = []
        n = len(line_list)
        for var in line_list:
            line = var.copy()
            if not "varx" in line.keys():
                line["varx"] = "t"
            if "var" in line.keys():
                line["vary"] = line.pop("var")
            lines.append(line)

        aSpec = AxeSpec(
            props={"nrow": 1, "ncol": 1, "ind": 1, "title": "Axe", "sharex": None},
            lines=lines,
        )
        spec = FigureSpec(props={"title": "Figure"}, axes=[aSpec])

        return spec


def plotFromLogger(
    log: Logger, id_x: str, id_y: str, axe: "AxesSubplot", **kwargs
) -> "Line2D":
    """Plots a value on a matplotlib axe

    Args:
      log
        :class:`SystemControl.Logger.Logger` instance
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
    if log is None:
        val_x = []
        val_y = []
    else:
        if type(id_x) == type(""):
            val_x = log.getValue(id_x)
        elif hasattr(id_x, "__iter__"):
            val_x = id_x
        else:
            raise SystemError(
                u"[ERROR]Unacceptable argument for id_x : %s" % (str(id_x))
            )

        if type(id_y) == type(""):
            val_y = log.getValue(id_y)
        elif hasattr(id_y, "__iter__"):
            val_y = id_y
        else:
            raise SystemError(
                u"[ERROR]Unacceptable argument for id_y : %s" % (str(id_y))
            )

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

    Returns:
      The matplotlib image generated

    """
    axe.grid(True)
    transform = kwargs.pop("transform", spg.default_transform)
    x_unit_mult = kwargs.pop("x_unit_mult", 1)
    x_unit_lbl = DSPLine.getUnitAbbrev(x_unit_mult)
    y_unit_mult = kwargs.pop("y_unit_mult", 1)
    y_unit_lbl = DSPLine.getUnitAbbrev(y_unit_mult)
    lbl = kwargs.pop("label", spg.name)

    ret = axe.imshow(
        transform(spg.img),
        aspect="auto",
        extent=(
            spg.generateXSerie(0),
            spg.generateXSerie(-1),
            spg.generateYSerie(0),
            spg.generateYSerie(-1),
        ),
        origin="lower",
        label=lbl,
        **kwargs
    )
    axe.set_xlabel(
        "%s (%s%s)"
        % (spg.__class__.name_of_x_var, x_unit_lbl, spg.__class__.unit_of_x_var)
    )
    axe.set_ylabel(
        "%s (%s%s)"
        % (spg.__class__.name_of_y_var, y_unit_lbl, spg.__class__.unit_of_y_var)
    )
    return ret


def plotBode(
    filt: DSPFilter, fs: float, axe_amp: "AxesSubplot", axe_pha: "AxesSubplot"
):
    """Plots the bode diagram of a filter

    Args:
      filt
        Filter to analyse
      fs (Hz)
        The sampling frequency of the signals that will be filtered through it
      axe_amp
        Matplotlib axe to draw the ampltiude on
      axe_pha
        Matplotlib axe to draw the unfolded phase on

    """
    n = 200
    b = filt.generateCoefficients(fs)

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
        Matplotlib axe to draw on
      kwargs
        Plotting options. The following extra keys are allowed:
        * transform for a different transform from the one given at instanciation
        * find_peaks to search peaks
        * x_unit_mult to have a more readable unit prefix

    """
    axe.grid(True)
    transform = kwargs.pop("transform", line.default_transform)
    find_peaks = kwargs.pop("find_peaks", 0)
    x_unit_mult = kwargs.pop("x_unit_mult", 1)
    x_unit_lbl = line.getUnitAbbrev(x_unit_mult)
    lbl = kwargs.pop("label", line.name)

    (ret,) = axe.plot(
        line.generateXSerie() / x_unit_mult,
        transform(line.y_serie),
        label=lbl,
        **kwargs
    )
    axe.set_xlabel(
        "%s (%s%s)"
        % (line.__class__.name_of_x_var, x_unit_lbl, line.__class__.unit_of_x_var)
    )

    if find_peaks > 0:
        lpeaks = line.findPeaksWithTransform(transform=transform, nb_peaks=find_peaks)
        for x in lpeaks:
            y = transform(line.getSample(x))
            axe.plot([x / x_unit_mult], [y], linestyle="", marker="o", color="red")
            axe.annotate(
                "(%.1f %s%s,%.1f)"
                % (x / x_unit_mult, x_unit_lbl, line.__class__.unit_of_x_var, y),
                xy=(x / x_unit_mult, y),
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


def plotBER(fic, output=""):
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

    c_n0 = np.array(snr) + 10 * log10(180e3)

    fig = plt.figure(dpi=150)
    axe = fig.add_subplot(111)
    axe.grid(True)
    axe.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    axe.semilogy(c_n0, ber, label="Simu BER")
    axe.legend()
    axe.set_xlabel("$C/N_0$ (dB)")
    axe.set_ylabel("BER")

    if output == "show":
        plt.show()
    elif output == "":
        pass
    else:
        plt.savefig(output)

    return fig


def plot3DEarth(trajectories: Iterable[Trajectory]) -> B3DPlotter:
    app = B3DPlotter()

    app.buildEarth()

    for traj in trajectories:
        app.buildTrajectory(traj)

    return app
