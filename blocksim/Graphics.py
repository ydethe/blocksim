from typing import Iterable

import numpy as np
from matplotlib import pyplot as plt


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


def plotFromLogger(log, id_x: str, id_y: str, axe, **kwargs):
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


def createFigureFromSpec(spec, log, fig=None):
    """Parses a :class:`FigureSpec` to build a matplotlib figure, and returns it

    Args:
      spec
        A :class:`FigureSpec` instance
      log
        A :class:`SystemControl.Logger.Logger` to read data from
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


if __name__ == "__main__":
    import doctest

    doctest.testmod()
