from typing import Iterable

from .AxeSpec import AxeSpec


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
            props={
                "nrow": 1,
                "ncol": 1,
                "ind": 1,
                "title": "Axe",
                "projection": "rectilinear",
                "sharex": None,
            },
            lines=lines,
        )
        spec = FigureSpec(props={"title": "Figure"}, axes=[aSpec])

        return spec
