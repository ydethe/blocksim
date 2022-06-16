from typing import Iterable, List
from enum import Enum
from dataclasses import dataclass


class DSPLineType(Enum):
    RECTILINEAR = 0
    HISTOGRAM = 1
    POLAR = 2
    NORTH_POLAR = 3


class DSPMapType(Enum):
    RECTILINEAR = 0
    POLAR = 1
    NORTH_POLAR = 2


class AxeProjection(Enum):
    #: For rectilinear plots (the most frequent use case)
    RECTILINEAR = 0
    LOGX = 2
    LOGY = 3
    LOGXY = 4
    #: For trigonometric polar plots
    POLAR = 5
    #: For north azimuthal plots
    NORTH_POLAR = 6
    #: For Mercator cartography
    PLATECARREE = 7
    #: For 3D plots
    DIM3D = 8


class FigureProjection(Enum):
    #: For matplotlib plots
    MPL = 0
    #: For panda3d 3d plots
    EARTH3D = 1


@dataclass(init=True)
class Annotation:
    coord: tuple
    text: str


class AxeSpec(object):
    """Class that provides a description of an axe, without data.
    It handles the lines to be drawn
    (with the name of the variables instead of a concrete set of data)

    Args:
        props: A dictionary. Supported keys :

            * coord for the position in the layout. Shall be a slice object
            * sharex is the numer of an axe whose X axe will be shared with the instance of AxeSpec
            * title for the title of the axe
            * projection for the axe projection. Can be 'map', 'rectilinear', 'north_polar' or 'polar'
        lines: List of dict to specify the lines' spec. Supported keys :

            * the matplotlib keyword arguments of the funcion *plot*
            * varx for the name of the X variable
            * vary for the name of the y variable

    """

    __slots__ = ["props", "lines"]

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
        props: A dictionary. Supported key:

        * title: figure title
        * nrow: number of rows in the BFigure layout
        * ncol: number of columns in the BFigure layout
        axes: List of blocksim.graphics.GraphicSpec to specify the axes' spec

    Examples:
        >>> fs = FigureSpec.specForOneAxeMultiLines([{'var':'th_mes','linestyle':'', 'marker':'+'}])

    """

    __slots__ = ["props", "axes"]

    def __init__(self, props: dict, axes: List[AxeSpec]):
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
    def specForOneAxeMultiLines(
        cls, line_list: Iterable[dict]
    ) -> "blocksim.graphics.GraphicSpec.FigureSpec":
        """Returns a FigureSpec to draw all the given variables on one same axe

        Args:
            line_list: List of dictionary, whose keys are :

              * the matplotlib keyword arguments of the funcion *plot*
              * varx for the name of the X variable. If not specified, varx will be assumed to be the time variable 't'
              * var or vary for the name of the y variable

        Returns
            A FigureSpec that describes the layout

        """
        lines = []
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
