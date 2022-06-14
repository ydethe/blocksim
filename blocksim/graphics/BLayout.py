class BGridElement(object):
    """This class stores the position of an axe in a grid

    Args:
        gs: Grid that gives axes' positions in a BFigure
        coord: Position of the BGridElement in the grid

    """

    __slots__ = ["axe", "gs", "coord"]

    def __init__(self, gs: "blocksim.graphics.BLayout.BGridSpec", coord: slice):
        self.axe = None
        self.gs = gs
        self.coord = coord

    def get_gridspec(self) -> "blocksim.graphics.BLayout.BGridSpec":
        """Returns the BGridSpec associated with the BGridElement

        Returns:
            The BGridSpec associated with the BGridElement

        """
        return self.gs

    def getFigure(self) -> "blocksim.graphics.BFigure.ABFigure":
        return self.gs.figure


class BGridSpec(object):
    """This class stores the layout of the axes in a BFigure

    Args:
        figure: Refering BFigure
        nrow: Number of rows of the layout
        ncol: Number of columns of the layout

    """

    __slots__ = ["figure", "nrow", "ncol"]

    def __init__(
        self, figure: "blocksim.graphics.BFigure.ABFigure", nrow: int, ncol: int
    ):
        self.figure = figure
        self.nrow = nrow
        self.ncol = ncol

    def __getitem__(self, ind) -> BGridElement:
        ge = BGridElement(gs=self, coord=ind)
        return ge
