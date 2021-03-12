import os
import unittest
from typing import Iterable
import pathlib
from inspect import currentframe, getframeinfo

import numpy as np
from matplotlib import pyplot as plt

from blocksim.Graphics import FigureSpec, AxeSpec, createFigureFromSpec


class TestBase(unittest.TestCase):
    def setUp(self):
        np.random.seed(1883647)

    def plotVerif(self, fig_title, *axes):
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
        fig = createFigureFromSpec(spec, self.log)

        if "SHOW_PLOT" in os.environ.keys():
            plt.show()

        return fig
