import sys
from pathlib import Path
import unittest

import numpy as np
from matplotlib import pyplot as plt
import pytest

from blocksim.Logger import Logger
from blocksim.graphics import (
    plotFromLogger,
    createFigureFromSpec,
    plotVerif,
)
from blocksim.graphics.AxeSpec import AxeSpec
from blocksim.graphics.FigureSpec import FigureSpec

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestGraphics(TestBase):
    def setUp(self):
        self.log = Logger()

        dt = 0.01
        f = 11
        ns = 1000

        self.tps_ref = np.arange(ns) * dt
        self.x_ref = np.sin(self.tps_ref * f * 2 * np.pi + 1)

        for (t, x) in zip(self.tps_ref, self.x_ref):
            self.log.log("t", t)
            self.log.log("x", x)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_plot_logger(self):
        err_t = np.max(np.abs(self.tps_ref - self.log.getValue("t")))
        err_x = np.max(np.abs(self.x_ref - self.log.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

        fig = plt.figure()
        fig.suptitle = "Essai logger"
        axe = fig.add_subplot(111)
        axe.grid(True)

        fc = 5.0
        dt = self.tps_ref[1] - self.tps_ref[0]
        fr = self.log.getFilteredValue("x", 64, 2 * fc * dt)

        plotFromLogger(self.log, "t", "x", axe, label="brut")
        plotFromLogger(self.log, "t", fr, axe, label="filtré")

        t = self.log.getValue("t")
        plotFromLogger(self.log, t + 1, "x", axe, label="brut,translated")

        axe.legend(loc="best")

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_figure_from_spec(self):
        aProp = dict()

        aProp["title"] = "Axe 1"
        aProp["nrow"] = 1
        aProp["ncol"] = 1
        aProp["ind"] = 1
        aProp["sharex"] = None

        x = np.arange(10)
        y = -2 * x
        lSpec = [{"varx": x, "vary": y}]

        aSpec = AxeSpec(aProp, lSpec)
        spec = FigureSpec({"title": "Figure title"}, axes=[aSpec])
        fig = createFigureFromSpec(spec=spec, log=None, fig=None)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_plot_verif(self):
        aProp = dict()

        aProp["title"] = "Axe 1"
        aProp["nrow"] = 1
        aProp["ncol"] = 1
        aProp["ind"] = 1
        aProp["sharex"] = None

        t = self.log.getValue("t")

        fig = plotVerif(
            self.log,
            "Figure title",
            [
                {
                    "title": "Axe title",
                    "nrow": 1,
                    "ncol": 1,
                    "ind": 1,
                },
                {"var": 1 + t},
            ],
        )

        return fig
