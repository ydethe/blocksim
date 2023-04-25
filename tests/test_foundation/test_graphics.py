from datetime import datetime, timedelta

import numpy as np
from numpy import pi, sqrt, exp, cos
from scipy.special import erf
import pytest

from blocksim.loggers.Logger import Logger
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.graphics import (
    createFigureFromSpec,
    plotVerif,
    showFigures,
)
from blocksim.graphics.GraphicSpec import AxeSpec, FigureSpec
from blocksim.graphics.BFigure import FigureFactory


from blocksim.testing import TestBase


class TestGraphics(TestBase):
    def setUp(self):
        super().setUp()

        self.log = Logger()

        dt = 0.01
        f = 11
        ns = 1000

        self.tps_ref = np.arange(ns) * dt
        self.x_ref = np.sin(self.tps_ref * f * 2 * np.pi + 1)

        for (t, x) in zip(self.tps_ref, self.x_ref):
            self.log.log(name="t", val=t, unit="s")
            self.log.log(name="x", val=x, unit="")

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_plot_datetime(self):
        fig = FigureFactory.create(title="Essai datetime")
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])

        t0 = datetime(year=2022, month=4, day=20, hour=13, minute=46)
        t = [t0 + timedelta(minutes=x) for x in range(20)]
        x = cos(np.arange(20))
        axe.plot((t, x))

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_plot_timedelta(self):
        fig = FigureFactory.create(title="Essai timedelta")
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])

        t = [timedelta(minutes=x) for x in range(20)]
        x = cos(np.arange(20))
        axe.plot((t, x))

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 150})
    def test_plot_logger(self):
        err_t = np.max(np.abs(self.tps_ref - self.log.getValue("t")))
        err_x = np.max(np.abs(self.x_ref - self.log.getValue("x")))
        self.assertAlmostEqual(err_t, 0.0, delta=1.0e-2)
        self.assertAlmostEqual(err_x, 0.0, delta=1.0e-2)

        fig = FigureFactory.create(title="Essai logger")
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])

        fc = 5.0
        dt = self.tps_ref[1] - self.tps_ref[0]
        fr = self.log.getFilteredValue("x", 64, 2 * fc * dt)

        axe.plot((self.log, "t", "x"), label="brut")
        axe.plot((self.log, "t", fr), label="filtré")

        t = self.log.getValue("t")
        axe.plot((self.log, t + 1, "x"), label="brut,translated")

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_histogram(self):
        ns = 10000
        a = np.random.normal(size=ns)
        serie = DSPSignal(name="serie", samplingStart=0, samplingPeriod=1, y_serie=a)
        hist = serie.histogram(name="hist", density=True)
        bins = hist.generateXSerie()

        def pdf(x):
            return 1 / sqrt(2 * pi) * exp(-(x**2) / 2)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="Histogram", spec=gs[0, 0])
        axe.plot(hist)
        axe.plot(plottable=(bins, pdf(bins)), color="red")

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_cdf(self):
        ns = 10000
        a = np.random.normal(size=ns)
        serie = DSPSignal(name="serie", samplingStart=0, samplingPeriod=1, y_serie=a)
        hist = serie.histogram(name="hist", density=True, cumulative=True)
        bins = hist.generateXSerie()

        def cdf(x):
            return 0.5 * (1 + erf(x / sqrt(2)))

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="Histogram (CDF)", spec=gs[0, 0])
        axe.plot(hist)
        axe.plot(plottable=(bins, cdf(bins)), color="red")

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_figure_from_spec(self):
        aProp = dict()

        aProp["title"] = "Axe 1"
        aProp["coord"] = 0
        aProp["sharex"] = None

        x = np.arange(10)
        y = -2 * x
        lSpec = [{"varx": x, "vary": y}]

        aSpec = AxeSpec(aProp, lSpec)
        spec = FigureSpec({"title": "Figure title", "nrow": 1, "ncol": 1}, axes=[aSpec])
        fig = createFigureFromSpec(spec=spec, log=None, fig=None)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_plot_verif(self):
        t = self.log.getValue("t")

        fig = plotVerif(
            self.log,
            "Figure title",
            [
                {
                    "title": "Axe title",
                    "coord": 0,
                },
                {"var": 1 + t},
            ],
        )

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def ntest_twinx(self):
        # Create some mock data
        t = np.arange(0.01, 10.0, 0.01)
        data1 = np.exp(t)
        data2 = np.sin(2 * np.pi * t)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(({"data": t, "unit": "s", "name": "Time"}, {"data": data1, "name": "exp"}))

        axe.plot(
            ({"data": t, "unit": "s", "name": "Time"}, {"data": data2, "name": "sin"}),
            # color="red",
            twinx=True,
        )

        return fig.render()


if __name__ == "__main__":
    # unittest.main()

    a = TestGraphics()
    a.setUp()
    # a.test_histogram()
    # a.test_cdf()
    # a.test_figure_from_spec()
    a.test_plot_timedelta()
    a.test_plot_datetime()
    # a.test_plot_logger()

    showFigures()
