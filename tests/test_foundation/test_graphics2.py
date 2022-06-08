import unittest
import pytest

import numpy as np
from numpy import pi, sqrt
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.dsp.DSPSpectrogram import DSPSpectrogram
from blocksim.graphics.wip import (
    AxeFactory,
    FigureFactory,
    render,
    AxeProjection,
    FigureProjection,
    showFigures,
)


class TestGraphics2(unittest.TestCase):
    def setUp(self):
        np.random.seed(1883647)

        self.sig = DSPSignal.fromLinearFM(
            name="sig",
            samplingPeriod=1e-2 / 100,
            samplingStart=0,
            tau=1e-2,
            fstart=-2000,
            fend=2000,
        )

        img = np.zeros((50, 100))
        xx = np.arange(-5, 5, 0.1)
        yy = np.arange(-5, 5, 0.2)
        X, Y = np.meshgrid(xx, yy)
        R2 = X**2 + Y**2
        img = np.sinc(sqrt(R2))
        self.spg = DSPSpectrogram(
            name="spg",
            samplingXStart=-5,
            samplingXPeriod=0.1,
            samplingYStart=-5,
            samplingYPeriod=0.2,
            img=img,
        )

    def test_3d_plot(self):
        fig = FigureFactory.create(title="Figure", projection=FigureProjection.EARTH3D)
        gs = fig.add_gridspec(2, 1)
        self.assertRaises(
            AssertionError,
            AxeFactory.create,
            spec=gs[0, 0],
            title="axe",
            projection=AxeProjection.RECTILINEAR,
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_rect_spg(self):
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.RECTILINEAR
        )

        axe.plot(plottable=self.spg, find_peaks=1)

        return render(fig)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_rect_spg_3d(self):
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.DIM3D
        )

        axe.plot(plottable=self.spg, fill="plot_surface")

        return render(fig)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_rect_spg_contour(self):
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0],
            title="axe",
            projection=AxeProjection.RECTILINEAR,
            aspect="equal",
        )

        axe.plot(plottable=self.spg, fill="contour", levels=20)

        return render(fig)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_2d_earth(self):
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.PLATECARREE
        )

        lon = np.linspace(-pi, pi, 50)
        lat = np.cos(lon)
        axe.plot(plottable=(lon, lat), color="red")

        self.assertRaises(AssertionError, axe.plot, self.sig)

        return render(fig)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_northpolar_plot(self):
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.NORTH_POLAR
        )

        theta = np.linspace(0, 2 * pi, 50)
        r = theta / (2 * pi)
        axe.plot(plottable=(theta, r), color="red")

        return render(fig)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_rectilinear(self):
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(2, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.RECTILINEAR
        )

        theta = np.linspace(0, 2 * pi, 50)
        r = theta / (2 * pi)
        axe.plot(plottable=(theta, r), color="red")

        return render(fig)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_multiple_plots(self):
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(2, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.RECTILINEAR
        )

        theta = np.linspace(0, 0.5, 50)
        r = theta * 2
        axe.plot(plottable=(theta, r), color="red")
        axe.plot(plottable=self.sig, color="red")

        axe = AxeFactory.create(
            spec=gs[1, 0], title="axe", sharex=axe, projection=AxeProjection.RECTILINEAR
        )
        axe.plot(plottable=(theta, np.cos(theta * 6 * pi)), color="blue", find_peaks=2)

        return render(fig)


if __name__ == "__main__":
    # unittest.main()

    a = TestGraphics2()
    a.setUp()
    a.test_multiple_plots()
    # a.test_rectilinear()
    # a.test_rect_spg_contour()
    # a.test_rect_spg()

    showFigures()
