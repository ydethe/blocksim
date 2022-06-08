import unittest
import pytest

import numpy as np
from numpy import pi
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.graphics.wip import (
    AxeFactory,
    FigureFactory,
    render,
    AxeProjection,
    FigureProjection,
)


class TestGraphics2(unittest.TestCase):
    def setUp(self):
        np.random.seed(1883647)

        self.sig = DSPSignal.fromLinearFM(
            name="sig",
            samplingPeriod=1e-2,
            samplingStart=0,
            tau=50e-2,
            fstart=-20,
            fend=20,
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


if __name__ == "__main__":
    unittest.main()
