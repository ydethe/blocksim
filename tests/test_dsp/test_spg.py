import sys
from pathlib import Path

import numpy as np
import pytest

from blocksim.dsp.DSPMap import DSPRectilinearMap
from blocksim.graphics.BFigure import FigureFactory
from blocksim.graphics.GraphicSpec import AxeProjection

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestDSPSpectrogram(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_ionex(self, ionex_pth: Path = Path("tests/test_dsp/c1pg1080.23i")):
        spg = DSPRectilinearMap.from_ionex(ionex_pth)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0], projection=AxeProjection.PLATECARREE)
        axe.plot(spg)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_2d_peak(self):
        def f(x, y):
            x = 2 * x - 1
            y = -y + 3
            z = -0.5 * x**2 - 0.2 * y**2
            return z

        x = np.linspace(-1.5, 1.5, 100)
        y = np.linspace(2, 4, 50)

        X, Y = np.meshgrid(x, y)
        Z = f(X, Y)

        spg = DSPRectilinearMap(
            name="spg",
            samplingXStart=x[0],
            samplingXPeriod=x[1] - x[0],
            samplingYStart=y[0],
            samplingYPeriod=y[1] - y[0],
            img=Z,
            default_transform=lambda x: x,
        )

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(spg, find_peaks=2)

        return fig.render()


if __name__ == "__main__":
    from blocksim.graphics import showFigures

    a = TestDSPSpectrogram()
    # a.test_2d_peak()
    a.test_ionex()

    showFigures()
