import sys
from pathlib import Path
import unittest

import numpy as np
from matplotlib import pyplot as plt
import pytest

from blocksim.dsp.DSPSpectrogram import DSPSpectrogram
from blocksim.graphics import plotSpectrogram

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase, plotAnalyticsolution


class TestDSPSpectrogram(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
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

        spg = DSPSpectrogram(
            name="spg",
            samplingXStart=x[0],
            samplingXPeriod=x[1] - x[0],
            samplingYStart=y[0],
            samplingYPeriod=y[1] - y[0],
            img=Z,
            default_transform=lambda x: x,
        )

        fig = plt.figure()
        axe = fig.add_subplot(111)

        plotSpectrogram(spg=spg, axe=axe, find_peaks=2)

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestDSPSpectrogram()
    a.test_2d_peak()

    plt.show()
