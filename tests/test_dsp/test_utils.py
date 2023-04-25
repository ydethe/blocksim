import numpy as np
from numpy import pi
import pytest

from blocksim.graphics.BFigure import FigureFactory
from blocksim.dsp import phase_unfold


from blocksim.testing import TestBase


class TestUtils(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_phase_unfold(self):
        fs = 20e6
        bp = fs / 5
        tau = 10e-6
        n = int(np.ceil(fs * tau))
        tps = np.arange(n) / fs

        pha = bp * tps * (tps - tau) / (2 * tau)
        x = np.exp(1j * pi * 2 * pha + 1j * pi / 4)
        y = np.hstack((np.zeros(n // 2), x, np.zeros(2 * n)))
        tps = np.arange(len(y)) / fs

        pha = phase_unfold(y)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(
            plottable=(
                {"data": tps, "name": "Time", "unit": "s"},
                {"data": pha * 180 / np.pi, "unit": "deg", "name": "Phase"},
            )
        )

        return fig.render()

    def test_phase_unfold_odd(self):
        y = np.zeros(10, dtype=np.complex128)
        pha = phase_unfold(y)
        self.assertAlmostEqual(np.max(np.abs(pha)), 0, delta=1e-10)

        y[-1] = np.exp(1j * pi / 4)
        pha = phase_unfold(y)
        self.assertAlmostEqual(np.max(np.abs(pha - pi / 4)), 0, delta=1e-10)


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    from blocksim.graphics import showFigures

    a = TestUtils()
    a.test_phase_unfold()

    showFigures()
