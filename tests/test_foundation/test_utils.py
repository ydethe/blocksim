import sys
from pathlib import Path
import unittest

import numpy as np
from numpy import pi, exp
from matplotlib import pyplot as plt
import pytest

from blocksim.dsp import phase_unfold

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestUtils(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
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

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.plot(tps * 1e6, pha * 180 / np.pi)
        axe.grid(True)
        axe.set_xlabel("Time (µs)")

        return fig

    def test_phase_unfold_odd(self):
        y = np.zeros(10, dtype=np.complex128)
        pha = phase_unfold(y)
        self.assertAlmostEqual(np.max(np.abs(pha)), 0, delta=1e-10)

        y[-1] = np.exp(1j * pi / 4)
        pha = phase_unfold(y)
        self.assertAlmostEqual(np.max(np.abs(pha - pi / 4)), 0, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
