import sys
from pathlib import Path
import unittest

import pytest
import numpy as np
from numpy import log10, sqrt, exp, pi, cos
from matplotlib import pyplot as plt

from blocksim import logger

from blocksim.dsp.DelayLine import InfiniteDelayLine, FiniteDelayLine

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestDelayLine(TestBase):
    def test_infinite_dl(self):
        dl = InfiniteDelayLine()

        f0 = 1.0
        fs = 10.0
        ns = 50
        tau = 1.2

        tps = np.arange(ns) / fs
        res = np.empty(ns)
        for k, t in enumerate(tps):
            t = k / fs
            z = exp(2 * pi * 1j * f0 * t)
            dl.addSample(t, z)
            zd = dl.getDelayedSample(tau)
            res[k] = np.real(zd)

        nd = int(tau * fs)
        diff = res[nd:] - cos(2 * pi * f0 * tps[:-nd])
        err = np.max(np.abs(diff))

        self.assertAlmostEqual(err, 0, delta=1e-13)

    def test_finite_dl(self):
        dl = FiniteDelayLine(size=100, dtype=np.complex128)

        f0 = 1.0
        fs = 10.0
        ns = 50
        tau = 1.2

        tps = np.arange(ns) / fs
        res = np.empty(ns)
        for k, t in enumerate(tps):
            t = k / fs
            z = exp(2 * pi * 1j * f0 * t)
            dl.addSample(t, z)
            zd = dl.getDelayedSample(tau)
            res[k] = np.real(zd)

        nd = int(tau * fs)
        diff = res[nd:] - cos(2 * pi * f0 * tps[:-nd])
        err = np.max(np.abs(diff))

        self.assertAlmostEqual(err, 0, delta=1e-13)


if __name__ == "__main__":
    a = TestDelayLine()
    a.setUp()
    # a.test_infinite_dl()
    a.test_finite_dl()
