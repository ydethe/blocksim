import os
import sys
import unittest

import numpy as np
from numpy import pi, exp
from matplotlib import pyplot as plt
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from blocksim.dsp.DSPSignal import DSPSignal


class TestSignal(TestBase):
    def test_mul(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        t2 = 0.3 / fs + np.arange(3 * ns) / (3 * fs)
        x2 = exp(-1j * 2 * pi * f0 * t2)
        s2 = DSPSignal(
            name="s2", samplingStart=0.3 / fs, samplingPeriod=1 / (3 * fs), y_serie=x2
        )

        z = s1 * s2

        self.assertAlmostEqual(np.max(np.abs(z.y_serie[1:-3] - 1)), 0, delta=1e-3)

    def test_smul(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        z = s1 * 2

        self.assertAlmostEqual(
            np.max(np.abs(z.y_serie - 2 * s1.y_serie)), 0, delta=1e-3
        )

        z2 = 3 * s1

        self.assertAlmostEqual(
            np.max(np.abs(z2.y_serie - 3 * s1.y_serie)), 0, delta=1e-3
        )

    def test_add(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        t2 = np.arange(3 * ns - 2) / (3 * fs)
        x2 = exp(-1j * 2 * pi * f0 * t2)
        s2 = DSPSignal(
            name="s2", samplingStart=0, samplingPeriod=1 / (3 * fs), y_serie=x2
        )

        z = s1 + s2
        z_ref = 2 * np.cos(2 * pi * f0 * t2)

        self.assertAlmostEqual(np.max(np.abs(z.y_serie - z_ref)), 0, delta=1e-3)

    def test_sadd(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        z = s1 + 2

        self.assertAlmostEqual(np.max(np.abs(z.y_serie - x1 - 2)), 0, delta=1e-3)

        z2 = 3 + s1

        self.assertAlmostEqual(np.max(np.abs(z2.y_serie - x1 - 3)), 0, delta=1e-3)

    def test_sub(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        t2 = np.arange(3 * ns - 2) / (3 * fs)
        x2 = exp(-1j * 2 * pi * f0 * t2)
        s2 = DSPSignal(
            name="s2", samplingStart=0, samplingPeriod=1 / (3 * fs), y_serie=x2
        )

        z = s1 - s2
        z_ref = 2 * 1j * np.sin(2 * pi * f0 * t2)

        self.assertAlmostEqual(np.max(np.abs(z.y_serie - z_ref)), 0, delta=1e-3)

    def test_ssub(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        z = s1 - 2

        self.assertAlmostEqual(np.max(np.abs(z.y_serie - x1 + 2)), 0, delta=1e-3)

        z2 = -3 + s1

        self.assertAlmostEqual(np.max(np.abs(z2.y_serie - x1 + 3)), 0, delta=1e-3)


if __name__ == "__main__":
    unittest.main()
