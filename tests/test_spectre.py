import os
import sys
import unittest

import numpy as np
from numpy import pi, exp
from matplotlib import pyplot as plt
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from blocksim.dsp.DSPSpectrum import DSPSpectrum
from blocksim.dsp.DSPSignal import DSPSignal


class TestSpectre(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_spectre(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1) + 2 * exp(1j * 2 * pi * 3 * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        fig = plt.figure()
        axe = fig.add_subplot(111)

        sp = s1.fft()
        sp.plot(axe)

        return fig

    def test_fft(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1) + 2 * exp(1j * 2 * pi * 3 * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        sp = s1.fft()
        s2 = sp.ifft()

        self.assertAlmostEqual(np.max(np.abs(s1 - s2)), 0, delta=1e-8)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_doppler(self):
        fs = 300
        f0 = 10
        fdop = 20
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        pha = 2 * pi * f0 * t1
        s1 = DSPSignal.fromPhaseLaw(name="s1", sampling_freq=fs, pha=pha)

        s2 = s1.applyDopplerFrequency(fdop=fdop)
        sp = s2.fft()

        ech = sp.getSample(f0 + fdop)
        self.assertAlmostEqual(ech, 1, delta=1e-8)

        fig = plt.figure()
        axe = fig.add_subplot(111)

        sp.plot(axe)

        return fig


if __name__ == "__main__":
    unittest.main()
