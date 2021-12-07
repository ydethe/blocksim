import sys
from pathlib import Path
import unittest

import numpy as np
from numpy import pi, exp
from matplotlib import pyplot as plt
import pytest

from blocksim.graphics import plotDSPLine
from blocksim.Logger import Logger
from blocksim.dsp.DSPSignal import DSPSignal

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestSignal(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_correlation(self):
        fs = 20e6
        bp = fs / 5
        tau = 10e-6
        n = int(np.ceil(fs * tau))
        tps = np.arange(n) / fs

        pha = bp * tps * (tps - tau) / (2 * tau)
        x = np.exp(1j * np.pi * 2 * pha)
        y = np.hstack((np.zeros(n // 2), x, np.zeros(2 * n)))

        # print("Pic de corrélation attendu à %.3f µs" % (n / fs / 2 * 1e6))

        rep = DSPSignal("rep", samplingStart=0, samplingPeriod=3 / fs, y_serie=x[::3])
        s = DSPSignal("s", samplingStart=-1e-3, samplingPeriod=1 / fs, y_serie=y)

        y = rep.correlate(rep)
        y1 = rep.correlate(s)
        y2 = s.correlate(rep)

        z = y2 - y1
        self.assertAlmostEqual(np.max(np.abs(z)), 0, delta=1e-8)

        fig = plt.figure()
        axe = fig.add_subplot(111)

        plotDSPLine(y1, axe, find_peaks=1, x_unit_mult=1e-6, linewidth=3, color="red")
        plotDSPLine(y2, axe, x_unit_mult=1e-6, color="black")

        return fig

    def test_convolution(self):
        fs = 20e6
        bp = fs / 5
        tau = 10e-6
        n = int(np.ceil(fs * tau))
        tps = np.arange(n) / fs

        pha = bp * tps * (tps - tau) / (2 * tau)
        x = np.exp(1j * np.pi * 2 * pha)
        y = np.hstack((np.zeros(n // 2), x, np.zeros(2 * n)))

        rep = DSPSignal("rep", samplingStart=0, samplingPeriod=3 / fs, y_serie=x[::3])
        s = DSPSignal("s", samplingStart=-1e-3, samplingPeriod=1 / fs, y_serie=y)

        y1 = s.correlate(rep)

        rep_conv = rep.reverse().conj()
        y3 = s.convolve(rep_conv)
        y4 = s @ rep_conv

        z = y3 - y1
        self.assertAlmostEqual(np.max(np.abs(z)), 0, delta=1e-8)

        z = y4 - y1
        self.assertAlmostEqual(np.max(np.abs(z)), 0, delta=1e-8)

    def test_phase_unfold(self):
        fs = 200
        f0 = fs / 10
        ns = 200

        tps = np.arange(ns) / fs

        pha_ref = 2 * pi * f0 * tps + pi / 2

        x = np.exp(1j * pha_ref)
        sig = DSPSignal(
            name="sig", samplingStart=tps[0], samplingPeriod=1 / fs, y_serie=x
        )

        pha = sig.getUnfoldedPhase()

        z = pha - pha_ref

        self.assertAlmostEqual(np.max(np.abs(z)), 0, delta=1e-10)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_resample(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)
        t1 = np.arange(ns) / fs

        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        s2 = s1.resample(
            samplingStart=-2.3 / fs,
            samplingPeriod=0.5 / fs,
            samplingStop=s1.samplingStop + 2 / fs,
        )

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.grid(True)

        plotDSPLine(s1, axe, linestyle="--", marker="x", label="orig.")
        tref = np.arange(int(50 * fs / f0)) / (50 * fs)
        axe.plot(tref * 1000, np.cos(2 * pi * f0 * tref))
        plotDSPLine(s2, axe, linestyle="--", marker="+", label="oodsp")
        axe.legend()

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_delay(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.grid(True)
        plotDSPLine(s1.delay(1 / 60), axe)
        plotDSPLine(s1, axe)

        return fig

    def test_from_logger(self):
        log = Logger()

        dt = 0.01
        f = 11
        ns = 1000

        for i in range(ns):
            log.log("t", i * dt)
            log.log("x", np.sin(i * dt * f * 2 * np.pi + 1))

        sig = DSPSignal.fromLogger(name="sin", log=log, param="x")
        err = np.max(np.abs(sig.y_serie - log.getValue("x")))

        self.assertAlmostEqual(err, 0, delta=1e-9)


if __name__ == "__main__":
    # unittest.main()

    a = TestSignal()
    a.test_resample()

    plt.show()
