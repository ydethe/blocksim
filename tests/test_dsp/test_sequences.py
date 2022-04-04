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

from blocksim_sigspace.dsp import createGoldSequence, createZadoffChu

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestSignal(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_zadoff_chu_crosscorr(self):
        s1 = createZadoffChu(name="s1", n_zc=1021, u=1, sampling_freq=1e6)
        s2 = createZadoffChu(name="s2", n_zc=1021, u=75, sampling_freq=1e6)

        y = s1.correlate(s2)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.grid(True)
        plotDSPLine(y, axe)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_zadoff_chu_autocorr(self):
        s1 = createZadoffChu(name="s1", n_zc=1021, u=1, sampling_freq=1e6)

        y = s1.correlate(s1)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.grid(True)
        plotDSPLine(y, axe)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_gold_crosscorr(self):
        s1 = createGoldSequence(
            name="s1", sv=[2, 6], repeat=1, chip_rate=1.023e6, sampling_factor=10
        )
        s2 = createGoldSequence(
            name="s2", sv=[3, 7], repeat=1, chip_rate=1.023e6, sampling_factor=10
        )

        y = s1.correlate(s2)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.grid(True)
        plotDSPLine(y, axe)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_gold_autocorr(self):
        s1 = createGoldSequence(
            name="s1", sv=[2, 6], repeat=1, chip_rate=1.023e6, sampling_factor=10
        )

        y = s1.correlate(s1)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.grid(True)
        plotDSPLine(y, axe)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_gold_corr_integ(self):
        # Reference Gold sequence
        y1 = createGoldSequence(
            name="s1", sv=[2, 6], repeat=1, chip_rate=1.023e6, sampling_factor=10
        )

        # Noisy received signal
        y = createGoldSequence(
            name="s1", sv=[2, 6], repeat=20, chip_rate=1.023e6, sampling_factor=10
        )
        y = y.applyGaussianNoise(pwr=200)

        # Correlation
        z = y.correlate(y1)

        # Integration
        zi = z.integrate(period=1e-3, offset=511 / (1.023e6))

        # Plotting
        fig = plt.figure()
        axe = fig.add_subplot(311)
        axe.grid(True)
        plotDSPLine(y, axe)
        axe.set_ylabel("Brut")

        axe = fig.add_subplot(312)
        axe.grid(True)
        plotDSPLine(z, axe)
        axe.set_ylabel("Corrélation")

        axe = fig.add_subplot(313)
        axe.grid(True)
        plotDSPLine(zi, axe)
        axe.set_ylabel("Intégration")

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestSignal()
    # a.test_zadoff_chu_crosscorr()
    # a.test_zadoff_chu_autocorr()
    a.test_gold_autocorr()
    a.test_gold_crosscorr()

    plt.show()
