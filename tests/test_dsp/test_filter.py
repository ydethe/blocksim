import os
import sys
import unittest

import numpy as np
from numpy import pi, exp
from matplotlib import pyplot as plt
import pytest


sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from blocksim.Graphics import plotBode, plotDSPLine
from blocksim.dsp.DSPFilter import DSPFilter
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.Simulation import Simulation


class TestFilter(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_bode(self):
        fs = 200
        f1 = 10
        f2 = 30
        filt = DSPFilter(
            name="filter",
            f_low=f1,
            f_high=f2,
            numtaps=256,
            samplingPeriod=1 / fs,
            win=("chebwin", -60),
        )

        fig = plt.figure()
        axe_amp = fig.add_subplot(211)
        axe_pha = fig.add_subplot(212, sharex=axe_amp)

        plotBode(filt, axe_amp, axe_pha)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_filtrage(self):
        sim = Simulation()

        fs = 200
        f0 = 20
        ns = 200

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1) + exp(1j * 2 * pi * 3 * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        sim.addComputer(s1)

        x2 = exp(1j * 2 * pi * f0 * t1)
        s2 = DSPSignal(name="s2", samplingStart=0, samplingPeriod=1 / fs, y_serie=x2)

        f1 = 10
        f2 = 30
        filt = DSPFilter(
            name="filter",
            f_low=f1,
            f_high=f2,
            numtaps=256,
            samplingPeriod=1 / fs,
            win=("chebwin", -60),
        )

        sim.addComputer(filt)

        sim.connect("s1.setpoint", "filter.unfilt")

        tps = s1.generateXSerie()

        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()
        y = DSPSignal.fromLogger(name="filt", log=log, param="filter_filt_sample")
        y = y.forceSamplingStart(-filt.getTransientPhaseDuration())

        fig = plt.figure()
        axe = fig.add_subplot(111)

        plotDSPLine(y, axe)
        plotDSPLine(s2, axe)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_filtrage_chirp(self):
        sim = Simulation()

        fs = 200
        bp = 40
        ns = 200
        tau = ns / fs
        t1 = np.arange(ns) / fs
        x1 = exp(1j * pi * t1 * bp * (t1 / tau - 1))
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)
        s2 = s1.resample(
            samplingStart=-1, samplingPeriod=1 / fs, samplingStop=s1.samplingStop + 1
        )
        s3 = s2.applyDopplerFrequency(fdop=50)
        sim.addComputer(s3)

        f1 = 47
        f2 = 53
        filt = DSPFilter(
            name="filter",
            f_low=f1,
            f_high=f2,
            samplingPeriod=1 / fs,
            numtaps=256,
            win=("chebwin", -60),
        )

        sim.addComputer(filt)

        sim.connect("s1.setpoint", "filter.unfilt")

        tps = s3.generateXSerie()
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()
        y = DSPSignal.fromLogger(name="filt", log=log, param="filter_filt_sample")
        y = y.forceSamplingStart(-filt.getTransientPhaseDuration())

        fig = plt.figure()
        axe = fig.add_subplot(111)

        plotDSPLine(y, axe, transform=np.abs)

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestFilter()
    # a.test_bode()
    # a.test_filtrage()
    a.test_filtrage_chirp()

    plt.show()