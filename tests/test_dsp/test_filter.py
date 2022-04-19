from os import fstat
import sys
from pathlib import Path
from math import factorial
import unittest

import numpy as np
from numpy import pi, exp
from scipy import linalg as lin
from scipy.signal import lfilter
from matplotlib import pyplot as plt
import pytest

from blocksim.graphics import plotBode, plotDSPLine
from blocksim.dsp import derivative_coeff
from blocksim.dsp.DSPFilter import ArbitraryDSPFilter, BandpassDSPFilter
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.Simulation import Simulation

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestFilter(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_bode(self):
        fs = 200
        f1 = 10
        f2 = 30
        filt = BandpassDSPFilter(
            name="filter",
            f_low=f1,
            f_high=f2,
            numtaps=256,
            samplingPeriod=1 / fs,
            win=("chebwin", -60),
        )

        fig = plt.figure()
        gs = fig.add_gridspec(2, 1)

        plotBode(filt, gs[0, 0], gs[1, 0])

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
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
        filt = BandpassDSPFilter(
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
        y_sim = DSPSignal.fromLogger(name="filt", log=log, param="filter_filt_sample")
        y = y_sim.forceSamplingStart(-filt.getTransientPhaseDuration())

        y_direct = filt.apply(s1)
        diff = y_direct - y
        crop = diff.resample(samplingStart=0.05)
        err = np.max(np.abs(crop.y_serie))
        self.assertAlmostEqual(err, 0, delta=0.2)

        fig = plt.figure()
        axe = fig.add_subplot(111)

        plotDSPLine(y, axe, label="simu")
        plotDSPLine(s2, axe, label="theoric")
        plotDSPLine(y_direct, axe, label="direct")
        axe.legend(loc="best")

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_filtrage_chirp(self):
        sim = Simulation()

        fs = 200
        bp = 40
        ns = 200
        tau = ns / fs
        t1 = np.arange(ns) / fs
        s1 = DSPSignal.fromLinearFM(
            name="s1",
            samplingStart=0,
            samplingPeriod=1 / fs,
            tau=tau,
            fstart=-bp / 2,
            fend=bp / 2,
        )
        s2 = s1.resample(
            samplingStart=-1, samplingPeriod=1 / fs, samplingStop=s1.samplingStop + 1
        )
        s3 = s2.applyDopplerFrequency(fdop=50)
        sim.addComputer(s3)

        f1 = 47
        f2 = 53
        filt = BandpassDSPFilter(
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
        y = y.forceSamplingStart(y.generateXSerie(0) - filt.getTransientPhaseDuration())

        fig = plt.figure()
        axe = fig.add_subplot(111)

        plotDSPLine(y, axe, transform=np.abs)

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_freq_estimator(self):
        fs = 200
        bp = 40
        ns = 200
        tau = ns / fs
        s1 = DSPSignal.fromLinearFM(
            name="s1",
            samplingStart=0,
            samplingPeriod=1 / fs,
            tau=tau,
            fstart=-bp / 2,
            fend=bp / 2,
        )
        sig = s1.resample(
            samplingStart=-1, samplingPeriod=1 / fs, samplingStop=s1.samplingStop + 1
        )
        # sig=sig.applyGaussianNoise(0.1)

        tps = np.arange(ns) / fs
        freq = -bp / 2 + bp * tps / tau

        fig = plt.figure()
        gs = fig.add_gridspec(1, 1)

        taps = derivative_coeff(rank=1, order=10)
        filt = ArbitraryDSPFilter(name="filt", samplingPeriod=1 / fs, btaps=taps * fs)

        psig = filt.apply(sig)
        res = -psig / sig / (2 * pi * 1j)
        axe = plotDSPLine(line=res, spec=gs[0, 0], transform=np.real)
        axe.plot(tps, freq)
        axe.set_ylabel("Frequency (Hz)")

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_iir_design(self):
        wp = 20
        ws = 30
        fs = 100
        gpass = 1
        gstop = 40

        fig = plt.figure()
        gs = fig.add_gridspec(2, 1)

        filt = ArbitraryDSPFilter.fromIIRSpecification(
            name="filt", wp=wp, ws=ws, gpass=gpass, gstop=gstop, fs=fs
        )
        plotBode(filt, spec_amp=gs[0, 0], spec_pha=gs[1, 0])

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_iir_filter(self):
        wp = 20
        ws = 30
        fs = 100
        gpass = 1
        gstop = 40

        filt = ArbitraryDSPFilter.fromIIRSpecification(
            name="filt", wp=wp, ws=ws, gpass=gpass, gstop=gstop, fs=fs
        )

        t = np.linspace(0, 2, 201)
        dt = t[1] - t[0]
        x = (
            np.sin(2 * np.pi * 0.75 * t * (1 - t) + 2.1)
            + 0.1 * np.sin(2 * np.pi * 1.25 * t + 1)
            + 0.18 * np.cos(2 * np.pi * 3.85 * t)
        )
        xn = x + np.random.normal(size=len(t)) * 0.08

        sig = DSPSignal(name="sig", samplingStart=t[0], samplingPeriod=dt, y_serie=xn)

        z = filt.apply(sig)

        fig = plt.figure()
        gs = fig.add_gridspec(1, 1)

        axe = plotDSPLine(sig, spec=gs[0, 0])
        plotDSPLine(z, spec=axe)
        axe.legend(("noisy signal", "filt.apply"), loc="best")

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestFilter()
    # a.test_bode()
    # a.test_filtrage()
    # a.test_filtrage_chirp()
    # a.test_phase_estimator()
    # a.test_iir_design()
    a.test_iir_filter()

    plt.show()
