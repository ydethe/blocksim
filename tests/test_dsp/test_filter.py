from os import fstat
import sys
from pathlib import Path
from math import factorial
import unittest

import numpy as np
from numpy import pi, exp
import pytest

from blocksim.graphics.BFigure import FigureFactory
from blocksim.dsp import derivative_coeff, phase_unfold_deg
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
        bode = filt.bodeDiagram(name="bode")

        fig = FigureFactory.create()
        gs = fig.add_gridspec(2, 1)

        axe_amp = fig.add_baxe(title="Amplitude", spec=gs[0, 0])
        axe_amp.plot(bode, transform=bode.to_db_lim(-100))

        axe_pha = fig.add_baxe(title="Phase", spec=gs[1, 0], sharex=axe_amp)
        axe_pha.plot(bode, transform=phase_unfold_deg)

        return fig.render()

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
        y = y_sim.forceSamplingStart(-filt.getGroupDelay())

        y_direct = filt.apply(s1)
        diff = y_direct - y
        crop = diff.resample(samplingStart=0.05)
        err = np.max(np.abs(crop.y_serie))
        self.assertAlmostEqual(err, 0, delta=0.2)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(y, label="simu")
        axe.plot(s2, label="theoric")
        axe.plot(y_direct, label="direct")

        return fig.render()

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
        y = y.forceSamplingStart(y.generateXSerie(0) - filt.getGroupDelay())

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(y, transform=np.abs)

        return fig.render()

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
        # sig = sig.applyGaussianNoise(2e-3)

        tps = np.arange(ns) / fs
        freq = -bp / 2 + bp * tps / tau

        taps = derivative_coeff(rank=1, order=10)
        filt = ArbitraryDSPFilter(name="filt", samplingPeriod=1 / fs, num=taps * fs)

        psig = filt.apply(sig)
        res = -psig / sig / (2 * pi * 1j)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(res, transform=np.real)
        axe.plot(
            plottable=(
                {"data": tps, "unit": "s", "name": "Time"},
                {"data": freq, "unit": "Hz", "name": "Frequency"},
            )
        )

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_fir_design(self):
        fs = 10.0  # Hz
        desired = (0, 1, 0)
        bands = (0, 1.5, 2, 3, 3.5, 5)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(2, 1)
        axe_amp = fig.add_baxe(title="Amplitude", spec=gs[0, 0])
        axe_pha = fig.add_baxe(title="Phase", spec=gs[1, 0], sharex=axe_amp)

        for method in ["firwin2", "remez", "ls"]:
            filt = ArbitraryDSPFilter.fromFIRSpecification(
                name=method,
                fs=fs,
                numtaps=73,
                method=method,
                bands=bands,
                desired=desired,
            )
            bode = filt.bodeDiagram(name="bode")
            axe_amp.plot(bode, transform=bode.to_db_lim(-100), label=method)
            axe_pha.plot(bode, transform=phase_unfold_deg)

        axe_amp.plot(
            plottable=((2, 3), (0, 0)), linestyle="--", color="black", linewidth=2
        )

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_iir_design(self):
        from scipy.signal import dlti

        wp = 20
        ws = 30
        fs = 100
        gpass = 1
        gstop = 40

        filt = ArbitraryDSPFilter.fromIIRSpecification(
            name="filt", wp=wp, ws=ws, gpass=gpass, gstop=gstop, fs=fs
        )
        bode = filt.bodeDiagram(name="bode")
        num, den = filt.generateCoefficients()

        fig = FigureFactory.create()
        gs = fig.add_gridspec(2, 1)
        axe_amp = fig.add_baxe(title="Amplitude", spec=gs[0, 0])
        axe_pha = fig.add_baxe(title="Phase", spec=gs[1, 0], sharex=axe_amp)

        axe_amp.plot(bode, transform=bode.to_db_lim(-100))
        axe_pha.plot(bode, transform=phase_unfold_deg)

        sys = dlti(num, den, dt=1 / fs)
        w, mag, phase = sys.bode()
        axe_amp.plot(
            plottable=(w / (2 * pi), mag), label="scipy"
        )  # Bode magnitude plot
        axe_pha.plot(plottable=(w / (2 * pi), phase), label="scipy")  # Bode phase plot

        return fig.render()

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

        sys = filt.to_dlti()
        _, z_sp = sys.output(xn, t)

        sig = DSPSignal(name="sig", samplingStart=t[0], samplingPeriod=dt, y_serie=xn)

        z = filt.apply(sig)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(sig, label="noisy signal")
        axe.plot(z, label="filt.apply")
        axe.plot(plottable=(t, z_sp[:, 0]), label="dlti.output")

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_transfer_function(self):
        # Study of H(z) = (-2.z + 1)/(z^2 + 1)
        # dt = 1/100.
        from scipy.signal import dlti, dfreqresp

        dt = 1 / 100.0
        num = [1.4, -1.8, 1.4]
        den = [1, 0, 0]

        filt = ArbitraryDSPFilter(name="filt", samplingPeriod=dt, num=num, den=den)
        bode = filt.bodeDiagram(name="bode", fpoints=100)
        sys = dlti(num, den, dt=dt)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(2, 1)
        axe_amp = fig.add_baxe(title="Amplitude", spec=gs[0, 0])
        axe_pha = fig.add_baxe(title="Phase", spec=gs[1, 0], sharex=axe_amp)

        axe_amp.plot(bode, transform=bode.to_db_lim(-100))
        axe_pha.plot(bode, transform=phase_unfold_deg)

        w, mag, phase = sys.bode(n=100)
        axe_amp.plot(
            plottable=(w / (2 * pi), mag), label="scipy"
        )  # Bode magnitude plot
        axe_pha.plot(plottable=(w / (2 * pi), phase), label="scipy")  # Bode phase plot

        return fig.render()


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    from blocksim.graphics import showFigures

    a = TestFilter()
    # a.test_bode()
    # a.test_filtrage()
    # a.test_filtrage_chirp()
    # a.test_freq_estimator()
    # a.test_iir_design()
    a.test_iir_filter()
    # a.test_transfer_function()
    # a.test_fir_design()

    showFigures()
