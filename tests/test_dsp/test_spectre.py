import sys
from pathlib import Path
from typing import Any
import unittest

from nptyping import NDArray
import numpy as np
from numpy import pi, exp
import pytest

from blocksim.graphics.BFigure import FigureFactory
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.dsp.DSPSpectrum import RecursiveSpectrumEstimator
from blocksim.Simulation import Simulation

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


def generate_lin_fm(ns: int, fs: float, f1: float, f2: float) -> NDArray[Any, Any]:
    t = np.arange(ns) / fs
    tau = ns / fs
    x = exp(1j * (pi * t * (2 * f1 * tau + f2 * t - f1 * t)) / tau)
    return x


class TestSpectre(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_spectre(self):
        fs = 200
        f0 = 9.1
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1) + 2 * exp(1j * 2 * pi * 3 * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)
        sp = s1.fft(nfft=64)
        sp0 = s1.fft()

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])

        axe.plot(([f0, f0], [0, 4]), linestyle="--", color="black", label="Expected")
        axe.plot(([3 * f0, 3 * f0], [0, 4]), linestyle="--", color="black")
        axe.plot(sp, label="nfft=64")
        axe.plot(sp0, label="nfft=None")

        return fig.render()

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

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
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

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(sp)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_recursive_spectrum_est(self):
        fs = 20
        f1 = -5
        f2 = 2.5
        tau = 12

        ns = int(tau * fs)
        t = np.arange(ns) / fs
        x = np.zeros(3 * ns, dtype=np.complex128)
        x[ns : 2 * ns] = generate_lin_fm(ns, fs, f1, f2)
        xtps = np.arange(3 * ns) / fs - tau
        nfft = 64

        sig = DSPSignal.fromTimeAndSamples(name="sig", tps=xtps, y_serie=x)

        spe = RecursiveSpectrumEstimator(name="spe", dt=1 / fs, nfft=nfft)

        sim = Simulation()

        sim.addComputer(sig)
        sim.addComputer(spe)

        sim.connect("sig.setpoint", "spe.measurement")

        sim.simulate(sig.generateXSerie(), progress_bar=False)

        log = sim.getLogger()
        spg = spe.getSpectrogram(log)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(spg)
        axe.plot((t, (f2 - f1) / tau * t + f1), linestyle="--", color="white")

        return fig.render()


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    from blocksim.graphics import showFigures

    a = TestSpectre()
    # a.test_spectre()
    a.test_recursive_spectrum_est()

    showFigures()
