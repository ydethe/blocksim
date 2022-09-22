import sys
from pathlib import Path
import unittest

import numpy as np
from numpy.testing import assert_array_almost_equal_nulp
from numpy import pi, exp

from blocksim.dsp.DSPSignal import DSPSignal

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestSignalOperations(TestBase):
    def test_energy(self):
        tau = 10e-6
        bp = 5e6
        fs = bp * 3
        eta = 0.1
        nrep = 50
        fdop = 1e3

        rep = DSPSignal.fromLinearFM(
            name="rep",
            samplingStart=0,
            samplingPeriod=1 / fs,
            tau=tau,
            fstart=-bp / 2,
            fend=bp / 2,
        )

        Et = rep.energy
        srep = rep.fft()

        Ef = srep.energy

        self.assertAlmostEqual(Et, Ef, delta=1e-12)
        self.assertAlmostEqual(Et, 150.0, delta=1e-12)

    def test_truncate(self):
        sig = DSPSignal(name="sig", samplingStart=0, samplingPeriod=1, y_serie=np.arange(10))
        sig2 = sig.truncate(samplingStart=-1.2, samplingStop=10.0, zero_padding=True)

        x_ref = np.arange(-1, 11)
        assert_array_almost_equal_nulp(sig2.generateXSerie(), x_ref)

        y_ref = x_ref.copy()
        y_ref[:2] = 0
        y_ref[-1] = 0
        y_ref[1:11] = np.arange(10)
        assert_array_almost_equal_nulp(sig2.y_serie, y_ref)

    def test_resample(self):
        sig = DSPSignal(name="sig", samplingStart=0, samplingPeriod=1, y_serie=np.arange(10))
        sig2 = sig.resample(samplingStart=-1.2, samplingStop=10.8)

        x_ref = np.arange(13) - 1.2
        assert_array_almost_equal_nulp(sig2.generateXSerie(), x_ref)

        y_ref = x_ref.copy()
        y_ref[:2] = 0
        y_ref[-2:] = 0
        y_ref[2:11] = np.arange(9) + 0.8
        assert_array_almost_equal_nulp(sig2.y_serie, y_ref, nulp=2)

    def test_mul(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=t1[0], samplingPeriod=1 / fs, y_serie=x1)

        t2 = -1 / fs + np.arange(3 * ns - 3) / (3 * fs)
        x2 = exp(-1j * 2 * pi * f0 * t2)
        s2 = DSPSignal(name="s2", samplingStart=t2[0], samplingPeriod=1 / (3 * fs), y_serie=x2)

        z = s1 * s2

        self.assertAlmostEqual(np.max(np.abs(z.y_serie[3:-4] - 1)), 0, delta=1.5e-2)

    def test_smul(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        z = s1 * 2

        self.assertAlmostEqual(np.max(np.abs(z.y_serie - 2 * s1.y_serie)), 0, delta=1e-3)

        z2 = 3 * s1

        self.assertAlmostEqual(np.max(np.abs(z2.y_serie - 3 * s1.y_serie)), 0, delta=1e-3)

    def test_add(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        t2 = np.arange(3 * ns - 2) / (3 * fs)
        x2 = exp(-1j * 2 * pi * f0 * t2)
        s2 = DSPSignal(name="s2", samplingStart=0, samplingPeriod=1 / (3 * fs), y_serie=x2)

        z = s1 + s2
        z_ref = 2 * np.cos(2 * pi * f0 * t2)

        self.assertAlmostEqual(np.max(np.abs(z.y_serie - z_ref)), 0, delta=1.5e-2)

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
        s2 = DSPSignal(name="s2", samplingStart=0, samplingPeriod=1 / (3 * fs), y_serie=x2)

        z = s1 - s2
        z_ref = 2 * 1j * np.sin(2 * pi * f0 * t2)

        self.assertAlmostEqual(np.max(np.abs(z.y_serie - z_ref)), 0, delta=1.5e-2)

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
    a = TestSignalOperations()
    # a.test_mul()
    # a.test_add()
    a.test_sub()
