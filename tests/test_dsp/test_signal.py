import numpy as np
from numpy import testing, pi, exp
import pytest

from blocksim.dsp.DSPLine import DSPRectilinearLine
from blocksim.graphics.BFigure import FigureFactory
from blocksim.loggers.Logger import Logger
from blocksim.dsp.DSPSignal import DSPSignal


from blocksim.testing import TestBase


class TestSignal(TestBase):
    def test_repeat_to_fit(self):
        sig = DSPRectilinearLine(
            name="sig", samplingStart=-5, samplingPeriod=1, y_serie=np.zeros(20)
        )
        rep = DSPRectilinearLine(
            name="rep", samplingStart=0, samplingPeriod=1, y_serie=np.arange(6)
        )

        rep2: DSPRectilinearLine = rep.repeatToFit(sig)

        actual = rep2.y_serie
        desired = np.array(
            [
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                0.0,
                1.0,
                2.0,
                3.0,
                4.0,
                5.0,
                0.0,
                1.0,
                2.0,
            ]
        )

        testing.assert_allclose(actual, desired)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_decimate(self):
        fs = 192 * 203
        f0 = 9.1
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1) + 2 * exp(1j * 2 * pi * 3 * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)
        s2 = s1.decimate(192)  # type: DSPSignal
        sp = s2.fft()
        print(1 / s2.samplingPeriod)

        print(sp.findPeaksWithTransform(nb_peaks=2))

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])

        axe.plot(sp)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
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

        rep.autoCorrelation()

        y = rep.correlate(rep)
        y1 = rep.correlate(s)
        y1.setDefaultTransform(y1.to_db_lim(-100), unit_of_y_var="dB")

        y2 = s.correlate(rep)
        y2.setDefaultTransform(y2.to_db_lim(-100), unit_of_y_var="dB")

        z = y2 - y1
        self.assertAlmostEqual(np.max(np.abs(z)), 0, delta=1e-8)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])

        axe.plot(y1, find_peaks=1, linewidth=3, color="red")
        axe.plot(y2, color="black")

        return fig.render()

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
        sig = DSPSignal(name="sig", samplingStart=tps[0], samplingPeriod=1 / fs, y_serie=x)

        pha = sig.getUnfoldedPhase()

        z = pha - pha_ref

        self.assertAlmostEqual(np.max(np.abs(z)), 0, delta=1e-10)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
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

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])

        axe.plot(s1, linestyle="--", marker="x", label="orig.")
        tref = np.arange(int(50 * fs / f0)) / (50 * fs)
        axe.plot(plottable=(tref * 1000, np.cos(2 * pi * f0 * tref)))
        axe.plot(s2, linestyle="--", marker="+", label="oodsp")

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_delay(self):
        fs = 200
        f0 = 10
        ns = int(fs / f0)

        t1 = np.arange(ns) / fs
        x1 = exp(1j * 2 * pi * f0 * t1)
        s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(s1.delay(1 / 60))
        axe.plot(s1)

        return fig.render()

    def test_from_logger(self):
        log = Logger()

        dt = 0.01
        f = 11
        ns = 1000

        for i in range(ns):
            log.log(name="x", val=np.sin(i * dt * f * 2 * np.pi + 1), unit="")
            log.log(name="t", val=i * dt, unit="s")

        sig = DSPSignal.fromLogger(name="sin", log=log, param="x")
        err = np.max(np.abs(sig.y_serie - log.getValue("x")))

        self.assertAlmostEqual(err, 0, delta=1e-9)

    def test_instanciation(self):
        # Unevenly spaced time samples
        tps = np.array([0, 2, 3, 4])
        self.assertRaises(
            ValueError,
            DSPSignal.fromTimeAndSamples,
            name="sig",
            tps=tps,
            y_serie=np.zeros_like(tps),
        )

        sig = DSPSignal.fromBinaryRandom(name="sig", samplingPeriod=0.1, size=10, seed=14887)
        testing.assert_equal(sig.y_serie, [0, 1, 0, 0, 0, 1, 0, 0, 0, 0])
        self.assertEqual(sig.energy, 2)

        self.assertFalse(sig.hasOutputComplex)

    def test_polyfit(self):
        a = 2 - 3j
        b = -4 + 1j
        x = np.arange(10)
        y = a * x + b
        line = DSPRectilinearLine(
            name="line", samplingStart=x[0], samplingPeriod=x[1] - x[0], y_serie=y
        )
        p = line.polyfit(deg=1)
        err = np.max(np.abs(y - p(x)))
        self.assertAlmostEqual(err, 0, delta=1e-10)

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 150})
    def test_superheterodyne(self):
        fs = 1000.0
        B = 100
        tau = 0.2
        fc = 2e2
        ntau = int(tau * fs)
        ns = 3 * ntau
        tps = np.arange(ns) / fs
        x = np.zeros(ns)
        x[ntau : 2 * ntau] = np.cos(
            2 * pi * fc * tps[:ntau] + pi * B * tps[:ntau] * (tps[:ntau] - tau) / tau
        )
        re = DSPSignal(name="re", samplingStart=0, samplingPeriod=1 / fs, y_serie=x)
        cp = re.superheterodyneIQ(carrier_freq=fc, bandwidth=B * 1.2)
        rep = DSPSignal.fromLinearFM(
            name="rep",
            samplingStart=0,
            samplingPeriod=1 / fs,
            tau=tau,
            fstart=-B / 2,
            fend=B / 2,
        )

        fig = FigureFactory.create()
        gs = fig.add_gridspec(2, 1)

        axe = fig.add_baxe(title="Linear Power of BB signal", spec=gs[0, 0])
        axe.plot(cp, transform=lambda x: np.abs(x) ** 2)

        axe = fig.add_baxe(title="Correlated signal", spec=gs[1, 0])
        axe.plot(
            cp.correlate(rep, win="hamming"),
            transform=DSPSignal.to_db_lim(-80),
        )

        return fig.render()


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    a = TestSignal()
    a.test_repeat_to_fit()
    # a.test_instanciation()
    # a.test_resample()
    # a.test_polyfit()
    # a.test_superheterodyne()
    # a.test_correlation()
    # a.test_delay()
    # a.test_phase_unfold()
    # a.test_decimate()

    # showFigures()
