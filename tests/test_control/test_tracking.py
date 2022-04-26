import sys
from pathlib import Path
import unittest

import numpy as np
from numpy import exp, pi, log10, sin, cos
import matplotlib.pyplot as plt
import pytest

from blocksim.loggers.Logger import Logger
from blocksim.control.Estimator import SteadyStateKalmanFilter, SpectrumEstimator
from blocksim.Simulation import Simulation
from blocksim.control.Route import IQExtract
from blocksim.graphics.FigureSpec import FigureSpec
from blocksim.graphics import plotSpectrogram, plotBode
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.dsp import phase_unfold

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


def generate_lin_fm(ns: int, fs: float, f1: float, f2: float) -> "array":
    t = np.arange(ns) / fs
    tau = ns / fs
    x = exp(1j * (pi * t * (2 * f1 * tau + f2 * t - f1 * t)) / tau)
    return x


class TestTrackingSteadyState(TestBase):
    def setUp(self):
        super().setUp()

        self.fs = 20
        self.dt = 1.0 / self.fs
        f1 = 3
        f2 = 10.0
        tau = 12
        self.tracks = np.arange(0, 20, 0.5) / self.fs
        ns = int(self.fs * tau)
        self.t = np.arange(ns) / self.fs
        x = generate_lin_fm(ns, self.fs, f1, f2)
        x[ns // 2 :] += exp(1j * 2 * pi * f2 * self.t[ns // 2 :])

        self.fchirp = (1 - self.t / tau) * f1 + self.t / tau * f2

        self.sig = (
            DSPSignal(
                name="sig", samplingStart=self.t[0], samplingPeriod=self.dt, y_serie=x
            )
            .resample(
                samplingStart=self.t[0] - 1,
                samplingPeriod=self.dt,
                samplingStop=self.t[-1] + 1,
            )
            .applyGaussianNoise(0.5)
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_tracking_steadystate_cplxe(self):
        nb_tracks = len(self.tracks)

        kal = SpectrumEstimator(
            name="kal",
            dt=self.dt,
            sname_output="x_kal",
            snames_state=["x_%i_est" % i for i in range(nb_tracks)],
            tracks=self.tracks * self.fs,
        )
        kal.matQ = np.eye(nb_tracks) / 10
        kal.matR = np.eye(1)

        sim = Simulation()

        sim.addComputer(self.sig)
        sim.addComputer(kal)

        sim.connect("sig.setpoint", "kal.measurement")

        sim.simulate(self.sig.generateXSerie(), progress_bar=False)

        K = kal.getConvergedGainMatrix()
        P = kal.getConvergedStateCovariance()
        ii = np.diag_indices(nb_tracks)

        self.assertAlmostEqual(np.max(np.abs(K - 1 / nb_tracks)), 0, delta=1e-2)
        self.assertAlmostEqual(np.max(np.abs(P[ii] - 1.97484567)), 0, delta=1e-7)

        log = sim.getLogger()
        spg = kal.getSpectrogram(log)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        plotSpectrogram(spg, axe)
        axe.plot(
            self.t,
            self.fchirp,
            linewidth=2,
            color="white",
            linestyle="--",
        )

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_bode_steadystate_cplxe(self):
        nb_tracks = len(self.tracks)

        kal = SpectrumEstimator(
            name="kal",
            dt=self.dt,
            sname_output="x_kal",
            snames_state=["x_%i_est" % i for i in range(nb_tracks)],
            tracks=self.tracks * self.fs,
        )
        kal.matQ = np.eye(nb_tracks) / 10
        kal.matR = np.eye(1)

        filt = kal.getEstimatingFilter("filt")

        fig = plt.figure()
        gs = fig.add_gridspec(2, 1)
        axe_amp, axe_pha = plotBode(
            filt, spec_amp=gs[0, 0], spec_pha=gs[1, 0], fpoints=np.arange(0, 20, 0.1)
        )

        return fig

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_bode3_steadystate_cplxe(self):
        nb_tracks = len(self.tracks)

        kal = SpectrumEstimator(
            name="kal",
            dt=self.dt,
            sname_output="x_kal",
            snames_state=["x_%i_est" % i for i in range(nb_tracks)],
            tracks=self.tracks * self.fs,
        )
        kal.matQ = np.eye(nb_tracks) / 10
        kal.matR = np.eye(1)

        filt = kal.getEstimatingFilter("filt", ma_freq=[3])

        fig = plt.figure()
        gs = fig.add_gridspec(2, 1)
        plotBode(
            filt, spec_amp=gs[0, 0], spec_pha=gs[1, 0], fpoints=np.arange(0, 20, 0.1)
        )

        return fig


if __name__ == "__main__":
    a = TestTrackingSteadyState()
    a.setUp()
    # a.test_tracking_steadystate_cplxe()
    # a.test_bode_steadystate_cplxe()
    a.test_bode3_steadystate_cplxe()

    plt.show()
