import numpy as np
from numpy import exp, pi
import pytest

from blocksim.control.Estimator import KalmanSpectrumEstimator
from blocksim.Simulation import Simulation
from blocksim.dsp import phase_unfold_deg
from blocksim.graphics.BFigure import FigureFactory
from blocksim.dsp.DSPSignal import DSPSignal


from blocksim.testing import TestBase
from blocksim.utils import FloatArr


def generate_lin_fm(ns: int, fs: float, f1: float, f2: float) -> FloatArr:
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
            DSPSignal(name="sig", samplingStart=self.t[0], samplingPeriod=self.dt, y_serie=x)
            .resample(
                samplingStart=self.t[0] - 1,
                samplingPeriod=self.dt,
                samplingStop=self.t[-1] + 1,
            )
            .applyGaussianNoise(0.5)
        )

    @pytest.mark.mpl_image_compare(tolerance=10, savefig_kwargs={"dpi": 150})
    def test_tracking_steadystate_cplxe(self):
        nb_tracks = len(self.tracks)

        kal = KalmanSpectrumEstimator(
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

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(spg)
        axe.plot(
            plottable=(self.t, self.fchirp),
            linewidth=2,
            color="white",
            linestyle="--",
        )

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=10, savefig_kwargs={"dpi": 150})
    def test_bode_steadystate_cplxe(self):
        nb_tracks = len(self.tracks)

        kal = KalmanSpectrumEstimator(
            name="kal",
            dt=self.dt,
            sname_output="x_kal",
            snames_state=["x_%i_est" % i for i in range(nb_tracks)],
            tracks=self.tracks * self.fs,
        )
        kal.matQ = np.eye(nb_tracks) / 10
        kal.matR = np.eye(1)

        filt = kal.getEstimatingFilter("filt")
        bode = filt.bodeDiagram(name="bode", fpoints=np.arange(0, 20, 0.1))

        fig = FigureFactory.create()
        gs = fig.add_gridspec(2, 1)

        axe_amp = fig.add_baxe(title="Amplitude", spec=gs[0, 0])
        axe_amp.plot(bode, transform=bode.to_db_lim(-100))

        axe_pha = fig.add_baxe(title="Phase", spec=gs[1, 0], sharex=axe_amp)
        axe_pha.plot(bode, transform=phase_unfold_deg)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=30, savefig_kwargs={"dpi": 150})
    def ntest_bode3_steadystate_cplxe(self):
        nb_tracks = len(self.tracks)

        kal = KalmanSpectrumEstimator(
            name="kal",
            dt=self.dt,
            sname_output="x_kal",
            snames_state=["x_%i_est" % i for i in range(nb_tracks)],
            tracks=self.tracks * self.fs,
        )
        kal.matQ = np.eye(nb_tracks) / 10
        kal.matR = np.eye(1)

        filt = kal.getEstimatingFilter("filt", ma_freq=[3])
        fpoints = np.arange(0, 20, 0.1)
        bode = filt.bodeDiagram(name="bode", fpoints=fpoints)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(2, 1)

        axe_amp = fig.add_baxe(title="Amplitude", spec=gs[0, 0])
        axe_amp.plot(bode, transform=bode.to_db_lim(-40))

        axe_pha = fig.add_baxe(title="Phase", spec=gs[1, 0], sharex=axe_amp)
        axe_pha.plot(bode, transform=phase_unfold_deg)

        #### TMP###
        # Check with scipy functions

        from scipy.signal import StateSpace

        matK = kal.getOutputByName("matK")
        matK.resetCallback(None)
        K = kal.getConvergedGainMatrix()

        n = len(kal.tracks)

        Ad, Bd, Cd, Dd = kal.discretize(kal.dt)
        Cf = np.zeros((1, n))
        Cf[0, 3] = 1.0
        sss = StateSpace(Ad - K @ Cd, K, Cf, Dd, dt=kal.dt)
        sss = sss.to_tf()
        _, mag, phase = sss.bode(w=fpoints * 2 * pi * self.dt)
        axe_pha.plot(plottable=(fpoints, phase), label="scipy bode")

        #### TMP###

        return fig.render()


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    from blocksim.graphics import showFigures

    a = TestTrackingSteadyState()
    a.setUp()
    a.test_tracking_steadystate_cplxe()
    # a.test_bode_steadystate_cplxe()
    # a.test_bode3_steadystate_cplxe()

    showFigures()
