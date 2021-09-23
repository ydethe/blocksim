r"""
Frequency tracking
==================
This example shows how to use a Kalman filter to estimate the frequencies
present in a signal.

"""
import numpy as np


fs = 20
dt = 1.0 / fs
f1 = 3
f2 = 10
tau = 12

###############################################################################
# Generation of a chirp with :class:`blocksim.dsp.DSPSignal.DSPSignal.fromLinearFM`

from blocksim.dsp.DSPSignal import DSPSignal

chirp = DSPSignal.fromLinearFM(
    "sig", samplingStart=0, samplingPeriod=dt, tau=tau, fstart=f1, fend=f2
)
t = chirp.generateXSerie()
fchirp = (1 - t / tau) * f1 + t / tau * f2

###############################################################################
# Generation of a jammer after tau/2
# The jammer emits a pure sinusoid at frequency f2. So this is a special case
# of a chirp with zero bandwith
jammer = DSPSignal.fromLinearFM(
    "jam", samplingStart=tau / 2, samplingPeriod=dt, tau=tau / 2, fstart=f2, fend=f2
)

###############################################################################
# The test signal is the superposition of the chirp and the jammer,
# wiht extra margins,
# and with some noise added
sig = (
    (chirp + jammer)
    .resample(samplingStart=t[0] - 1, samplingPeriod=dt, samplingStop=t[-1] + 1,)
    .applyGaussianNoise(0.5)
)

###############################################################################
# We configure the frequency estimator :class:`blocksim.control.Estimator.SpectrumEstimator`

from blocksim.control.Estimator import SpectrumEstimator

tracks = np.arange(0, 20, 0.5) / fs
nb_tracks = len(tracks)

X = np.zeros(nb_tracks, dtype=np.complex128)
kal = SpectrumEstimator(
    name="kal",
    dt=dt,
    shape_cmd=(1),
    snames_output=["x_kal"],
    snames_state=["x_%i_est" % i for i in range(nb_tracks)],
    tracks=tracks * fs,
)
kal.matQ = np.eye(nb_tracks) / 10
kal.matR = np.eye(1) / 10
kal.setInitialStateForOutput(X, "state")

###############################################################################
# We define the simulation

from blocksim.control.SetPoint import Step
from blocksim.Simulation import Simulation

ctrl = Step("ctrl", snames=["u"], cons=np.zeros(1))

sim = Simulation()

sim.addComputer(ctrl)
sim.addComputer(sig)
sim.addComputer(kal)

sim.connect("sig.setpoint", "kal.measurement")
sim.connect("ctrl.setpoint", "kal.command")

sim.simulate(sig.generateXSerie(), progress_bar=False)

###############################################################################
# We plot the spectrogram, which is computed by :class:`blocksim.control.Estimator.SpectrumEstimator.getSpectrogram`
# The spectrogram is an instance of :class:`blocksim.dsp.DSPSpectrogram.DSPSpectrogram`
# It has a method plot which allows to visualize the spectrogram

log = sim.getLogger()
spg = kal.getSpectrogram(log)

from blocksim.Graphics import plotSpectrogram
import matplotlib.pyplot as plt

fig = plt.figure()
axe = fig.add_subplot(111)
plotSpectrogram(spg, axe)
axe.plot(
    t, fchirp, linewidth=2, color="white", linestyle="--",
)
axe.set_xlabel("Time (s)")
axe.set_ylabel("Frequency (Hz)")

plt.show()
