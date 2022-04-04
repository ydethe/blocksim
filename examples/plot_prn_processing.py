r"""
GPS PRN proccessing
===================

"""

###############################################################################
# Creation of the signal
# ----------------------

from blocksim.dsp.DSPSignal import DSPSignal
from blocksim_sigspace.dsp import createGoldSequence

###############################################################################
# Reference Gold sequence. The argument sv=[2, 6] is the

y1 = createGoldSequence(name="s1", sv=[2, 6], repeat=1, chip_rate=1.023e6)

###############################################################################
# Noisy received signal

y = createGoldSequence(name="s1", sv=[2, 6], repeat=20, chip_rate=1.023e6)
y = y.applyGaussianNoise(pwr=200)

###############################################################################
# Processing
# ----------

###############################################################################
# Correlation

z = y.correlate(y1)

###############################################################################
# Integration

zi = z.integrate(period=1e-3, offset=511 / (1.023e6))

###############################################################################
# Plotting
# --------

from matplotlib import pyplot as plt

from blocksim.graphics import plotDSPLine

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
plotDSPLine(zi, axe, find_peaks=1)
axe.set_ylabel("Intégration")

plt.show()
