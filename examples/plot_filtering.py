r"""
FIR filtering
=============

"""
###############################################################################
# Main libraries import
# ---------------------

import numpy as np
from numpy import pi, exp
from matplotlib import pyplot as plt

###############################################################################
# Definition of the filter
# ------------------------
# We define a FIR pass-band filter between f1 (Hz) and f2 (Hz)
# A Chebychev window is applied to lower the side-lobes

from blocksim.dsp.DSPFilter import DSPFilter
from blocksim.Graphics import plotBode

fs = 200
f1 = 10
f2 = 30
filt = DSPFilter("filtre", f_low=f1, f_high=f2, numtaps=256, win=("chebwin", -60))

###############################################################################
# We plot the Bode diagram

fig = plt.figure()
axe_amp = fig.add_subplot(211)
axe_pha = fig.add_subplot(212, sharex=axe_amp)

plotBode(filt, fs, axe_amp, axe_pha)

###############################################################################
# Application of the filter
# -------------------------
# We construct s1 as a signal with f0 and 3*f0 frequencies

from blocksim.Graphics import plotDSPLine
from blocksim.dsp.DSPSignal import DSPSignal

f0 = 20
ns = 200
t1 = np.arange(ns) / fs
x1 = exp(1j * 2 * pi * f0 * t1) + exp(1j * 2 * pi * 3 * f0 * t1)
s1 = DSPSignal(name="s1", samplingStart=0, samplingPeriod=1 / fs, y_serie=x1)

###############################################################################
# s2 is the expected signal at the output of the filter

x2 = exp(1j * 2 * pi * f0 * t1)
s2 = DSPSignal(name="s2", samplingStart=0, samplingPeriod=1 / fs, y_serie=x2)

###############################################################################
# We apply the filter

y = filt.apply(s1)

###############################################################################
# We plot s2 and y

fig = plt.figure()
axe = fig.add_subplot(111)

plotDSPLine(y, axe, label="Output (y)")
plotDSPLine(s2, axe, label="Expected (s2)")
axe.legend()

plt.show()
