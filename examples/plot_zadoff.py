r"""
Zadoff-Chu sequence
===================

"""

###############################################################################
# Creation of a Zadoff-Chu sequence
# ---------------------------------

from blocksim.dsp.DSPSignal import DSPSignal

s1 = DSPSignal.fromZadoffChu(name="s1", n_zc=1021, u=1, sampling_freq=1e6)

###############################################################################
# Correlation of the sequence with itself

y = s1.correlate(s1)

###############################################################################
# Plotting
# --------

from matplotlib import pyplot as plt

from blocksim.Graphics import plotDSPLine

fig = plt.figure()
axe = fig.add_subplot(111)
axe.grid(True)
plotDSPLine(y, axe)

plt.show()
