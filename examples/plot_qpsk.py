r"""
QPSK modulation
===============

"""
###############################################################################
# Main libraries import
# ---------------------

import numpy as np
from numpy import log10, sqrt
from matplotlib import pyplot as plt

from blocksim import logger
from blocksim.dsp.QPSKMod import QPSKMapping, QPSKDemapping

###############################################################################
# Instanciation of QPSK modulator and demodulator
# -----------------------------------------------

qpsk_co = QPSKMapping(name="map")
qpsk_dec = QPSKDemapping(name="demap")

###############################################################################
# Generation of a random bitstream

ntot = 256
data = np.random.randint(low=0, high=2, size=ntot)

###############################################################################
# Generation of the QPSK stream of symbols

qpsk_payload = qpsk_co.process(data)

###############################################################################
# Adding gaussian noise to the QPSK stream of symbols

n = len(qpsk_payload)
qpsk_payload += (np.random.normal(size=n) + 1j * np.random.normal(size=n)) * sqrt(
    0.05 / 2
)

###############################################################################
# Demodulation of the noisy QPSK symbols stream
# and verification of the error

data2 = qpsk_dec.process(qpsk_payload)
err = np.abs(data - data2)
inok = np.where(err == 1)[0]
len(inok) / ntot

###############################################################################
# Plotting
# --------

fig = plt.figure()
axe = fig.add_subplot(111)
axe.grid(True)
qpsk_dec.plotOutput(qpsk_payload, axe)

plt.show()
