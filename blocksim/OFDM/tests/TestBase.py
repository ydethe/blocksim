import unittest

import numpy as np
from numpy import exp, pi, log10


class TestBase(unittest.TestCase):
    def setUp(self):
        self.K = 12  # number of OFDM subcarriers

        # Length of a symbol
        self.nsamp = 2048

        # The known value each pilot transmits
        self.pilotValue = 3 + 3j

        # Now, let us define some index sets that describe which carriers transmit
        # pilots and which carriers contain payload.
        self.allCarriers = np.arange(self.K)
        self.pilotCarriers = self.allCarriers[::3]  # Pilots is every (K/P)th carrier.

        # data carriers are all remaining carriers
        self.dataCarriers = np.delete(self.allCarriers, self.pilotCarriers)

        # bits per symbol (i.e. QPSK)
        self.mu = 2

        self.channelResponse = np.array([1])

        self.n_bits = 16 * (14 * 3 // 3)
        self.bits = np.random.randint(low=0, high=2, size=self.n_bits)

        np.random.seed(155462157)
