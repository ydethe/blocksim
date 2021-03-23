import os
import sys
import unittest

import pytest
import numpy as np
from numpy import log10, sqrt
import sk_dsp_comm.digitalcom as dc

from blocksim import logger
from blocksim.dsp.QPSKMod import QPSKMapping, QPSKDemapping

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class TestQPSK(TestBase):
    def test_qpsk(self):
        qpsk_co = QPSKMapping()
        qpsk_dec = QPSKDemapping()

        ntot = 256

        data = np.zeros((ntot, 2))
        data[:, 0] = np.random.randint(low=0, high=2, size=ntot)
        data[:, 1] = np.random.randint(low=0, high=2, size=ntot)

        qpsk_payload = qpsk_co.process(data)
        data2 = qpsk_dec.process(qpsk_payload)

        self.assertAlmostEqual(np.max(np.abs(data - data2)), 0, delta=1e-9)

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_qpsk_constellation(self):
        qpsk_co = QPSKMapping()

        axe = qpsk_co.plotConstellation()

        return axe.figure

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_qpsk_noise(self):
        qpsk_co = QPSKMapping()
        qpsk_dec = QPSKDemapping()

        ntot = 256

        data = np.zeros((ntot, 2))
        data[:, 0] = np.random.randint(low=0, high=2, size=ntot)
        data[:, 1] = np.random.randint(low=0, high=2, size=ntot)

        qpsk_payload = qpsk_co.process(data)

        n = len(qpsk_payload)
        qpsk_payload += (np.random.normal(size=n) + 1j * np.random.normal(size=n)) / 2

        data2 = qpsk_dec.process(qpsk_payload)

        self.assertAlmostEqual(np.max(np.abs(data - data2)), 0, delta=1)

        axe = qpsk_dec.plotOutput()

        return axe.figure


if __name__ == "__main__":
    unittest.main()
