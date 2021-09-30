import os
import sys
import unittest

import pytest
import numpy as np
from numpy import log10, sqrt
from matplotlib import pyplot as plt

from blocksim import logger
from blocksim.dsp.QPSKMod import QPSKMapping, QPSKDemapping

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class TestQPSK(TestBase):
    def test_qpsk(self):
        qpsk_co = QPSKMapping(name="map")
        qpsk_dec = QPSKDemapping(name="demap")

        ntot = 256

        data = np.random.randint(low=0, high=2, size=ntot)

        qpsk_payload = qpsk_co.process(data)
        data2 = qpsk_dec.process(qpsk_payload)

        self.assertAlmostEqual(np.max(np.abs(data - data2)), 0, delta=1e-9)

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_qpsk_noise(self):
        qpsk_co = QPSKMapping(name="map")
        qpsk_dec = QPSKDemapping(name="demap")

        ntot = 256

        data = np.random.randint(low=0, high=2, size=ntot)

        qpsk_payload = qpsk_co.process(data)

        n = len(qpsk_payload)
        qpsk_payload += (
            np.random.normal(size=n) + 1j * np.random.normal(size=n)
        ) * sqrt(0.05 / 2)

        data2 = qpsk_dec.process(qpsk_payload)

        self.assertAlmostEqual(np.max(np.abs(data - data2)), 0, delta=0.5)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.grid(True)
        qpsk_dec.plotOutput(qpsk_payload, axe)

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestQPSK()
    a.test_qpsk_noise()

    plt.show()
