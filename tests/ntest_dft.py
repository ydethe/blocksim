import os
import sys
import unittest

import pytest
import numpy as np

from blocksim import logger
from blocksim.dsp.DFT import DFT, IDFT

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class TestDFT(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_dft(self):
        nsymb = 3
        ntot = self.nsamp * nsymb
        data = np.zeros((self.K, nsymb), dtype=np.complex128)
        data += np.random.randint(low=-5, high=6, size=(self.K, nsymb)) * 1j
        data += np.random.randint(low=-5, high=6, size=(self.K, nsymb))

        idft = IDFT(self.nsamp)
        dft = DFT(self.K, self.nsamp)

        sig = idft.process(data)
        data2 = dft.process(sig)

        self.assertAlmostEqual(np.max(np.abs(data - data2)), 0, delta=1e-9)

        axe = dft.plotOutput(df_khz=15)

        return axe.figure

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_idft(self):
        nsymb = 3
        ntot = self.nsamp * nsymb
        data = np.zeros((self.K, nsymb), dtype=np.complex128)
        data += np.random.randint(low=-5, high=6, size=(self.K, nsymb)) * 1j
        data += np.random.randint(low=-5, high=6, size=(self.K, nsymb))

        idft = IDFT(self.nsamp)

        sig = idft.process(data)

        axe = idft.plotOutput(dt_us=1e6 / (15000 * 2048),)

        return axe.figure


if __name__ == "__main__":
    unittest.main()
