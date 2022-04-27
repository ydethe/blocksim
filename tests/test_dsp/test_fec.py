import sys
from pathlib import Path
import unittest

import pytest
import numpy as np
from numpy import log10, sqrt
from matplotlib import pyplot as plt

from blocksim import logger

from blocksim.dsp.FEC import FECCoder, FECDecoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestFEC(TestBase):
    def setUp(self):
        self.data = np.random.randint(low=0, high=2, size=(2, 500))

    def test_fec_coding(self):
        fec_co = FECCoder(name="fec", output_size=6)
        self.assertEqual(fec_co.k_cc, 3)
        pl = fec_co.process(self.data)

        fec_deco = FECDecoder(name="fec", output_size=2)
        self.assertEqual(fec_deco.k_cc, 3)
        pl2 = fec_deco.process(pl)

        err = np.max(np.abs(pl2 - self.data))

        self.assertEqual(err, 0)


if __name__ == "__main__":
    a = TestFEC()
    a.setUp()
    a.test_fec_coding()
