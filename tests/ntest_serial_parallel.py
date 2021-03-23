import os
import sys
import unittest

import numpy as np
from numpy import exp, pi

from blocksim.dsp.SerialParallel import SerialToParallel, ParallelToSerial

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class TestSerialParallel(TestBase):
    def test_serial_parallel(self):
        sp = SerialToParallel(self.mu)
        ps = ParallelToSerial(self.mu)

        par = sp.process(self.bits)
        res = ps.process(par)

        err = np.sum(np.abs(res - self.bits))

        self.assertEqual(err, 0)


if __name__ == "__main__":
    unittest.main()
