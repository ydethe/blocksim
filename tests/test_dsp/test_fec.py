import sys
from pathlib import Path
import unittest

import tqdm
import numpy as np
from numpy import log10
import sk_dsp_comm.digitalcom as dc

from blocksim import logger
from blocksim.dsp.FEC import FECCoder, FECDecoder

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestFEC(TestBase):
    def test_fec(self):
        fec_co = FECCoder(name="coder", output_size=6)
        fec_dec = FECDecoder(name="decoder", output_size=2)

        N_bits_per_frame = 1023

        # Create N_bits_per_frame random 0/1 bits
        np.random.seed(165467)
        ref = np.random.randint(0, 2, size=(2, N_bits_per_frame))

        y = fec_co.process(ref)
        est = fec_dec.process(y)

        err = np.max(np.abs(ref[:, :-12] - est))

        self.assertEqual(err, 0)


if __name__ == "__main__":
    # unittest.main()

    a = TestFEC()
    a.test_fec()
