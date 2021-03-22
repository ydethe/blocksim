import unittest

import numpy as np
from numpy import log10
import sk_dsp_comm.digitalcom as dc

from blocksim import logger
from blocksim.blocs.FEC import FECCoder, FECDecoder
from tests.TestBase import TestBase


class TestFEC(TestBase):
    def test_fec(self):
        fec_co = FECCoder()
        fec_dec = FECDecoder()

        N_bits_per_frame = 10000
        SNR = 17.55
        EsN0 = SNR + 10 * log10(180e3 / (15000 * 2048))
        total_bit_errors = 0
        total_bit_count = 0
        while total_bit_errors < 100:
            # Create N_bits_per_frame random 0/1 bits
            x = np.random.randint(0, 2, N_bits_per_frame)

            y = fec_co.process(x)

            # Add channel noise to bits, include antipodal level shift to [-1,1]
            yn = dc.cpx_AWGN(2 * y - 1, EsN0, 1)

            yn_demap = (yn.real + 1) / 2
            z = fec_dec.process(yn_demap)

            # Count bit errors
            bit_count, bit_errors = dc.bit_errors(x, z)
            total_bit_errors += bit_errors
            total_bit_count += bit_count

        ber = total_bit_errors / total_bit_count

        self.assertAlmostEqual(ber, 0.0839013, delta=1e-6)


if __name__ == "__main__":
    unittest.main()
