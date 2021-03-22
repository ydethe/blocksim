import numpy as np
from numpy import sqrt, sign
from matplotlib import pyplot as plt

import sk_dsp_comm.fec_conv as fec

from blocksim import logger
from blocksim.blocs.ProcessingBlock import ProcessingBlock


class FECCoder(ProcessingBlock):
    def __init__(self):
        # Convolutional encoder 7, 1/3
        self.k_cc = 3
        self.cc = fec.fec_conv(("11111", "11011", "10101"), 25)

    def __update__(self, data: np.array) -> np.array:
        fec_bits, _ = self.cc.conv_encoder(data, state="0000")
        return fec_bits


class FECDecoder(ProcessingBlock):
    def __init__(self):
        # Convolutional encoder 7, 1/3
        self.k_cc = 3
        self.cc = fec.fec_conv(("11111", "11011", "10101"), 25)

    def __update__(self, data: np.array) -> np.array:
        bits_est = self.cc.viterbi_decoder(data * 7, "soft", quant_level=3)

        return bits_est.astype(np.int64)
