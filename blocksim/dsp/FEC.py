import numpy as np
from numpy import sqrt, sign

import sk_dsp_comm.fec_conv as fec

from .. import logger
from .ADSPComputer import ADSPComputer


class FECCoder(ADSPComputer):
    __slots__ = ["__cc", "__state"]

    def __init__(self, name: str, output_size: int = 3):
        # Convolutional encoder 7, 1/3
        self.__cc = fec.FECConv(G=("11111", "11011", "10101"), Depth=25)
        k_cc = 3

        if output_size % k_cc != 0:
            raise ValueError(
                "[%s]output_size must be divisible by %i. Got %i"
                % (name, k_cc, output_size)
            )

        ADSPComputer.__init__(
            self,
            name=name,
            input_name="raw",
            output_name="coded",
            input_size=output_size // k_cc,
            output_size=output_size,
            input_dtype=np.int64,
            output_dtype=np.int64,
        )

        self.createParameter("k_cc", value=k_cc, read_only=True)

        self.__state = "0000"

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        raw: np.array,
        coded: np.array,
    ) -> dict:
        strm = self.flatten(raw)
        fec_strm, self.__state = self.__cc.conv_encoder(strm, state=self.__state)
        fec_bits = self.unflatten(fec_strm)

        outputs = {}
        outputs["coded"] = fec_bits
        return outputs


class FECDecoder(ADSPComputer):
    __slots__ = ["__cc"]

    def __init__(self, name: str, output_size: int = 1):
        # Convolutional encoder 7, 1/3
        self.__cc = fec.FECConv(G=("11111", "11011", "10101"), Depth=25)
        k_cc = 3

        ADSPComputer.__init__(
            self,
            name=name,
            input_name="coded",
            output_name="raw",
            input_size=output_size * k_cc,
            output_size=output_size,
            input_dtype=np.int64,
            output_dtype=np.int64,
        )

        self.createParameter("k_cc", value=k_cc, read_only=True)

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        coded: np.array,
        raw: np.array,
    ) -> dict:
        strm = self.flatten(coded)
        bits = self.__cc.viterbi_decoder(strm, "hard")
        raw = self.unflatten(bits)

        outputs = {}
        outputs["raw"] = raw
        return outputs
