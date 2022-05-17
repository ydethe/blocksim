from numpy.typing import ArrayLike
import numpy as np
from numpy import sqrt, sign

from commpy.channelcoding.convcode import Trellis, conv_encode, viterbi_decode

from .ADSPComputer import ADSPComputer

from .. import logger


class FECCoder(ADSPComputer):
    """Forward Error Correction encoder for NB-IoT (TODO : for any FEC)
    For an input vector of n bits, this computer will generate an output vector of 3*n bits

    Args:
        name: Name of the computer
        output_size: Number of scalar in the output vector. Shall be divisible by 3.

    """

    __slots__ = ["__trellis", "__state"]

    def __init__(self, name: str, output_size: int = 3):
        # Convolutional encoder 7, 1/3
        G = np.array([[0o133, 0o171, 0o165]])
        _, k_cc = G.shape
        depth = 7
        self.__trellis = Trellis(
            memory=np.array([depth]), g_matrix=G, polynomial_format="MSB"
        )

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

        self.createParameter("G", value=G, read_only=True)
        self.createParameter("depth", value=depth, read_only=True)

    @property
    def k_cc(self):
        _, k_cc = self.G.shape
        return k_cc

    def update(
        self,
        t1: float,
        t2: float,
        raw: ArrayLike,
        coded: ArrayLike,
    ) -> dict:
        strm = self.flatten(raw)
        fec_strm = conv_encode(strm, self.__trellis)
        fec_bits = self.unflatten(fec_strm[: -self.k_cc * self.depth])

        outputs = {}
        outputs["coded"] = fec_bits
        return outputs


class FECDecoder(ADSPComputer):
    """Forward Error Correction decoder for NB-IoT (TODO : for any FEC)
    For an input vector of n bits, this computer will generate an output vector of n//3 bits

    Args:
        name: Name of the computer
        output_size: Number of scalar in the output vector.

    """

    __slots__ = ["__trellis"]

    def __init__(self, name: str, output_size: int = 1):
        # Convolutional encoder 7, 1/3
        G = np.array([[0o133, 0o171, 0o165]])
        _, k_cc = G.shape
        depth = 7
        self.__trellis = Trellis(
            memory=np.array([depth]), g_matrix=G, polynomial_format="MSB"
        )

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

        self.createParameter("G", value=G, read_only=True)
        self.createParameter("depth", value=depth, read_only=True)

    @property
    def k_cc(self):
        _, k_cc = self.G.shape
        return k_cc

    def update(
        self,
        t1: float,
        t2: float,
        coded: ArrayLike,
        raw: ArrayLike,
    ) -> dict:
        strm = self.flatten(coded)
        bits = viterbi_decode(
            coded_bits=strm, trellis=self.__trellis, decoding_type="hard"
        )
        raw = self.unflatten(bits)

        outputs = {}
        outputs["raw"] = raw
        return outputs
