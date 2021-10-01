import numpy as np
from numpy import sqrt, sign, pi, exp, cos, sin, log2

from .. import logger
from ..core.Node import AComputer
from .CircularBuffer import CircularBuffer


class PSKMapping(AComputer):
    """Phase Shift Keying modulator.
    To determine the type of modulation, the *mapping* argument must tell what keying is used :
    For example, in 8-PSK, the block '010' in decimal is 2. So the phase is mapping[2], and the symbol is exp(1j*mapping[2])
    The number of values in *mapping* shall be a power of 2: len(mapping) == 2**mu.

    To initiate the modulator, mu-1 samples shall be processed. The mu-th sample will trigger the conversion of the mu previous samples into a symbol.
    Then the following mu-1 samples are memorised and the last symbol is repeated on the output mu-1 times.

    Args:
      name
        Name of the computer
      mapping
        List of phase values. For example, in QPSK, [pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4]

    """

    __slots__ = ["__nb_samples", "__acc_bits", "__prev_symb"]

    def __init__(self, name: str, mapping: list):
        AComputer.__init__(self, name=name)
        self.defineInput("input", shape=1, dtype=np.int64)
        self.defineOutput("output", snames=["symb"], dtype=np.complex128)
        self.createParameter("mapping", value=mapping, read_only=True)

        mu = int(np.round(log2(len(mapping)), 0))
        if 2 ** mu != len(mapping):
            raise ValueError("Mapping size must be a power of 2. Got %i" % len(mapping))

        self.createParameter("mu", value=mu, read_only=True)

        self.__nb_samples = 0
        self.__acc_bits = 0
        self.__prev_symb = 0

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
    ) -> dict:
        assert len(input) == 1
        nmap = len(self.mapping)

        if self.__nb_samples == self.mu - 1:
            self.__acc_bits += input[0] * nmap // 2
            symb = exp(1j * self.mapping[self.__acc_bits])
            self.__acc_bits = 0
            self.__nb_samples = 0
            self.__prev_symb = symb
        else:
            self.__acc_bits += input[0] * 2 ** self.__nb_samples
            self.__nb_samples += 1
            symb = self.__prev_symb

        outputs = {}
        outputs["output"] = np.array([symb], dtype=np.complex128)
        return outputs

    def process(self, data: np.array) -> np.array:
        """Processes a bitstream by calling compute_outputs.
        This proccessing removes the initial null symbol,
        due to the fact that the modulator is not initialised.

        Args:
          data
            A bitstream of length n, n even. Bits are either 0, or 1

        Returns:
          An array of QPSK symbols

        """
        n = len(data)
        res = np.empty(n, dtype=np.complex128)
        for k in range(n + self.mu - 1):
            if k < n:
                outputs = self.compute_outputs(t1=0, t2=0, input=data[[k]], output=None)
                symb = outputs["output"][0]

            p = k - self.mu + 1
            if p >= 0:
                res[p] = symb

        return res


class PSKDemapping(AComputer):
    """Phase Shift Keying demodulator.
    To determine the type of modulation, the *mapping* argument must tell what keying is used :
    For example, in 8-PSK, the block '010' in decimal is 2. So the phase is mapping[2], and the symbol is exp(1j*mapping[2])
    The number of values in *mapping* shall be a power of 2: len(mapping) == 2**mu.

    To initiate the demodulator, 1 symbol shall be processed to store the mu bits.
    Then the following mu samples are the dequeing of the stored samples.
    The symbols on the input are not supposed to change during this period

    Args:
      name
        Name of the computer
      mapping
        List of phase values. For example, in QPSK, [pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4]

    """

    __slots__ = ["__prev_symb", "__num_bit"]

    def __init__(self, name: str, mapping: list):
        AComputer.__init__(self, name=name)
        self.defineInput("input", shape=1, dtype=np.complex128)
        self.defineOutput("output", snames=["bit"], dtype=np.int64)
        self.createParameter("mapping", value=mapping, read_only=True)
        self.createParameter(
            "cmapping", value=exp(1j * np.array([mapping])), read_only=True
        )

        mu = int(np.round(log2(len(mapping)), 0))
        if 2 ** mu != len(mapping):
            raise ValueError("Mapping size must be a power of 2. Got %i" % len(mapping))

        self.createParameter("mu", value=mu, read_only=True)

        self.__prev_symb = [0] * self.mu
        self.__num_bit = self.mu - 2

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
    ) -> dict:
        assert len(input) == 1

        if self.__num_bit == self.mu - 1:
            bit = self.__prev_symb[self.__num_bit]
            self.__num_bit = 0
            symb_in = input[0]
            isymb = np.argmin(np.abs(self.cmapping - symb_in))
            ss = bin(isymb)[2:].zfill(self.mu)
            self.__prev_symb = [int(x) for x in ss[::-1]]
        else:
            bit = self.__prev_symb[self.__num_bit]
            self.__num_bit += 1

        outputs = {}
        outputs["output"] = np.array([bit], dtype=np.int64)
        return outputs

    def process(self, symbols: np.array) -> np.array:
        """Processes a bitstream by calling compute_outputs.
        This proccessing removes the initial null bits,
        due to the fact that the demodulator is not initialised.

        Args:
          symbols
            An array of QPSK symbols

        Returns:
          A bitstream of length n, n even. Bits are either 0, or 1

        """
        n = len(symbols)
        res = np.empty(n, dtype=np.int64)
        for k in range(n + self.mu):
            if k < n:
                symb = symbols[[k]]

            outputs = self.compute_outputs(t1=0, t2=0, input=symb, output=None)

            bit = outputs["output"]

            p = k - self.mu
            if p >= 0:
                res[p] = bit

        return res
