import numpy as np
from numpy import sqrt, sign, pi, exp, cos, sin, log2

from .. import logger
from ..core.Node import AComputer
from .CircularBuffer import CircularBuffer


class PSKMapping(AComputer):
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
        """Processes a bitstream by calling compute_outputs

        Args:
          data
            A bitstream of length n, n even. Bits are either 0, or 1

        Returns:
          An array of QPSK symbols

        """
        n = len(data)
        res = np.empty(n, dtype=np.complex128)
        for k in range(n):
            outputs = self.compute_outputs(t1=0, t2=0, input=data[[k]], output=None)
            symb = outputs["output"]
            res[k] = symb
        return res


class PSKDemapping(AComputer):
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

    def process(self, data: np.array) -> np.array:
        """Processes a bitstream by calling compute_outputs

        Args:
          data
            An array of QPSK symbols

        Returns:
          A bitstream of length n, n even. Bits are either 0, or 1

        """
        n = len(data)
        res = np.empty(n, dtype=np.int64)
        for k in range(n):
            outputs = self.compute_outputs(t1=0, t2=0, input=data[[k]], output=None)
            bit = outputs["output"]
            res[k] = bit
        return res

    def plotOutput(self, payload: np.array, axe):
        axe.set_aspect("equal")

        isymb = np.arange(4)
        x_const = cos(pi / 4 + pi / 2 * isymb)
        y_const = sin(pi / 4 + pi / 2 * isymb)

        axe.scatter(np.real(payload), np.imag(payload), marker="+", color="blue")

        axe.scatter(x_const, y_const, marker="o", color="red")
