import numpy as np
from numpy import sqrt, sign, pi, exp, cos, sin

from .. import logger
from ..core.Node import AComputer


class QPSKMapping(AComputer):
    __slots__ = []

    def __init__(self, name: str):
        AComputer.__init__(self, name=name)
        self.defineInput("input", shape=(2,), dtype=np.int64)
        self.defineOutput("output", snames=["symb"], dtype=np.complex128)
        self.createParameter("mu", value=2)
        self.createParameter("inv_sq_2", value=1 / sqrt(2))

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
    ) -> dict:
        x = input[::2]
        y = input[1::2]
        symb = ((x + 1j * y) * 2 - 1 - 1j) * self.inv_sq_2
        outputs = {}
        outputs["output"] = symb
        return outputs

    def process(self, data: np.array) -> np.array:
        """Processes a bitstream by calling compute_outputs

        Args:
          data
            A bitstream of length n, n even. Bits are either 0, or 1

        Returns:
          An array of QPSK symbols

        """
        outputs = self.compute_outputs(t1=0, t2=0, input=data, output=None)
        return outputs["output"]


class QPSKDemapping(AComputer):
    __slots__ = []

    def __init__(self, name: str):
        AComputer.__init__(self, name=name)
        self.defineInput("input", shape=(1,), dtype=np.complex128)
        self.defineOutput("output", snames=["x", "y"], dtype=np.int64)
        self.createParameter("mu", value=2)
        self.createParameter("inv_sq_2", value=1 / sqrt(2))

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
    ) -> dict:
        x = (np.sign(input.real) + 1) / 2
        y = (np.sign(input.imag) + 1) / 2

        data = np.empty((x.size + y.size,))
        data[0::2] = x
        data[1::2] = y

        outputs = {}
        outputs["output"] = data
        return outputs

    def process(self, data: np.array) -> np.array:
        """Processes a bitstream by calling compute_outputs

        Args:
          data
            An array of QPSK symbols

        Returns:
          A bitstream of length n, n even. Bits are either 0, or 1

        """
        outputs = self.compute_outputs(t1=0, t2=0, input=data, output=None)
        return outputs["output"]

    def plotOutput(self, payload: np.array, axe):
        axe.set_aspect("equal")

        isymb = np.arange(4)
        x_const = cos(pi / 4 + pi / 2 * isymb)
        y_const = sin(pi / 4 + pi / 2 * isymb)

        axe.scatter(np.real(payload), np.imag(payload), marker="+", color="blue")

        axe.scatter(x_const, y_const, marker="o", color="red")
