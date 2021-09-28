import numpy as np
from numpy import sqrt, sign, pi, exp

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
        x, y = input
        symb = ((x + 1j * y) * 2 - 1 - 1j) * self.inv_sq_2
        outputs = {}
        outputs["output"] = np.array([symb])
        return outputs


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
        (symb,) = input

        x = (np.sign(symb.real) + 1) / 2
        y = (np.sign(symb.imag) + 1) / 2

        outputs = {}
        outputs["output"] = np.array([x, y])
        return outputs
