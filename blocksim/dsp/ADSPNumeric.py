from itertools import product
from abc import abstractmethod

import numpy as np

from ..core.Node import AComputer


class ADSPNumeric(AComputer):
    __slots__ = []

    def __init__(self, name: str, shape_in: tuple, shape_out: tuple):
        AComputer.__init__(self, name=name)
        self.defineInput("input", shape=shape_in, dtype=np.complex128)

        # Creating iterables, to handle the case where
        # the output 'setpoint' is a matrix
        it = []
        for k in shape_out:
            it.append(range(k))

        snames = []
        # Iterate over all dimensions
        for iscal in product(*it):
            k = "_".join(iscal)
            snames.append(k)

        self.defineOutput("output", snames=snames, dtype=np.complex128)

    @abstractmethod
    def process(self, t: float, input: np.array) -> np.array:
        pass

    def compute_outputs(self, t1: float, t2: float, input: np.array) -> dict:
        out = self.process(t2, input)

        outputs = {}
        outputs["output"] = out

        return outputs
