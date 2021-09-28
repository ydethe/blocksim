import numpy as np

from .. import logger
from ..core.Node import AComputer


class SerialToParallel(AComputer):
    __slots__ = []

    def __init__(self, name: str, nb_parallel: int):
        AComputer.__init__(self, name=name)
        self.createParameter("nb_parallel", value=nb_parallel)
        self.defineInput("input", shape=(nb_parallel,), dtype=np.complex128)
        self.defineOutput(
            "output",
            snames=["x%i" % i for i in range(nb_parallel)],
            dtype=np.complex128,
        )

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
    ) -> dict:
        outputs = {}
        outputs["output"] = input.reshape(1, self.nb_parallel)
        return outputs


class ParallelToSerial(AComputer):
    __slots__ = []

    def __init__(self, name: str, nb_parallel: int):
        AComputer.__init__(self, name=name)
        self.createParameter("nb_parallel", value=nb_parallel)

    def __update__(self, data: np.array) -> np.array:
        n, r = data.shape
        if r != self.nb_parallel:
            raise ArgumentError("The parallel data must have %i row" % self.nb_parallel)
        return data.reshape((-1,))
