import numpy as np
from numpy import sqrt, sign, pi, exp

from .. import logger
from blocksim.core.Node import AComputer


class OFDMMapping(AComputer):
    __slots__ = []

    def __init__(
        self,
        name: str,
        allCarriers: list,
        pilotCarriers: list,
        dataCarriers: list,
        pilotValue: np.complex128,
    ):
        AComputer.__init__(self, name=name)
        self.defineInput("input", shape=(len(dataCarriers),), dtype=np.complex128)
        self.defineOutput(
            "output",
            snames=["x%i" % i for i in range(len(allCarriers))],
            dtype=np.complex128,
        )
        self.createParameter("allCarriers", value=allCarriers)
        self.createParameter("pilotCarriers", value=pilotCarriers)
        self.createParameter("dataCarriers", value=dataCarriers)
        self.createParameter("pilotValue", value=pilotValue)

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
    ) -> dict:
        K = len(self.allCarriers)
        nsymb = 1

        # the overall K subcarriers
        symbols = np.zeros(K, dtype=np.complex128)

        # allocate the pilot subcarriers
        symbols[self.pilotCarriers, :] = self.pilotValue

        # allocate the data subcarriers
        symbols[self.dataCarriers, :] = input

        outputs = {}
        outputs["output"] = symbols
        return outputs


class OFDMDemapping(AComputer):
    __slots__ = []

    def __init__(
        self,
        name: str,
        allCarriers: list,
        pilotCarriers: list,
        dataCarriers: list,
        pilotValue: np.complex128,
    ):
        AComputer.__init__(self, name=name)
        self.defineInput("input", shape=(len(allCarriers),), dtype=np.complex128)
        self.defineOutput(
            "output",
            snames=["x%i" % i for i in range(len(dataCarriers))],
            dtype=np.complex128,
        )
        self.createParameter("allCarriers", value=list(allCarriers))
        self.createParameter("pilotCarriers", value=list(pilotCarriers))
        self.createParameter("dataCarriers", value=list(dataCarriers))
        self.createParameter("pilotValue", value=pilotValue)

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
    ) -> dict:
        QAM_payload = input[self.dataCarriers]

        outputs = {}
        outputs["output"] = QAM_payload
        return outputs
