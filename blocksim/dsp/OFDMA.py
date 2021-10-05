import numpy as np
from numpy import sqrt, sign, pi, exp
from numpy.fft import fft, ifft

from .. import logger
from .ADSPComputer import ADSPComputer


class OFDMMapping(ADSPComputer):
    __slots__ = []

    def __init__(
        self,
        name: str,
        output_size: int,
        allCarriers: int,
        pilotCarriers: list,
        dataCarriers: list,
        pilotValue: np.complex128,
    ):
        ADSPComputer.__init__(
            self,
            name=name,
            input_name="input",
            output_name="output",
            input_size=len(dataCarriers),
            output_size=output_size,
            input_dtype=np.complex128,
            output_dtype=np.complex128,
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
        if len(input.shape) == 1:
            nsymb = 1
            ny = self.input_size
            input = input.reshape((ny, nsymb))
        else:
            ny, nsymb = input.shape

        K = self.allCarriers

        # the overall K subcarriers
        data = np.zeros((K, nsymb), dtype=np.complex128)

        # allocate the pilot subcarriers
        data[self.pilotCarriers, :] = self.pilotValue

        # allocate the data subcarriers
        data[self.dataCarriers, :] = input

        s = np.empty((self.output_size, nsymb), dtype=np.complex128)
        for k in range(nsymb):
            # Calcul du symbole
            symb = ifft(data[:, k], n=self.output_size)

            # On met le symbole OFDM dans le signal
            s[:, k] = symb

        if nsymb == 1:
            s = s.reshape(self.output_size)

        outputs = {}
        outputs["output"] = s
        return outputs


class OFDMDemapping(ADSPComputer):
    __slots__ = []

    def __init__(
        self,
        name: str,
        input_size: int,
        allCarriers: int,
        pilotCarriers: list,
        dataCarriers: list,
        pilotValue: np.complex128,
    ):
        ADSPComputer.__init__(
            self,
            name=name,
            input_name="input",
            output_name="output",
            input_size=input_size,
            output_size=len(dataCarriers),
            input_dtype=np.complex128,
            output_dtype=np.complex128,
        )

        self.createParameter("allCarriers", value=allCarriers)
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
        if len(input.shape) == 1:
            nsymb = 1
            ny = self.input_size
            input = input.reshape((ny, nsymb))
        else:
            ny, nsymb = input.shape

        K = self.allCarriers

        demod = np.empty((K, nsymb), dtype=np.complex128)
        for k in range(nsymb):
            buf = input[:, k]
            demod[:, k] = fft(buf)[:K]

        QAM_payload = demod[self.dataCarriers]

        if nsymb == 1:
            QAM_payload = QAM_payload.reshape(self.output_size)

        outputs = {}
        outputs["output"] = QAM_payload
        return outputs
