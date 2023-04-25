import numpy as np
from numpy.fft import fft, ifft

from ..utils import FloatArr

from .ADSPComputer import ADSPComputer


class OFDMMapping(ADSPComputer):
    """OFDMA modulator
    Takes symbols as input, and maps them to the data carriers.
    Adds pilot symbols on dedicated carriers

    Args:
        name: Name of the computer
        output_size: Number of samples in the generated signal
        allCarriers: Total number of sub carriers
        pilotCarriers: List of the indices of the carriers dedicated to pilot symbols
        dataCarriers: List of the indices of the carriers dedicated to data symbols
        pilotValue: Pilot symbols to add on the pilot sub carriers

    """

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

    def update(
        self,
        t1: float,
        t2: float,
        input: FloatArr,
        output: FloatArr,
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
    """OFDMA demodulator
    Takes a time signal (represented as a vector) and extracts data symbols

    Args:
        name: Name of the computer
        input_size: Number of samples in the input signal
        allCarriers: Total number of sub carriers
        pilotCarriers: List of the indices of the carriers dedicated to pilot symbols
        dataCarriers: List of the indices of the carriers dedicated to data symbols
        pilotValue: Pilot symbols to add on the pilot sub carriers

    """

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

    def update(
        self,
        t1: float,
        t2: float,
        input: FloatArr,
        output: FloatArr,
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
