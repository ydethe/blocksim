import numpy as np
from numpy import sqrt, sign, pi, exp
from matplotlib import pyplot as plt

from OFDM import logger
from OFDM.blocs.ProcessingBlock import ProcessingBlock


class OFDMMapping(ProcessingBlock):
    def __init__(self, allCarriers, pilotCarriers, dataCarriers, pilotValue):
        self.allCarriers = allCarriers
        self.pilotCarriers = pilotCarriers
        self.dataCarriers = dataCarriers
        self.pilotValue = pilotValue

    def __update__(self, data: np.array) -> np.array:
        n = len(data)

        K = len(self.allCarriers)
        nsymb = n // len(self.dataCarriers)

        # the overall K subcarriers
        symbols = np.zeros((K, nsymb), dtype=np.complex128)

        # allocate the pilot subcarriers
        symbols[self.pilotCarriers, :] = self.pilotValue

        # allocate the data subcarriers
        symbols[self.dataCarriers, :] = data.reshape(
            (len(self.dataCarriers), nsymb), order="F"
        )

        return symbols

    def plotCarriers(self, axe=None):
        if axe is None:
            fig = plt.figure()
            axe = fig.add_subplot(111)
            axe.grid(True)

        axe.plot(
            self.pilotCarriers, np.zeros_like(self.pilotCarriers), "bo", label="pilot"
        )
        axe.plot(
            self.dataCarriers, np.zeros_like(self.dataCarriers), "ro", label="data"
        )
        axe.legend()

        return axe

    def plotOutput(self, df_khz=1, axe=None):
        if axe is None:
            fig = plt.figure()
            axe = fig.add_subplot(111)
            axe.set_xlabel("Fréquence (kHz)")
            axe.grid(True)

        n = self.out.shape[0]
        frq = np.arange(n) * df_khz
        axe.plot(frq, np.abs(self.out[:, 0]), label="TX signal, no CP, demod")
        axe.legend(fontsize=10)

        return axe


class OFDMDemapping(ProcessingBlock):
    def __init__(self, allCarriers, pilotCarriers, dataCarriers, pilotValue):
        self.allCarriers = allCarriers
        self.pilotCarriers = pilotCarriers
        self.dataCarriers = dataCarriers
        self.pilotValue = pilotValue

    def __update__(self, data: np.array) -> np.array:
        _, nsymb = data.shape
        n_data = len(self.dataCarriers)
        QAM_payload = data[self.dataCarriers, :].reshape(n_data * nsymb, order="F")

        return QAM_payload

    def plotOutput(self, axe=None):
        if axe is None:
            fig = plt.figure()
            axe = fig.add_subplot(111)
            axe.set_aspect("equal")
            axe.grid(True)
            axe.set_xlabel("I")
            axe.set_ylabel("Q")
            axe.set_title("Constellation QPSK")

        x = np.linspace(0, 2 * pi, 100)
        circle = exp(1j * x)
        axe.plot(self.out.real, self.out.imag, "bo")
        axe.plot(circle.real, circle.imag, linestyle="--", color="black")

        return axe
