import numpy as np
from numpy import sqrt, sign, pi, exp
from matplotlib import pyplot as plt

from OFDM import logger
from OFDM.blocs.ProcessingBlock import ProcessingBlock


class QPSKMapping(ProcessingBlock):
    def __init__(self):
        self.mu = 2
        self.inv_sq_2 = 1 / sqrt(2)

    def __update__(self, data: np.array) -> np.array:
        n, r = data.shape
        if r != self.mu:
            raise ArgumentError("The parallel data must have %i row" % self.mu)
        return ((data[:, 0] + 1j * data[:, 1]) * 2 - 1 - 1j) * self.inv_sq_2

    def plotConstellation(self, axe=None):
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
        for b1 in [0, 1]:
            for b0 in [0, 1]:
                B = np.array([[b1, b0]])
                Q = self.process(B)[0]
                axe.plot(Q.real, Q.imag, "bo")
                axe.text(
                    Q.real, Q.imag + 0.1, "".join(str(x) for x in B[0]), ha="center"
                )
                axe.plot(circle.real, circle.imag, linestyle="--", color="black")

        return axe


class QPSKDemapping(ProcessingBlock):
    def __init__(self):
        self.mu = 2
        self.inv_sq_2 = 1 / sqrt(2)

    def __update__(self, data: np.array) -> np.array:
        n = len(data)
        bitstream = np.empty((n, 2), dtype=np.int64)

        bitstream[:, 0] = (np.sign(data.real) + 1) / 2
        bitstream[:, 1] = (np.sign(data.imag) + 1) / 2

        return bitstream

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
        qm = QPSKMapping()
        axe.plot(circle.real, circle.imag, linestyle="--", color="black")
        for qam, bits in zip(self.inp, self.out):
            hard = qm.process(np.array([bits]))[0]
            axe.plot([qam.real, hard.real], [qam.imag, hard.imag], "b-o")
            axe.plot([hard.real], [hard.imag], "ro")

        return axe
