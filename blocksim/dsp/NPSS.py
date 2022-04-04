import numpy as np
from numpy import sqrt, sign, pi, exp
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy import signal as sig

from blocksim.core.Node import AComputer

from .. import logger
from . import zadoff_chu


class NPSSGenerator(AComputer):
    __slots__ = []

    def __init__(self, name: str):
        AComputer.__init__(self, name=name)

    def __update__(self, data: np.array) -> np.array:
        sc_nb = 11

        spec_seq = zadoff_chu(u=5, n=sc_nb)
        mod_symb = np.array([0, 0, 0, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1])
        nb_symb = len(mod_symb)

        npss = np.zeros(sc_nb * nb_symb, dtype=np.complex128)
        for nf in range(nb_symb):
            npss[nf * sc_nb : (nf + 1) * sc_nb] = mod_symb[nf] * spec_seq

        return npss


class NPSSCorrelator(AComputer):
    __slots__ = []

    def __init__(self, name: str, nech_symb: int):
        AComputer.__init__(self, name=name)
        self.createParameter("nech_symb", value=nech_symb)
        self.createParameter("sc_nb", value=11)
        self.createParameter("u", value=5)

    def __update__(self, data: np.array) -> np.array:
        nsymb = len(data) // self.nech_symb

        spec_seq = zadoff_chu(u=self.u, n=self.sc_nb)
        rep = np.conj(ifft(spec_seq, n=self.nech_symb)[::-1])

        l_corr = self.nech_symb
        out = np.empty(l_corr * nsymb, dtype=np.complex128)
        for k in range(nsymb):
            x = data[k * self.nech_symb : (k + 1) * self.nech_symb]
            out[k * l_corr : (k + 1) * l_corr] = fft(rep * x)[:l_corr]

        return out

    # def plotOutput(self, dt_us=1, axe=None):
    # if axe is None:
    # fig = plt.figure()
    # axe = fig.add_subplot(111)
    # axe.set_xlabel("Time (µs)")
    # axe.set_ylabel("SNR (dB)")
    # axe.grid(True)

    # dt = (self.sc_nb - 1) * dt_us

    # n = len(self.out)
    # tps = np.arange(n) * dt_us - dt
    # axe.plot(
    # tps, ProcessingBlock.conv_sig_to_db(self.out), label="NPSS correlation"
    # )

    # axe.legend(fontsize=10)

    # return axe
