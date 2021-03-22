import numpy as np
from numpy import sqrt, sign, pi, exp
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy import signal as sig
from matplotlib import pyplot as plt

from OFDM import logger
from OFDM.blocs.ProcessingBlock import ProcessingBlock


def zadoff_chu(u, n):
    k = np.arange(n)
    return exp(-1j * pi * u * k * (k + 1) / n)


class NPSSGenerator(ProcessingBlock):
    def __init__(self):
        pass

    def __update__(self, data: np.array) -> np.array:
        sc_nb = 11

        spec_seq = zadoff_chu(u=5, n=sc_nb)
        mod_symb = np.array([0, 0, 0, 1, 1, 1, 1, -1, -1, 1, 1, 1, -1, 1])
        nb_symb = len(mod_symb)

        npss = np.zeros(sc_nb * nb_symb, dtype=np.complex128)
        for nf in range(nb_symb):
            npss[nf * sc_nb : (nf + 1) * sc_nb] = mod_symb[nf] * spec_seq

        return npss


class NPSSCorrelator(ProcessingBlock):
    def __init__(self, nech_symb):
        self.nech_symb = nech_symb
        self.sc_nb = 11

    def __update__(self, data: np.array) -> np.array:
        nsymb = len(data) // self.nech_symb

        spec_seq = zadoff_chu(u=5, n=self.sc_nb)
        rep = np.conj(ifft(spec_seq, n=self.nech_symb)[::-1])

        # l_corr=2 * self.sc_nb - 1
        l_corr = self.nech_symb
        out = np.empty(l_corr * nsymb, dtype=np.complex128)
        for k in range(nsymb):
            x = data[k * self.nech_symb : (k + 1) * self.nech_symb]
            out[k * l_corr : (k + 1) * l_corr] = fft(rep * x)[:l_corr]

        return out

    def plotOutput(self, dt_us=1, axe=None):
        if axe is None:
            fig = plt.figure()
            axe = fig.add_subplot(111)
            axe.set_xlabel("Time (µs)")
            axe.set_ylabel("SNR (dB)")
            axe.grid(True)

        dt = (self.sc_nb - 1) * dt_us

        n = len(self.out)
        tps = np.arange(n) * dt_us - dt
        axe.plot(
            tps, ProcessingBlock.conv_sig_to_db(self.out), label="NPSS correlation"
        )

        axe.legend(fontsize=10)

        return axe
