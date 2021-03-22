import numpy as np
from numpy import sqrt, sign
from numpy.fft import fft, ifft, fftshift, fftfreq
from scipy import signal as sig
from matplotlib import pyplot as plt

from blocksim import logger
from blocksim.blocs.ProcessingBlock import ProcessingBlock


class IDFT(ProcessingBlock):
    def __init__(self, nech_symb):
        self.nech_symb = nech_symb

    def __update__(self, data: np.array) -> np.array:
        _, nsymb = data.shape
        n_samp_tot = self.nech_symb * nsymb
        s = np.empty(n_samp_tot, dtype=np.complex128)
        for k in range(nsymb):
            # Indices de début et de fin du symbole k
            k_deb_symb = k * self.nech_symb
            k_fin_symb = (k + 1) * self.nech_symb

            # Calcul du symbole
            symb = ifft(data[:, k], n=self.nech_symb)

            # On met le symbole blocksim dans le signal
            s[k_deb_symb:k_fin_symb] = symb

        return s

    def plotOutput(self, dt_us=1, axe=None):
        if axe is None:
            fig = plt.figure()
            axe = fig.add_subplot(111)
            axe.set_xlabel("Time (µs)")
            axe.set_ylabel("$Re x(t)$")
            axe.grid(True)

        n = len(self.out)
        tps = np.arange(n) * dt_us
        axe.plot(tps, np.real(self.out), label="IDFT signal")
        axe.legend(fontsize=10)

        return axe


class DFT(ProcessingBlock):
    def __init__(self, nb_carriers, nech_symb):
        self.nech_symb = nech_symb
        self.nb_carriers = nb_carriers

    def __update__(self, data: np.array) -> np.array:
        n = len(data)
        nsymb = n // self.nech_symb

        demod = np.empty((self.nb_carriers, nsymb), dtype=np.complex128)
        self.spectrum = np.empty((self.nech_symb, nsymb), dtype=np.complex128)
        for k in range(nsymb):
            # Indices de début et de fin du symbole k
            k_deb_symb = k * self.nech_symb
            k_fin_symb = (k + 1) * self.nech_symb

            buf = data[k_deb_symb:k_fin_symb]
            self.spectrum[:, k] = fft(buf)

            demod[:, k] = self.spectrum[: self.nb_carriers, k]

        return demod

    def plotOutput(self, df_khz=1, axe=None):
        if axe is None:
            fig = plt.figure()
            axe = fig.add_subplot(111)
            axe.set_xlabel("Fréquence (kHz)")
            axe.grid(True)

        n = self.out.shape[0]
        frq = np.arange(n) * df_khz
        axe.plot(frq, np.abs(self.out[:, 0]), label="RX signal, no CP, demod")
        axe.legend(fontsize=10)

        return axe
