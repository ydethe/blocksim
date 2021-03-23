import numpy as np
from numpy import sqrt, sign, pi, exp, log10
import scipy.interpolate

from . import logger
from ..core.Node import AComputer


class AWGNChannel(AComputer):
    __slots__ = []

    def __init__(self, response, snr_db, norm_dop_freq):
        """

        Args:
          response
          snr_db
          norm_dop_freq
            Fréquence Doppler normalisée par la fréquence d'échantillonnage (15000 * 2048 en NB-IoT)

        """
        self.response = response
        self.snr_db = snr_db
        self.norm_dop_freq = norm_dop_freq

    def __update__(self, data: np.array) -> np.array:
        nr = len(self.response)
        if nr == 1:
            convolved = data * self.response[0]
        else:
            convolved = np.convolve(data, self.response, mode="full")
            convolved = convolved[: -nr + 1]

        signal_power = np.mean(abs(convolved ** 2))

        # Calculate noise power based on signal power and SNR
        # The SNR is multplied by the ratio 2048*15000 / 180e3 = 521 / 3 to take
        # the bandwidth of the receiver into account
        sigma2 = signal_power * 10 ** (-(self.snr_db - 10 * log10(512 / 3)) / 10)

        # Generate complex noise with given variance
        n = len(convolved)
        noise = np.sqrt(sigma2 / 2) * (
            np.random.normal(size=n) + 1j * np.random.normal(size=n)
        )

        t = np.arange(n)
        pha_dop = 2 * pi * self.norm_dop_freq * t
        dop = exp(1j * pha_dop)

        return convolved * dop + noise

    # def plotOutput(self, dt_us=1, axe=None):
        # if axe is None:
            # fig = plt.figure()
            # axe = fig.add_subplot(111)
            # axe.set_xlabel("Time (µs)")
            # axe.set_ylabel("$Re x(t)$")
            # axe.grid(True)

        # n = len(self.out)
        # tps = np.arange(n) * dt_us
        # axe.plot(tps, np.real(self.out), label="RX signal")
        # axe.legend(fontsize=10)

        # return axe

    # def plotTransferFunction(self, nfft, axe=None):
        # if axe is None:
            # fig = plt.figure()
            # axe = fig.add_subplot(111)
            # axe.set_xlabel("Carrier index")
            # axe.set_ylabel("$|H(f)|$")
            # axe.grid(True)

        # H_exact = np.fft.fft(self.response, n=nfft)
        # axe.plot(abs(H_exact), label="Correct Channel")
        # axe.legend(fontsize=10)

        # return axe


class AWGNChannelEstimator(AComputer):
    __slots__ = []

    def __init__(self, allCarriers, pilotCarriers, dataCarriers, pilotValue):
        self.allCarriers = allCarriers
        self.pilotCarriers = pilotCarriers
        self.dataCarriers = dataCarriers
        self.pilotValue = pilotValue

    def estimate(self, data: np.array):
        # extract the pilot values from the RX signal
        pilots = np.mean(data[self.pilotCarriers, :], axis=1)
        Hest_at_pilots = (
            pilots / self.pilotValue
        )  # divide by the transmitted pilot values

        # Perform interpolation between the pilot carriers to get an estimate
        # of the channel in the data carriers.

        # Here, we interpolate real part and imaginary part separately
        Hest_re = scipy.interpolate.interp1d(
            self.pilotCarriers,
            Hest_at_pilots.real,
            fill_value="extrapolate",
            kind="cubic",
        )(self.allCarriers)
        Hest_im = scipy.interpolate.interp1d(
            self.pilotCarriers,
            Hest_at_pilots.imag,
            fill_value="extrapolate",
            kind="cubic",
        )(self.allCarriers)

        self.Hest = Hest_re + 1j * Hest_im
        self.Hest_at_pilots = Hest_at_pilots

    # def plotEstimation(self, axe=None):
        # if axe is None:
            # fig = plt.figure()
            # axe = fig.add_subplot(111)
            # axe.grid(True)
            # axe.set_xlabel("Carrier index")
            # axe.set_ylabel("$|H(f)|$")

        # axe.stem(self.pilotCarriers, abs(self.Hest_at_pilots), label="Pilot estimates")
        # axe.plot(
            # self.allCarriers,
            # abs(self.Hest),
            # label="Estimated channel via interpolation",
        # )
        # axe.legend(fontsize=10)

        # return axe

    def __update__(self, data: np.array):
        self.estimate(data)
        _, nsymb = data.shape
        K = len(self.Hest)
        symb_Hest = np.hstack(nsymb * (self.Hest.reshape((K, 1)),))
        return data / symb_Hest
