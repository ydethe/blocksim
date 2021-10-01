# ofdm.py
# https://dspillustrations.com/pages/posts/misc/python-ofdm-example.html
import logging

import tqdm
import numpy as np
import scipy.interpolate
from scipy import signal as sig
from numpy import array, sin, cos, pi, sqrt, exp, log10, sign
from scipy.special import erfc
from numpy.fft import fft, ifft, fftshift, fftfreq
import sk_dsp_comm.digitalcom as dc
import sk_dsp_comm.fec_conv as fec
import matplotlib.pyplot as plt

from blocksim import logger
from blocksim.dsp.FEC import FECCoder, FECDecoder
from blocksim.dsp.SerialParallel import SerialToParallel, ParallelToSerial
from blocksim.dsp.PSKMod import PSKMapping, PSKDemapping
from blocksim.dsp.OFDMA import OFDMMapping, OFDMDemapping
from blocksim.dsp.DFT import IDFT, DFT
from blocksim.dsp.Channel import AWGNChannel, AWGNChannelEstimator


class Simu(object):
    def __init__(self):
        self.K = 12  # number of blocksim subcarriers

        # Length of a symbol
        self.nsamp = 2048

        # The known value each pilot transmits
        self.pilotValue = exp(1j * 1.3) * 2

        # Now, let us define some index sets that describe which carriers transmit
        # pilots and which carriers contain payload.
        self.allCarriers = np.arange(
            self.K
        )  # indices of all subcarriers ([0, 1, ... K-1])
        self.pilotCarriers = self.allCarriers[::3]  # Pilots is every (K/P)th carrier.

        # data carriers are all remaining carriers
        self.dataCarriers = np.delete(self.allCarriers, self.pilotCarriers)

        # bits per symbol (i.e. QPSK)
        self.mu = 2

        self.channelResponse = np.array([1])

        np.random.seed(155462157)

        self.fec_co = FECCoder()
        self.sp = SerialToParallel(self.mu)
        self.qpsk_co = QPSKMapping()
        self.ofdm_co = OFDMMapping(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, self.pilotValue
        )
        self.idft = IDFT(self.nsamp)
        self.dft = DFT(self.K, self.nsamp)
        self.chan_est = AWGNChannelEstimator(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, self.pilotValue
        )
        self.ofdm_dec = OFDMDemapping(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, self.pilotValue
        )
        self.qpsk_dec = QPSKDemapping()
        self.ps = ParallelToSerial(self.mu)
        self.fec_dec = FECDecoder()

    def run(self, snr_db, n_bits):
        ######################################
        #                 TX                 #
        ######################################
        bits = np.random.randint(low=0, high=2, size=n_bits)

        fec_bits = self.fec_co.process(bits)

        par = self.sp.process(fec_bits)

        qpsk_payload = self.qpsk_co.process(par)

        ofdm_payload = self.ofdm_co.process(qpsk_payload)

        idft_payload = self.idft.process(ofdm_payload)

        fs = 15000 * 2048
        self.chan = AWGNChannel(
            self.channelResponse, snr_db=snr_db, norm_dop_freq=50e3 / fs
        )
        rx_sig = self.chan.process(idft_payload)

        ######################################
        #                 RX                 #
        ######################################
        ofdm_payload = self.dft.process(rx_sig)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        axe.set_xlabel("Fréquence (kHz)")
        axe.grid(True)
        sp = self.dft.spectrum[:, 0]
        n = sp.shape[0]
        frq = np.arange(n) * 15e3
        axe.plot(frq / 1000, np.abs(sp))

        # equalized_payload = self.chan_est.process(ofdm_payload)

        # qpsk_payload = self.ofdm_dec.process(equalized_payload)

        # par = self.qpsk_dec.process(qpsk_payload)

        # fec_bits = self.ps.process(par)

        # rx_bits = self.fec_dec.process(fec_bits)

        # bit_count, bit_errors = dc.bit_errors(rx_bits, bits)
        bit_count, bit_errors = 1, 0
        return bit_count, bit_errors

    def plot(self):
        # self.chan.plotOutput(dt_us=1e6 / (15000 * 2048))
        axe_odfm = self.ofdm_co.plotOutput(df_khz=15)
        self.dft.plotOutput(df_khz=15, axe=axe_odfm)


def main():
    snr = 6
    sim = Simu()
    # Simu d'un message dont la taille (en nb de bits) est telle que
    # il occupe 3 subframes
    bit_count, bit_errors = sim.run(snr, 16 * (14 * 3 // 3))
    logger.info(
        "Bits Received = %d, Bit errors = %d, BER = %1.2e"
        % (
            bit_count,
            bit_errors,
            bit_errors / bit_count,
        )
    )
    # sim.plot()
    plt.show()


if __name__ == "__main__":
    main()
