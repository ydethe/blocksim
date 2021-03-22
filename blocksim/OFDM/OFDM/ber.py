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

from OFDM import logger
from OFDM.blocs.FEC import FECCoder, FECDecoder
from OFDM.blocs.SerialParallel import SerialToParallel, ParallelToSerial
from OFDM.blocs.QPSKMod import QPSKMapping, QPSKDemapping
from OFDM.blocs.OFDMA import OFDMMapping, OFDMDemapping
from OFDM.blocs.DFT import IDFT, DFT
from OFDM.blocs.Channel import AWGNChannel, AWGNChannelEstimator


class Simu(object):
    def __init__(self):
        self.K = 12  # number of OFDM subcarriers

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

        # self.channelResponse = np.array(
        # [1 * exp(0.6j), 0.5 * exp(-0.5j), 0.25 * exp(-1j)]
        # )
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

    def run(self, snr_db, n_bits, with_fec=True):
        ######################################
        #                 TX                 #
        ######################################
        bits = np.random.randint(low=0, high=2, size=n_bits)

        if with_fec:
            fec_bits = self.fec_co.process(bits)
        else:
            fec_bits = bits

        par = self.sp.process(fec_bits)

        qpsk_payload = self.qpsk_co.process(par)

        ofdm_payload = self.ofdm_co.process(qpsk_payload)

        idft_payload = self.idft.process(ofdm_payload)

        self.chan = AWGNChannel(self.channelResponse, snr_db=snr_db, norm_dop_freq=0)
        rx_sig = self.chan.process(idft_payload)

        ######################################
        #                 RX                 #
        ######################################
        ofdm_payload = self.dft.process(rx_sig)

        equalized_payload = self.chan_est.process(ofdm_payload)

        qpsk_payload = self.ofdm_dec.process(equalized_payload)

        par = self.qpsk_dec.process(qpsk_payload)

        fec_bits = self.ps.process(par)

        if with_fec:
            rx_bits = self.fec_dec.process(fec_bits)
        else:
            rx_bits = fec_bits

        bit_count, bit_errors = dc.bit_errors(rx_bits, bits)

        return bit_count, bit_errors

    def plot(self):
        # self.chan.plotOutput(dt_us=1e6 / (15000 * 2048))
        self.idft.plotOutput(dt_us=1e6 / (15000 * 2048))
        axe_odfm = self.ofdm_co.plotOutput(df_khz=15)
        self.dft.plotOutput(df_khz=15, axe=axe_odfm)


def mtcl():
    snr = np.arange(-5, 8, 0.5)
    n_snr = len(snr)
    ber = np.zeros(n_snr)
    sim = Simu()

    for k in tqdm.trange(n_snr):
        s = snr[k]

        total_bit_errors = 0
        total_bit_count = 0
        iter = 0
        while total_bit_errors < 300:
            bit_count, bit_errors = sim.run(s, 16 * 2 * 32, with_fec=True)
            total_bit_errors += bit_errors
            total_bit_count += bit_count
            iter += 1
            if iter > 1000:
                break

        logger.info(
            "SNR = %.1f dB, it=%i, Bits Received = %d, Bit errors = %d, BER = %1.2e"
            % (
                s,
                iter,
                total_bit_count,
                total_bit_errors,
                total_bit_errors / total_bit_count,
            )
        )

        ber[k] = total_bit_errors / total_bit_count

    cn0 = snr + 10 * log10(180e3)

    fig = plt.figure(dpi=150)
    axe = fig.add_subplot(111)
    axe.grid(True)
    axe.grid(b=True, which="minor", color="#999999", linestyle="-", alpha=0.2)
    # axe.semilogy(snr, ber)
    axe.semilogy(cn0, ber)
    axe.set_xlabel("$C/N_0$ (dB)")
    axe.set_ylabel("BER")


def single():
    sim = Simu()
    # Simu d'un message dont la taille (en nb de bits) est telle que
    # il occupe 3 subframes
    bit_count, bit_errors = sim.run(5, 16 * (14 * 3 // 3), with_fec=True)
    logger.info(
        "Bits Received = %d, Bit errors = %d, BER = %1.2e"
        % (
            bit_count,
            bit_errors,
            bit_errors / bit_count,
        )
    )
    sim.plot()


if __name__ == "__main__":
    mtcl()
    # single()
    plt.show()
