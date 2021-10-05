import os
import sys
import unittest

import pytest
import numpy as np
from numpy import exp, pi
import sk_dsp_comm.digitalcom as dc
from matplotlib import pyplot as plt

from blocksim import logger
from blocksim.dsp.FEC import FECCoder, FECDecoder
from blocksim.dsp.SerialParallel import SerialToParallel, ParallelToSerial
from blocksim.dsp.PSKMod import PSKMapping, PSKDemapping
from blocksim.dsp.OFDMA import OFDMMapping, OFDMDemapping
from blocksim.dsp.DFT import IDFT, DFT
from blocksim.dsp.Channel import AWGNChannel, AWGNChannelEstimator

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class TestChaine(TestBase):
    def run_nbiot_sim(self, snr, fig=None):
        fec = FECCoder()
        fec_bits = fec.process(self.bits)

        sp = SerialToParallel(self.mu)
        par = sp.process(fec_bits)

        qpsk = QPSKMapping()
        qpsk_payload = qpsk.process(par)

        blocksim = OFDMMapping(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, self.pilotValue
        )
        ofdm_payload = blocksim.process(qpsk_payload)

        if not fig is None:
            axe_odfm = fig.add_subplot(311)
            axe_odfm.grid(True)
            blocksim.plotOutput(axe=axe_odfm)

        idft = IDFT(self.nsamp)
        idft_payload = idft.process(ofdm_payload)

        chan = AWGNChannel(self.channelResponse, snr_db=snr, norm_dop_freq=0)
        rx_sig = chan.process(idft_payload)

        if not fig is None:
            axe_h_chan = fig.add_subplot(312)
            axe_h_chan.grid(True)
            chan.plotTransferFunction(nfft=self.K, axe=axe_h_chan)

        dft = DFT(self.K, self.nsamp)
        ofdm_payload = dft.process(rx_sig)
        if not fig is None:
            dft.plotOutput(axe=axe_odfm)

        chan_est = AWGNChannelEstimator(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, self.pilotValue
        )

        equalized_payload = chan_est.process(ofdm_payload)
        if not fig is None:
            chan_est.plotEstimation(axe=axe_h_chan)

        blocksim = OFDMDemapping(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, self.pilotValue
        )
        qpsk_payload = blocksim.process(equalized_payload)

        qpsk = QPSKDemapping()
        par = qpsk.process(qpsk_payload)
        if not fig is None:
            axe_qpsk = fig.add_subplot(313)
            axe_qpsk.grid(True)
            axe_qpsk.set_aspect("equal")
            qpsk.plotOutput(axe=axe_qpsk)

        ps = ParallelToSerial(self.mu)
        fec_bits = ps.process(par)

        fec = FECDecoder()
        rx_bits = fec.process(fec_bits)

        bit_count, bit_errors = dc.bit_errors(rx_bits, self.bits)

        return bit_count, bit_errors

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_chaine(self):
        snr = 5

        bit_count, bit_errors = self.run_nbiot_sim(snr)

        ber = bit_errors / bit_count

        self.assertAlmostEqual(ber, 0.405, delta=1e-4)

        fig = plt.figure()

        self.run_nbiot_sim(snr, fig)

        fig.tight_layout()

        return fig


if __name__ == "__main__":
    unittest.main()
