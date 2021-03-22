import unittest

import pytest
import numpy as np

from OFDM import logger
from OFDM.blocs.OFDMA import OFDMMapping
from OFDM.blocs.DFT import IDFT
from OFDM.blocs.Channel import AWGNChannel
from OFDM.blocs.NPSS import NPSSGenerator, NPSSCorrelator
from tests.TestBase import TestBase


class TestNPSS(TestBase):
    def setUp(self):
        self.K = 12  # number of OFDM subcarriers

        # Length of a symbol
        self.nsamp = 2048

        # indices of all subcarriers ([0, 1, ... K-1])
        self.allCarriers = np.arange(self.K)
        self.pilotCarriers = []

        # data carriers are all remaining carriers
        self.dataCarriers = np.arange(self.K - 1)

        self.channelResponse = np.array([1])

        self.npss_gen = NPSSGenerator()
        self.npss_corr = NPSSCorrelator(self.nsamp)
        self.ofdm_co = OFDMMapping(
            self.allCarriers, self.pilotCarriers, self.dataCarriers, pilotValue=0
        )
        self.idft = IDFT(self.nsamp)

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_npss(self):
        ######################################
        #                 TX                 #
        ######################################
        npss_payload = self.npss_gen.process()

        ofdm_payload = self.ofdm_co.process(npss_payload)

        idft_payload = self.idft.process(ofdm_payload)

        ######################################
        #                 RX                 #
        ######################################
        self.chan = AWGNChannel(self.channelResponse, snr_db=5, norm_dop_freq=0)

        rx_sig = self.chan.process(idft_payload)

        corr = self.npss_corr.process(rx_sig)

        axe = self.npss_corr.plotOutput(dt_us=1e6 / (15000 * 2048))

        return axe.figure


if __name__ == "__main__":
    unittest.main()
