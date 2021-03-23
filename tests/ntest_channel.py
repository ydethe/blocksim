import os
import sys
import unittest

import pytest
import numpy as np
from numpy import exp, pi, log10
from numpy.fft import fft
from matplotlib import pyplot as plt

from blocksim.dsp.ProcessingBlock import ProcessingBlock
from blocksim.dsp.Channel import AWGNChannel, AWGNChannelEstimator

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase


class TestChannel(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_channel(self):
        fs = 15000 * 2048
        ns = 2048
        df = fs / ns
        # 4th sub carrier
        f0 = np.round(4 * 15e3 / df / 2, 0) * df * 2
        t = np.arange(ns) / fs
        frq = np.arange(ns) / ns * fs
        data = np.exp(1j * pi * 2 * f0 * t)

        chan = AWGNChannel(
            response=self.channelResponse,
            snr_db=5,
            norm_dop_freq=f0 / 2 / fs,
        )
        sig = chan.process(data)

        fig = plt.figure()
        axe = fig.add_subplot(211)
        axe.grid(True)
        axe.set_xlabel("Fréquence (kHz)")

        sp = fft(sig) / ns

        axe.plot(
            frq / 1000,
            ProcessingBlock.conv_sig_to_db(fft(data) / ns),
            label="Théorique",
        )
        axe.plot(frq / 1000, ProcessingBlock.conv_sig_to_db(sp), label="Bruité")
        axe.legend(loc="best")

        axe = fig.add_subplot(212)
        axe.grid(True)
        axe.set_xlabel("Temps (ms)")

        axe.plot(1000 * t, data.real, label="Théorique")
        axe.plot(1000 * t, sig.real, label="Bruité")
        axe.legend(loc="best")

        return fig

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_plots_channel(self):
        fs = 15000 * 2048
        ns = 2048
        df = fs / ns
        # 4th sub carrier
        f0 = np.round(4 * 15e3 / df / 2, 0) * df * 2
        t = np.arange(ns) / fs
        frq = np.arange(ns) / ns * fs
        data = np.exp(1j * pi * 2 * f0 * t)

        chan = AWGNChannel(
            response=self.channelResponse,
            snr_db=5,
            norm_dop_freq=f0 / 2 / fs,
        )
        sig = chan.process(data)

        axe = chan.plotOutput()

        return axe.figure

    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 300})
    def test_channel_transfer_function(self):
        fs = 15000 * 2048
        ns = 2048
        df = fs / ns
        # 4th sub carrier
        f0 = np.round(4 * 15e3 / df / 2, 0) * df * 2
        t = np.arange(ns) / fs
        frq = np.arange(ns) / ns * fs
        data = np.exp(1j * pi * 2 * f0 * t)

        chan = AWGNChannel(
            response=self.channelResponse,
            snr_db=5,
            norm_dop_freq=f0 / 2 / fs,
        )

        axe = chan.plotTransferFunction(nfft=12)

        return axe.figure


if __name__ == "__main__":
    unittest.main()
