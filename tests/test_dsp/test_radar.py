import sys
from pathlib import Path
import unittest

import pytest
from scipy.signal import correlate
from numpy import pi, exp, sin, cos
import numpy as np
from matplotlib import pyplot as plt

from blocksim.dsp.DSPSignal import DSPSignal
from blocksim import logger
from blocksim.constants import c
from blocksim.graphics import plotSpectrogram, plotDSPLine
from blocksim.dsp import analyse_DV

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestRadar(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 150})
    def test_analyse_dv(self):
        tau = 10e-6
        bp = 5e6
        fs = bp * 3
        eta = 0.1
        nrep = 50
        fdop = 1e3

        rep = DSPSignal.fromLinearFM(
            name="rep",
            samplingStart=0,
            samplingPeriod=1 / fs,
            tau=tau,
            fstart=-bp / 2,
            fend=bp / 2,
        )

        ns_rep = len(rep)

        ns = int(ns_rep / eta)
        tps = np.arange(nrep * ns) / fs
        y_sig = np.zeros(nrep * ns, dtype=np.complex128)
        for k in range(nrep):
            y_sig[k * ns : k * ns + ns_rep] = rep.y_serie * k
        sig = (
            DSPSignal.fromTimeAndSamples(name="sig", tps=tps, y_serie=y_sig)
            .applyDopplerFrequency(fdop=fdop)
            .applyGaussianNoise(pwr=5)
        )

        wl = 0.2
        Tr = tau / eta
        Tana = nrep * Tr
        Rf = 1 / Tana
        Rv = Rf * wl
        Rd = c / bp * 1.4
        # print("Rv: %.2f m/s" % Rv)
        # print("Rd: %.2f ns" % (Rd/c*1e9))
        # vamb = 3 * Rv
        # damb = 3 * Rd
        vamb = wl / 2 / Tr
        # damb = Tr*c/2
        damb = 2 * tau * c

        spg = analyse_DV(
            wavelength=wl,
            period=Tr,
            dist0=tau * c,
            damb=damb,
            vrad0=-wl * fdop,
            vamb=vamb,
            seq=rep,
            rxsig=sig,
            coherent=True,
            nv=100,
            corr_window="hamming",
        )

        trf = DSPSignal.to_db_lim(-80)
        (peak,) = spg.findPeaksWithTransform(transform=trf, nb_peaks=1)
        self.assertAlmostEqual(peak.coord[0], 0, delta=0.05)  # radial velocity
        self.assertAlmostEqual(peak.coord[1], 0, delta=1e-5)  # delay
        self.assertAlmostEqual(peak.value, 27.85, delta=1e-2)

        fig = plt.figure()
        axe = fig.add_subplot(111)
        plotSpectrogram(
            spg,
            axe,
            transform=trf,
            search_fig=False,
            find_peaks=1,
        )
        axe.set_title("Power (dB)")

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestRadar()
    a.setUp()
    a.test_analyse_dv()

    plt.show()
