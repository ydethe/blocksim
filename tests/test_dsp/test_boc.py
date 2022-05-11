import sys
from pathlib import Path
import unittest

import pytest
import numpy as np
from numpy import pi, exp, log10, sqrt
from matplotlib import pyplot as plt

from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.Simulation import Simulation
from blocksim.graphics import plotDSPLine

from blocksim.dsp.BOCMod import BOCMapping
from blocksim.dsp import createGoldSequence

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestBOC(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=40, savefig_kwargs={"dpi": 150})
    def test_boc_spectrum(self):
        fs = 1.023e6
        p_samp = 10
        n = 1
        m = 1
        boc = BOCMapping(name="BOC", f_ref=fs, m=m, n=n, p_samp=p_samp, input_size=1)
        prn = createGoldSequence(
            name="PRN", sv=[3, 7], chip_rate=1.023e6, sampling_factor=p_samp * m
        )

        sim = Simulation()
        sim.addComputer(prn)
        sim.addComputer(boc)
        sim.connect("PRN.setpoint", "BOC.input")

        tps = prn.generateXSerie()
        sim.simulate(tps, progress_bar=False)
        self.log = sim.getLogger()
        mod = self.log.getFlattenOutput("BOC_output", dtype=np.complex128)

        sig = DSPSignal.fromTimeAndSamples(name="sig", tps=tps, y_serie=mod)

        fig = plt.figure()
        axe = fig.add_subplot(111)

        sp = sig.fft()
        plotDSPLine(sp, axe, label="BOC(%i,%i)" % (boc.m, boc.n), transform=sp.to_db)

        axe.legend()

        return fig

    def test_boc_parallel(self):
        fs = 1.023e6
        p_samp = 10
        n = 1
        m = 1
        boc = BOCMapping(name="BOC", f_ref=fs, m=m, n=n, p_samp=p_samp, input_size=2)
        bits = np.random.randint(0, 2, size=(2, 5)) * 2 - 1

        res = boc.process(bits)

        self.assertEqual(res.shape, (20, 5))

    @pytest.mark.mpl_image_compare(tolerance=10, savefig_kwargs={"dpi": 150})
    def test_boc_autocorr(self):
        fs = 1.023e6
        p_samp = 10
        n = 1
        m = 1
        boc = BOCMapping(name="BOC", f_ref=fs, m=m, n=n, p_samp=p_samp, input_size=1)

        fig = plt.figure()
        axe = fig.add_subplot(111)

        y = boc.boc_seq.autoCorrelation()
        plotDSPLine(y, axe, transform=np.real)

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestBOC()
    a.test_boc_spectrum()
    # a.test_boc_autocorr()
    # a.test_boc_parallel()

    plt.show()
