import sys
from pathlib import Path
import unittest

import pytest
import numpy as np

from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.Simulation import Simulation
from blocksim.graphics.BFigure import FigureFactory
from blocksim.dsp.BOCMod import BOCMapping
from blocksim.dsp import createGNSSSequence

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestBOC(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=40, savefig_kwargs={"dpi": 150})
    def test_boc_spectrum(self):
        fc = 1.023e6
        m = 1
        n = 1
        prn = createGNSSSequence(
            name="PRN", modulation="L1CA", sv=2, chip_rate=fc, samples_per_chip=10
        )
        boc = BOCMapping(name="BOC", f_ref=fc, m=m, n=n)

        sim = Simulation()
        sim.addComputer(prn)
        sim.addComputer(boc)
        sim.connect("PRN.setpoint", "BOC.input")

        tps = prn.generateXSerie()
        sim.simulate(tps, progress_bar=False)
        self.log = sim.getLogger()
        mod = self.log.getFlattenOutput("BOC_output", dtype=np.complex128)

        tps = boc.adaptTimeSerie(tps)
        sig = DSPSignal.fromTimeAndSamples(name="sig", tps=tps, y_serie=mod)

        sp = sig.fft()

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(sp, label="BOC(%i,%i)" % (boc.m, boc.n), transform=sp.to_db)

        return fig.render()

    def test_boc_parallel(self):
        fs = 1.023e6
        p_samp = 10
        n = 1
        m = 1
        boc = BOCMapping(name="BOC", f_ref=fs, m=m, n=n, p_samp=p_samp, input_size=2)
        bits = np.random.randint(0, 2, size=(2, 5)) * 2 - 1

        res = boc.process(bits)

        self.assertEqual(res.shape, (40, 5))

    @pytest.mark.mpl_image_compare(tolerance=10, savefig_kwargs={"dpi": 150})
    def test_boc_autocorr(self):
        fs = 1.023e6
        p_samp = 10
        n = 1
        m = 1
        boc = BOCMapping(name="BOC", f_ref=fs, m=m, n=n, p_samp=p_samp, input_size=1)

        y = boc.boc_seq.autoCorrelation()

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(y, transform=np.real)

        return fig.render()


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    from blocksim.graphics import showFigures

    a = TestBOC()
    a.test_boc_spectrum()
    # a.test_boc_autocorr()
    # a.test_boc_parallel()

    showFigures()
