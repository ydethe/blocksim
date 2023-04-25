import pytest
import numpy as np
from numpy import pi
from scipy import linalg as lin

from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.dsp.DSPAWGN import DSPAWGN
from blocksim.graphics.BFigure import FigureFactory
from blocksim.Simulation import Simulation

from blocksim.dsp import createGNSSSequence
from blocksim.dsp.PSKMod import PSKMapping, PSKDemapping


from blocksim.testing import TestBase


class TestBPSK(TestBase):
    def test_bpsk(self):
        mapping = [0, pi]

        psk_co = PSKMapping(name="map", mapping=mapping, output_size=1)
        psk_dec = PSKDemapping(name="demap", mapping=mapping, output_size=1)

        ntot = 256
        data = np.random.randint(low=0, high=2, size=(psk_co.mu, ntot))

        qpsk_payload = psk_co.process(data)
        data2 = psk_dec.process(qpsk_payload)

        self.assertAlmostEqual(np.max(np.abs(data - data2)), 0, delta=1e-9)

    def test_bpsk_noise(self):
        mapping = [0, pi]
        ntot = 1023
        fs = 1.023e6

        sim = Simulation()

        bs0 = DSPSignal.fromBinaryRandom(name="bs0", samplingPeriod=1 / fs, size=ntot, seed=9948457)

        psk_co = PSKMapping(name="map", mapping=mapping, output_size=1)
        awgn = DSPAWGN(
            name="awgn",
            mean=np.array([0.0]),
            cov=np.array([[0.05]]),
            dtype=np.complex128,
        )
        self.assertRaises(ValueError, awgn.setCovariance, np.zeros(2))
        self.assertRaises(ValueError, awgn.setMean, np.zeros(2))
        err_cov = lin.norm(awgn.getCovariance() - 0.05)
        err_mean = lin.norm(awgn.getMean() - 0.0)
        self.assertAlmostEqual(err_mean, 0, 0)
        self.assertAlmostEqual(err_cov, 0, 0)

        psk_dec = PSKDemapping(name="demap", mapping=mapping, output_size=1)

        sim.addComputer(bs0)
        sim.addComputer(psk_co)
        sim.addComputer(awgn)
        sim.addComputer(psk_dec)

        sim.connect("bs0.setpoint", "map.input")
        sim.connect("map.output", "awgn.noiseless")
        sim.connect("awgn.noisy", "demap.input")

        tps = bs0.generateXSerie()
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        log.getValue("awgn_noisy_n0")

        ref = log.getValue("bs0_setpoint_bs0")
        est = log.getValue("demap_output_s0")

        ber = len(np.where(ref != est)[0]) / ntot

        self.assertLess(ber, 1e-4)

    @pytest.mark.mpl_image_compare(tolerance=40, savefig_kwargs={"dpi": 150})
    def test_bpsk_spectrum(self):
        p_samp = 7
        prn = createGNSSSequence(
            name="PRN",
            modulation="L1CA",
            sv=1,
            chip_rate=1.023e6,
            samples_per_chip=p_samp,
            bitmap=[0, 1],
        )
        bpsk = PSKMapping(name="bpsk", mapping=[0, pi])

        sim = Simulation()
        sim.addComputer(prn, bpsk)
        sim.connect("PRN.setpoint", "bpsk.input")

        tps = prn.generateXSerie()
        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()
        mod = self.log.getFlattenOutput("bpsk_output", dtype=np.complex128)

        sig = DSPSignal.fromTimeAndSamples(name="sig", tps=tps, y_serie=mod)
        sp = sig.fft()

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(sp, transform=sp.to_db)

        return fig.render()


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    from blocksim.graphics import showFigures

    a = TestBPSK()
    # a.test_bpsk()
    # a.test_bpsk_noise()
    a.test_bpsk_spectrum()

    showFigures()
