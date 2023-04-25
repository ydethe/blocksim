import numpy as np
from numpy import pi, exp

from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.dsp.DSPAWGN import DSPAWGN
from blocksim.control.Route import Group
from blocksim.Simulation import Simulation
from blocksim.dsp.PSKMod import PSKMapping, PSKDemapping


from blocksim.testing import TestBase


class TestQPSK(TestBase):
    def test_qpsk(self):
        mapping = [pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4]

        qpsk_co = PSKMapping(name="map", mapping=mapping, output_size=1)
        qpsk_dec = PSKDemapping(name="demap", mapping=mapping, output_size=2)

        ntot = 256
        data = np.random.randint(low=0, high=2, size=(qpsk_co.mu, ntot))

        qpsk_payload = qpsk_co.process(data)
        data2 = qpsk_dec.process(qpsk_payload)

        self.assertAlmostEqual(np.max(np.abs(data - data2)), 0, delta=1e-9)

    def test_qpsk_noise(self):
        mapping = np.array([pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4])
        ntot = 1023
        fs = 1.023e6

        sim = Simulation()

        b_even = DSPSignal.fromBinaryRandom(
            name="beven", samplingPeriod=1 / fs, size=ntot, seed=9948457
        )
        b_odd = DSPSignal.fromBinaryRandom(
            name="bodd", samplingPeriod=1 / fs, size=ntot, seed=167847
        )

        grp_inp = dict()
        grp_inp["in1"] = (1,)
        grp_inp["in2"] = (1,)
        grp = Group(name="grp", inputs=grp_inp, snames=["even", "odd"], dtype=np.int32)

        qpsk_co = PSKMapping(name="map", mapping=mapping, output_size=1)
        awgn = DSPAWGN(
            name="awgn",
            mean=np.array([0.0]),
            cov=np.array([[0.05]]),
            dtype=np.complex128,
        )
        qpsk_dec = PSKDemapping(name="demap", mapping=mapping, output_size=2)

        sim.addComputer(b_even)
        sim.addComputer(b_odd)
        sim.addComputer(grp)
        sim.addComputer(qpsk_co)
        sim.addComputer(awgn)
        sim.addComputer(qpsk_dec)

        sim.connect("beven.setpoint", "grp.in1")
        sim.connect("bodd.setpoint", "grp.in2")
        sim.connect("grp.grouped", "map.input")
        sim.connect("map.output", "awgn.noiseless")
        sim.connect("awgn.noisy", "demap.input")

        tps = b_even.generateXSerie()
        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        log.getValue("awgn_noisy_n0")
        exp(1j * mapping)

        ref = log.getValue("beven_setpoint_beven")
        est = log.getValue("demap_output_s0")

        self.assertAlmostEqual(np.max(np.abs(ref - est)), 0, delta=1e-9)


if __name__ == "__main__":
    # unittest.main()

    a = TestQPSK()
    a.test_qpsk()
    a.test_qpsk_noise()
