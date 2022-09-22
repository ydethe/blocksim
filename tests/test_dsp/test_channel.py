from pathlib import Path
import sys
import unittest
from datetime import datetime

import pytest
from skyfield.api import utc
import numpy as np
from numpy import exp, pi
from matplotlib import pyplot as plt

from blocksim.control.SetPoint import Step
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.graphics.BFigure import FigureFactory
from blocksim.Simulation import Simulation
from blocksim.utils import azelalt_to_itrf, itrf_to_azeld
from blocksim.constants import Req, c
from blocksim.dsp.DSPChannel import DSPChannel
from blocksim.satellite.Satellite import SGP4Satellite

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestChannel(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 150})
    def test_channel(self):
        t0 = datetime(
            year=2021,
            month=10,
            day=13,
            hour=6,
            minute=0,
            second=0,
            microsecond=0,
            tzinfo=utc,
        )

        sat = SGP4Satellite.fromOrbitalElements(
            name="sat",
            tsync=t0,
            a=Req + 630e3,
            ecc=0,
            argp=0.0,
            inc=56.0 * pi / 180,
            mano=0.0,
            node=0.0,
        )
        pv = sat.getGeocentricITRFPositionAt(0)
        lon, lat = sat.subpoint(itrf_pos_vel=pv)
        self.assertAlmostEqual(lon * 180 / pi, -112.08124806457887, delta=1e-9)
        self.assertAlmostEqual(lat * 180 / pi, -0.0843360495637092, delta=1e-9)

        # Find a point with 45° elevation
        elev_min = pi / 4
        pos = azelalt_to_itrf(azelalt=(-pi, elev_min, 0), sat=pv)
        az, el, dist, azr, elr, vr = itrf_to_azeld(obs=np.hstack((pos, np.zeros(3))), sat=pv)
        self.assertAlmostEqual(az, -pi, delta=2e-8)
        self.assertAlmostEqual(el, elev_min, delta=2e-7)

        stp = Step(
            name="stp",
            snames=["px", "py", "pz", "vx", "vy", "vz"],
            cons=np.hstack((pos, np.zeros(3))),
        )

        chn = DSPChannel(
            name="chn",
            wavelength=c / 2e9,
            antenna_gain=0.0,
            antenna_temp=290.0,
            bandwidth=1e6,
            noise_factor=2.5,
            alpha=[0.3820e-7, 0.1490e-7, -0.1790e-6, 0.0000],
            beta=[0.1430e6, 0.0000, -0.3280e6, 0.1130e6],
        )

        f0 = 10e3
        fs = 100e3
        tps = np.arange(100 * 5) / fs
        x = exp(1j * 2 * pi * f0 * tps) * 100
        inok = np.where(tps > 1e-3)[0]
        x[inok] = 0.0
        sig = DSPSignal.fromTimeAndSamples(name="sig", tps=tps, y_serie=x)

        sim = Simulation()

        sim.addComputer(sat)
        sim.addComputer(stp)
        sim.addComputer(chn)
        sim.addComputer(sig)

        sim.connect("sig.setpoint", "chn.txsig")
        sim.connect("sat.itrf", "chn.txpos")
        sim.connect("stp.setpoint", "chn.rxpos")

        sim.simulate(tps, progress_bar=False)
        log = sim.getLogger()

        rxsig = log.getSignal("chn_rxsig_y")
        sp = rxsig.fft()

        fig = FigureFactory.create()
        gs = fig.add_gridspec(2, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(rxsig, transform=rxsig.to_db_lim(-200))

        axe = fig.add_baxe(title="", spec=gs[1, 0])
        axe.plot(sp)

        return fig.render()


if __name__ == "__main__":
    unittest.main()
    exit(0)

    from blocksim.graphics import showFigures

    a = TestChannel()
    a.setUp()
    a.test_channel()

    showFigures()
