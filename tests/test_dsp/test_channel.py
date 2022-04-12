from pathlib import Path
import sys
import unittest
from datetime import datetime, timezone

import pytest
from skyfield.api import utc
import numpy as np
from numpy import exp, pi, log10, sqrt, cos, sin
from matplotlib import pyplot as plt

from blocksim.control.SetPoint import Step
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.graphics import plotDSPLine
from blocksim.Simulation import Simulation

from blocksim.utils import geodetic_to_itrf
from blocksim.constants import Req, c
from blocksim.dsp.DSPChannel import DSPChannel
from blocksim.graphics.EarthPlotter import EarthPlotter
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
        r = Req + 630e3
        elev_min = pi / 4
        d_lim = sqrt(r**2 - Req**2 * cos(elev_min) ** 2) - Req * sin(elev_min)
        alpha_lim = np.arccos((Req**2 + r**2 - d_lim**2) / (2 * r * Req))
        rad = alpha_lim * Req
        self.assertAlmostEqual(rad, 550321.5722903715, delta=1e-3)

        from cartopy.geodesic import Geodesic

        g = Geodesic()
        val = g.circle(lon * 180 / pi, lat * 180 / pi, radius=rad)
        c_lon = val[0, 0] * pi / 180
        c_lat = val[0, 1] * pi / 180

        pos = geodetic_to_itrf(c_lon, c_lat, 0.0)
        pos_ref = np.array([-2388997.88859598, -5888920.68052473, 540338.96059696])
        err = np.max(np.abs(pos - pos_ref))
        self.assertAlmostEqual(err, 0, delta=1e-6)

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

        fig = plt.figure()
        axe = fig.add_subplot(211)
        axe.grid(True)
        plotDSPLine(rxsig, axe, transform=rxsig.to_db_lim(-200))

        axe = fig.add_subplot(212)
        axe.grid(True)
        plotDSPLine(sp, axe)

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestChannel()
    a.setUp()
    a.test_channel()

    plt.show()
