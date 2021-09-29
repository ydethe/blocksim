import sys
import os
import unittest
from datetime import datetime, timezone
from collections import OrderedDict

import numpy as np
from numpy import cos, sin, sqrt, exp, pi
from skyfield.api import utc
from matplotlib import pyplot as plt
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from blocksim.control.Route import Group
from blocksim.source.Satellite import createSatellites
from blocksim.control.GNSSTracker import GNSSTracker
from blocksim.control.GNSSReceiver import GNSSReceiver
from blocksim.Simulation import Simulation
from blocksim.utils import geodetic_to_itrf, rad


class TestGNSS(TestBase):
    def test_gnss(self):
        fmt = "%Y/%m/%d %H:%M:%S.%f"
        sts = "2021/04/15 09:29:54.996640"
        tsync = datetime.strptime(sts, fmt)
        tsync = tsync.replace(tzinfo=utc)

        satellites = createSatellites("tests/galileo.tle", tsync)
        nsat = len(satellites)

        lon = 1.4415632156260192
        lat = 43.60467117912294
        alt = 143.0
        rec = GNSSReceiver(name="UE", nsat=nsat, lon=lon, lat=lat, alt=alt, tsync=tsync)
        rec.algo = "ranging"
        rec.optim = "trust-constr"

        sim = Simulation()

        nom_coord = ["px", "py", "pz", "vx", "vy", "vz"]

        grp_snames = []
        grp_inp = OrderedDict()
        for k, sat in enumerate(satellites):
            sim.addComputer(sat)

            grp_inp["itrf%i" % k] = (6,)
            grp_snames.extend(["%s%i" % (n, k) for n in nom_coord])

        grp = Group("grp", inputs=grp_inp, snames=grp_snames,)

        uere = 0.0
        ueve = 0.0
        tkr = GNSSTracker(name="tkr", nsat=nsat)
        tkr.elev_mask = 5.0
        tkr.dp = 300.0
        tkr.dv = 50.0
        cov = np.zeros((2 * nsat, 2 * nsat))
        for k in range(nsat):
            cov[2 * k, 2 * k] = uere ** 2
            cov[2 * k + 1, 2 * k + 1] = ueve ** 2
        tkr.setCovariance(cov, oname="measurement")

        # Receiver shall be updated last, so we must add it first
        sim.addComputer(grp)
        sim.addComputer(tkr)
        sim.addComputer(rec)

        for k, sat in enumerate(satellites):
            sim.connect("%s.itrf" % sat.getName(), "grp.itrf%i" % k)

        sim.connect("UE.realpos", "tkr.ueposition")
        sim.connect("grp.grouped", "tkr.state")
        sim.connect("tkr.measurement", "UE.measurements")
        sim.connect("tkr.ephemeris", "UE.ephemeris")

        tps = np.linspace(0, 3, 3)
        sim.simulate(tps, progress_bar=False)

        log = sim.getLogger()

        x_ref, y_ref, z_ref = geodetic_to_itrf(rad(lon), rad(lat), alt)

        x = log.getValue("UE_estpos_x")[-1]
        y = log.getValue("UE_estpos_y")[-1]
        z = log.getValue("UE_estpos_z")[-1]
        dp_est = log.getValue("UE_estclkerror_dp")[-1]

        self.assertAlmostEqual(x, x_ref, delta=0.5)
        self.assertAlmostEqual(y, y_ref, delta=0.5)
        self.assertAlmostEqual(z, z_ref, delta=0.5)
        self.assertAlmostEqual(dp_est, tkr.dp, delta=0.5)


if __name__ == "__main__":
    # unittest.main()

    a = TestGNSS()
    a.test_gnss()

    plt.show()
