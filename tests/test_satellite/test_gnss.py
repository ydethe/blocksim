import sys
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
from numpy import cos, sin, sqrt, exp, pi, nan, testing
from skyfield.api import utc
from matplotlib import pyplot as plt
import pytest

from blocksim.control.Route import Group
from blocksim.Simulation import Simulation
from blocksim.utils import rad

from blocksim.utils import geodetic_to_itrf
from blocksim.satellite.Satellite import createSatellites
from blocksim.gnss.GNSSTracker import GNSSTracker
from blocksim.gnss.GNSSReceiver import GNSSReceiver

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestGNSS(TestBase):
    def setUp(self):
        TestBase.setUp(self)

        fmt = "%Y/%m/%d %H:%M:%S.%f"
        sts = "2021/04/15 09:29:54.996640"
        tsync = datetime.strptime(sts, fmt)
        tsync = tsync.replace(tzinfo=utc)

        pth = Path(__file__).parent / "galileo.tle"
        satellites = createSatellites(str(pth), tsync)
        nsat = len(satellites)

        lon = 1.4415632156260192
        lat = 43.60467117912294
        alt = 143.0

        x_ref, y_ref, z_ref = geodetic_to_itrf(rad(lon), rad(lat), alt)

        rec = GNSSReceiver(name="UE", nsat=nsat, lon=lon, lat=lat, alt=alt, tsync=tsync)
        rec.algo = "ranging"
        rec.optim = "trust-constr"

        sim = Simulation()

        nom_coord = ["px", "py", "pz", "vx", "vy", "vz"]

        grp_snames = []
        grp_inp = dict()
        for k, sat in enumerate(satellites):
            sim.addComputer(sat)

            grp_inp["itrf%i" % k] = (6,)
            grp_snames.extend(["%s%i" % (n, k) for n in nom_coord])

        grp = Group(
            "grp",
            inputs=grp_inp,
            snames=grp_snames,
        )

        uere = 0.0
        ueve = 0.0
        tkr = GNSSTracker(name="tkr", nsat=nsat)
        tkr.elev_mask = 5.0
        tkr.dp = 300.0 * 0
        tkr.dv = 50.0 * 0
        cov = np.zeros((2 * nsat, 2 * nsat))
        for k in range(nsat):
            cov[2 * k, 2 * k] = uere**2
            cov[2 * k + 1, 2 * k + 1] = ueve**2
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

        self.sim = sim
        self.x_ref = x_ref
        self.y_ref = y_ref
        self.z_ref = z_ref
        self.rec = rec
        self.tkr = tkr
        self.nsat = nsat

        self.meas_ref = np.array(
            [
                nan,
                nan,
                nan,
                nan,
                2.62293241e07,
                5.71638940e02,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                2.72965169e07,
                -5.49811518e02,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                2.68422624e07,
                -3.57744855e02,
                nan,
                nan,
                nan,
                nan,
                2.35502306e07,
                -9.55992754e01,
                nan,
                nan,
                nan,
                nan,
                2.45367209e07,
                1.38689940e02,
                nan,
                nan,
                2.45213992e07,
                2.97477727e02,
                nan,
                nan,
                nan,
                nan,
                2.68662616e07,
                -4.08471655e02,
                2.79461122e07,
                2.24106211e02,
                nan,
                nan,
                nan,
                nan,
                2.63652561e07,
                -2.34177719e02,
            ]
        )
        self.ephem_ref = np.array(
            [
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                2.90598289e07,
                -3.36238001e06,
                -4.49993023e06,
                -4.52970927e02,
                4.77088949e01,
                -2.95481327e03,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                2.19432761e07,
                1.94303026e07,
                -4.11674854e06,
                3.83309524e02,
                2.08090076e02,
                3.02186415e03,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                -1.17375627e06,
                -1.70722449e07,
                2.41612361e07,
                2.35710354e03,
                -5.53478167e02,
                -2.75419805e02,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                1.79918553e07,
                -6.75327682e06,
                2.25075292e07,
                -7.16778009e02,
                2.14341784e03,
                1.21640558e03,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                2.00971851e07,
                -1.55749296e07,
                1.51666779e07,
                1.47509794e03,
                -3.65388043e02,
                -2.33120489e03,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                1.18771506e07,
                1.42133682e07,
                2.30841044e07,
                -6.51220645e02,
                2.22449530e03,
                -1.03385270e03,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                1.36777615e07,
                -2.48453264e07,
                8.46856337e06,
                -3.48165812e02,
                7.81673971e02,
                2.85671344e03,
                -8.93540357e06,
                1.38072106e07,
                2.46169550e07,
                -1.68517924e03,
                -1.80827821e03,
                4.03583849e02,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                nan,
                9.10630339e06,
                2.38078901e07,
                1.50421077e07,
                -9.47094081e02,
                -1.18407086e03,
                2.44926006e03,
            ]
        )

    def test_gnss_ranging(self):
        tps = np.linspace(0, 3, 3)
        self.sim.simulate(tps, progress_bar=False)

        log = self.sim.getLogger()

        x = log.getValue("UE_estpos_x")[-1]
        y = log.getValue("UE_estpos_y")[-1]
        z = log.getValue("UE_estpos_z")[-1]
        dp_est = log.getValue("UE_estclkerror_dp")[-1]
        vissat = log.getValue("tkr_vissat_n")[-1]
        meas = log.getMatrixOutput("tkr_measurement")[:, -1]
        ephem = log.getMatrixOutput("tkr_ephemeris")[:, -1]

        testing.assert_allclose(
            actual=meas, desired=self.meas_ref, rtol=5e-4, equal_nan=True
        )
        testing.assert_allclose(
            actual=ephem, desired=self.ephem_ref, rtol=5e-4, equal_nan=True
        )
        self.assertEqual(vissat, 9)

        self.assertAlmostEqual(x, self.x_ref, delta=0.5)
        self.assertAlmostEqual(y, self.y_ref, delta=0.5)
        self.assertAlmostEqual(z, self.z_ref, delta=0.5)
        self.assertAlmostEqual(dp_est, self.tkr.dp, delta=0.5)

    def test_gnss_bancroft(self):
        self.rec.optim = "bancroft"

        tps = np.linspace(0, 3, 3)
        self.sim.simulate(tps, progress_bar=False)

        log = self.sim.getLogger()

        x = log.getValue("UE_estpos_x")[-1]
        y = log.getValue("UE_estpos_y")[-1]
        z = log.getValue("UE_estpos_z")[-1]
        dp_est = log.getValue("UE_estclkerror_dp")[-1]

        self.assertAlmostEqual(x, self.x_ref, delta=0.5)
        self.assertAlmostEqual(y, self.y_ref, delta=0.5)
        self.assertAlmostEqual(z, self.z_ref, delta=0.5)
        self.assertAlmostEqual(dp_est, self.tkr.dp, delta=0.5)

    def test_gnss_doppler(self):
        self.rec.algo = "doppler"

        tps = np.linspace(0, 3, 3)
        self.sim.simulate(tps, progress_bar=False)

        log = self.sim.getLogger()

        pr = log.getRawValue("tkr_measurement_pr2")
        vr = log.getRawValue("tkr_measurement_vr2")

        x = log.getRawValue("UE_estpos_x")[-1]
        y = log.getRawValue("UE_estpos_y")[-1]
        z = log.getRawValue("UE_estpos_z")[-1]
        dv_est = log.getRawValue("UE_estclkerror_dv")[-1]

        self.assertAlmostEqual(x, self.x_ref, delta=0.5)
        self.assertAlmostEqual(y, self.y_ref, delta=0.5)
        self.assertAlmostEqual(z, self.z_ref, delta=0.5)
        self.assertAlmostEqual(dv_est, self.tkr.dv, delta=0.5)

    def test_gnss_dv(self):
        self.rec.algo = "doppler-ranging"

        tps = np.linspace(0, 3, 3)
        self.sim.simulate(tps, progress_bar=False)

        log = self.sim.getLogger()

        x = log.getValue("UE_estpos_x")[-1]
        y = log.getValue("UE_estpos_y")[-1]
        z = log.getValue("UE_estpos_z")[-1]
        dp_est = log.getValue("UE_estclkerror_dp")[-1]
        dv_est = log.getValue("UE_estclkerror_dv")[-1]

        self.assertAlmostEqual(x, self.x_ref, delta=0.5)
        self.assertAlmostEqual(y, self.y_ref, delta=0.5)
        self.assertAlmostEqual(z, self.z_ref, delta=0.5)
        self.assertAlmostEqual(dp_est, self.tkr.dp, delta=0.5)
        self.assertAlmostEqual(dv_est, self.tkr.dv, delta=0.5)


if __name__ == "__main__":
    # unittest.main()

    a = TestGNSS()
    a.setUp()
    a.test_gnss_ranging()

    # a.setUp()
    # a.test_gnss_bancroft()

    # a.setUp()
    # a.test_gnss_doppler()

    # a.setUp()
    # a.test_gnss_dv()
