import sys
from pathlib import Path
import unittest
from datetime import datetime, timezone

import numpy as np
from numpy import cos, sin, sqrt, exp, pi
from scipy import linalg as lin
from matplotlib import pyplot as plt
import pytest

from blocksim.constants import Req, omega, mu
from blocksim.source.Satellite import Satellite
from blocksim.source.Trajectory import Trajectory
from blocksim.EarthPlotter import EarthPlotter
from blocksim.Simulation import Simulation

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestSatellite(TestBase):
    def test_satellite(self):
        t_init = datetime(
            year=2020, month=11, day=19, hour=0, minute=0, second=0, tzinfo=timezone.utc
        )
        satellite = Satellite.fromOrbitalElements(
            name="sat",
            t=t_init,
            a=Req + 630e3,  # semi-major axis
            ecc=0,  # eccentricity
            argp=0,  # argument of perigee (radians)
            inc=0,  # inclination (radians)
            mano=0,  # mean anomaly (radians)
            node=0,  # nodeo: right ascension of ascending node (radians)
        )
        pv0 = satellite.compute_outputs(0, 0, subpoint=None, itrf=None)["itrf"]

        r = satellite.orbit_periapsis
        ws = sqrt(mu / r ** 3)
        t = 2 * pi / (ws - satellite.orbital_precession - omega)
        pv = satellite.compute_outputs(0, t, subpoint=None, itrf=None)["itrf"]

        err = lin.norm(pv - pv0)
        self.assertAlmostEqual(err, 0, delta=500)

        sp = satellite.subpoint(t_init)
        lon_ref = -1.0207588638091307
        lat_ref = 0.0
        self.assertAlmostEqual(sp[0], lon_ref, delta=1e-9)
        self.assertAlmostEqual(sp[1], lat_ref, delta=1e-9)

        a, ex, ey, hx, hy, lv = satellite.toEquinoctialOrbit()

        sat2 = Satellite.fromEquinoctialOrbit(
            name="sat2", t=t_init, a=a, ex=ex, ey=ey, hx=hx, hy=hy, lv=lv
        )

        self.assertAlmostEqual(satellite.orbit_mano, sat2.orbit_mano, delta=0)
        self.assertAlmostEqual(
            satellite.orbit_eccentricity, sat2.orbit_eccentricity, delta=0
        )
        self.assertAlmostEqual(
            satellite.orbit_semi_major_axis, sat2.orbit_semi_major_axis, delta=0
        )
        self.assertAlmostEqual(
            satellite.orbit_inclination, sat2.orbit_inclination, delta=0
        )
        self.assertAlmostEqual(satellite.orbit_argp, sat2.orbit_argp, delta=0)
        self.assertAlmostEqual(satellite.orbit_node, sat2.orbit_node, delta=0)
        self.assertAlmostEqual(satellite.orbit_bstar, sat2.orbit_bstar, delta=0)
        self.assertAlmostEqual(satellite.orbit_ndot, sat2.orbit_ndot, delta=0)
        self.assertAlmostEqual(satellite.orbit_nddot, sat2.orbit_nddot, delta=0)
        self.assertAlmostEqual(satellite.orbit_periapsis, sat2.orbit_periapsis, delta=0)
        self.assertAlmostEqual(satellite.orbit_apoapsis, sat2.orbit_apoapsis, delta=0)
        self.assertAlmostEqual(
            satellite.orbital_precession, sat2.orbital_precession, delta=0
        )
        self.assertAlmostEqual(satellite.orbit_period, sat2.orbit_period, delta=0)
        self.assertAlmostEqual(
            satellite.epoch.timestamp(), sat2.epoch.timestamp(), delta=0
        )

    def test_iss(self):
        pth = Path(__file__).parent / "iss.tle"
        satellite = Satellite.fromTLE(str(pth))

        t_calc = datetime(
            year=2021,
            month=10,
            day=14,
            hour=14,
            minute=38,
            second=10,
            tzinfo=timezone.utc,
        )
        lon, lat = satellite.subpoint(t_calc)
        self.assertAlmostEqual(lat * 180 / pi, 37.16605088834936, delta=1e-6)
        self.assertAlmostEqual(lon * 180 / pi, 12.690443402340033, delta=1e-6)

        dt = satellite.orbit_period
        t = dt.total_seconds()
        pv = satellite.compute_outputs(0, t, subpoint=None, itrf=None)["itrf"]

        traj = satellite.geocentricITRFTrajectory(
            number_of_periods=1, number_of_position=100
        )
        self.assertEqual(len(traj), 100)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_ground_track(self):
        pt = (-74.0542275, 40.7004153)

        pth = Path(__file__).parent / "iss.tle"
        iss = Satellite.fromTLE(str(pth))

        sim = Simulation()
        sim.addComputer(iss)

        ns = 200
        tps = np.linspace(0, 14400, ns)
        sim.simulate(tps, progress_bar=False)

        log = sim.getLogger()
        lon = log.getValue("deg(iss_subpoint_lon)")
        lat = log.getValue("deg(iss_subpoint_lat)")

        traj = Trajectory.fromLogger(
            log=log,
            name="traj",
            npoint=ns,
            params=("iss_itrf_px", "iss_itrf_py", "iss_itrf_pz"),
            color="red",
        )

        fig = plt.figure()
        ep = EarthPlotter()
        axe = ep.createAxe(fig)
        ep.plotTrajectory(axe, traj)
        ep.plotDeviceReach(axe, coord=pt, elev_min=pi / 8, sat_alt=600e3)
        ep.plotPoint(axe, coord=pt)

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestSatellite()
    # a.test_satellite()
    a.test_iss()
    # a.test_ground_track()

    # plt.show()
