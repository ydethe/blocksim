import sys
from pathlib import Path
import unittest
from datetime import datetime, timezone

from skyfield.api import load
from skyfield.framelib import itrs
import numpy as np
from numpy import cos, sin, sqrt, exp, pi, testing
from scipy import linalg as lin
from matplotlib import pyplot as plt
import pytest

from blocksim.Simulation import Simulation

from blocksim.constants import Req, omega, mu
from blocksim.source.Satellite import (
    SGP4Satellite,
    CircleSatellite,
    createSatellites,
)
from blocksim.utils import *
from blocksim.source.Trajectory import Trajectory
from blocksim.graphics.EarthPlotter import EarthPlotter

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestSatellite(TestBase):
    def test_azeld(self):
        obs = np.array([1, 0, 0, 0, 0, 0], dtype=np.float64)
        sat = np.array([2, 1, 0, 0, 1, 0], dtype=np.float64)
        az, el, dist, vr, vs, va = itrf_to_azeld(obs, sat)
        self.assertAlmostEqual(az, 90, delta=1e-10)
        self.assertAlmostEqual(el, 45, delta=1e-10)
        self.assertAlmostEqual(dist, sqrt(2), delta=1e-10)
        self.assertAlmostEqual(vr, sqrt(2) / 2, delta=1e-10)
        self.assertAlmostEqual(vs, 45, delta=1e-10)
        self.assertAlmostEqual(va, 90, delta=1e-10)

    def test_azeld2(self):
        obs = np.array([6378137.0, -10000, 0, 0, 0, 0])
        ps = np.array([-1.48138924e07, -2.10421715e07, -1.46534295e07])
        vs = np.array([2.70050410e03, -1.76191617e02, -2.47601263e03])
        sat = np.hstack((ps, vs))
        _, _, dist, vr, _, _ = itrf_to_azeld(obs, sat)

        dt = 0.1
        sat2 = np.hstack((ps + vs * dt, vs))
        _, _, dist2, vr2, _, _ = itrf_to_azeld(obs, sat2)

        sat2 = np.hstack((ps + 2 * vs * dt, vs))
        _, _, dist3, vr3, _, _ = itrf_to_azeld(obs, sat2)

        self.assertAlmostEqual(dist2, dist + vr * dt, delta=5e-3)
        self.assertAlmostEqual(vr2, (dist3 - dist) / (2 * dt), delta=1e-7)

    def test_sgp4(self):
        ts = load.timescale()
        pth = Path(__file__).parent / "iss.tle"
        satellite = load.tle_file(str(pth))[0]
        t = ts.utc(2022, 2, 5, 13, 0, 0)
        geocentric = satellite.at(t)
        pos, vel = geocentric.frame_xyz_and_velocity(itrs)
        pos_ref = pos.m
        vel_ref = vel.m_per_s

        t0 = datetime(2022, 2, 5, 13, 0, 0, tzinfo=timezone.utc)
        sat = SGP4Satellite.fromTLE(tsync=t0, tle_file=str(pth))
        pv = sat.getGeocentricITRFPositionAt(0)
        pos = pv[:3]
        vel = pv[3:]

        testing.assert_allclose(actual=pos, desired=pos_ref, rtol=50, equal_nan=True)

        testing.assert_allclose(actual=vel, desired=vel_ref, rtol=2e-5, equal_nan=True)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_circle_satellite(self):
        fig = plt.figure()
        sp = EarthPlotter()
        axe = sp.createAxe(fig, 1, 1, 1)

        tle_pth = Path(__file__).parent / "gs1_gs2.tle"
        # 2021/04/15 12:00:00.0
        t0 = datetime(
            year=2022, month=2, day=3, hour=12, minute=0, second=0, tzinfo=timezone.utc
        )
        sats = createSatellites(tle_file=tle_pth, tsync=t0, prop=SGP4Satellite)

        ns = 200
        tps = np.linspace(0, 5 * 60, ns)
        for s in sats:
            traj = Trajectory(name=s.getName(), color="red")
            for t in tps:
                dat = s.compute_outputs(t1=t, t2=t, itrf=None, subpoint=None)
                x, y, z, vx, vy, vz = dat["itrf"]
                traj.addPosition(x, y, z)
            sp.plotTrajectory(axe, traj, linewidth=4)

        sats = createSatellites(tle_file=tle_pth, tsync=t0, prop=CircleSatellite)
        for s in sats:
            traj = Trajectory(name=s.getName(), color="blue")
            for t in tps:
                dat = s.compute_outputs(t1=t, t2=t, itrf=None, subpoint=None)
                x, y, z, vx, vy, vz = dat["itrf"]
                traj.addPosition(x, y, z)
            sp.plotTrajectory(axe, traj)

        return fig

    def test_satellite(self):
        t_init = datetime(
            year=2020, month=11, day=19, hour=0, minute=0, second=0, tzinfo=timezone.utc
        )
        satellite = SGP4Satellite.fromOrbitalElements(
            name="sat",
            tsync=t_init,
            a=Req + 630e3,  # semi-major axis
            ecc=0,  # eccentricity
            argp=0,  # argument of perigee (radians)
            inc=0,  # inclination (radians)
            mano=0,  # mean anomaly (radians)
            node=0,  # nodeo: right ascension of ascending node (radians)
        )
        pv0 = satellite.compute_outputs(0, 0, subpoint=None, itrf=None)["itrf"]

        r = satellite.orbit_periapsis
        ws = sqrt(mu / r**3)
        t = 2 * pi / (ws - satellite.orbital_precession - omega)
        pv = satellite.compute_outputs(0, t, subpoint=None, itrf=None)["itrf"]

        err = lin.norm(pv - pv0)
        self.assertAlmostEqual(err, 0, delta=1700)

        sp = satellite.subpoint(itrf_pos_vel=pv0)
        lon_ref = -1.020771898706975
        lat_ref = 0.0
        self.assertAlmostEqual(sp[0], lon_ref, delta=1e-9)
        self.assertAlmostEqual(sp[1], lat_ref, delta=1e-9)

        a, ex, ey, hx, hy, lv = satellite.toEquinoctialOrbit()

        sat2 = SGP4Satellite.fromEquinoctialOrbit(
            name="sat2", tsync=t_init, a=a, ex=ex, ey=ey, hx=hx, hy=hy, lv=lv
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
            satellite.tsync.timestamp(), sat2.tsync.timestamp(), delta=0
        )

    def test_iss(self):
        pth = Path(__file__).parent / "iss.tle"
        t0 = datetime.strptime("Feb 5 2022  01:00AM", "%b %d %Y %I:%M%p")
        t0 = t0.replace(tzinfo=timezone.utc)
        satellite = SGP4Satellite.fromTLE(tsync=t0, tle_file=str(pth))

        t_calc = datetime(
            year=2021,
            month=10,
            day=14,
            hour=14,
            minute=38,
            second=10,
            tzinfo=timezone.utc,
        )
        dt = (t_calc - satellite.tsync).total_seconds()
        pv = satellite.getGeocentricITRFPositionAt(dt)
        lon, lat = satellite.subpoint(itrf_pos_vel=pv)
        self.assertAlmostEqual(lat * 180 / pi, 43.576772126654895, delta=5e-4)
        self.assertAlmostEqual(lon * 180 / pi, 0.33138740555705554, delta=5e-4)

        dt = satellite.orbit_period
        t = dt.total_seconds()
        pv = satellite.compute_outputs(0, t, subpoint=None, itrf=None)["itrf"]

        traj = satellite.geocentricITRFTrajectory(
            number_of_periods=1, number_of_position=100
        )
        self.assertEqual(len(traj), 100)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_ground_track(self):
        pt = (-74.0542275, 40.7004153)

        pth = Path(__file__).parent / "iss.tle"
        t0 = datetime.strptime("Feb 5 2022  01:00AM", "%b %d %Y %I:%M%p")
        t0 = t0.replace(tzinfo=timezone.utc)
        iss = SGP4Satellite.fromTLE(tsync=t0, tle_file=str(pth))

        sim = Simulation()
        sim.addComputer(iss)

        ns = 200
        tps = np.linspace(0, 14400, ns)
        sim.simulate(tps, progress_bar=False)

        log = sim.getLogger()

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
    a.test_teme_itrf()
    # a.test_satellite()
    # a.test_iss()
    # a.test_ground_track()
    # a.test_circle_satellite()

    # plt.show()
