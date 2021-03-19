import sys
import os
import unittest
from datetime import datetime, timezone

import numpy as np
from numpy import cos, sin, sqrt, exp
from matplotlib import pyplot as plt
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from blocksim.constants import Req, omega
from blocksim.blocks.Satellite import Satellite


class TestSatellite(TestBase):
    def test_sat_param(self):
        t_init = datetime(
            year=2020, month=11, day=19, hour=0, minute=0, second=0, tzinfo=timezone.utc
        )
        sat = Satellite.fromOrbitalElements(
            name="sat",
            t=t_init,
            a=Req + 630e3,  # semi-major axis
            ecc=0.1,  # eccentricity
            argp=0.2,  # argument of perigee (radians)
            inc=0.3,  # inclination (radians)
            mano=0.4,  # mean anomaly (radians)
            node=0.5,  # nodeo: right ascension of ascending node (radians)
        )
        a, ex, ey, hx, hy, lv = sat.toEquinoctialOrbit()
        sat2 = Satellite.fromEquinoctialOrbit("sat2", t_init, a, ex, ey, hx, hy, lv)
        self.assertAlmostEqual(sat2.orbit_mano, sat.orbit_mano, delta=1e-9)
        self.assertAlmostEqual(
            sat2.orbit_semi_major_axis, sat.orbit_semi_major_axis, delta=1e-9
        )
        self.assertAlmostEqual(
            sat2.orbit_inclination, sat.orbit_inclination, delta=1e-9
        )
        self.assertAlmostEqual(sat2.orbit_argp, sat.orbit_argp, delta=1e-9)
        self.assertAlmostEqual(sat2.orbit_node, sat.orbit_node, delta=1e-9)
        self.assertAlmostEqual(sat2.orbit_bstar, sat.orbit_bstar, delta=1e-9)
        self.assertAlmostEqual(sat2.orbit_ndot, sat.orbit_ndot, delta=1e-9)
        self.assertAlmostEqual(sat2.orbit_nddot, sat.orbit_nddot, delta=1e-9)
        self.assertAlmostEqual(sat2.orbit_periapsis, sat.orbit_periapsis, delta=1e-9)
        self.assertAlmostEqual(sat2.orbit_apoapsis, sat.orbit_apoapsis, delta=1e-9)
        self.assertAlmostEqual(
            sat2.orbital_precession, sat.orbital_precession, delta=1e-9
        )
        self.assertAlmostEqual(sat2.orbit_period, sat.orbit_period, delta=1e-9)
        self.assertAlmostEqual(sat2.epoch, sat.epoch, delta=1e-9)

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

        dt = satellite.orbit_period
        t = dt.total_seconds()
        pv = satellite.compute_outputs(0, t, subpoint=None, itrf=None)["itrf"]

    def test_iss(self):
        satellite = Satellite.fromTLE("tests/iss.tle")
        pv0 = satellite.compute_outputs(0, 0, subpoint=None, itrf=None)["itrf"]

        dt = satellite.orbit_period
        t = dt.total_seconds()
        pv = satellite.compute_outputs(0, t, subpoint=None, itrf=None)["itrf"]


if __name__ == "__main__":
    unittest.main()
