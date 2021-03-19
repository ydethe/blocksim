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
from blocksim.EarthPlotter import EarthPlotter


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

        dt = satellite.orbit_period
        t = dt.total_seconds()
        pv = satellite.compute_outputs(0, t, subpoint=None, itrf=None)["itrf"]

    def test_iss(self):
        satellite = Satellite.fromTLE("tests/iss.tle")
        pv0 = satellite.compute_outputs(0, 0, subpoint=None, itrf=None)["itrf"]

        dt = satellite.orbit_period
        t = dt.total_seconds()
        pv = satellite.compute_outputs(0, t, subpoint=None, itrf=None)["itrf"]

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 300})
    def test_ground_track(self):
        iss = Satellite.fromTLE("tests/iss.tle")
        ns = 200
        tps = np.linspace(0, 14400, ns)
        lon = np.empty(ns)
        lat = np.empty(ns)
        for k in range(ns):
            t = tps[k]
            lon[k], lat[k] = iss.compute_outputs(0, t, subpoint=None, itrf=None)[
                "subpoint"
            ]

        fig = plt.figure()
        ep = EarthPlotter()
        axe = ep.createAxe(fig)
        ep.plotGroundTrack(axe, lon * 180 / np.pi, lat * 180 / np.pi)

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestSatellite()
    # a.test_satellite()
    a.test_ground_track()

    plt.show()
