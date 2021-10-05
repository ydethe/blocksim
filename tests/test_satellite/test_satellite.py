import sys
from pathlib import Path
import unittest
from datetime import datetime, timezone

import numpy as np
from numpy import cos, sin, sqrt, exp, pi
from matplotlib import pyplot as plt
import pytest

from blocksim.constants import Req, omega
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

        dt = satellite.orbit_period
        t = dt.total_seconds()
        pv = satellite.compute_outputs(0, t, subpoint=None, itrf=None)["itrf"]

    def test_iss(self):
        pth = Path(__file__).parent / "iss.tle"
        satellite = Satellite.fromTLE(str(pth))
        pv0 = satellite.compute_outputs(0, 0, subpoint=None, itrf=None)["itrf"]

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
        ep.plotGroundTrack(axe, lon, lat)
        ep.plotDeviceReach(axe, coord=pt, elev_min=pi / 8, sat_alt=600e3)
        ep.plotPoint(axe, coord=pt)

        return fig


if __name__ == "__main__":
    # unittest.main()

    a = TestSatellite()
    # a.test_satellite()
    a.test_ground_track()

    plt.show()
