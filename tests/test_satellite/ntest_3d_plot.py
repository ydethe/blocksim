from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestB3DPlotter(TestBase):
    def test_3d_plot(self):
        from datetime import datetime, timedelta, timezone

        import numpy as np
        from numpy import sqrt, cos, sin, pi
        from scipy import linalg as lin

        from blocksim.constants import Req
        from blocksim.gnss.GNSSReceiver import GNSSReceiver
        from blocksim.satellite.Satellite import SGP4Satellite
        from blocksim.graphics.B3DPlotter import B3DPlotter

        # Parametres orbite
        t_init = datetime(
            year=2020, month=11, day=19, hour=0, minute=0, second=0, tzinfo=timezone.utc
        )
        satellite = SGP4Satellite.fromOrbitalElements(
            name="sat",
            tsync=t_init,
            a=Req + 630e3,  # semi-major axis
            ecc=0.0001,  # eccentricity
            argp=4.253109078380886 * pi / 180,  # argument of perigee (radians)
            inc=98 * np.pi / 180,  # inclination (radians)
            mano=4.401503952702452 * pi / 180,  # mean anomaly (radians)
            node=0,  # nodeo: right ascension of ascending node (radians)
            bstar=0.21445e-4,
            ndot=0.00000053,
        )
        sim = SGP4Satellite.fromOrbitalElements(
            name="sim",
            tsync=t_init,
            a=Req + 630e3 + 0.608887,  # semi-major axis
            ecc=0.0,  # eccentricity
            argp=0,  # argument of perigee (radians)
            inc=98 * np.pi / 180 + 0.000379,  # inclination (radians)
            mano=(4.253109078380886 + 4.401503952702452) * pi / 180
            - 0.000790,  # mean anomaly (radians)
            node=-0.005174,  # nodeo: right ascension of ascending node (radians)
            bstar=0.0,
            ndot=0.0,
        )

        device = GNSSReceiver(
            name="rec",
            tsync=t_init,
            nsat=1,
            lat=8.50510613183644,
            lon=-59.68293668545234,
            alt=0,
        )

        nb_per = 3
        traj = satellite.geocentricITRFTrajectory(
            number_of_periods=nb_per, color=(1, 0, 0, 1)
        )
        traj_sim = sim.geocentricITRFTrajectory(
            number_of_periods=nb_per, color=(1, 1, 0, 1)
        )

        psat = satellite.getGeocentricITRFPositionAt(0)[:3]
        psat_sim = sim.getGeocentricITRFPositionAt(0)[:3]
        dsat = device.getGeocentricITRFPositionAt(0)[:3]
        u1 = dsat / lin.norm(dsat)
        u2 = (psat - dsat) / lin.norm(psat - dsat)
        print(180 / pi * np.arcsin(u1 @ u2))

        app = B3DPlotter()

        app.plotEarth()

        # app.buildLine(color=(0, 1, 0, 1), itrf_positions=[(0,0,0), app.sun_light.getPos()*Req])
        app.plotLine(color=(0, 1, 0, 1), itrf_positions=[psat, dsat])

        app.plotTrajectory(traj)
        app.plotTrajectory(traj_sim)

        app.plotCube(itrf_position=dsat, size=100e3, color=(0, 0, 1, 1))

        app.plotCube(itrf_position=psat, size=100e3, color=(1, 0, 0, 1))
        app.plotCube(itrf_position=psat_sim, size=100e3, color=(1, 1, 0, 1))

        return app


if __name__ == "__main__":
    a = TestB3DPlotter()
    app = a.test_3d_plot()

    app.run()
