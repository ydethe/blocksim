"""
Satellite ground track
======================

"""
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy import cos, sin, sqrt, exp, pi
from matplotlib import pyplot as plt

from blocksim.Simulation import Simulation
from blocksim.constants import Req, omega
from blocksim.source.Satellite import SGP4Satellite
from blocksim.graphics.EarthPlotter import EarthPlotter


tle_pth = Path(__file__).parent / "iss.tle"
pt = (-74.0542275, 40.7004153)
tsync = datetime(year=2022, month=1, day=1, hour=13, tzinfo=timezone.utc)
iss = SGP4Satellite.fromTLE(tsync=tsync, tle_file=str(tle_pth))

sim = Simulation()
sim.addComputer(iss)

ns = 200
tps = np.linspace(0, 14400, ns)
sim.simulate(tps, progress_bar=False)

log = sim.getLogger()
lon = log.getValue("deg(iss_subpoint_lon)")
lat = log.getValue("deg(iss_subpoint_lat)")

fig = plt.figure()
ep = EarthPlotter()
axe = ep.createAxe(fig)
ep.plotGroundTrack(axe, lon, lat)
ep.plotDeviceReach(axe, coord=pt, elev_min=pi / 8, sat_alt=600e3, color="red")
ep.plotPoint(axe, coord=pt, marker="*", color="red")

plt.show()
