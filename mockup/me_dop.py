from datetime import datetime, timezone

import numpy as np
from numpy import sqrt, cos, sin, pi
from scipy import linalg as lin

from blocksim.gnss.utils import computeDOP
from blocksim.utils import (
    azelalt_to_itrf,
    azeld_to_itrf,
    build_local_matrix,
    deg,
    geodetic_to_itrf,
    itrf_to_azeld,
    itrf_to_geodetic,
    itrf_to_llavpa,
    llavpa_to_itrf,
    mean_variance_3dgauss_norm,
    rad,
)
from blocksim.satellite.Satellite import CircleSatellite
from blocksim.constants import Req, c, mu, omega
from blocksim.graphics import showFigures
from blocksim.graphics.BFigure import FigureFactory
from blocksim.graphics.GraphicSpec import AxeProjection


def calcul(alt_sat: float, pos_lat_ue: float, azim_vsat: float, latency: float, nb_epoch: int):
    """
    Args:
        alt_sat: Altitude of the satellite (m)
        pos_lat_ue: Distance between UE and satellite's ground track (m). Positive if
        azim_vsat: Azimuth of the satellite's velocity (deg)

    """
    azim_vsat = rad(azim_vsat)

    t0 = datetime(year=2022, month=7, day=26, hour=13, minute=0, second=0, tzinfo=timezone.utc)
    r = alt_sat + Req
    xs, ys, zs = r, 0.0, 0.0

    vs_env = sqrt(mu / r) * np.array([sin(azim_vsat), cos(azim_vsat), 0])
    M = build_local_matrix(np.array([xs, ys, zs]))
    vxs, vys, vzs = M @ vs_env

    pv_sat0 = np.array([xs, ys, zs, vxs, vys, vzs])
    sat: CircleSatellite = CircleSatellite.fromITRF(name="sat", tsync=t0, pv_itrf=pv_sat0)

    ue_orig = np.array([Req, 0, 0])
    pv_ue_env = pos_lat_ue * np.array([cos(azim_vsat), -sin(azim_vsat), 0.0])
    M = build_local_matrix(ue_orig)
    pv_ue = ue_orig + M @ pv_ue_env
    pv_ue = np.hstack((pv_ue, np.zeros(3)))
    llavpa = itrf_to_llavpa(pv_ue)
    llavpa[2] = 0
    pv_ue = llavpa_to_itrf(llavpa)

    ephem = np.empty(6 * nb_epoch)
    lat = np.empty(nb_epoch)
    lon = np.empty(nb_epoch)
    for k, t in enumerate(np.linspace(0, latency, nb_epoch)):
        pv_sat = sat.getGeocentricITRFPositionAt(t)
        lon[k], lat[k] = sat.subpoint(pv_sat)
        az, el, _, _, _, _ = itrf_to_azeld(obs=pv_ue, sat=pv_sat)
        _, _, alts = itrf_to_geodetic(position=pv_sat)
        # print(
        #     f"t={t:.2f} s, elev_sat={deg(el):.2f} deg, azim_sat={deg(az):.2f} deg, alt={alts/1000:.2f} km"
        # )
        ephem[6 * k : 6 * k + 6] = pv_sat

    Q = computeDOP(algo="ranging", ephem=ephem, pv_ue=pv_ue, elev_mask=0)
    om = Q[:3, :3]
    m, v = mean_variance_3dgauss_norm(om)

    iQ = np.diag_indices(4)
    if np.any(Q[iQ] < 0):
        EDOP = np.nan
        NDOP = np.nan
        VDOP = np.nan
        TDOP = np.nan
        PDOP = np.nan
    else:
        EDOP = np.sqrt(Q[0, 0])
        NDOP = np.sqrt(Q[1, 1])
        VDOP = np.sqrt(Q[2, 2])
        TDOP = np.sqrt(Q[3, 3])
        PDOP = lin.norm((EDOP, VDOP, NDOP))

    # print(f"EDOP: {EDOP:.3f}")
    # print(f"NDOP: {NDOP:.3f}")
    # print(f"VDOP: {VDOP:.3f}")
    # print(f"PDOP: {PDOP:.3f}")
    # print(f"TDOP: {TDOP:.3f}")
    # print(f"With 50 ns UERE: {PDOP*50e-9*c*1e-3:.3f} km")

    # llavpa = itrf_to_llavpa(pv_ue)

    # fig = FigureFactory.create(title="Geometry")
    # gs = fig.add_gridspec(1, 1)
    # axe = fig.add_baxe("", gs[0, 0], projection=AxeProjection.PLATECARREE)
    # axe.plot((lon, lat))
    # axe.scatter((np.array([llavpa[0]]), np.array([llavpa[1]])))

    print(
        ",".join(
            [
                str(alt_sat / 1000),
                str(pos_lat_ue / 1000),
                str(deg(azim_vsat)),
                str(latency),
                str(nb_epoch),
                # str(EDOP),
                # str(NDOP),
                # str(VDOP),
                # str(PDOP),
                # str(TDOP),
                # str(PDOP * 50e-9 * c * 1e-3),
                str(sqrt(v)),
                str(sqrt(v) * 50e-9 * c * 1e-3),
            ]
        )
    )


print(
    f"Scenario Hsat (km), pos UE (km), azimut vsat (deg), latency (s), nb epochs, EDOP, NDOP, VDOP, PDOP, TDOP, Global error with 50 ns UERE"
)

for azim_vsat in [0, 90]:
    for pos_ue in [0, 12.5e3, 50e3]:
        for latency in [500]:
            calcul(
                alt_sat=600e3, pos_lat_ue=pos_ue, azim_vsat=azim_vsat, latency=latency, nb_epoch=5
            )

# showFigures()
