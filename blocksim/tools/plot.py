import rich.progress as rp
import numpy as np
from numpy import pi
from scipy import linalg as lin  # type: ignore
from munch import Munch
import pandas as pd  # type: ignore

from ..loggers.Logger import Logger
from ..satellite.Satellite import ASatellite
from ..satellite.Trajectory import Trajectory
from ..utils import itrf_to_azeld
from ..dsp.DSPSignal import DSPSignal
from ..graphics.BFigure import FigureFactory
from ..graphics.GraphicSpec import AxeProjection, FigureProjection, Annotation
from .. import logger
from .config import create_simulation


def plot_histogram(cfg: Munch, bin: int, axe=None):
    if axe is None:
        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="Mean SV per elevation", spec=gs[0, 0])

    log = Logger()
    log.loadLogFile(cfg.setup.logfile)

    oc = log.getMatrixOutput("tkr_obscoord_elev")
    nsat, ns = oc.shape

    res = np.empty(ns * nsat)

    iel = 0
    for ksat in range(nsat):
        for ktime in range(ns):
            el = oc[ksat, ktime] * 180 / np.pi
            if el >= 0:
                res[iel] = el
                iel += 1

    bins = np.arange(0, 90 + bin, bin)

    serie = DSPSignal(name="elev", samplingStart=0, samplingPeriod=1, y_serie=res[:iel])
    hist = serie.histogram(
        name="hist", bins=bins, density=False, bin_unit="deg", bin_name="Elevation"
    )

    axe.plot(hist / ns)

    return axe.figure


def plot_polar_view(cfg: Munch, tplot=None, axe=None):
    title = cfg.cfg_file

    if axe is None:
        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title=title, spec=gs[0, 0], projection=AxeProjection.NORTH_POLAR)

    list_az = []
    list_el = []

    sim = create_simulation(cfg)
    rec = sim.getComputerByName("UE")
    t0 = rec.tsync

    log = Logger()
    log.loadLogFile(cfg.setup.logfile)

    ns = log.getDataSize()
    if tplot is None:
        kplot = ns - 1
    else:
        t = log.getRawValue("t")
        kplot = np.argmin((t - tplot) ** 2)

    logger.info("Computation time : %s UTC", t0)

    upx = log.getRawValue("UE_realpos_px")[kplot]
    upy = log.getRawValue("UE_realpos_py")[kplot]
    upz = log.getRawValue("UE_realpos_pz")[kplot]
    uvx = log.getRawValue("UE_realpos_vx")[kplot]
    uvy = log.getRawValue("UE_realpos_vy")[kplot]
    uvz = log.getRawValue("UE_realpos_vz")[kplot]
    rpos = np.array([upx, upy, upz, uvx, uvy, uvz])

    dx = log.getRawValue("UE_estdop_sx")[kplot]
    dy = log.getRawValue("UE_estdop_sy")[kplot]
    dz = log.getRawValue("UE_estdop_sz")[kplot]

    if np.isnan(dx):
        pdopp = np.nan
        pdopv = np.nan
    else:
        pdop = np.array([dx, dy, dz])
        pdopp = lin.norm(np.real(pdop))
        pdopv = lin.norm(np.imag(pdop))

    logger.info(f"PDOP (d,v):\t{pdopp:.3g},\t{pdopv:.3g}")

    tkr = sim.getComputerByName("tkr")

    spos = np.empty(6)
    COORDS = ["px", "py", "pz", "vx", "vy", "vz"]
    columns = [
        "Name",
        "Azimut (deg)",
        "Elevation (deg)",
        "Distance (m)",
        "Radial velocity (m/s)",
    ]
    rows = []

    annotations = []
    for comp in rp.track(sim.iterComputersList()):
        if not isinstance(comp, ASatellite):
            continue

        satname = comp.getName()

        for kc, c in enumerate(COORDS):
            name = "%s_itrf_%s" % (satname, c)
            x = log.getRawValue(name)[kplot]
            spos[kc] = x

        # Calcul azimut / elevation
        az, el, dist, vr, _, _ = itrf_to_azeld(rpos, spos)

        # Validation avec le masque d'elevation
        if el > tkr.elev_mask:
            list_el.append(el * 180 / pi)
            list_az.append(az)
            annotations.append(Annotation(coord=(az, el * 180 / pi), text=satname))
            rows.append((satname, az, el, dist, vr))

    data = pd.DataFrame.from_records(data=rows, columns=columns)

    axe.plot((list_az, list_el), annotations=annotations, linestyle="", marker="o")
    axe.set_ylim(90, 0)

    return axe.figure, data


def plot_dop(cfg: Munch):
    title = cfg.cfg_file

    log = Logger()
    log.loadLogFile(cfg.setup.logfile)

    tps = log.getRawValue("t")
    pdop_x = log.getRawValue("UE_estdop_sx")
    pdop_y = log.getRawValue("UE_estdop_sy")
    pdop_z = log.getRawValue("UE_estdop_sz")
    n_vissat = log.getRawValue("tkr_vissat_n")

    ns = len(pdop_x)
    if ns <= 1:
        logger.warning("Only %i time samples simulated" % ns)

    pdopp = np.empty(ns)
    pdopv = np.empty(ns)
    for k in rp.track(range(ns)):
        if np.isnan(pdop_x[k]):
            pdopp[k] = np.nan
            pdopv[k] = np.nan
        else:
            pdop = np.array([pdop_x[k], pdop_y[k], pdop_z[k]])
            pdopp[k] = lin.norm(np.real(pdop))
            pdopv[k] = lin.norm(np.imag(pdop))

    fig = FigureFactory.create()
    gs = fig.add_gridspec(2, 1)

    axe = fig.add_baxe(title=title, spec=gs[0, 0])
    axe.plot(
        (
            {"data": tps, "unit": "s", "name": "Time"},
            {"data": pdopp, "name": r"$DOP_p$"},
        )
    )
    axe.plot(
        (
            {"data": tps, "unit": "s", "name": "Time"},
            {"data": pdopv, "unit": "s", "name": r"$DOP_v$"},
        ),
        twinx=True,
    )

    axe = fig.add_baxe(title=title, spec=gs[1, 0], sharex=axe)
    axe.plot(
        (
            {"data": tps, "unit": "s", "name": "Time"},
            {"data": n_vissat, "name": "N. sat"},
        )
    )

    return fig


def plot_mtcl(cfg: Munch):
    title = cfg.cfg_file

    log = Logger()
    log.loadLogFile(cfg.setup.logfile)

    # Estimated positions
    xe = log.getValue("UE_estpos_px-UE_realpos_px")
    ye = log.getValue("UE_estpos_py-UE_realpos_py")
    ze = log.getValue("UE_estpos_pz-UE_realpos_pz")

    sx = log.getRawValue("UE_estdop_sx")[-1]
    sy = log.getRawValue("UE_estdop_sy")[-1]
    sz = log.getRawValue("UE_estdop_sz")[-1]
    ns = len(xe)
    err3d = np.empty(ns)
    for k in rp.track(range(ns)):
        err3d[k] = lin.norm((xe[k], ye[k], ze[k]))
    err3d = DSPSignal.fromTimeAndSamples(name="err3d", tps=np.arange(ns), y_serie=err3d)

    df = pd.DataFrame(
        {
            "min": [np.min(xe), np.min(ye), np.min(ze)],
            "max": [np.max(xe), np.max(ye), np.max(ze)],
            "avg": [np.mean(xe), np.mean(ye), np.mean(ze)],
            "std": [np.std(xe), np.std(ye), np.std(ze)],
        }
    )
    df.index = ["dx", "dy", "dz"]
    if np.isnan(sx):
        pdop = np.nan * 1j
    else:
        pdop = np.array([sx, sy, sz])
        pdopp = lin.norm(np.real(pdop))
        pdopv = lin.norm(np.imag(pdop))

    logger.info(f"PDOP (d,v):\t{pdopp:.3g},\t{pdopv:.3g}")

    fig = FigureFactory.create()
    gs = fig.add_gridspec(1, 1)
    axe = fig.add_baxe(title=title, spec=gs[0, 0])

    hist3d = err3d.histogram(
        name="hist3d",
        bins="auto",
        density=True,
        cumulative=True,
        bin_unit="m",
        bin_name="3D position error",
    )
    axe.plot(hist3d)

    return fig


def plot_3d(cfg: Munch, npoint: int):

    log = Logger()
    log.loadLogFile(cfg.setup.logfile)

    lsat = []
    for kn in log.getParametersName():
        if kn.endswith("_itrf_px"):
            sn = kn[:-8]
            lsat.append(sn)

    fig = FigureFactory.create(projection=FigureProjection.EARTH3D)
    gs = fig.add_gridspec(1, 1)
    axe = fig.add_baxe(title="", spec=gs[0, 0])

    for sn in rp.track(lsat):
        traj = Trajectory.fromLogger(
            log,
            npoint=npoint,
            name=sn,
            params=("%s_itrf_p%s" % (sn, c) for c in ("x", "y", "z")),
        )
        axe.plot(traj)

    return fig


def plot_ground_track(cfg: Munch, npoint: int):

    log = Logger()
    log.loadLogFile(cfg.setup.logfile)

    lsat = []
    for kn in log.getParametersName():
        if kn.endswith("_itrf_px"):
            sn = kn[:-8]
            lsat.append(sn)

    fig = FigureFactory.create()
    gs = fig.add_gridspec(1, 1)
    axe = fig.add_baxe(title="", spec=gs[0, 0], projection=AxeProjection.PLATECARREE)
    for sn in rp.track(lsat):
        traj = Trajectory.fromLogger(
            log,
            npoint=npoint,
            name=sn,
            params=("%s_itrf_p%s" % (sn, c) for c in ("x", "y", "z")),
            raw_value=True,
        )
        axe.plot(traj)

    return fig
