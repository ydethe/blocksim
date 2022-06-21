import yaml
from datetime import datetime, timezone
import requests

import numpy as np
from munch import munchify, Munch
from skyfield.api import utc

from ..Simulation import Simulation
from ..control.Route import Group
from ..satellite.Satellite import (
    CircleSatellite,
    SGP4Satellite,
    createSatellites,
)
from ..utils import rad, deg, geodetic_to_itrf, azelalt_to_itrf, itrf_to_geodetic
from ..gnss.GNSSTracker import GNSSTracker
from ..gnss.GNSSReceiver import GNSSReceiver
from ..graphics import getUnitAbbrev
from .. import logger


def load_config(cfg_file: str, simid: int = -1) -> Munch:
    cfg_file = str(cfg_file)
    if cfg_file.startswith("https://") or cfg_file.startswith("http://"):
        myfile = requests.get(cfg_file, allow_redirects=True, verify=False)
        data = myfile.text
    else:
        data = open(cfg_file, "r")

    try:
        dat = yaml.safe_load(data)
    except yaml.YAMLError as exc:
        logger.error(exc)
        return None

    dat["cfg_file"] = cfg_file

    valid = True
    if not "setup" in dat.keys():
        logger.error("No setup section in configuration file")
        valid = False
    for p in ["tsync"]:
        if not p in dat["setup"].keys():
            logger.error("No %s parameter in setup section" % p)
            valid = False
    if not "logfile" in dat["setup"].keys():
        logger.error("No logfile parameter in setup section")
        valid = False
    elif simid > 0:
        dat["setup"]["logfile"] += "?sim_id=%i" % simid
    if not "nb_points" in dat["setup"].keys():
        dat["setup"]["nb_points"] = 1
    if not "tend" in dat["setup"].keys():
        dat["setup"]["tend"] = dat["setup"]["tsync"]
    if not "n_epoch" in dat["setup"].keys():
        dat["setup"]["n_epoch"] = 1
        dat["setup"]["dt_epoch"] = 0

    fmt = "%Y/%m/%d %H:%M:%S.%f"

    sts = dat["setup"]["tsync"]
    tsync = datetime.strptime(sts, fmt)
    tsync = tsync.replace(tzinfo=utc)
    dat["setup"]["tsync"] = datetime.combine(tsync.date(), tsync.time(), timezone.utc)

    sts = dat["setup"]["tend"]
    tend = datetime.strptime(sts, fmt)
    tend = tend.replace(tzinfo=utc)
    dat["setup"]["tend"] = datetime.combine(tend.date(), tend.time(), timezone.utc)

    if not "satellites" in dat.keys():
        logger.error("No satellites section in configuration file")
        valid = False
    if not "TLE" in dat["satellites"].keys():
        logger.error("No TLE parameter in satellites section")
        valid = False
    else:
        if len(dat["satellites"]["TLE"]) == 0:
            logger.error("No TLE in satellites section")
            valid = False
    if "prop" in dat["satellites"].keys():
        prop = dat["satellites"]["prop"]
        if not prop in ["circle", "SGP4"]:
            raise AssertionError(
                "satellites.prop=%s. Should be 'circle' or 'SGP4'" % prop
            )
    else:
        dat["satellites"]["prop"] = "SGP4"

    if not "receiver" in dat.keys():
        logger.error("No receiver section")
        valid = False
    else:
        if not "algo" in dat["receiver"].keys():
            logger.error("No %s parameter in receiver section" % p)
            valid = False

        lla_ok = True
        for p in ["lat", "lon", "alt"]:
            if not p in dat["receiver"].keys():
                lla_ok = False

        azeld_ok = True
        for p in ["azimut", "elevation"]:
            if not p in dat["receiver"].keys():
                azeld_ok = False

        if azeld_ok and not lla_ok:
            az = rad(dat["receiver"]["azimut"])
            el = rad(dat["receiver"]["elevation"])
            alt = rad(dat["receiver"]["alt"])
            tle = dat["satellites"]["TLE"][0]
            sats = createSatellites(
                tle, tsync=dat["setup"]["tsync"], prop=CircleSatellite
            )
            sat = sats[0]
            pv0 = sat.getGeocentricITRFPositionAt(0)
            pos_ue = azelalt_to_itrf(azelalt=(az, el, alt), sat=pv0)
            lon, lat, alt = itrf_to_geodetic(pos_ue)
        elif not azeld_ok and lla_ok:
            lon = rad(dat["receiver"]["lon"])
            lat = rad(dat["receiver"]["lat"])
            alt = rad(dat["receiver"]["alt"])
            pos_ue = geodetic_to_itrf(lon, lat, alt)
        else:
            raise ValueError(
                "Incoherent receiver config. Specifiy (lat,lon,alt) OR (azimut,elevation)"
            )

        dat["receiver"]["lon"] = lon
        dat["receiver"]["lat"] = lat
        dat["receiver"]["alt"] = alt

        if not "optim" in dat["receiver"].keys():
            dat["receiver"]["optim"] = "trust-constr"

    if not "tracker" in dat.keys():
        valid = False
        logger.error("No tracker section")
    else:
        for p in ["dp", "dv", "elev_mask", "uere", "ueve"]:
            if not p in dat["tracker"].keys():
                logger.error("No %s parameter in tracker section" % p)
                valid = False
        dat["tracker"]["elev_mask"] = rad(dat["tracker"]["elev_mask"])

    if not valid:
        return None

    cfg = munchify(dat)

    return cfg


def create_simulation(cfg: Munch) -> Simulation:
    logfile = cfg.setup.logfile

    satellites = []
    prop = cfg.satellites.prop
    if prop == "circle":
        cls = CircleSatellite
    elif prop == "SGP4":
        cls = SGP4Satellite
    for tle_list in cfg.satellites.TLE:
        new_sat = createSatellites(tle_list, cfg.setup.tsync, prop=cls)
        satellites.extend(new_sat)
    nsat = len(satellites)

    nom_coord = ["px", "py", "pz", "vx", "vy", "vz"]

    sim = Simulation()

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

    sim.addComputer(grp)

    for k, sat in enumerate(satellites):
        sim.connect("%s.itrf" % sat.getName(), "grp.itrf%i" % k)

    lon = cfg.receiver.lon
    lat = cfg.receiver.lat
    alt = cfg.receiver.alt

    rec = GNSSReceiver(
        name="UE", nsat=nsat, lon=lon, lat=lat, alt=alt, tsync=cfg.setup.tsync
    )
    rec.algo = cfg.receiver.algo
    rec.optim = cfg.receiver.optim

    sim.addComputer(rec)

    em = cfg.tracker.elev_mask
    dp = cfg.tracker.dp
    dv = cfg.tracker.dv
    uere = cfg.tracker.uere
    ueve = cfg.tracker.ueve
    tkr = GNSSTracker(name="tkr", nsat=nsat)
    tkr.elev_mask = em
    tkr.dp = dp
    tkr.dv = dv
    cov = np.zeros((2 * nsat, 2 * nsat))
    for k in range(nsat):
        cov[2 * k, 2 * k] = uere**2
        cov[2 * k + 1, 2 * k + 1] = ueve**2
    tkr.setCovariance(cov, oname="measurement")

    # Receiver shall be updated last, so we must add it first
    sim.addComputer(tkr)

    sim.connect("grp.grouped", "tkr.state")

    sim.connect("UE.realpos", "tkr.ueposition")
    sim.connect("tkr.measurement", "UE.measurements")
    sim.connect("tkr.ephemeris", "UE.ephemeris")

    if not cfg.receiver.algo:
        logger.info("grp logged, not the satellites")
        for k, sat in enumerate(satellites):
            sat.setLogged(False)
        grp.setLogged(True)

    return sim


def print_config(cfg):
    msg = "\n"
    msg += 72 * "=" + "\n"
    msg += "File:\t%s\n" % cfg.cfg_file
    msg += 72 * "=" + "\n"
    msg += "Setup\n"
    msg += "  Tsync:\t%s\n" % cfg.setup.tsync
    if cfg.setup.nb_points > 1:
        msg += "  Tend:\t%s\n" % cfg.setup.tend
        msg += "  N. points:\t%i\n" % cfg.setup.nb_points
    if cfg.setup.n_epoch > 1:
        scaled_dt, mult, lbl, unit = getUnitAbbrev(samp=cfg.setup.dt_epoch, unit="s")
        msg += "  Nb. epoch:\t%i\n" % cfg.setup.n_epoch
        msg += "  dt epoch:\t%.3f %s%s\n" % (scaled_dt, lbl, unit)
    msg += "  Logfile:\t%s\n" % cfg.setup.logfile
    msg += "\n"
    msg += "Satellites\n"
    for s in cfg.satellites.TLE:
        msg += "  %s\n" % s
    msg += "\n"
    if hasattr(cfg, "receiver"):
        msg += "Receiver\n"
        msg += "  Longitude (deg):\t%.7f\n" % deg(cfg.receiver.lon)
        msg += "  Latitude (deg):\t%.7f\n" % deg(cfg.receiver.lat)
        msg += "  Altitude (m):\t%.1f\n" % cfg.receiver.alt
        msg += "  Algo PVT:\t%s\n" % cfg.receiver.algo
        msg += "  Optimiseur:\t%s\n" % cfg.receiver.optim
        msg += "\n"
    if hasattr(cfg, "tracker"):
        msg += "Trackers\n"
        msg += "  Ranging bias (m):\t%.1f\n" % cfg.tracker.dp
        msg += "  Radial velocity bias (m/s):\t%.3f\n" % cfg.tracker.dv
        msg += "  Elevation mask (deg):\t%.1f\n" % deg(cfg.tracker.elev_mask)
        msg += "  UERE (m):\t%.1f\n" % cfg.tracker.uere
        msg += "  UEVE (m/s):\t%.3f\n" % cfg.tracker.ueve
        msg += "\n"
    if hasattr(cfg, "mtcl"):
        msg += "Monte-Carlo\n"
        msg += "  Number simu:\t%i\n" % cfg.mtcl.ns
    msg += 72 * "=" + "\n"
    msg += "\n"

    logger.info(msg)
