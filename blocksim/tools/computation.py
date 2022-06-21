import numpy as np
from munch import Munch

from ..loggers.Logger import Logger
from .. import logger
from .config import create_simulation


def buildTimeArray(cfg: Munch):
    n_epoch = cfg.setup.n_epoch
    dt_epoch = cfg.setup.dt_epoch
    dt = cfg.setup.tend - cfg.setup.tsync
    tsim = dt.total_seconds()
    nbp = cfg.setup.nb_points

    tps = []
    for k in range(nbp):
        if nbp == 1:
            t = 0.0
        else:
            t = k / (nbp - 1) * tsim
        epoch = [t + p * dt_epoch for p in range(n_epoch)]
        tps.extend(epoch)

    tps = np.array(tps)

    return tps


def mtcl(cfg: Munch) -> "figure":
    """
    Runs a Monte-Carlo simulation
    """
    if cfg.setup.logfile == "":
        logger.error("No logfile specified")
        exit(1)

    sim = create_simulation(cfg)
    ns = cfg.mtcl.ns

    sim.simulate(np.linspace(0, 1e-9, ns))
    log = sim.getLogger()
    log.export(cfg.setup.logfile)

    return log


def simu(cfg: Munch) -> Logger:
    """
    Runs a simple simulation
    """
    sim = create_simulation(cfg)
    tps = buildTimeArray(cfg)

    if cfg.setup.nb_points > 1:
        pb = True
    else:
        pb = False

    logger.info("Simulating setup")
    sim.simulate(tps, progress_bar=pb)
    log = sim.getLogger()
    log.export(cfg.setup.logfile)

    return log
