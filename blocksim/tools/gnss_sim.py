import os
import datetime

import typer
import rich.progress as rp
import numpy as np
from numpy import sqrt
from scipy import linalg as lin  # type: ignore
from matplotlib import pyplot as plt  # type: ignore

from ..utils import geodetic_to_itrf
from ..gnss.utils import computeDOP
from ..loggers.Logger import Logger
from .. import logger
from .config import load_config, print_config, create_simulation


app = typer.Typer()


@app.command()
def dop(fic_cfg: str):
    """
    Runs a simple simulation
    """
    cfg = load_config(fic_cfg)
    print_config(cfg)
    sim = create_simulation(cfg)
    grp = sim.getComputerByName("grp")

    log = Logger()
    log.loadLogFile(cfg.setup.logfile)

    # Real position
    pv_ue = geodetic_to_itrf(cfg.receiver.lon, cfg.receiver.lat, cfg.receiver.alt)
    pv_ue = np.hstack((pv_ue, np.zeros(3)))

    # PDOP
    ephems = log.getValueForComputer(grp, output_name="grouped")
    nv, ns = ephems.shape

    logger.info("Computing DOP")
    pdop_r = np.empty(ns)
    pdop_d = np.empty(ns)
    pdop_dr = np.empty((2, ns))
    di = np.diag_indices(3)
    for k in rp.track(range(ns)):
        Q = computeDOP(
            algo="ranging",
            ephem=ephems[:, k],
            pv_ue=pv_ue,
            elev_mask=cfg.tracker.elev_mask,
        )
        if Q is None:
            pdop_r[k] = np.nan
        else:
            pdop_r[k] = sqrt(np.sum(Q[di]))

        Q = computeDOP(
            algo="doppler",
            ephem=ephems[:, k],
            pv_ue=pv_ue,
            elev_mask=cfg.tracker.elev_mask,
        )
        if Q is None:
            pdop_d[k] = np.nan
        else:
            pdop_d[k] = sqrt(np.sum(Q[di]))

        Q = computeDOP(
            algo="doppler-ranging",
            ephem=ephems[:, k],
            pv_ue=pv_ue,
            elev_mask=cfg.tracker.elev_mask,
        )
        if Q is None:
            pdop_dr[:, k] = np.nan
        else:
            pdop_dr[:, k] = [sqrt(np.sum(np.real(Q[di]))), sqrt(np.sum(np.imag(Q[di])))]

    logger.info(f"PDOP ranging:\t{np.nanmean(pdop_r)} -")
    logger.info(f"PDOP rad. vel.:\t{np.nanmean(pdop_d)} s")
    dr, dv = np.nanmean(pdop_dr, axis=1)
    logger.info(f"PDOP ranging + rad. vel.:\t{dr} -,\t{dv} s")


@app.command()
def sim(fic_cfg: str):
    """
    Runs a simple simulation
    """
    from .computation import simu as fct_sim

    cfg = load_config(fic_cfg)
    print_config(cfg)
    log = fct_sim(cfg)

    if not hasattr(cfg, "receiver") or cfg.receiver.algo is False:
        return

    if not hasattr(cfg, "tracker") or not hasattr(cfg, "receiver"):
        return

    # Estimated position
    xe = log.getRawValue("UE_estpos_px")[-1]
    ye = log.getRawValue("UE_estpos_py")[-1]
    ze = log.getRawValue("UE_estpos_pz")[-1]
    pos_est = np.array([xe, ye, ze])

    # Real position
    xr = log.getRawValue("UE_realpos_px")[-1]
    yr = log.getRawValue("UE_realpos_py")[-1]
    zr = log.getRawValue("UE_realpos_pz")[-1]
    pos_sim = np.array([xr, yr, zr])
    ur = pos_sim / lin.norm(pos_sim)

    # Position error
    err = pos_est - pos_sim
    err_3d = lin.norm(err)
    err_v = err @ ur
    err_h = np.sqrt(err_3d**2 - err_v**2)

    logger.info("Erreur position (m):\t%.3g" % err_3d)
    logger.info("Erreur position H (m):\t%.3g" % err_h)
    logger.info("Erreur position V (m):\t%.3g" % err_v)

    # PDOP
    sx, sy, sz = (
        log.getRawValue("UE_estdop_sx")[-1],
        log.getRawValue("UE_estdop_sy")[-1],
        log.getRawValue("UE_estdop_sz")[-1],
    )
    if np.isnan(sx):
        pdopp = np.nan
        pdopv = np.nan
    else:
        pdop = np.array([sx, sy, sz])
        pdopp = lin.norm(np.real(pdop))
        pdopv = lin.norm(np.imag(pdop))

    logger.info(f"PDOP (d,v):\t{pdopp:.3g},\t{pdopv:.3g}")


@app.command()
def mtcl(fic_cfg: str):
    """
    Runs a Monte-Carlo simulation
    """
    from .computation import mtcl as fct_mtcl

    cfg = load_config(fic_cfg)
    print_config(cfg)
    fct_mtcl(cfg)

    logger.info("Finished. You can plot the results with 'gnss_sim plot %s'" % fic_cfg)


@app.command()
def hist(
    fic_cfg: str,
    bin: int = typer.Option(5, help="Size of a bin (deg)"),
    save: str = typer.Option("", help="file to write the plot into"),
    simid: int = typer.Option(-1, help="id of the simulation to read"),
):
    """
    Makes a elevation histogram
    """
    from .plot import plot_histogram

    cfg = load_config(fic_cfg, simid=simid)

    fig = plot_histogram(cfg, bin)

    save_str = f"{save}"

    if save_str == "":
        plt.show()
    else:
        fig.savefig(
            save_str,
            dpi=200,
            transparent=True,
            bbox_inches="tight",
            metadata={
                "Title": cfg.cfg_file,  # Short (one line) title or caption for image.
                "Author": os.getlogin(),  # Name of image's creator.
                "Description": open(
                    cfg.cfg_file, "r"
                ).read(),  # Description of image (possibly long).
                "Creation Time": datetime.datetime.utcnow().strftime(
                    "%a, %d %b %Y %H:%M:%S GMT"
                ),  # Time of original image creation (usually RFC 1123 format).
                "Software": "gnss_sim",  # Software used to create the image.
                # Miscellaneous comment; conversion from other image format.
                "Comment": "https://git:8443/projects/DNFSND/repos/blocksim/browse",
            },
        )


@app.command()
def iterate(fic_cfg: str):
    """
    Iterates through one parameter
    """
    from .computation import simu as fct_sim

    cfg = load_config(fic_cfg)
    print_config(cfg)

    lat_list = np.linspace(0, cfg["iterate"]["lat_max"], cfg["iterate"]["nb_lat"])
    min_sat = np.empty_like(lat_list)

    for k, lat in rp.track(enumerate(lat_list)):
        cfg["receiver"]["lat"] = lat
        cfg["setup"]["logfile"] = "data_file_%.1f.log" % lat
        log = fct_sim(cfg)
        n_vissat = log.getRawValue("tkr_vissat_n")
        min_sat[k] = np.min(n_vissat)

    for lat, ms in zip(lat_list, min_sat):
        print(lat, ms)


@app.command()
def plot(
    fic_cfg: str,
    mtcl: bool = typer.Option(False, help="MTCL plotting"),
    dop: bool = typer.Option(False, help="DOP plotting"),
    polar: bool = typer.Option(False, help="polar view plotting"),
    tplot: float = typer.Option(None, help="time of plotting for polar view plotting"),
    tocsv: str = typer.Option(None, help="file to export the polar view data"),
    globe: bool = typer.Option(False, help="3D globe plotting"),
    gtrack: bool = typer.Option(False, help="Ground track plotting"),
    save: str = typer.Option("", help="file to write the plot into"),
    npoint: int = typer.Option(0, help="number of records to read"),
    simid: int = typer.Option(-1, help="id of the simulation to read"),
):
    """
    Plots results
    """
    from .plot import plot_mtcl, plot_polar_view, plot_dop, plot_3d, plot_ground_track

    cfg = load_config(fic_cfg, simid=int(simid))

    if mtcl:
        fig = plot_mtcl(cfg)
    elif dop:
        fig = plot_dop(cfg)
    elif polar:
        fig, data = plot_polar_view(cfg, tplot=tplot)
        if tocsv is not None:
            data.to_csv(tocsv, index=False)
    elif globe:
        app3d = plot_3d(cfg, npoint=int(npoint))
        app3d.run()
        return
    elif gtrack:
        fig = plot_ground_track(cfg, npoint=npoint)
    else:
        logger.error("Please provide the type of plot you want. See the help")

    save_str = f"{save}"

    if save_str == "":
        plt.show()
    else:
        fig.savefig(
            save_str,
            dpi=200,
            transparent=True,
            bbox_inches="tight",
            metadata={
                "Title": cfg.cfg_file,  # Short (one line) title or caption for image.
                "Author": os.getlogin(),  # Name of image's creator.
                "Description": open(
                    cfg.cfg_file, "r"
                ).read(),  # Description of image (possibly long).
                "Creation Time": datetime.datetime.utcnow().strftime(
                    "%a, %d %b %Y %H:%M:%S GMT"
                ),  # Time of original image creation (usually RFC 1123 format).
                "Software": "gnss_sim",  # Software used to create the image.
                # Miscellaneous comment; conversion from other image format.
                "Comment": "https://git:8443/projects/DNFSND/repos/blocksim/browse",
            },
        )


def main():
    app()
