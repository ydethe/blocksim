from typing import Any

import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
from nptyping import NDArray
import numpy as np
from numpy import sqrt, pi, cos, sin

from ..constants import *
from ..satellite.Trajectory import Trajectory

__all__ = ["EarthPlotter"]


class EarthPlotter(object):
    """Helper object to plot on a 2D Earth projection

    The methods of this class assume a cartopy compatile axe.
    To create such an axe, you can either use blocksim.EarthPlotter.EarthPlotter.createAxe::

        sp = EarthPlotter()
        axe = sp.createAxe(fig, 1, 1, 1)

    or do it manually::

        axe = fig.add_subplot(111, projection=ccrs.PlateCarree())
        axe.stock_img()
        axe.gridlines(crs=ccrs.PlateCarree())

    Args:
        projection: The projection used for the map.
          See https://scitools.org.uk/cartopy/docs/latest/crs/projections.html

    """

    def __init__(self, projection=ccrs.PlateCarree()):
        self.proj = projection

    def createAxe(self, fig, *args, **kwargs) -> "axe":
        """Creates an axe compatible with cartopy

        Args:
            fig: A matplotlib figure
            *args: Arguments list for fig.add_subplot
            **kwargs: Matplotlib options for axe.gridlines.
              See https://scitools.org.uk/cartopy/docs/latest/matplotlib/geoaxes.html#cartopy.mpl.geoaxes.GeoAxes.gridlines
              for the available options for gridlines

        Returns:
            Axe compatible with cartopy.
            See blocksim.EarthPlotter.EarthPlotter.plotGroundTrack

        """
        axe = fig.add_subplot(*args, projection=self.proj)
        axe.stock_img()
        axe.gridlines(crs=self.proj, **kwargs)
        return axe

    def plotPoint(self, axe, coord: tuple, **kwargs):
        """Draws a point on an axe compatible with cartopy.

        See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html for the possible values in kwargs

        Args:
            axe: A matplotlib axe, compatible with cartopy
            coord: The position of the point, in longitude/latitude (rad)
            **kwargs: Matplotlib options for the plot

        """
        lon, lat = coord
        g_lon = np.array([lon])
        g_lat = np.array([lat])

        if not "marker" in kwargs.keys():
            kwargs["marker"] = "*"

        axe.plot(g_lon * 180 / pi, g_lat * 180 / pi, linestyle="", **kwargs)

    def plotDeviceReach(
        self, axe, coord: tuple, elev_min: float, sat_alt: float, **kwargs
    ):
        """Plots a line that represents the device reach

        See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html for the possible values in kwargs

        Args:
            axe: A matplotlib axe, compatible with cartopy
            coord: The position of the point, in longitude/latitude (rad)
            elev_min: Minimum elevation angle (rad)
            sat_alt: Satellite altitude, **assuming circular orbit** (m)
            **kwargs: Matplotlib options for the plot

        """

        g_lon, g_lat = coord

        # https://scitools.org.uk/cartopy/docs/v0.17/cartopy/geodesic.html#cartopy.geodesic.Geodesic.circle
        r = Req + sat_alt
        d_lim = sqrt(r**2 - Req**2 * cos(elev_min) ** 2) - Req * sin(elev_min)
        alpha_lim = np.arccos((Req**2 + r**2 - d_lim**2) / (2 * r * Req))
        rad = alpha_lim * Req

        g = Geodesic()
        val = g.circle(g_lon * 180 / pi, g_lat * 180 / pi, radius=rad)
        c_lon = val[:, 0]
        c_lat = val[:, 1]

        axe.plot(c_lon, c_lat, **kwargs)

    def plotGroundTrack(
        self, axe, lon: NDArray[Any, Any], lat: NDArray[Any, Any], **kwargs
    ):
        """Draws the ground track on an axe compatible with cartopy.

        See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html for the possible values in kwargs

        Args:
            axe: A matplotlib axe, compatible with cartopy
            lon: Array of longitudes (rad)
            lat: Array of latitudes (rad)
            **kwargs: Matplotlib options for the plot

        """
        lon *= 180 / pi
        lat *= 180 / pi
        dlon = np.abs(np.diff(lon))
        list_ilmax = np.where(dlon > 180)[0]
        ns = len(lon)

        decal = 0
        for k in range(len(list_ilmax)):
            ilmax = list_ilmax[k] + decal
            if ilmax > 0 and ilmax < decal + ns - 1:
                new_lat = (lat[ilmax] + lat[ilmax + 1]) / 2
                lon = np.insert(lon, ilmax + 1, [180, np.nan, -180])
                lat = np.insert(lat, ilmax + 1, [new_lat, np.nan, new_lat])
                decal += 3
        # np.nan intercalés

        axe.plot(lon, lat, **kwargs)

    def plotTrajectory(self, axe, traj: Trajectory, **kwargs):
        """Draws the ground track on an axe compatible with cartopy.

        See https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.plot.html for the possible values in kwargs

        Args:
            axe: A matplotlib axe, compatible with cartopy
            traj: Trajectory instance
            **kwargs: Matplotlib options for the plot

        """
        lon, lat = traj.getGroundTrack()
        if "color" in kwargs.keys():
            col = kwargs.pop("color")
        else:
            col = traj.color

        self.plotGroundTrack(axe, lon, lat, color=col, **kwargs)
