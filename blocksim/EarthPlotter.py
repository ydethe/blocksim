from typing import Tuple, List
from datetime import datetime, timedelta, timezone

import cartopy.crs as ccrs
from cartopy.geodesic import Geodesic
import matplotlib.pyplot as plt
import numpy as np
from numpy import sqrt, pi, cos, sin

from .constants import *


__all__ = ["EarthPlotter"]


class EarthPlotter(object):
    """

    The methods of this class assume a cartopy compatile axe.
    To create such an axe, you can either use :class:`blocksim.EarthPlotter.EarthPlotter.createAxe`::

            sp = EarthPlotter()
            axe = sp.createAxe(fig, 1, 1, 1)

        or do it manually::

            axe = fig.add_subplot(111, projection=ccrs.PlateCarree())
            axe.stock_img()
            axe.gridlines(crs=ccrs.PlateCarree())

    Args:
      projection
        The projection used for the map.
        See https://scitools.org.uk/cartopy/docs/latest/crs/projections.html

    """

    def __init__(self, projection=ccrs.PlateCarree()):
        self.proj = projection

    def createAxe(self, fig, *args, **kwargs) -> "axe":
        """
        Creates an axe compatible with cartopy

        Args:
          fig
            A matplotlib figure
          *args
            Arguments list for fig.add_subplot
          **kwargs
            Matplotlib options for axe.gridlines.
            See https://scitools.org.uk/cartopy/docs/latest/matplotlib/geoaxes.html#cartopy.mpl.geoaxes.GeoAxes.gridlines
            for the available options for gridlines

        Returns:
          Axe compatible with cartopy.
          See :class:`blocksim.STKPlotter.STKPlotter.plotGroundTrack`

        """
        axe = fig.add_subplot(*args, projection=self.proj)
        axe.stock_img()
        axe.gridlines(crs=self.proj, **kwargs)
        return axe

    def plotPoint(self, axe, coord: tuple, **kwargs):
        """
        Draws a point on an axe compatible with cartopy.

        Args:
          axe
            A matplotlib axe, compatible with cartopy
          coord
            The position of the point, in longitude/latitude (deg)
          **kwargs
            Matplotlib options for the plot

        """
        lon, lat = coord
        g_lon = np.array([lon])
        g_lat = np.array([lat])

        axe.plot(g_lon, g_lat, linestyle="", **kwargs)

    def plotDeviceReach(
        self, axe, coord: tuple, elev_min: float, sat_alt: float, **kwargs
    ):
        """

        Args:
          axe
            A matplotlib axe, compatible with cartopy
          coord
            The position of the point, in longitude/latitude (deg)
          elev_min (rad)
            Minimum elevatin angle
          sat_alt (m)
            Satellite altitude, **assuming circular orbit**
          **kwargs
            Matplotlib options for the plot

        """

        g_lon, g_lat = coord

        # https://scitools.org.uk/cartopy/docs/v0.17/cartopy/geodesic.html#cartopy.geodesic.Geodesic.circle
        r = Req + sat_alt
        d_lim = sqrt(r ** 2 - Req ** 2 * cos(elev_min) ** 2) - Req * sin(elev_min)
        alpha_lim = np.arccos((Req ** 2 + r ** 2 - d_lim ** 2) / (2 * r * Req))
        rad = alpha_lim * Req

        g = Geodesic()
        val = g.circle(g_lon, g_lat, radius=rad)
        c_lon = val[:, 0]
        c_lat = val[:, 1]

        axe.plot(c_lon, c_lat, **kwargs)

    def plotGroundTrack(self, axe, lon: np.array, lat: np.array, **kwargs):
        """
        Draws the ground track on an axe compatible with cartopy.

        Args:
          axe
            A matplotlib axe, compatible with cartopy
          lon (deg)
            Array of longitudes
          lat (deg)
            Array of latitudes
          **kwargs
            Matplotlib options for the plot

        """
        dlon = np.abs(np.diff(lon))
        list_ilmax = np.where(dlon > 180)[0]
        ns = len(lon)

        decal = 0
        for k in range(len(list_ilmax)):
            ilmax = list_ilmax[k] + decal
            if ilmax > 0 and ilmax < ns - 1:
                new_lat = (lat[ilmax] + lat[ilmax + 1]) / 2
                lon = np.insert(lon, ilmax + 1, [180, None, -180])
                lat = np.insert(lat, ilmax + 1, [new_lat, None, new_lat])
                decal += 3
        # None intercalés

        axe.plot(lon, lat, **kwargs)
