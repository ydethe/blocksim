from enum import Enum
from abc import ABCMeta, abstractmethod
from pathlib import Path
from typing import Callable, List, Any
import re
from datetime import datetime, timezone

from nptyping import NDArray
import numpy as np
from numpy import pi
from scipy.interpolate import interp2d
import fortranformat as ff

from .. import logger
from .DSPLine import DSPRectilinearLine
from ..utils import find2dpeak, Peak, geocentric_to_geodetic, geodetic_to_geocentric
from ..graphics.GraphicSpec import AxeProjection, DSPMapType
from ..gnss.utils import read_ionex_metadata


__all__ = ["ADSPMap", "DSPRectilinearMap", "DSPPolarMap", "DSPNorthPolarMap"]


class ADSPMap(metaclass=ABCMeta):
    """Representation of a map

    Attributes:
        name: Name of the map
        samplingXStart: First date of the sample of the map
        samplingXPeriod: Time spacing of the map
        samplingYStart: First frequency of the sample of the map
        samplingYPeriod: Frequency spacing of the map
        img: Matrix of complex samples
        default_transform: Function to apply to the samples before plotting
        name_of_x_var: Name of x variable. Default: "Time"
        unit_of_x_var: Unit of x variable. Default: "s"
        name_of_y_var: Name of y variable. Default: "Frequency"
        unit_of_y_var: Unit of y variable. Default: "Hz"

    Args:
        name: Name of the map
        samplingXStart: First date of the sample of the map
        samplingXPeriod: Time spacing of the map
        samplingYStart: First frequency of the sample of the map
        samplingYPeriod: Frequency spacing of the map
        img: Matrix of complex samples
        default_transform: Function to apply to the samples before plotting.
          Shall be vectorized

    """

    def __init__(
        self,
        name: str,
        samplingXStart: float = None,
        samplingXPeriod: float = None,
        samplingYStart: float = None,
        samplingYPeriod: float = None,
        img: NDArray[Any, Any] = None,
        default_transform=np.abs,
    ):
        self.name = name
        self.samplingXStart = samplingXStart
        self.samplingXPeriod = samplingXPeriod
        self.samplingYStart = samplingYStart
        self.samplingYPeriod = samplingYPeriod
        self.img = img
        self.default_transform = default_transform
        self.name_of_x_var = "Time"
        self.unit_of_x_var = "s"
        self.name_of_y_var = "Frequency"
        self.unit_of_y_var = "Hz"

        x = self.generateXSerie()
        y = self.generateYSerie()
        self.__itp2d_re = interp2d(x, y, np.real(img), kind="cubic")
        self.__itp2d_im = interp2d(x, y, np.imag(img), kind="cubic")

    def interpolate(self, x: float, y: float) -> float:
        """Interpolates in the data.
        If x or y is an array, the returned value is a 2D array.
        Else, it is a scalar.

        Args:
            x: X coordinates
            y: Y coordinates

        Returns:
            The interpolated data

        """
        res_re = self.__itp2d_re(x, y)

        if self.img.dtype in [np.complex128, np.complex64]:
            res_im = 1j * self.__itp2d_im(x, y)
        else:
            res_im = 0.0

        res = res_re + res_im
        if not hasattr(x, "__iter__") and not hasattr(y, "__iter__"):
            return res[0]
        else:
            return res

    def sectionX(self, y0: float) -> DSPRectilinearLine:
        """Section of the map following X axis

        Args:
            y0: Y coordinate where the section is made

        Returns:
            A `DSPRectilinearLine` that represents the section

        """
        x = self.generateXSerie()

        z = self.interpolate(x, y0)

        res = DSPRectilinearLine(
            name=self.name,
            samplingStart=x[0],
            samplingPeriod=x[1] - x[0],
            y_serie=z,
            default_transform=self.default_transform,
        )
        res.name_of_x_var = self.name_of_x_var
        res.unit_of_x_var = self.unit_of_x_var

        return res

    def sectionY(self, x0: float) -> DSPRectilinearLine:
        """Section of the map following Y axis

        Args:
            x0: X coordinate where the section is made

        Returns:
            A `DSPRectilinearLine` that represents the section

        """
        y = self.generateYSerie()

        z = self.interpolate(x0, y)

        res = DSPRectilinearLine(
            name=self.name,
            samplingStart=y[0],
            samplingPeriod=y[1] - y[0],
            y_serie=z,
            default_transform=self.default_transform,
        )
        res.name_of_x_var = self.name_of_y_var
        res.unit_of_x_var = self.unit_of_y_var

        return res

    @property
    @abstractmethod
    def dspmap_type(self) -> DSPMapType:
        pass

    def __add__(self, x):
        if issubclass(x.__class__, ADSPMap):
            new_img = self.img + x.img
        else:
            new_img = self.img + x

        res = self.__class__(
            name=self.name,
            samplingXStart=self.samplingXStart,
            samplingXPeriod=self.samplingXPeriod,
            samplingYStart=self.samplingYStart,
            samplingYPeriod=self.samplingYPeriod,
            img=new_img,
            default_transform=self.default_transform,
        )
        res.name_of_x_var = self.name_of_x_var
        res.unit_of_x_var = self.unit_of_x_var
        res.name_of_y_var = self.name_of_y_var
        res.unit_of_y_var = self.unit_of_y_var

        return res

    def __neg__(self):
        res = self.__class__(
            name=self.name,
            samplingXStart=self.samplingXStart,
            samplingXPeriod=self.samplingXPeriod,
            samplingYStart=self.samplingYStart,
            samplingYPeriod=self.samplingYPeriod,
            img=-self.img,
            default_transform=self.default_transform,
        )
        res.name_of_x_var = self.name_of_x_var
        res.unit_of_x_var = self.unit_of_x_var
        res.name_of_y_var = self.name_of_y_var
        res.unit_of_y_var = self.unit_of_y_var

        return res

    def __sub__(self, x):
        return self + (-x)

    def generateXSerie(self, index: int = None) -> NDArray[Any, Any]:
        """Generates the x samples of the map

        Args:
            index: If given, returns only the x coord at the position given by index

        Returns:
            The x coordinate(s)

        """
        n = self.img.shape[1]
        if index is None:
            index = np.arange(n)
        elif index < 0:
            index += n
        x = index * self.samplingXPeriod + self.samplingXStart
        return x

    def generateYSerie(self, index: int = None) -> NDArray[Any, Any]:
        """Generates the y samples of the map

        Args:
            index: If given, returns only the y coord at the position given by index

        Returns:
            The y coordinate(s)

        """
        n = self.img.shape[0]
        if index is None:
            index = np.arange(n)
        elif index < 0:
            index += n
        y = index * self.samplingYPeriod + self.samplingYStart
        return y

    def getAdaptedAxeProjection(self) -> AxeProjection:
        if self.dspmap_type == DSPMapType.RECTILINEAR:
            return AxeProjection.RECTILINEAR
        elif self.dspmap_type == DSPMapType.POLAR:
            return AxeProjection.POLAR
        elif self.dspmap_type == DSPMapType.NORTH_POLAR:
            return AxeProjection.NORTH_POLAR

    def quickPlot(self, **kwargs) -> "blocksim.graphics.BAxe.ABaxe":
        """Quickly plots the map

        Args:
            kwargs: Plotting options

        Returns:
            The ABaxe created

        """
        from ..graphics.BFigure import FigureFactory

        axe = kwargs.pop("axe", None)
        if axe is None:
            fig = FigureFactory.create()
            gs = fig.add_gridspec(1, 1)
            proj = self.getAdaptedAxeProjection()
            axe = fig.add_baxe(title="", spec=gs[0, 0], projection=proj)

        axe.plot(self, **kwargs)

        return axe

    def findPeaksWithTransform(self, transform: Callable = None, nb_peaks: int = 1) -> List[Peak]:
        """Finds the peaks
        The search is performed on the tranformed samples (with the argument *transform*, or the attribute *default_transform*)

        Args:
            transform: A callable applied on samples before looking for the peaks
            nb_peaks: Max number of peaks to seach. Only the highest are kept

        Returns:
            The list of detected peaks, sorted by descreasing value of the peak

        """
        if transform is None:
            transform = self.default_transform

        dat = transform(self.img)
        lpeak = find2dpeak(
            nb_peaks=nb_peaks,
            xd=self.generateXSerie(),
            yd=self.generateYSerie(),
            zd=dat,
            name_of_x_var=self.name_of_x_var,
            unit_of_x_var=self.unit_of_x_var,
            name_of_y_var=self.name_of_y_var,
            unit_of_y_var=self.unit_of_y_var,
        )

        return lpeak

    @classmethod
    def to_db_lim(cls, low: float) -> Callable:
        """Returns a function that turns a complex signal into the power serie of the signal, in dB.

        Args:
            low: The min value to clamp to (dB)

        Returns:
            The function to map on a complex map

        Examples:
            >>> f = DSPRectilinearMap.to_db_lim(low=-80)
            >>> f(1e-3)
            -60.0
            >>> f(1e-4)
            -80.0
            >>> f(1e-5)
            -80.0

        """

        def _to_db(x):
            low_lin = 10 ** (low / 10)
            pwr = np.real(np.conj(x) * x)
            pwr = np.clip(pwr, low_lin, None)
            return 10 * np.log10(pwr)

        return _to_db

    @classmethod
    def to_db(cls, x: NDArray[Any, Any], lim_db: float = -100) -> NDArray[Any, Any]:
        """Converts the samples into their power, in dB.
        If a sample's power is below *low*, the dB value in clamped to *low*.

        Args:
            x: The array of samples
            lim_db: Limit to clamp the power (dB)
                Pass None to avoid any clipping

        Returns:
            The power of the serie *x* in dB

        """
        pwr = np.real(np.conj(x) * x)
        return np.clip(10 * np.log10(pwr), a_min=lim_db, a_max=None)


class DSPRectilinearMap(ADSPMap):
    @property
    def dspmap_type(self) -> DSPMapType:
        return DSPMapType.RECTILINEAR

    @classmethod
    def from_ionex(cls, pth: Path, map_index: int = 0):
        """Builds the TEC map from a IONEX filex. The map stores the TEC in \( 10^{15} \\mathrm{el}/\\mathrm{m}^2 \)

        Args:
            pth: Path to the IONEX file
            map_index: index of the map read in the file

        Returns:
            The TEC map

        """
        # Adapted from : https://notebook.community/daniestevez/jupyter_notebooks/IONEX
        with open(pth.expanduser().resolve()) as f:
            ionex = f.read()

        sections = ionex.split("START OF TEC MAP")
        metadata = read_ionex_metadata(sections[0])
        tecmap = [t for t in sections[1:]][map_index]
        for k in tecmap.split("\n"):
            if "EPOCH OF CURRENT MAP" in k:
                ff_fmt = ff.FortranRecordReader("(6I6,24X)")
                yr, mo, da, hr, mn, sc = ff_fmt.read(k)

                map_epoch = datetime(yr, mo, da, hr, mn, sc, tzinfo=timezone.utc)
                metadata["map_epoch"] = map_epoch
        tecmap = re.split(".*END OF TEC MAP", tecmap)[0]
        img = (
            np.stack(
                [
                    np.fromstring(l, sep=" ")
                    for l in re.split(".*LAT/LON1/LON2/DLON/H\\n", tecmap)[1:]
                ]
            )
            * 10 ** metadata["exponent"]
        )

        ny, nx = img.shape
        x = (
            np.arange(
                metadata["lon1"],
                metadata["lon2"] + metadata["dlon"],
                metadata["dlon"],
            )
            * pi
            / 180
        )
        y = (
            np.arange(
                metadata["lat1"],
                metadata["lat2"] + metadata["dlat"],
                metadata["dlat"],
            )
            * pi
            / 180
        )

        assert len(x) == nx
        assert len(y) == ny

        # IONEX TEC grid are given in geocentric latitudes
        r0 = (metadata["base_radius"] + metadata["hgt1"]) * 1000
        y_geod = np.empty_like(y)
        for k in range(len(y)):
            _, y_geod[k], _ = geocentric_to_geodetic(0, y[k], r0)

        itp = interp2d(x, y_geod, img, kind="cubic")
        img2 = np.empty_like(img)

        for i in range(ny):
            for j in range(nx):
                img2[i, j] = itp(x[j], y[i])

        res = cls(
            name="tec",
            samplingXStart=x[0],
            samplingXPeriod=x[1] - x[0],
            samplingYStart=y[0],
            samplingYPeriod=y[1] - y[0],
            img=img2,
        )
        res.unit_of_x_var = "deg"
        res.unit_of_y_var = "deg"
        res.name_of_x_var = "Longitude"
        res.name_of_y_var = "Latitude"

        res.metadata = metadata

        return res


class DSPPolarMap(ADSPMap):
    @property
    def dspmap_type(self) -> DSPMapType:
        return DSPMapType.POLAR


class DSPNorthPolarMap(ADSPMap):
    @property
    def dspmap_type(self) -> DSPMapType:
        return DSPMapType.NORTH_POLAR
