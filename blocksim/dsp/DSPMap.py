from enum import Enum
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Any

from nptyping import NDArray
import numpy as np

from .. import logger
from ..utils import find2dpeak, Peak
from ..graphics.GraphicSpec import AxeProjection, DSPMapType

__all__ = ["DSPRectilinearMap", "DSPPolarMap", "DSPNorthPolarMap"]


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

    @property
    @abstractmethod
    def dspmap_type(self) -> DSPMapType:
        pass

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

    def quickPlot(self, **kwargs) -> "AxesSubplot":
        """Quickly plots the map

        Args:
            kwargs: Plotting options

        Returns:
            The Axes used by matplotlib

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

    def findPeaksWithTransform(
        self, transform: Callable = None, nb_peaks: int = 1
    ) -> List[Peak]:
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


class DSPPolarMap(ADSPMap):
    @property
    def dspmap_type(self) -> DSPMapType:
        return DSPMapType.POLAR


class DSPNorthPolarMap(ADSPMap):
    @property
    def dspmap_type(self) -> DSPMapType:
        return DSPMapType.NORTH_POLAR
