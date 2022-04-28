from typing import Callable, List

import numpy as np
from scipy import linalg as lin

from .. import logger
from .DSPLine import DSPLine
from .Peak import Peak

__all__ = ["DSPSpectrogram"]


class DSPSpectrogram(object):
    """Representation of a spectrogram

    Attributes:
        name: Name of the spectrogram
        samplingXStart: First date of the sample of the spectrogram
        samplingXPeriod: Time spacing of the spectrogram
        samplingYStart: First frequency of the sample of the spectrogram
        samplingYPeriod: Frequency spacing of the spectrogram
        img: Matrix of complex samples
        default_transform: Function to apply to the samples before plotting
        name_of_x_var: Name of x variable. Default: "Time"
        unit_of_x_var: Unit of x variable. Default: "s"
        name_of_y_var: Name of y variable. Default: "Frequency"
        unit_of_y_var: Unit of y variable. Default: "Hz"
        projection: Axe projection. Can be 'rectilinear' or 'polar'

    Args:
        name: Name of the spectrogram
        samplingXStart: First date of the sample of the spectrogram
        samplingXPeriod: Time spacing of the spectrogram
        samplingYStart: First frequency of the sample of the spectrogram
        samplingYPeriod: Frequency spacing of the spectrogram
        img: Matrix of complex samples
        projection: Axe projection. Can be 'rectilinear' or 'polar'
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
        img: np.array = None,
        projection: str = "rectilinear",
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
        self.projection = projection

    def generateXSerie(self, index: int = None) -> "array":
        """Generates the x samples of the spectrogram

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

    def generateYSerie(self, index: int = None) -> "array":
        """Generates the y samples of the spectrogram

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

        ep = 2
        eq = 2
        iDtr = 1 / (2 * ep**2 * eq**2)
        iA = iDtr * np.array(
            [
                [0, -ep * eq**2, 0, 0, ep * eq**2],
                [0, 0, -(ep**2) * eq, ep**2 * eq, 0],
                [-2 * eq**2, eq**2, 0, 0, eq**2],
                [-2 * ep**2, 0, ep**2, ep**2, 0],
            ]
        )

        dat = transform(self.img)
        Np, Nq = dat.shape
        lpeak = []
        for p0 in range(ep, Np - ep):
            for q0 in range(eq, Nq - eq):
                Z00 = dat[p0, q0]
                B = np.array(
                    [
                        Z00,
                        dat[p0 - ep, q0],
                        dat[p0, q0 - eq],
                        dat[p0, q0 + eq],
                        dat[p0 + ep, q0],
                    ]
                )
                if np.any(B[1:] >= Z00):
                    continue

                X0 = iA @ B
                b, c, d, f = X0

                dp = -b / (2 * d)
                dq = -c / (2 * f)
                dval = (-(b**2) * f - c**2 * d) / (4 * d * f)

                if -ep / 2 <= dp and dp < ep / 2 and -eq / 2 <= dq and dq < eq / 2:
                    p = Peak(
                        coord_label=(self.name_of_x_var, self.name_of_y_var),
                        coord_unit=(self.unit_of_x_var, self.unit_of_y_var),
                        coord=(
                            self.generateXSerie(q0 + dq),
                            self.generateYSerie(p0 + dp),
                        ),
                        value=Z00 + dval,
                    )
                    lpeak.append(p)
                else:
                    raise AssertionError(dp, dq)

        lpeak.sort(key=lambda x: x.value, reverse=True)

        if len(lpeak) > nb_peaks:
            lpeak = lpeak[:nb_peaks]

        return lpeak

    @classmethod
    def to_db_lim(cls, low: float) -> Callable:
        """Returns a function that turns a complex signal into the power serie of the signal, in dB.

        Args:
            low: The min value to clamp to (dB)

        Returns:
            The function to map on a complex map

        Examples:
            >>> f = DSPLine.to_db_lim(low=-80)
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
    def to_db(cls, x: "array", lim_db: float = -100) -> "array":
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
