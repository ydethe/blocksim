from typing import Callable, List

import numpy as np
from scipy import linalg as lin

from .. import logger
from .DSPLine import DSPLine
from .Peak import Peak

__all__ = ["DSPSpectrogram"]


class DSPSpectrogram(object):
    """Representation of a spectrogram

    Args:
      name
        Name of the spectrum
      samplingXStart (s)
        First date of the sample of the spectrogram
      samplingXPeriod (s)
        Time spacing of the spectrogram
      samplingYStart (Hz)
        First frequency of the sample of the spectrogram
      samplingYPeriod (Hz)
        Frequency spacing of the spectrogram
      img
        Matrix of complex samples
      default_transform
        Function to apply to the samples before plotting.
        Shall be vectorized

    """

    name_of_x_var = "Time"
    unit_of_x_var = "s"
    name_of_y_var = "Frequency"
    unit_of_y_var = "Hz"

    def __init__(
        self,
        name: str,
        samplingXStart: float = None,
        samplingXPeriod: float = None,
        samplingYStart: float = None,
        samplingYPeriod: float = None,
        img: np.array = None,
        default_transform=np.abs,
    ):
        self.name = name
        self.samplingXStart = samplingXStart
        self.samplingXPeriod = samplingXPeriod
        self.samplingYStart = samplingYStart
        self.samplingYPeriod = samplingYPeriod
        self.img = img
        self.default_transform = default_transform

    def generateXSerie(self, index: int = None) -> np.array:
        """Generates the x samples of the spectrogram

        Args:
          index
            If given, returns only the x coord at the position given by index

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

    def generateYSerie(self, index: int = None) -> np.array:
        """Generates the y samples of the spectrogram

        Args:
          index
            If given, returns only the y coord at the position given by index

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
          transform
            A callable applied on samples before looking for the peaks
          nb_peaks
            Max number of peaks to seach. Only the highest are kept

        Returns:
          The list of detected peaks, sorted by descreasing value of the peak

        """
        if transform is None:
            transform = self.default_transform

        ep = 2
        eq = 2
        iDtr = 1 / (2 * ep ** 2 * eq ** 2)
        iA = iDtr * np.array(
            [
                [0, -ep * eq ** 2, 0, 0, ep * eq ** 2],
                [0, 0, -(ep ** 2) * eq, ep ** 2 * eq, 0],
                [-2 * eq ** 2, eq ** 2, 0, 0, eq ** 2],
                [-2 * ep ** 2, 0, ep ** 2, ep ** 2, 0],
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
                dval = (-(b ** 2) * f - c ** 2 * d) / (4 * d * f)

                if -ep / 2 <= dp and dp < ep / 2 and -eq / 2 <= dq and dq < eq / 2:
                    p = Peak(
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
