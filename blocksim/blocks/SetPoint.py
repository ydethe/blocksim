from typing import Iterable

import numpy as np

from ..core.Frame import Frame
from ..core.Node import AComputer

__all__ = [
    "ASetPoint",
    "InterpolatedSetPoint",
    "Step",
    "Ramp",
    "Rectangular",
    "Sinusoid",
]


class ASetPoint(AComputer):
    """Abstract class for a set point

    Implement the method **updateAllOutput** to make it concrete

    This element has no input
    The output name of the computer is **setpoint**

    Args:
      name
        Name of the element
      nscal
        Number of scalars in the data expected by the input
      dtype
        Data type (typically np.float64 or np.complex128)

    """

    def __init__(self, name: str, nscal: int, dtype):
        AComputer.__init__(self, name)
        self.defineOutput("setpoint", nscal=nscal, dtype=dtype)


class Step(ASetPoint):
    """Step set point

    This element has no input
    The output name of the computer is **setpoint**

    Args:
      name
        Name of the element
      cons
        Amplitude of the steps

    """

    def __init__(self, name: str, cons: np.array):
        nscal = cons.shape
        dtype = cons.dtype
        ASetPoint.__init__(self, name, nscal=nscal, dtype=dtype)
        otp = self.getOutputByName("setpoint")
        otp.setInitialState(cons)

    def updateAllOutput(self, frame: Frame):
        pass


class InterpolatedSetPoint(ASetPoint):
    """Set point that interpolates into an array

    This element has no input
    The output name of the computer is **setpoint**

    The parameters are :

    * interpolators : Dictionary of the interpolators

    Args:
      name
        Name of the element

    """

    def __init__(self, name: str, nscal: int, dtype=np.float64):
        ASetPoint.__init__(self, name, nscal=nscal, dtype=dtype)
        otp = self.getOutputByName("setpoint")
        otp.setInitialState(np.zeros(nscal, dtype=dtype))
        self.interpolators = dict()

    def setInterpolatorForOutput(
        self,
        iscal: int,
        t_interp: np.array,
        sp_interp: np.array,
        kind: str = "linear",
    ):
        """

        Args:
          iscal
            Index in the output's state vector whose interpolation function is beeing set
          t_interp
            Array of dates (s)
          sp_interp
            Array of set points (s)
          kind
            * linear
            * nearest
            * zero
            * slinear
            * quadratic
            * cubic
            * previous
            * next

        """
        f = interp1d(
            t_interp,
            sp_interp,
            kind=kind,
            copy=True,
            bounds_error=False,
            fill_value="extrapolate",
            assume_sorted=True,
        )
        self.interpolators[iscal] = f

    def updateAllOutput(self, frame: Frame):
        t2 = frame.getStopTimeStamp()
        otp = self.getOutputByName("setpoint")
        n = otp.getNumberScalar()
        sp = np.empty(n)
        for iscal in range(n):
            f = self.interpolators[iscal]
            sp[i] = f(t2)
        otp.setData(sp)


class Sinusoid(ASetPoint):
    """Sinusoid set point : for each output

    :math:`A.sin(2.\pi.f+\phi)`

    This element has no input
    The output name of the computer is **setpoint**

    The parameters are :

    * amp : Amplitude A
    * freq : Frequency f
    * pha : Initial phase phi

    Args:
      name
        Name of the element

    """

    def __init__(self, name: str, nscal: int, dtype=np.float64):
        ASetPoint.__init__(self, name, nscal=nscal, dtype=dtype)
        otp = self.getOutputByName("setpoint")
        otp.setInitialState(np.zeros(nscal, dtype=dtype))
        self.amp = np.empty(nscal)
        self.freq = np.empty(nscal)
        self.pha = np.empty(nscal)

    def updateAllOutput(self, frame: Frame):
        t2 = frame.getStopTimeStamp()
        otp = self.getOutputByName("setpoint")
        n = otp.getNumberScalar()
        sp = np.empty(n)
        for iscal in range(n):
            sp[i] = self.amp[iscal] * np.sin(
                2 * np.pi * self.freq[iscal] * t2 + self.pha[iscal]
            )
        otp.setData(sp)


class Ramp(ASetPoint):
    """Ramp set point

    This element has no input
    The output name of the computer is **setpoint**

    The parameters are :

    * slopes : Gradients of the slopes

    Args:
      name
        Name of the element
      slopes
        Gradients of the slopes

    """

    def __init__(self, name: str, slopes: np.array):
        nscal = slopes.shape
        dtype = slopes.dtype
        ASetPoint.__init__(self, name, nscal=nscal, dtype=dtype)
        otp = self.getOutputByName("setpoint")
        self.slopes = slopes
        otp.setInitialState(np.zeros(nscal, dtype=dtype))

    def updateAllOutput(self, frame: Frame):
        t2 = frame.getStopTimeStamp()
        otp = self.getOutputByName("setpoint")
        n = otp.getNumberScalar()
        sp = np.empty(n)
        for iscal in range(n):
            sp[i] = self.slopes[iscal] * t2
        otp.setData(sp)


class Rectangular(ASetPoint):
    """Door window

    This element has no input
    The output name of the computer is **setpoint**

    The parameters are :

    * slopes : Gradients of the slopes

    The parameters are :

    * doors : Doors descriptions

    Args:
      name
        Name of the element
      doors
        Each element of doors is a tuple :

        * tdeb : date of the beginning of the door
        * xon : value of the door inside [tdeb,tfin]
        * xoff : value of the door outside [tdeb,tfin]
        * tfin : date of the end of the door

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        doors: Iterable[tuple],
        dtype=np.float64,
    ):
        nscal = len(doors)
        ASetPoint.__init__(self, name, nscal=nscal, dtype=dtype)
        self.doors = doors

    def updateAllOutput(self, frame: Frame):
        t2 = frame.getStopTimeStamp()
        otp = self.getOutputByName("setpoint")
        n = otp.getNumberScalar()
        res = np.empty(n)
        for k in range(n):
            tdeb, xon, xoff, tfin = self.doors[k]
            if t2 >= tdeb and t2 <= tfin:
                res[k] = xon
            else:
                res[k] = xoff

        return res
