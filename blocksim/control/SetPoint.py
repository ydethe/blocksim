from typing import Iterable
from itertools import product

import numpy as np
from scipy.interpolate import interp1d

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

    Implement the method **update** to make it concrete

    This element has no input
    The output name of the computer is **setpoint**

    Args:
        name: Name of the element
        snames: Name of each of the scalar components of the setpoint.
            Its shape defines the shape of the data
        dtype: Data type (typically np.float64 or np.complex128)

    """

    __slots__ = []

    def __init__(self, name: str, snames: Iterable[str], dtype):
        AComputer.__init__(self, name)
        self.defineOutput("setpoint", snames=snames, dtype=dtype)


class Step(ASetPoint):
    """Step set point

    This element has no input
    The output name of the computer is **setpoint**

    Args:
        name: Name of the element
        snames: Name of each of the scalar components of the setpoint.
            Its shape defines the shape of the data
        cons: Amplitude of the steps

    """

    __slots__ = []

    def __init__(self, name: str, snames: Iterable[str], cons: np.array):
        dtype = cons.dtype
        ASetPoint.__init__(self, name=name, snames=snames, dtype=dtype)
        otp = self.getOutputByName("setpoint")
        otp.setInitialState(cons)

    def update(
        self,
        t1: float,
        t2: float,
        setpoint: np.array,
    ) -> dict:
        outputs = {}
        outputs["setpoint"] = setpoint

        return outputs


class InterpolatedSetPoint(ASetPoint):
    """Set point that interpolates into an array

    This element has no input
    The output name of the computer is **setpoint**

    Attributes:
        interpolators: Dictionary of the interpolators

    Args:
        name: Name of the element
        snames: Name of each of the scalar components of the setpoint.
            Its shape defines the shape of the data

    """

    __slots__ = []

    def __init__(self, name: str, snames: Iterable[str], dtype=np.float64):
        ASetPoint.__init__(self, name, snames=snames, dtype=dtype)
        otp = self.getOutputByName("setpoint")
        otp.setInitialState(np.zeros(otp.getDataShape(), dtype=dtype))
        self.createParameter("interpolators", value=dict())

    def evalState(self, t: float) -> "array":
        """Perform interpolation at simulation time t

        Args:
            t: timestamp of the interpolation

        Returns:
            The interpolated vector

        """
        otp = self.getOutputByName("setpoint")
        ns = otp.getDataShape()
        x0 = np.empty(ns, dtype=otp.getDataType())

        # Creating iterables, to handle the case where
        # the output 'setpoint' is a matrix
        it = []
        for k in ns:
            it.append(range(k))

        # Iterate over all dimensions
        for iscal in product(*it):
            f = self.interpolators[iscal]
            x0[iscal] = f(t)

        return x0

    def setInterpolatorForOutput(
        self,
        iscal: int,
        t_interp: np.array,
        sp_interp: np.array,
        kind: str = "linear",
    ):
        """Sets the interpolator for the scalar iscal

        Args:
            iscal: Index in the output's state vector whose interpolation function is beeing set
            t_interp: Array of dates (s)
            sp_interp: Array of set points (s)
            kind: Interpolation method

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

    def update(
        self,
        t1: float,
        t2: float,
        setpoint: np.array,
    ) -> dict:
        sp = self.evalState(t2)

        outputs = {}
        outputs["setpoint"] = sp

        return outputs


class Sinusoid(ASetPoint):
    """Sinusoid set point : for each output

    $$ A.sin(2.\pi.f+\phi) $$

    This element has no input
    The output name of the computer is **setpoint**

    Attributes:
        amp: Amplitude A
        freq: Frequency f (Hz)
        pha: Initial phase phi (rad)

    Args:
        name: Name of the element
        snames: Name of each of the scalar components of the setpoint.
            Its shape defines the shape of the data

    """

    __slots__ = []

    def __init__(self, name: str, snames: Iterable[str], dtype=np.float64):
        ASetPoint.__init__(self, name, snames=snames, dtype=dtype)
        otp = self.getOutputByName("setpoint")
        shape = otp.getDataShape()
        otp.setInitialState(np.zeros(shape, dtype=dtype))
        self.createParameter("amp", value=np.empty(shape))
        self.createParameter("freq", value=np.empty(shape))
        self.createParameter("pha", value=np.empty(shape))

    def evalState(self, t: float) -> "array":
        """Computes the output at simulation time t

        Args:
            t: timestamp of the interpolation

        Returns:
            The interpolated vector

        """
        otp = self.getOutputByName("setpoint")
        ns = otp.getDataShape()
        x0 = np.empty(ns, dtype=otp.getDataType())

        # Creating iterables, to handle the case where
        # the output 'setpoint' is a matrix
        it = []
        for k in ns:
            it.append(range(k))

        # Iterate over all dimensions
        for iscal in product(*it):
            x0[iscal] = self.amp[iscal] * np.sin(
                2 * np.pi * self.freq[iscal] * t + self.pha[iscal]
            )

        return x0

    def update(
        self,
        t1: float,
        t2: float,
        setpoint: np.array,
    ) -> dict:
        sp = self.evalState(t2)

        outputs = {}
        outputs["setpoint"] = sp

        return outputs


class Ramp(ASetPoint):
    """Ramp set point

    This element has no input
    The output name of the computer is **setpoint**

    Attributes:
        slopes: Gradients of the slopes

    Args:
        name: Name of the element
        snames: Name of each of the scalar components of the setpoint.
            Its shape defines the shape of the data
        slopes: Gradients of the slopes

    """

    __slots__ = []

    def __init__(self, name: str, snames: Iterable[str], slopes: np.array):
        dtype = slopes.dtype
        ASetPoint.__init__(self, name, snames=snames, dtype=dtype)
        otp = self.getOutputByName("setpoint")
        otp.setInitialState(np.zeros(otp.getDataShape(), dtype=dtype))
        self.createParameter("slopes", value=slopes)

    def update(
        self,
        t1: float,
        t2: float,
        setpoint: np.array,
    ) -> dict:
        ns = setpoint.shape
        sp = np.empty(ns, dtype=setpoint.dtype)

        # Creating iterables, to handle the case where
        # the output 'setpoint' is a matrix
        it = []
        for k in ns:
            it.append(range(k))

        # Iterate over all dimensions
        for iscal in product(*it):
            sp[iscal] = self.slopes[iscal] * t2

        sp0 = self.getInitialStateForOutput("setpoint")

        outputs = {}
        outputs["setpoint"] = sp0 + sp

        return outputs


class Rectangular(ASetPoint):
    """Door window

    This element has no input
    The output name of the computer is **setpoint**

    Attributes:
        doors : Doors descriptions
            Each key of doors is a the coordinate in the data vector
            Each value of doors is a tuple :

            * tdeb : date of the beginning of the door
            * xon : value of the door inside [tdeb,tfin[
            * xoff : value of the door outside [tdeb,tfin[
            * tfin : date of the end of the door

    Args:
        name: Name of the element
        snames: Name of each of the scalar components of the setpoint.
            Its shape defines the shape of the data

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        snames: Iterable[str],
        dtype=np.float64,
    ):
        ASetPoint.__init__(self, name, snames=snames, dtype=dtype)
        self.createParameter("doors", value=[])

    def evalState(self, t: float) -> "array":
        """Computes the output at simulation time t

        Args:
            t: timestamp of the interpolation

        Returns:
            The interpolated vector

        """
        otp = self.getOutputByName("setpoint")
        ns = otp.getDataShape()
        res = np.empty(ns, dtype=otp.getDataType())

        # Creating iterables, to handle the case where
        # the output 'setpoint' is a matrix
        it = []
        for k in ns:
            it.append(range(k))

        # Iterate over all dimensions
        for iscal in product(*it):
            tdeb, xon, xoff, tfin = self.doors[iscal]
            if t >= tdeb and t < tfin:
                res[iscal] = xon
            else:
                res[iscal] = xoff

        return res

    def update(
        self,
        t1: float,
        t2: float,
        setpoint: np.array,
    ) -> dict:
        sp = self.evalState(t2)

        outputs = {}
        outputs["setpoint"] = sp

        return outputs
