import numpy as np

from ..core.Frame import Frame
from ..core.Node import AComputer

__all__ = ["ASetPoint", "Step"]


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
