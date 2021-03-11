from abc import abstractmethod

import numpy as np
from scipy.integrate import ode

from ..core.Frame import Frame
from ..core.Node import AComputer


__all__ = ["ASystem"]


class ASystem(AComputer):
    """Abstract class for a physical system

    Implement the method **transition** to make it concrete
    You can also implement the method **jacobian**
    (see :class:`blocksim.blocks.System.ASystem.example_jacobian`)
    to use the integrators that need the jacobian.

    The input name of the computer is **command**
    The output name of the computer is **state**

    Args:
      name
        Name of the system
      nscal_command
        Number of scalars in the data expected by the command
      nscal_state
        Number of scalars in the data expected by the state
      dtype
        Data type (typically np.float64 or np.complex128)
      method, optional
        Integrator selected for scipy.ode. Default : 'dop853'.
        See `SciPy doc`_.

    .. _SciPy doc: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode

    """

    def __init__(
        self,
        name: str,
        nscal_command: int,
        nscal_state: int,
        dtype=np.float64,
        method: str = "dop853",
    ):
        AComputer.__init__(self, name)
        self.defineInput("command", nscal_command, dtype)
        self.defineOutput("state", nscal_state, dtype)

        has_jacobian = hasattr(self, "jacobian")

        if has_jacobian:
            self.__integ = ode(self.transition, self.jacobian)
        else:
            self.__integ = ode(self.transition)

        self.__integ.set_integrator(method, nsteps=10000)

    @abstractmethod
    def transition(self, t: float, y: np.array, u: np.array) -> np.array:
        """Defines the transition function f(t,x,u) :

        x' = f(t,x,u)

        Args:
          t
            Date of the current state
          x
            Current state
          u
            Command applied to the system

        Returns:
          The derivative of the state

        """
        pass

    def example_jacobian(self, t: float, x: np.array, u: np.array) -> np.array:
        """Defines the jacobian of
        the transition function f(t,x,u) with respect to x:

        x' = f(t,x,u)

        Args:
          t
            Date of the current state
          x
            Current state
          u
            Command applied to the system

        Returns:
          The jacobian of the transition function with respect to x

        """
        return

    def updateAllOutput(self, frame: Frame):
        u = self.getDataForInput(frame, name="command")
        y0 = self.getDataForOutput(frame, name="state")

        t0 = frame.getStartTimeStamp()
        t1 = frame.getStopTimeStamp()

        self.__integ.set_initial_value(y0, t0).set_f_params(u).set_jac_params(u)
        try:
            y1 = self.__integ.integrate(t1)
        except Exception as e:
            print(72 * "=")
            print("When updating '%s'" % self.getName())
            print("t", t0)
            print("x", y0)
            print("u", u)
            print("dx", self.transition(t0, y0, u))
            print(72 * "=")
            raise e

        otp = self.getOutputByName("state")
        otp.setData(y1)
