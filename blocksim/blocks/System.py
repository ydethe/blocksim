from abc import abstractmethod
from typing import Iterable

import numpy as np
from scipy.integrate import ode
from scipy.signal import cont2discrete
import scipy.linalg as lin

from ..exceptions import *
from ..utils import quat_to_matrix, quat_to_euler
from ..core.Frame import Frame
from ..core.Node import AComputer


__all__ = ["ASystem", "LTISystem", "G6DOFSystem"]


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
    def transition(self, t: float, x: np.array, u: np.array) -> np.array:
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


class LTISystem(ASystem):
    """Models a SISO LTI system :

    :math:`dX/dt = A.X + B.u`

    The input name of the element is **command**
    The output name of the computer is **state**

    The matrix A and B are parameters that must be defined by the user :

    * A : (n x n) State (or system) matrix
    * B : (n x m) Input matrix

    with :

    * n number of states (nscal_state in the init arguments)
    * m number of commands (nscal_command in the init arguments)

    Args:
      name
        Name of the system
      nscal_command
        Number of scalars in the data expected by the command
      nscal_state
        Number of scalars in the data expected by the state
      dtype
        Data type (typically np.float64 or np.complex128)
      method
        Integrator selected for scipy.ode. Default : 'dop853'

    """

    def __init__(
        self,
        name: str,
        nscal_command: int,
        nscal_state: int,
        dtype=np.float64,
        method: str = "dop853",
    ):
        ASystem.__init__(
            self,
            name=name,
            nscal_command=nscal_command,
            nscal_state=nscal_state,
            dtype=dtype,
            method=method,
        )

    def getDiscreteMatrices(
        self, dt: float, method: str = "zoh", alpha: float = None
    ) -> Iterable[np.array]:
        """

        Args:
          dt
            The discretization time step
          method
            * gbt: generalized bilinear transformation
            * bilinear: Tustin’s approximation (“gbt” with alpha=0.5)
            * euler: Euler (or forward differencing) method (“gbt” with alpha=0)
            * backward_diff: Backwards differencing (“gbt” with alpha=1.0)
            * zoh: zero-order hold (default)
          alpha
            Within [0, 1]
            The generalized bilinear transformation weighting parameter, which should only be specified with method=”gbt”, and is ignored otherwise

        Returns:
          Ad
            Discrete dynamics matrix
          Bd
            Discrete input matrix

        Examples:
          >>> m = 1. # Mass
          >>> k = 40. # Spring rate
          >>> sys = LTISystem('sys', nscal_command=1, nscal_state=2)
          >>> sys.A = np.array([[0,1],[-k/m,0]])
          >>> sys.B = np.array([[0,1/m]]).T
          >>> Kk = 1/m
          >>> Ka = np.sqrt(k/m)
          >>> def A(t1, t2):
          ...    return np.array([[np.cos(Ka * (t2 - t1)), np.sin(Ka * (t2 - t1)) / Ka],
          ...                     [-Ka * np.sin(Ka * (t2 - t1)), np.cos(Ka * (t2 - t1))]])
          >>> Ad,Bd = sys.getDiscreteMatrices(dt=0.1)
          >>> D = A(2, 2.1) - Ad
          >>> lin.norm(D) < 1.e-14
          True

        """
        n = self.getOutputByName("state").getNumberScalar()
        m = self.getInputByName("command").getNumberScalar()

        C = np.zeros((1, n))
        D = np.zeros((1, m))
        sys = (self.A, self.B, C, D)
        Ad, Bd, _, _, _ = cont2discrete(sys, dt, method, alpha)
        return Ad, Bd

    def transition(self, t: float, x: np.array, u: np.array) -> np.array:
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
        dX = self.A @ x + self.B @ u
        return dx

    def jacobian(self, t: float, x: np.array, u: np.array) -> np.array:
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
        return self.A


class G6DOFSystem(ASystem):
    """Generic 6 DOF rigid body
    The attitude state is handled with quaternions
    The integration scheme is 5-stage Crouch-Grossman method "CG4",
    which garantees that the attitude quaternion's norm remains equal to 1

    http://ancs.eng.buffalo.edu/pdf/ancs_papers/2013/geom_int.pdf

    The input of the element is **command**
    The output name of the computer is **state**

    The following parameters must be defined by the user :

    * m : Mass of the body. Default : :math:`1`
    * J : Inertia tensor of the body. Default : :math:`10^{-3}.I_3`
    * max_q_denorm : If N the square norm of the attitude quaternion is N > 1+max_q_denorm or N<1-max_q_denorm, raise an exception

    command : 3D force in the fixed frame
              3D torque in the fixed frame
    state:
    * px
    * py
    * pz
    * vx
    * vy
    * vz
    * roll
    * pitch
    * yaw
    * wx
    * wy
    * wz

    Args:
      name
        Name of the system

    """

    def __init__(self, name):
        ASystem.__init__(
            self,
            name,
            nscal_command=6,
            nscal_state=12,
            dtype=np.float64,
            method="dop853",
        )
        self.m = 1
        self.J = np.eye(3) * 1e-3
        self.max_q_denorm = 1e-6
        self.setInitialStateForOutput(
            np.array([0, 0, 0, 0, 0, 0, 1.0, 0.0, 0, 0, 0, 0, 0]), output_name="state"
        )

    def vecBodyToEarth(self, frame: Frame, x: np.array) -> np.array:
        """Expresses a vector from the body frame to the Earth's frame

        Args:
          frame
            The time frame
          x
            Vector expressed in the body frame

        Returns:
          Vector x expressed in Earth's frame

        """
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = self.getDataForOutput(
            frame, name="state"
        )
        R = quat_to_matrix(qw, qx, qy, qz)
        return R @ x

    def vecEarthToBody(self, frame: Frame, x: np.array) -> np.array:
        """Expresses a vector from Earth's frame to the body's frame

        Args:
          frame
            The time frame
          x
            Vector expressed in Earth's frame

        Returns:
          Vector x expressed in the body frame

        """
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = self.getDataForOutput(
            frame, name="state"
        )
        R = quat_to_matrix(qw, qx, qy, qz)
        return R.T @ x

    def transition(self, t: float, x: np.array, u: np.array) -> np.array:
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
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = x
        force = u[:3]
        torque = u[3:6]

        q = np.array([qw, qx, qy, qz])

        w = np.array([wx, wy, wz])
        W = np.array(
            [[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]]
        )

        dvx, dvy, dvz = force / self.m
        dwx, dwy, dwz = lin.inv(self.J) @ (-np.cross(w, self.J @ w) + torque)
        dqw, dqx, dqy, dqz = 1.0 / 2.0 * W @ q
        dX = np.array([vx, vy, vz, dvx, dvy, dvz, dqw, dqx, dqy, dqz, dwx, dwy, dwz])
        return dX

    def updateAllOutput(self, frame: Frame):
        """Updates the state of the element

        Called at each simulation step

        Args:
          frame
            The time frame

        """
        # Crouch-Grossman method CG4
        # http://ancs.eng.buffalo.edu/pdf/ancs_papers/2013/geom_int.pdf
        s = 5
        a = np.zeros((s, s))
        b = np.empty(s)
        c = np.empty(s)
        a[1, 0] = 0.8177227988124852
        b[0] = 0.1370831520630755
        a[2, 0] = 0.3199876375476427
        b[1] = -0.0183698531564020
        a[2, 1] = 0.0659864263556022
        b[2] = 0.7397813985370780
        a[3, 0] = 0.9214417194464946
        b[3] = -0.1907142565505889
        a[3, 1] = 0.4997857776773573
        b[4] = 0.3322195591068374
        a[3, 2] = -1.0969984448371582
        c[0] = 0.0
        a[4, 0] = 0.3552358559023322
        c[1] = 0.8177227988124852
        a[4, 1] = 0.2390958372307326
        c[2] = 0.3859740639032449
        a[4, 2] = 1.3918565724203246
        c[3] = 0.3242290522866937
        a[4, 3] = -1.1092979392113565
        c[4] = 0.8768903263420429

        u = self.getDataForInput(frame, name="command")
        x = self.getDataForOutput(frame, name="state")
        otp = self.getOutputByName("state")

        t1 = frame.getStartTimeStamp()
        t2 = frame.getStopTimeStamp()
        h = t2 - t1

        ns = otp.getNumberScalar()
        k = np.zeros((ns, s))
        q = x[6:10]
        w = x[10:13]
        for i in range(s):
            dx = np.zeros(ns)
            for j in range(i):
                dx += a[i, j] * k[:, j]
            k[:, i] = h * self.transition(t1 + c[i] * h, x + dx, u)

            wk = w + c[i] * k[10:13, 0]
            Nwk = lin.norm(wk)
            wx, wy, wz = wk
            W = np.array(
                [
                    [0, -wx, -wy, -wz],
                    [wx, 0, wz, -wy],
                    [wy, -wz, 0, wx],
                    [wz, wy, -wx, 0],
                ]
            )
            # We use the sinc function to handle the case where Nwk is close to zero
            E = np.cos(1.0 / 2.0 * h * b[i] * Nwk) * np.eye(4) + W * h * b[
                i
            ] / 2 * np.sinc(h * b[i] * Nwk / (2 * np.pi))
            q = E @ q

        Nq2 = np.sum(q ** 2)
        if np.abs(Nq2 - 1) > self.max_q_denorm:
            raise DenormalizedQuaternion(self.getName(), q)

        res = x.copy()
        for i in range(s):
            res += b[i] * k[:, i]
        res[6:10] = q

        otp.setData(res)
