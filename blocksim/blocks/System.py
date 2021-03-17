from abc import abstractmethod
from typing import Iterable

import numpy as np
from scipy.integrate import ode
from scipy.signal import cont2discrete
import scipy.linalg as lin

from ..exceptions import *
from ..utils import vecBodyToEarth, vecEarthToBody, quat_to_euler
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
      shape_command
        Shape of the data expected by the command
      snames_state
        Name of each of the scalar components of the state.
        Its shape defines the shape of the data
      dtype
        Data type (typically np.float64 or np.complex128)
      method, optional
        Integrator selected for scipy.ode. Default : 'dop853'.
        See `SciPy doc`_.

    .. _SciPy doc: https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode

    """

    __slots__ = ["__integ"]

    def __init__(
        self,
        name: str,
        shape_command: int,
        snames_state: Iterable[str],
        dtype=np.float64,
        method: str = "dop853",
    ):
        AComputer.__init__(self, name)
        self.defineInput("command", shape_command, dtype)
        self.defineOutput("state", snames_state, dtype)

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

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        command: np.array,
        state: np.array,
    ) -> dict:
        self.__integ.set_initial_value(state, t1).set_f_params(command).set_jac_params(
            command
        )

        if t1 != t2:
            try:
                state = self.__integ.integrate(t2)
            except Exception as e:
                print(72 * "=")
                print("When updating '%s'" % self.getName())
                print("t", t1)
                print("x", state)
                print("u", command)
                print("dx", self.transition(t1, state, command))
                print(72 * "=")
                raise e

        outputs = {}
        outputs["state"] = state

        return outputs


class LTISystem(ASystem):
    """Models a SISO LTI system :

    :math:`dX/dt = A.X + B.u`

    The input name of the element is **command**
    The output name of the computer is **state**

    The matrix A and B are parameters that must be defined by the user :

    * matA : (n x n) State (or system) matrix
    * matB : (n x m) Input matrix

    with :

    * n number of states (shape of snames_state in the init arguments)
    * m number of commands (shape_command in the init arguments)

    Args:
      name
        Name of the system
      shape_command
        Shape of the data expected by the command
      snames_state
        Name of each of the scalar components of the state.
        Its shape defines the shape of the data
      dtype
        Data type (typically np.float64 or np.complex128)
      method
        Integrator selected for scipy.ode. Default : 'dop853'

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        shape_command: int,
        snames_state: Iterable[str],
        dtype=np.float64,
        method: str = "dop853",
    ):
        ASystem.__init__(
            self,
            name=name,
            shape_command=shape_command,
            snames_state=snames_state,
            dtype=dtype,
            method=method,
        )
        self.createParameter("matA", value=0)
        self.createParameter("matB", value=0)

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
          >>> sys = LTISystem('sys', shape_command=1, snames_state=['x','v'])
          >>> sys.matA = np.array([[0,1],[-k/m,0]])
          >>> sys.matB = np.array([[0,1/m]]).T
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
        n = self.getOutputByName("state").getDataShape()[0]
        m = self.getInputByName("command").getDataShape()[0]

        C = np.zeros((1, n))
        D = np.zeros((1, m))
        sys = (self.matA, self.matB, C, D)
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
        dX = self.matA @ x + self.matB @ u
        return dX

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
        return self.matA


class G6DOFSystem(ASystem):
    """Generic 6 DOF rigid body
    The attitude state is handled with quaternions
    The integration scheme is 5-stage Crouch-Grossman method "CG4",
    which garantees that the attitude quaternion's norm remains equal to 1

    http://ancs.eng.buffalo.edu/pdf/ancs_papers/2013/geom_int.pdf

    The input of the element is **command**
    The outputs name of the computer are **state** and **euler**

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
    * qr
    * qx
    * qy
    * qz
    * wx
    * wy
    * wz
    euler:
    * roll
    * pitch
    * yaw

    Args:
      name
        Name of the system

    """

    __slots__ = []

    def __init__(self, name):
        ASystem.__init__(
            self,
            name,
            shape_command=6,
            snames_state=[
                "px",
                "py",
                "pz",
                "vx",
                "vy",
                "vz",
                "qr",
                "qx",
                "qy",
                "qz",
                "wx",
                "wy",
                "wz",
            ],
            dtype=np.float64,
            method="dop853",
        )
        self.defineOutput(
            name="euler", snames=["roll", "pitch", "yaw"], dtype=np.float64
        )
        self.setInitialStateForOutput(
            np.array([0, 0, 0, 0, 0, 0, 1.0, 0.0, 0, 0, 0, 0, 0]), output_name="state"
        )
        self.setInitialStateForOutput(np.array([0, 0, 0]), output_name="euler")

        self.createParameter("m", 1)
        self.createParameter("J", np.eye(3) * 1e-3)
        self.createParameter("max_q_denorm", 1e-6)

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
        return vecBodyToEarth(np.array([qw, qx, qy, qz]), x)

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
        return vecEarthToBody(np.array([qw, qx, qy, qz]), x)

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

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        command: np.array,
        state: np.array,
        euler: np.array,
    ) -> dict:
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

        ns = state.shape[0]

        h = t2 - t1

        k = np.zeros((ns, s))
        q = state[6:10]
        w = state[10:13]
        for i in range(s):
            dx = np.zeros(ns)
            for j in range(i):
                dx += a[i, j] * k[:, j]
            k[:, i] = h * self.transition(t1 + c[i] * h, state + dx, command)

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

        res = state.copy()
        for i in range(s):
            res += b[i] * k[:, i]
        res[6:10] = q

        euler_att = np.array(quat_to_euler(*res[6:10]))

        outputs = {}
        outputs["euler"] = euler_att
        outputs["state"] = res

        return outputs
