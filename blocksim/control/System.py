from abc import abstractmethod
from typing import Iterable
from functools import lru_cache


import numpy as np
from scipy.integrate import ode
from scipy.signal import (
    cont2discrete,
    tf2ss,
    TransferFunction,
)
import scipy.linalg as lin

from ..exceptions import DenormalizedQuaternion
from ..utils import vecBodyToEarth, vecEarthToBody, quat_to_euler, FloatArr
from ..core.Node import AComputer


__all__ = ["ASystem", "LTISystem", "G6DOFSystem", "TransferFunctionSystem"]


class ASystem(AComputer):
    """Abstract class for a physical system

    Implement the method **transition** to make it concrete
    You can also implement the method **jacobian**
    (see `ASystem.example_jacobian`)
    to use the integrators that need the jacobian.

    The input name of the computer is **command**
    The output name of the computer is **state**

    Args:
        name: Name of the system
        shape_command: Shape of the data expected by the command
        snames_state: Name of each of the scalar components of the state.
          Its shape defines the shape of the data
        dtype: Data type (typically np.float64 or np.complex128)
        method (optional): Integrator selected for scipy.ode. Default : 'dop853'.
          See https://docs.scipy.org/doc/scipy-0.18.1/reference/generated/scipy.integrate.ode.html#scipy.integrate.ode

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
    def transition(self, t: float, x: FloatArr, u: FloatArr) -> FloatArr:  # pragma: no cover
        """Defines the transition function f(t,x,u) :

        $$ x' = f(t,x,u) $$

        Args:
            t: Date of the current state
            x: Current state
            u: Command applied to the system

        Returns:
            The derivative of the state

        """
        pass

    def example_jacobian(self, t: float, x: FloatArr, u: FloatArr) -> FloatArr:
        """Defines the jacobian of
        the transition function f(t,x,u) with respect to x:

        $$ x' = f(t,x,u) $$

        Args:
            t: Date of the current state
            x: Current state
            u: Command applied to the system

        Returns:
            The jacobian of the transition function with respect to x

        """
        return

    def update(
        self,
        t1: float,
        t2: float,
        command: FloatArr,
        state: FloatArr,
    ) -> dict:
        self.__integ.set_initial_value(state, t1).set_f_params(command).set_jac_params(command)

        if t1 != t2:
            try:
                state = self.__integ.integrate(t2)
            except BaseException as e:
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
    """Models a MIMO LTI system :

    $$ dX/dt = A.X + B.u $$

    The input name of the element is **command**
    The output name of the computer is **state**

    Attributes:
        matA : (n x n) State (or system) matrix
        matB : (n x m) Input matrix

    with :

    * n number of states (shape of snames_state in the init arguments)
    * m number of commands (shape_command in the init arguments)

    Args:
        name: Name of the system
        shape_command: Shape of the data expected by the command
        snames_state: Name of each of the scalar components of the state.
          Its shape defines the shape of the data
        dtype (optional): Data type (typically np.float64 or np.complex128)
        method (optional): Integrator selected for scipy.ode. Default : 'dop853'

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
    ) -> Iterable[FloatArr]:
        """

        Args:
            dt: The discretization time step
            method: Discretization method:

              * gbt: generalized bilinear transformation
              * bilinear: Tustin’s approximation (“gbt” with alpha=0.5)
              * euler: Euler (or forward differencing) method (“gbt” with alpha=0)
              * backward_diff: Backwards differencing (“gbt” with alpha=1.0)
              * zoh: zero-order hold (default)
            alpha: Parameter for the gbt method, within [0, 1]
              The generalized bilinear transformation weighting parameter,
              which should only be specified with method=”gbt”, and is ignored otherwise

        Returns:
            A tuple containing:

            * The discrete dynamics matrix
            * The discrete input matrix

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

    def transition(self, t: float, x: FloatArr, u: FloatArr) -> FloatArr:
        dX = self.matA @ x + self.matB @ u
        return dX

    def jacobian(self, t: float, x: FloatArr, u: FloatArr) -> FloatArr:
        return self.matA


class G6DOFSystem(ASystem):
    """Generic 6 DOF rigid body

    http://ancs.eng.buffalo.edu/pdf/ancs_papers/2013/geom_int.pdf

    The input of the element is **command**
    The outputs name of the computer are **state** and **euler**

    Attributes:
        m : Mass of the body (kg). Default 1
        J : Inertia tensor of the body (kg.m^2). Default : \(( 10^{-3}.I_3 \))
        max_q_denorm : If N the square norm of the attitude quaternion is
            N > 1+max_q_denorm or N<1-max_q_denorm, raise an exception

    command:

    * 3D force in the fixed frame (N)
    * 3D torque in the fixed frame (N.m)

    state:

    * px, py, pz (m)
    * vx, vy, vz (m/s)
    * qr, qx, qy, qz
    * wx, wy, wz (rad/s)

    euler:

    * roll, pitch, yaw (rad)

    Args:
        name: Name of the system

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
        self.defineOutput(name="euler", snames=["roll", "pitch", "yaw"], dtype=np.float64)
        self.setInitialStateForOutput(
            np.array([0, 0, 0, 0, 0, 0, 1.0, 0.0, 0, 0, 0, 0, 0]), output_name="state"
        )
        self.setInitialStateForOutput(np.array([0, 0, 0]), output_name="euler")

        self.createParameter("m", 1)
        self.createParameter("J", np.eye(3) * 1e-3)
        self.createParameter("max_q_denorm", 1e-6)

    def vecBodyToEarth(self, x: FloatArr) -> FloatArr:
        """Expresses a vector from the body frame to the Earth's frame

        Args:
            x: Vector expressed in the body frame

        Returns:
            Vector x expressed in Earth's frame

        """
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = self.getDataForOutput(oname="state")
        return vecBodyToEarth(np.array([qw, qx, qy, qz]), x)

    def vecEarthToBody(self, x: FloatArr) -> FloatArr:
        """Expresses a vector from Earth's frame to the body's frame

        Args:
            x: Vector expressed in Earth's frame

        Returns:
            Vector x expressed in the body frame

        """
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = self.getDataForOutput(oname="state")
        return vecEarthToBody(np.array([qw, qx, qy, qz]), x)

    def transition(self, t: float, x: FloatArr, u: FloatArr) -> FloatArr:
        px, py, pz, vx, vy, vz, qw, qx, qy, qz, wx, wy, wz = x
        force = u[:3]
        torque = u[3:6]

        q = np.array([qw, qx, qy, qz])

        w = np.array([wx, wy, wz])
        W = np.array([[0, -wx, -wy, -wz], [wx, 0, wz, -wy], [wy, -wz, 0, wx], [wz, wy, -wx, 0]])

        dvx, dvy, dvz = force / self.m

        dwx, dwy, dwz = lin.inv(self.J) @ (-np.cross(w, self.J @ w) + torque)
        dqw, dqx, dqy, dqz = 0.5 * W @ q

        dX = np.array([vx, vy, vz, dvx, dvy, dvz, dqw, dqx, dqy, dqz, dwx, dwy, dwz])

        return dX

    def update(
        self,
        t1: float,
        t2: float,
        command: FloatArr,
        state: FloatArr,
        euler: FloatArr,
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
            E = np.cos(1.0 / 2.0 * h * b[i] * Nwk) * np.eye(4) + W * h * b[i] / 2 * np.sinc(
                h * b[i] * Nwk / (2 * np.pi)
            )
            q = E @ q

        Nq2 = np.sum(q**2)
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


class TransferFunctionSystem(AComputer):
    r"""Linear Time Invariant system class in transfer function form.

    Represents the system as the continuous-time transfer function

    $$ H(s)=\frac{\sum_{i=0}^N b[N-i] s^i}{\sum_{j=0}^M a[M-j] s^j} $$

    Sequences representing the coefficients of the numerator and denominator polynomials,
    in order of descending degree. The denominator needs to be at least as long as the numerator.

    Attributes:
        num: Numerator coefficients
        den: Denominator coefficients
        dt: Time step (s)
        matA: Continuous state matrix
        matB: Continuous input matrix
        matC: Continuous output matrix
        matD: Continuous feedthrough matrix

    Args:
        name: Name of the AComputer
        sname: Name of the output state
        num: Coefficients of the numerator
        den: Coefficients of the denominator
        dt: Time step (s)

    """

    def __init__(
        self,
        name: str,
        sname: str,
        num: FloatArr,
        den: FloatArr,
        dt: float,
        dtype=np.float64,
    ):
        A, B, C, D = tf2ss(num, den)

        AComputer.__init__(self, name=name)

        ns, _ = A.shape
        self.defineOutput(name="inner", snames=[f"is{i}" for i in range(ns)], dtype=dtype)
        self.defineOutput(name="state", snames=[sname], dtype=dtype)
        self.defineInput("command", (1,), dtype)

        self.createParameter(name="num", value=num, read_only=True)
        self.createParameter(name="den", value=den, read_only=True)
        self.createParameter(name="dt", value=dt, read_only=True)
        self.createParameter(name="matA", value=A, read_only=True)
        self.createParameter(name="matB", value=B, read_only=True)
        self.createParameter(name="matC", value=C, read_only=True)
        self.createParameter(name="matD", value=D, read_only=True)

    @lru_cache(maxsize=None)
    def discretize(self, method: str = "zoh", alpha: float = None) -> Iterable[np.array]:
        """Turns the continous system into a discrete one

        Args:
            method: Discretization method:

              * gbt: generalized bilinear transformation
              * bilinear: Tustin’s approximation (“gbt” with alpha=0.5)
              * euler: Euler (or forward differencing) method (“gbt” with alpha=0)
              * backward_diff: Backwards differencing (“gbt” with alpha=1.0)
              * zoh: zero-order hold (default)
            alpha: Parameter for the gbt method, within [0, 1]
              The generalized bilinear transformation weighting parameter,
              which should only be specified with method=”gbt”, and is ignored otherwise

        Returns:
            Ad: Discrete state matrix
            Bd: Discrete input matrix
            Cd: Discrete output matrix
            Dd: Discrete direct term

        """
        sys = (self.matA, self.matB, self.matC, self.matD)
        Ad, Bd, Cd, Dd, dt = cont2discrete(sys, self.dt, method=method, alpha=alpha)
        return Ad, Bd, Cd, Dd

    def update(
        self,
        t1: float,
        t2: float,
        command: FloatArr,
        state: FloatArr,
        inner: FloatArr,
    ) -> dict:
        Ad, Bd, Cd, Dd = self.discretize()

        state = Cd @ inner + Dd @ command
        inner = Ad @ inner + Bd @ command

        outputs = {}
        outputs["state"] = state
        outputs["inner"] = inner

        return outputs

    def to_continuous_tf(self) -> TransferFunction:
        """Returns an instance of TransferFunctionContinuous from scipy.signal
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html

        """
        sys = TransferFunction(self.num, self.den)
        return sys

    def to_continuous_ss(self) -> TransferFunction:
        """Returns an instance StateSpaceContinuous from scipy.signal
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.StateSpace.html

        """
        sys = TransferFunction(self.num, self.den)
        return sys.to_ss()

    def to_discrete_tf(self) -> TransferFunction:
        """Returns an instance of TransferFunctionDiscrete from scipy.signal
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html

        """
        sys = TransferFunction(self.num, self.den, dt=self.dt)
        return sys

    def to_discrete_ss(self) -> TransferFunction:
        """Returns an instance StateSpaceDiscrete from scipy.signal
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.StateSpace.html

        """
        sys = TransferFunction(self.num, self.den, dt=self.dt)
        return sys.to_ss()
