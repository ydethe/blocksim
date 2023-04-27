from typing import Iterable

import numpy as np
from scipy import linalg as lin

from ..utils import FloatArr
from ..core.Node import AComputer


__all__ = [
    "AController",
    "PIDController",
    "AntiWindupPIDController",
    "LQRegulator",
]


class AController(AComputer):
    """Abstract class for a scalar controller

    Implement the method **update** to make it concrete

    The inputs of the computer are **estimation** and **setpoint**
    The output of the computer is **command**

    Args:
        name: Name of the element
        shape_setpoint: Shape of the setpoint data
        shape_estimation: Shape of the estimation data
        snames: Name of each of the scalar components of the command.
          Its shape defines the shape of the data

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        shape_setpoint: tuple,
        shape_estimation: tuple,
        snames: Iterable[str],
    ):
        AComputer.__init__(self, name)
        self.defineInput("setpoint", shape=shape_setpoint, dtype=np.float64)
        self.defineInput("estimation", shape=shape_estimation, dtype=np.float64)
        self.defineOutput("command", snames=snames, dtype=np.float64)


class PIDController(AController):
    """One-dimensional PID controller

    The inputs of the computer are **estimation** and **setpoint**
    The outputs of the computer are **command** and **integral**

    The **estimation** \\( \hat{X} \\) must contain the state you want to control \\( X \\)
    and its derivative \\( \dot{X} \\) in this order:

    $$ \hat{X} = (X, \dot{X}, ...)^T $$

    Attributes:
        P: Proportinnal gain
        I: Integral gain
        D: Derivative gain

    Args:
        name: Name of the element
        shape_estimation: Shape of the data expected by the estimation (> 2)
        snames:
        coeffs: Coefficients of the retroaction (P, I, D)

    """

    __slots__ = []

    def __init__(self, name: str, shape_estimation: tuple, snames: Iterable[str], coeffs: float):
        AController.__init__(
            self,
            name,
            shape_setpoint=(1,),
            shape_estimation=shape_estimation,
            snames=snames,
        )
        self.defineOutput("integral", snames=["int"], dtype=np.float64)
        self.setInitialStateForOutput(np.array([0]), "integral")
        (Kprop, Kinteg, Kderiv) = coeffs
        self.createParameter("P", value=Kprop)
        self.createParameter("I", value=Kinteg)
        self.createParameter("D", value=Kderiv)

    def update(
        self,
        t1: float,
        t2: float,
        integral: FloatArr,
        setpoint: FloatArr,
        estimation: FloatArr,
        command: FloatArr,
    ) -> dict:
        (ix,) = integral
        x = estimation[0]
        if self.D == 0.0:
            dx = 0.0
        else:
            dx = estimation[1]
        (c,) = setpoint

        u = self.P * (x - c) + self.I * ix + self.D * dx

        dt = t2 - t1
        ix = np.array([ix + dt * (x - c)])

        outputs = {}
        outputs["command"] = -np.array([u])
        outputs["integral"] = ix

        return outputs


class AntiWindupPIDController(AController):
    """One-dimensional PID controller with anti-windup

    The inputs of the computer are **estimation** and **setpoint**
    The outputs of the computer are **command** and **integral**

    The **estimation** \\( \hat{X} \\) must contain the state you want to control \\( X \\)
    and its derivative \\( \dot{X} \\) in this order:

    $$ \hat{X} = (X, \dot{X}, ...)^T $$

    Attributes:
        P: Proportinnal gain
        I: Integral gain
        D: Derivative gain
        Umin: if the command u is < Umin, then u = Umin
        Umax: if the command u is > Umax, then u = Umax
        Ks: gain of the anti-windup effect

    Args:
        name: Name of the element
        shape_estimation: Shape of the data expected by the estimation (> 2)
        coeffs: Coefficients of the retroaction (P, I, D, Umin, Umax, Ks)

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        shape_estimation: tuple,
        snames: Iterable[str],
        coeffs: float = (0, 0, 0, 0, 0, 0),
    ):
        AController.__init__(
            self,
            name,
            shape_setpoint=(1,),
            shape_estimation=shape_estimation,
            snames=snames,
        )
        self.defineOutput("integral", snames=["int", "corr"], dtype=np.float64)
        (Kprop, Kinteg, Kderiv, Umin, Umax, Ks) = coeffs
        self.createParameter("P", value=Kprop)
        self.createParameter("I", value=Kinteg)
        self.createParameter("D", value=Kderiv)
        self.createParameter("Umin", value=Umin)
        self.createParameter("Umax", value=Umax)
        self.createParameter("Ks", value=Ks)

    def update(
        self,
        t1: float,
        t2: float,
        integral: FloatArr,
        setpoint: FloatArr,
        estimation: FloatArr,
        command: FloatArr,
    ) -> dict:
        int_x, corr = integral

        x = estimation[0]
        dx1 = estimation[1]
        (x_c,) = setpoint

        u = -(self.P * (x - x_c) + self.I * int_x + self.D * dx1)
        int_x += (x - x_c + corr) * (t2 - t1)

        u_sat = np.clip(u, self.Umin, self.Umax)
        corr = (u_sat - u) * self.Ks

        outputs = {}
        outputs["command"] = np.array([u_sat])
        outputs["integral"] = np.array([int_x, corr])

        return outputs


class LQRegulator(AController):
    """LQ regulator. See https://en.wikipedia.org/wiki/Linear-quadratic_regulator

    The inputs of the element are *estimation* and *setpoint*
    The outputs of the computer are **command**
    The size of the setpoint vector must be the same as the command vector

    The following attributes are to be defined by the user:

    Attributes:
        Q: State weight matrices
        R: Input weight matrices

    The following attributes are computed thanks to the method `LQRegulator.computeGain`

    Attributes:
        K: State feedback gains
        S: Solution to Riccati equation
        E: Eigenvalues of the closed loop system
        N: Precompensation gain

    Args:
        name: Name of the element
        shape_setpoint:  Shape of the setpoint data
        shape_estimation: Shape of the data expected by the estimation (> 2)

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        shape_setpoint: tuple,
        shape_estimation: tuple,
        snames: Iterable[str],
    ):
        AController.__init__(
            self,
            name,
            shape_setpoint=shape_setpoint,
            shape_estimation=shape_estimation,
            snames=snames,
        )
        self.createParameter(name="matA", value=0.0)
        self.createParameter(name="matB", value=0.0)
        self.createParameter(name="matC", value=0.0)
        self.createParameter(name="matD", value=0.0)
        self.createParameter(name="matQ", value=0.0)
        self.createParameter(name="matR", value=0.0)

        self.createParameter(name="matK", value=0.0)
        self.createParameter(name="matS", value=0.0)
        self.createParameter(name="matE", value=0.0)
        self.createParameter(name="matN", value=0.0)

    def computeGain(self):
        """Computes the optimal gain K, and the correct precompensation gain N.
        Called automatically at beginning of simulation

        K minimizes the quadratic cost:

        $$ J = \\int_0^\\infty (x' Q x + u' R u + 2 x' N u) dt $$

        and N is such that the steady-state error is 0

        Needs to have the following parameters set:

        * matA (n x n) State (or system) matrix
        * matB (n x m) Input matrix
        * matC (p x n) Output matrix
        * matD (p x m) Feedthrough (or feedforward) matrix
        * matQ covariance matrix of model noise
        * matR covariance matrix of measurement noise

        At the end of the call, the following attributes are updated :

        * matK : optimal gain
        * matS : solution to Riccati equation
        * matE : eigenvalues of the closed loop system
        * matN : precompensation gain

        """
        import control

        Q = self.matQ
        R = self.matR
        A = self.matA
        B = self.matB
        C = self.matC
        D = self.matD

        K, S, E = control.lqr(A, B, Q, R, method="scipy")

        self.matK = K
        self.matS = S
        self.matE = E

        # Computation of the precompensation gain
        iABK = lin.inv(A - B @ self.matK)
        M = -(C - D @ self.matK) @ iABK @ B + D
        self.matN = lin.inv(M)

    def resetCallback(self, t0: float):
        super().resetCallback(t0)

        self.computeGain()

    def update(
        self,
        t1: float,
        t2: float,
        setpoint: FloatArr,
        estimation: FloatArr,
        command: FloatArr,
    ) -> dict:
        u = self.matN @ setpoint - self.matK @ estimation

        outputs = {}
        outputs["command"] = u

        return outputs
