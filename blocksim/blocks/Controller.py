from typing import Iterable

import numpy as np
from scipy import linalg as lin

from ..core.Frame import Frame
from ..core.Node import AComputer
from .System import LTISystem


__all__ = [
    "AController",
    "PIDController",
    "AntiWindupPIDController",
    "LQRegulator",
]


class AController(AComputer):
    """Abstract class for a scalar controller

    Implement the method **compute_outputs** to make it concrete

    The inputs of the computer are **estimation** and **setpoint**
    The output of the computer is **command**

    Args:
      name
        Name of the element
      shape_setpoint
        Shape of the setpoint data
      shape_estimation
        Shape of the estimation data
      snames
        Name of each of the scalar components of the estimation.
        Its shape defines the shape of the data

    """

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

    The **estimation** is an estimation of the system, with the following constraints:
    * the first value of **estimation** is the position
    * the second value of **estimation** is the velocity

    Args:
      name
        Name of the element
      shape_estimation
        Shape of the data expected by the estimation (> 2)
      coeffs
        Coefficients of the retroaction (P, I, D)

    """

    def __init__(
        self, name: str, shape_estimation: tuple, snames: Iterable[str], coeffs: float
    ):
        AController.__init__(
            self,
            name,
            shape_setpoint=(1,),
            shape_estimation=shape_estimation,
            snames=snames,
        )
        self.defineOutput("integral", snames=["int"], dtype=np.float64)
        self.setInitialStateForOutput(np.array([0]), "integral")
        self.__coeffs = coeffs

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        integral: np.array,
        setpoint: np.array,
        estimation: np.array,
        command: np.array,
    ) -> dict:
        (ix,) = integral
        x = estimation[0]
        dx = estimation[1]
        (c,) = setpoint

        P, I, D = self.__coeffs
        u = P * (x - c) + I * ix + D * dx

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

    The **estimation** :math:`\hat{X}` must contain the state you want to control :math:`X` and its derivative :math:`\dot{X}`.:

    :math:`\hat{X} = (X, \dot{X}, ...)^T`

    The parameters P ,I and D are to be defined by the user :

    * P : Proportinnal gain
    * I : Integral gain
    * D : Derivative gain
    * Umin : if the command u is < Umin, then u = Umin
    * Umax : if the command u is > Umax, then u = Umax
    * Ks : gain of the anti-windup effect

    Args:
      name
        Name of the element
      shape_estimation
        Shape of the data expected by the estimation (> 2)
      coeffs
        Coefficients of the retroaction (P, I, D, Umin, Umax, Ks)

    """

    def __init__(
        self, name: str, shape_estimation: tuple, snames: Iterable[str], coeffs: float
    ):
        AController.__init__(
            self,
            name,
            shape_setpoint=(1,),
            shape_estimation=shape_estimation,
            snames=snames,
        )
        self.defineOutput("integral", snames=["usat", "int", "corr"], dtype=np.float64)
        self.setInitialStateForOutput(np.array([0]), "integral")
        self.__coeffs = coeffs

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        integral: np.array,
        setpoint: np.array,
        estimation: np.array,
        command: np.array,
    ) -> dict:
        _, int_x, corr = integral

        x = estimation[0]
        dx1 = estimation[1]
        (x_c,) = setpoint

        u = -(self.P * (x - x_c) + self.I * int_x + self.D * dx1)
        int_x += (x - x_c + corr) * (t2 - t1)

        u_sat = np.clip(u, self.Umin, self.Umax)
        corr = (u_sat - u) * self.Ks

        outputs = {}
        outputs["command"] = np.array([u])
        outputs["integral"] = np.array([u_sat, int_x, corr])

        return outputs


class LQRegulator(AController):
    """LQ regulator

    The inputs of the element are *estimation* and *setpoint*
    The outputs of the computer are **command**
    The size of the setpoint vector must be the same as the command vector

    The parameters Q and R are to be defined by the user :

    * Q : State weight matrices
    * R : Input weight matrices

    The parameters K, S, E and N are computed thanks to the method LQRegulator.computeGain

    * K : State feedback gains
    * S : Solution to Riccati equation
    * E : Eigenvalues of the closed loop system
    * N : Precompensation gain

    Args:
      name
        Name of the element
      shape_setpoint
        Shape of the setpoint data
      shape_estimation
        Shape of the data expected by the estimation (> 2)

    """

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

    def computeGain(
        self,
        Q: np.array = None,
        R: np.array = None,
        sys: LTISystem = None,
        precomp: bool = True,
    ):
        """Computes the optimal gain K, and the correct precompensation gain N

        K minimizes the quadratic cost
        :math:`J = \\int_0^\\infty (x' Q x + u' R u + 2 x' N u) dt`

        and N is such that the steady-state error is 0

        At the end of the call, the following attributes are updated :

        * Q : copy of the given Q matrix
        * R : copy of the given R matrix
        * K : optimal gain
        * S : solution to Riccati equation
        * E : eigenvalues of the closed loop system
        * N : precompensation gain

        Args:
          sys
            The system you want to control. Must be an instance of :class:`SystemControl.blocks.System.LTISystem`
            If None, LQRegulator uses the parameters A, B, C and D
          Q
            The state weight matrix
            If None, LQRegulator uses the parameters A, B, C and D
          R
            The command weight matrix
            If None, LQRegulator uses the parameters A, B, C and D
          precomp
            If True, also computes the N matrix such as for a setpoint c,
            the command u = N.c - K.x suppresses the steady-state error.

        """
        import control

        control.use_numpy_matrix(flag=False)

        if Q is None:
            Q = self.Q

        if R is None:
            R = self.R

        if sys is None:
            A = self.A
            B = self.B
        else:
            A = sys.A
            B = sys.B

        C = self.C
        D = self.D

        K, S, E = control.lqr(A, B, Q, R)

        self.Q = Q.copy()
        self.R = R.copy()
        self.K = K
        self.S = S
        self.E = E

        if precomp:
            # Computation of the precompensation gain
            iABK = lin.inv(A - B @ self.K)
            M = -(C - D @ self.K) @ iABK @ B + D
            self.N = lin.inv(M)
        else:
            nout = D.shape[0]
            self.N = np.eye(nout)

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        setpoint: np.array,
        estimation: np.array,
        command: np.array,
    ) -> dict:
        u = self.N @ setpoint - self.K @ estimation

        outputs = {}
        outputs["command"] = u

        return outputs
