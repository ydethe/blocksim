from abc import ABC, abstractmethod
from typing import Iterable
from functools import lru_cache

import numpy as np
from numpy import cos, sin, cosh, sinh, sqrt, exp, pi
from scipy import linalg as lin
from scipy.signal import firwin, cont2discrete

from ..exceptions import *
from .System import LTISystem
from ..core.Node import AComputer, Input, Output
from ..core.Frame import Frame
from ..Logger import Logger
from ..utils import quat_to_euler, euler_to_quat, assignVector


__all__ = [
    "AEstimator",
    "AKalmanFilter",
    "TimeInvariantKalmanFilter",
    "SteadyStateKalmanFilter",
    "SpectrumEstimator",
    "MadgwickFilter",
    "MahonyFilter",
]


class ConvergedGainMatrix(Output):
    def __init__(self, name: str, state: Output, meas: Input, dtype):
        ny = state.getDataShape()[0]
        nx = meas.getDataShape()[0]
        snames = [["" for _ in range(nx)] for _ in range(ny)]
        for i in range(ny):
            for j in range(nx):
                snames[i][j] = "K%i%i" % (i, j)

        Output.__init__(self, name=name, snames=snames, dtype=dtype)

        nc = self.getDataShape()
        cov0 = np.zeros(nc, dtype=dtype)
        self.setInitialState(cov0)

    def resetCallback(self, frame: Frame):
        import control

        estim = self.getComputer()

        Ad, _, Cd, _ = estim.discretize(estim.dt)

        # We solve the Discrete Algebraic Riccati Equation (DARE)
        # The matrix Pp is the prediction error covariance matrix in steady state which is the positive solution of the DARE
        Pp, _, _ = control.dare(Ad.T, Cd.T, estim.matQ, estim.matR)

        Pp = np.array(Pp)

        # Converged gain matrix
        K = Pp @ Cd.T @ lin.inv(Cd @ Pp @ Cd.T + estim.matR)

        self.setData(K)


class ConvergedStateCovariance(Output):
    def __init__(self, name: str, state: Output, dtype):
        nx = state.getDataShape()[0]
        state_names = state.getScalarNames()
        snames = [["" for _ in range(nx)] for _ in range(nx)]
        for i in range(nx):
            for j in range(nx):
                snames[i][j] = "cov%s%s" % (state_names[i], state_names[j])

        Output.__init__(self, name=name, snames=snames, dtype=dtype)

        nc = self.getDataShape()
        cov0 = np.zeros(nc, dtype=dtype)
        self.setInitialState(cov0)

    def resetCallback(self, frame: Frame):
        import control

        n, _ = self.getDataShape()
        estim = self.getComputer()

        Ad, _, Cd, _ = estim.discretize(estim.dt)

        # We solve the Discrete Algebraic Riccati Equation (DARE)
        # The matrix Pp is the prediction error covariance matrix in steady state which is the positive solution of the DARE
        Pp, _, _ = control.dare(Ad.T, Cd.T, estim.matQ, estim.matR)

        Pp = np.array(Pp)

        # Converged gain matrix
        K = Pp @ Cd.T @ lin.inv(Cd @ Pp @ Cd.T + estim.matR)

        # The matrix P is the estimation error covariance matrix in steady state
        P = (np.eye(n) - K @ Cd) @ Pp

        self.setData(P)


class AEstimator(AComputer):
    """Abstract class for a state estimator

    Implement the method **updateAllOutput** to make it concrete

    The input name of the element are **command** and **measurement**
    The outputs of the computer are **state** and **output**

    Args:
      name
        Name of the element
      shape_cmd
        Shape of the command
      shape_meas
        Shape of the measurement
      snames_state
        Name of each of the scalar components of the state.
        Its shape defines the shape of the data
      snames_output
        Name of each of the scalar components of the output.
        Its shape defines the shape of the data

    """

    def __init__(
        self,
        name: str,
        shape_cmd: tuple,
        shape_meas: tuple,
        snames_state: Iterable[str],
        snames_output: Iterable[str],
        dtype=np.float64,
    ):
        AComputer.__init__(self, name)
        self.defineInput("command", shape=shape_cmd, dtype=dtype)
        self.defineInput("measurement", shape=shape_meas, dtype=dtype)
        self.defineOutput("state", snames_state, dtype=dtype)
        self.defineOutput("output", snames_output, dtype=dtype)


class AKalmanFilter(AEstimator):
    """Definition of the matrices which define the filter

    The inputs of the element are **command** and **measurement**
    The outputs of the computer are **state**, **output**, **statecov** and **matK**

    https://en.wikipedia.org/wiki/State-space_representation#Linear_systems

    :math:`X_{k+1} = A.X_k + B.u_k + N_x`

    :math:`Y_k = C.X_k + D.u_k + N_y`

    with :

    * n number of states
    * m number of commands
    * p number of measured states

    * X state vector (n,1)
    * Y output vector (p,1)
    * u input (or control) vector (m,1)

    * A (n x n) State (or system) matrix
    * B (n x m) Input matrix
    * C (p x n) Output matrix
    * D (p x m) Feedthrough (or feedforward) matrix
    * Nx gaussian noise with covariance Q
    * Ny gaussian noise with covariance R

    The matrices A, B, C, D, Q, R must be implemented for AKalmanFilter to be used

    Args:
      name
        Name of the element
      shape_cmd
        Shape of the command
      shape_meas
        Shape of the measurement
      snames_state
        Name of each of the scalar components of the state.
        Its shape defines the shape of the data
      snames_output
        Name of each of the scalar components of the output.
        Its shape defines the shape of the data

    """

    def __init__(
        self,
        name: str,
        shape_cmd: tuple,
        shape_meas: tuple,
        snames_state: Iterable[str],
        snames_output: Iterable[str],
        dtype=np.float64,
    ):
        AEstimator.__init__(
            self,
            name=name,
            shape_cmd=shape_cmd,
            shape_meas=shape_meas,
            snames_state=snames_state,
            snames_output=snames_output,
            dtype=dtype,
        )

        state = self.getOutputByName("state")
        meas = self.getInputByName("measurement")
        nx = state.getDataShape()[0]
        ny = meas.getDataShape()[0]
        state_names = state.getScalarNames()

        cov_snames = [["" for _ in range(nx)] for _ in range(nx)]
        matK_snames = [["" for _ in range(ny)] for _ in range(nx)]
        for i in range(nx):
            for j in range(nx):
                cov_snames[i][j] = "cov%s%s" % (state_names[i], state_names[j])
            for j in range(ny):
                matK_snames[i][j] = "K%i%i" % (i, j)

        otp = self.defineOutput("statecov", snames=cov_snames, dtype=dtype)
        nc = otp.getDataShape()
        cov0 = np.zeros(nc, dtype=dtype)
        otp.setInitialState(cov0)

        otp = self.defineOutput("matK", snames=matK_snames, dtype=dtype)
        nc = otp.getDataShape()
        matK0 = np.zeros(nc, dtype=dtype)
        otp.setInitialState(matK0)

    @abstractmethod
    def A(self, frame: Frame) -> np.array:
        """(n x n) State (or system) matrix

        Args:
          frame
            Time frame

        """
        pass

    @abstractmethod
    def B(self, frame: Frame) -> np.array:
        """(n x m) Input matrix

        Args:
          frame
            Time frame

        """
        pass

    @abstractmethod
    def C(self, frame: Frame) -> np.array:
        """(p x n) Output matrix

        Args:
          frame
            Time frame

        """
        pass

    @abstractmethod
    def D(self, frame: Frame) -> np.array:
        """(p x m) Feedthrough (or feedforward) matrix

        Args:
          frame
            Time frame

        """
        pass

    @abstractmethod
    def Q(self, frame: Frame) -> np.array:
        """(n x n) Gaussian noise covariance for the state vector

        Args:
          frame
            Time frame

        """
        pass

    @abstractmethod
    def R(self, frame: Frame) -> np.array:
        """(n x n) Gaussian noise covariance for the measurement vector

        Args:
          frame
            Time frame

        """
        pass

    def _prediction(self, xest: np.array, P: np.array, u: np.array, frame: Frame):
        if np.abs(frame.getTimeStep()) < 1e-9:
            return xest.copy(), self.C(frame) @ xest + self.D(frame) @ u, P.copy()

        xest_pred = self.A(frame) @ xest + self.B(frame) @ u
        meas_pred = self.C(frame) @ xest_pred + self.D(frame) @ u
        P_pred = self.A(frame) @ P @ np.transpose(self.A(frame)) + self.Q(frame)

        return xest_pred, meas_pred, P_pred

    def _update(self, xest_pred, meas_pred, P_pred, u, meas, frame):
        y = meas - meas_pred

        S = self.C(frame) @ P_pred @ np.transpose(self.C(frame)) + self.R(frame)
        K = P_pred @ np.transpose(self.C(frame)) @ lin.inv(S)

        xest = xest_pred + K @ y
        P = (np.eye(len(xest_pred)) - K @ self.C(frame)) @ P_pred

        return xest, K, P

    def updateAllOutput(self, frame: Frame):
        u = self.getDataForInput(frame, name="command")
        meas = self.getDataForInput(frame, name="measurement")

        state = self.getOutputByName("state")
        output = self.getOutputByName("output")
        matK = self.getOutputByName("matK")
        statecov = self.getOutputByName("statecov")

        xest = state.getDataForFrame(frame)
        P = statecov.getDataForFrame(frame)

        xest_pred, meas_pred, P_pred = self._prediction(xest, P, u, frame)
        xest, K, P = self._update(xest_pred, meas_pred, P_pred, u, meas, frame)

        output.setData(meas_pred)
        statecov.setData(P)
        matK.setData(K)
        state.setData(xest)


class TimeInvariantKalmanFilter(AKalmanFilter):
    """Definition of the matrices which define the filter

    The inputs of the element are **command** and **measurement**
    The outputs of the computer are **state**, **output**, and **statecov**

    https://en.wikipedia.org/wiki/State-space_representation#Linear_systems

    :math:`X_{k+1} = A.X_k + B.u_k + N_x`

    :math:`Y_k = C.X_k + D.u_k + N_y`

    with :

    * n number of states
    * m number of commands
    * p number of measured states

    * X state vector (n,1)
    * Y output vector (p,1)
    * u input (or control) vector (m,1)

    The matrices A, B, C, D, Q, R are provided with attributes, and A and B matrices are those of the *continuous time* system

    **The attributes are the following:**

      matA
        (n x n) Continuous state matrix
      matB
        (n x m) Continuous input matrix
      matC
        (p x n) Output matrix
      matD
        (p x m) Feedthrough (or feedforward) matrix
      matQ
        (n x n) N_x covariance
      matR
        (p x p) N_y covariance

    Args:
      name
        Name of the element
      shape_cmd
        Shape of the command
      shape_meas
        Shape of the measurement
      snames_state
        Name of each of the scalar components of the state.
        Its shape defines the shape of the data
      snames_output
        Name of each of the scalar components of the output.
        Its shape defines the shape of the data

    """

    def __init__(
        self,
        name: str,
        shape_cmd: tuple,
        shape_meas: tuple,
        snames_state: Iterable[str],
        snames_output: Iterable[str],
        dtype=np.float64,
    ):
        AKalmanFilter.__init__(
            self,
            name=name,
            shape_cmd=shape_cmd,
            shape_meas=shape_meas,
            snames_state=snames_state,
            snames_output=snames_output,
            dtype=dtype,
        )

    @lru_cache(maxsize=None)
    def discretize(
        self, dt: float, method: str = "zoh", alpha: float = None
    ) -> Iterable[np.array]:
        """

        Args:
          dt
            Fixed time step of the simulation
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
          Cd
            Discrete output matrix
          Dd
            Discrete direct term

        """
        sys = (self.matA, self.matB, self.matC, self.matD)
        Ad, Bd, Cd, Dd, dt = cont2discrete(sys, dt, method=method, alpha=alpha)
        return Ad, Bd, Cd, Dd

    def A(self, frame: Frame) -> np.array:
        """(n x n) State (or system) matrix

        Args:
          frame
            Time frame

        """
        dt = frame.getTimeStep()
        Ad, Bd, Cd, Dd = self.discretize(dt)
        return Ad

    def B(self, frame: Frame) -> np.array:
        """(n x m) Input matrix

        Args:
          frame
            Time frame

        """
        dt = frame.getTimeStep()
        Ad, Bd, Cd, Dd = self.discretize(dt)
        return Bd

    def C(self, frame: Frame) -> np.array:
        """(p x n) Output matrix

        Args:
          frame
            Time frame

        """
        return self.matC

    def D(self, frame: Frame) -> np.array:
        """(p x m) Feedthrough (or feedforward) matrix

        Args:
          frame
            Time frame

        """
        return self.matD

    def Q(self, frame: Frame) -> np.array:
        """(n x n) Gaussian noise covariance for the state vector

        Args:
          frame
            Time frame

        """
        return self.matQ

    def R(self, frame: Frame) -> np.array:
        """(n x n) Gaussian noise covariance for the measurement vector

        Args:
          frame
            Time frame

        """
        return self.matR


class SteadyStateKalmanFilter(TimeInvariantKalmanFilter):
    """Definition of the matrices which define the filter

    The inputs of the element are **command** and **measurement**
    The outputs of the computer are **state**, **output**, **statecov** and **matK**

    https://en.wikipedia.org/wiki/State-space_representation#Linear_systems

    :math:`X_{k+1} = A.X_k + B.u_k + N_x`

    :math:`Y_k = C.X_k + D.u_k + N_y`

    with :

    * n number of states
    * m number of commands
    * p number of measured states

    * X state vector (n,1)
    * Y output vector (p,1)
    * u input (or control) vector (m,1)

    The matrices A, B, C, D, Q, R are provided with attributes, and A and B matrices are those of the *continuous time* system
    SteadyStateKalmanFilter also takes dt as attribute, since the converged gain matrix depends on the time step, which shall be constant.
    After the simulation (or a call to SteadyStateKalmanFilter.reset), the attribute matK contains the gain matrix in steady state

    **The attributes are the following:**

      dt
        Time step of the discretized Kalman filter
      matA
        (n x n) Continuous state matrix
      matB
        (n x m) Continuous input matrix
      matC
        (p x n) Output matrix
      matD
        (p x m) Feedthrough (or feedforward) matrix
      matQ
        (n x n) N_x covariance
      matR
        (p x p) N_y covariance

    Args:
      name
        Name of the element
      dt
        Time step used for solving the Discrete Algebraic Riccati Equation
      shape_cmd
        Shape of the command
      shape_meas
        Shape of the measurement
      snames_state
        Name of each of the scalar components of the state.
        Its shape defines the shape of the data
      snames_output
        Name of each of the scalar components of the output.
        Its shape defines the shape of the data

    """

    def __init__(
        self,
        name: str,
        dt: float,
        shape_cmd: tuple,
        shape_meas: tuple,
        snames_state: Iterable[str],
        snames_output: Iterable[str],
        dtype=np.float64,
    ):
        TimeInvariantKalmanFilter.__init__(
            self,
            name=name,
            shape_cmd=shape_cmd,
            shape_meas=shape_meas,
            snames_state=snames_state,
            snames_output=snames_output,
            dtype=dtype,
        )
        self.dt = dt

        statecov = ConvergedStateCovariance(
            "statecov", state=self.getOutputByName("state"), dtype=dtype
        )
        self.replaceOutput(old_name="statecov", new_output=statecov)

        matK = ConvergedGainMatrix(
            "matK",
            state=self.getOutputByName("state"),
            meas=self.getInputByName("measurement"),
            dtype=dtype,
        )
        self.replaceOutput(old_name="matK", new_output=matK)

    def updateAllOutput(self, frame: Frame):
        dt = frame.getTimeStep()
        assert np.abs(dt) / self.dt < 1e-5 or np.abs(dt - self.dt) / self.dt < 1e-5
        u = self.getDataForInput(frame, name="command")
        meas = self.getDataForInput(frame, name="measurement")

        state = self.getOutputByName("state")
        output = self.getOutputByName("output")
        matK = self.getOutputByName("matK")
        statecov = self.getOutputByName("statecov")

        xest = state.getDataForFrame(frame)
        P = statecov.getDataForFrame(frame)
        K = matK.getDataForFrame(frame)

        xest_pred, meas_pred, P_pred = self._prediction(xest, P, u, frame)

        # Modified update with converged gain matrix
        y = meas - meas_pred
        xest = xest_pred + K @ y
        # [END]Modified update with converged gain matrix

        state.setData(xest)
        output.setData(meas_pred)
