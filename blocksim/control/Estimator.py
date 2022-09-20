from abc import abstractmethod
from typing import Iterable, Any
from functools import lru_cache

from nptyping import NDArray, Shape
import numpy as np
from numpy import exp, pi
from scipy import linalg as lin
from scipy.signal import cont2discrete, TransferFunction

from ..exceptions import *
from ..core.Node import AComputer, Input, Output
from ..loggers.Logger import Logger
from ..dsp.DSPMap import DSPRectilinearMap
from ..dsp.DSPFilter import ArbitraryDSPFilter
from ..utils import quat_to_euler, assignVector


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

    __slots__ = ["__K"]

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

    def resetCallback(self, t0: float):
        super().resetCallback(t0)

        # from control import dare
        from scipy.linalg import solve_discrete_are as dare

        estim = self.getComputer()

        Ad, _, Cd, _ = estim.discretize(estim.dt)

        # We solve the Discrete Algebraic Riccati Equation (DARE)
        # The matrix Pp is the prediction error covariance matrix in steady state which is the positive solution of the DARE
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_discrete_are.html
        Pp = dare(a=Ad.T, b=Cd.T, q=estim.matQ, r=estim.matR)
        # a = Ad.T
        # b = Cd.T
        # q = estim.matQ
        # r = estim.matR
        # x = Pp
        # aH = np.conj(a.T)
        # bH = np.conj(b.T)

        # v = aH @ x @ a - x - (aH @ x @ b) @ lin.inv(r + bH @ x @ b) @ (bH @ x @ a) + q
        # err = np.max(np.abs(v))
        # if err > 1e-10:
        #     raise AssertionError(f"DARE check failed : err = {err:.3g}")

        # Converged gain matrix
        self.__K = Pp @ Cd.T @ lin.inv(Cd @ Pp @ Cd.T + estim.matR)

        self.setData(self.__K)
        self.setInitialState(self.__K)

    def getConvergedGainMatrix(self) -> NDArray[Any, Any]:
        """Returns the offline gain matrix K

        Returns:
            The offline gain matrix K

        """
        return self.__K.copy()


class ConvergedStateCovariance(Output):

    __slots__ = ["__P"]

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

    def resetCallback(self, t0: float):
        super().resetCallback(t0)

        # from control import dare
        from scipy.linalg import solve_discrete_are as dare

        n, _ = self.getDataShape()
        estim = self.getComputer()

        Ad, _, Cd, _ = estim.discretize(estim.dt)

        # We solve the Discrete Algebraic Riccati Equation (DARE)
        # The matrix Pp is the prediction error covariance matrix in steady state which is the positive solution of the DARE
        Pp = dare(Ad.T, Cd.T, estim.matQ, estim.matR)

        # Converged gain matrix
        K = Pp @ Cd.T @ lin.inv(Cd @ Pp @ Cd.T + estim.matR)

        # The matrix P is the estimation error covariance matrix in steady state
        self.__P = (np.eye(n) - K @ Cd) @ Pp

        self.setData(self.__P)
        self.setInitialState(self.__P)

    def getConvergedStateCovariance(self) -> NDArray[Any, Any]:
        """Returns the offline covariance matrix P

        Returns:
            The offline covariance matrix P

        """
        return self.__P.copy()


class AEstimator(AComputer):
    """Abstract class for a state estimator

    Implement the method **update** to make it concrete

    The input name of the element are **command** and **measurement**
    The outputs of the computer are **state** and **output**
    The **output** vector and the **measurement** vector shall have the same shape.

    Args:
        name: Name of the element
        shape_cmd: Shape of the command
        snames_state: Name of each of the scalar components of the state.
          Its shape defines the shape of the data
        snames_output: Name of each of the scalar components of the output.
          Its shape defines the shape of the data

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        shape_cmd: tuple,
        snames_state: Iterable[str],
        snames_output: Iterable[str],
        dtype=np.float64,
    ):
        AComputer.__init__(self, name)
        shape_meas = (len(snames_output),)
        self.defineInput("command", shape=shape_cmd, dtype=dtype)
        self.defineInput("measurement", shape=shape_meas, dtype=dtype)
        self.defineOutput("state", snames_state, dtype=dtype)
        self.defineOutput("output", snames_output, dtype=dtype)


class AKalmanFilter(AEstimator):
    """Definition of the matrices which define the filter

    The inputs of the element are **command** and **measurement**
    The outputs of the computer are **state**, **output**, **statecov** and **matK**

    https://en.wikipedia.org/wiki/State-space_representation#Linear_systems

    $$ X_{k+1} = A.X_k + B.u_k + N_x $$
    $$ Y_k = C.X_k + D.u_k + N_y $$

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
        name: Name of the element
        shape_cmd: Shape of the command
        snames_state: Name of each of the scalar components of the state.
          Its shape defines the shape of the data
        snames_output: Name of each of the scalar components of the output.
          Its shape defines the shape of the data

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        shape_cmd: tuple,
        snames_state: Iterable[str],
        snames_output: Iterable[str],
        dtype=np.float64,
    ):
        AEstimator.__init__(
            self,
            name=name,
            shape_cmd=shape_cmd,
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
    def A(self, t1: float, t2: float) -> NDArray[Any, Any]:  # pragma: no cover
        """(n x n) State (or system) matrix

        Args:
            t1: Timestamp of the begining of the simulation step (s)
            t2: Timestamp of the end of the simulation step (s)

        Returns:
            The state matrix

        """
        pass

    @abstractmethod
    def B(self, t1: float, t2: float) -> NDArray[Any, Any]:  # pragma: no cover
        """(n x m) Input matrix

        Args:
            t1: Timestamp of the begining of the simulation step (s)
            t2: Timestamp of the end of the simulation step (s)

        Returns:
            The input matrix

        """
        pass

    @abstractmethod
    def C(self, t: float) -> NDArray[Any, Any]:  # pragma: no cover
        """(p x n) Output matrix

        Args:
            t1: Timestamp of the begining of the simulation step (s)
            t2: Timestamp of the end of the simulation step (s)

        Returns:
            The output matrix

        """
        pass

    @abstractmethod
    def D(self, t: float) -> NDArray[Any, Any]:  # pragma: no cover
        """(p x m) Feedthrough (or feedforward) matrix

        Args:
            t1: Timestamp of the begining of the simulation step (s)
            t2: Timestamp of the end of the simulation step (s)

        Returns:
            The feedthrough matrix

        """
        pass

    @abstractmethod
    def Q(self, t: float) -> NDArray[Any, Any]:  # pragma: no cover
        """(n x n) Gaussian noise covariance for the state vector

        Args:
            t1: Timestamp of the begining of the simulation step (s)
            t2: Timestamp of the end of the simulation step (s)

        Returns:
            The covariance matrix for the state vector

        """
        pass

    @abstractmethod
    def R(self, t: float) -> NDArray[Any, Any]:  # pragma: no cover
        """(n x n) Gaussian noise covariance for the measurement vector

        Args:
            t1: Timestamp of the begining of the simulation step (s)
            t2: Timestamp of the end of the simulation step (s)

        Returns:
            The covariance matrix for the measurement vector

        """
        pass

    def _prediction(
        self,
        xest: NDArray[Any, Any],
        P: NDArray[Any, Any],
        u: NDArray[Any, Any],
        t1: float,
        t2: float,
    ):
        if np.abs(t2 - t1) < 1e-9:
            return xest.copy(), self.C(t2) @ xest + self.D(t2) @ u, P.copy()

        xest_pred = self.A(t1, t2) @ xest + self.B(t1, t2) @ u
        meas_pred = self.C(t2) @ xest_pred + self.D(t2) @ u
        P_pred = self.A(t1, t2) @ P @ np.transpose(self.A(t1, t2)) + self.Q(t2)

        return xest_pred, meas_pred, P_pred

    def _update(self, xest_pred, meas_pred, P_pred, u, meas, t1, t2):
        y = meas - meas_pred

        S = self.C(t2) @ P_pred @ np.transpose(self.C(t2)) + self.R(t2)
        K = P_pred @ np.transpose(self.C(t2)) @ lin.inv(S)

        xest = xest_pred + K @ y
        P = (np.eye(len(xest_pred)) - K @ self.C(t2)) @ P_pred

        return xest, K, P

    # def resetCallback(self, t0:float):
    #     super().resetCallback(t0)

    #     state = self.getOutputByName("state")

    #     state.setInitialState(X0)

    def update(
        self,
        t1: float,
        t2: float,
        command: NDArray[Any, Any],
        measurement: NDArray[Any, Any],
        state: NDArray[Any, Any],
        output: NDArray[Any, Any],
        statecov: NDArray[Any, Any],
        matK: NDArray[Any, Any],
    ) -> dict:
        xest_pred, meas_pred, P_pred = self._prediction(
            state, statecov, command, t1, t2
        )
        xest, K, P = self._update(
            xest_pred, meas_pred, P_pred, command, measurement, t1, t2
        )

        outputs = {}
        outputs["output"] = meas_pred
        outputs["statecov"] = P
        outputs["matK"] = K
        outputs["state"] = xest

        return outputs


class TimeInvariantKalmanFilter(AKalmanFilter):
    """Definition of the matrices which define the filter

    The inputs of the element are **command** and **measurement**
    The outputs of the computer are **state**, **output**, and **statecov**

    https://en.wikipedia.org/wiki/State-space_representation#Linear_systems

    $$ X_{k+1} = A.X_k + B.u_k + N_x $$
    $$ Y_k = C.X_k + D.u_k + N_y $$

    with :

    * n number of states
    * m number of commands
    * p number of measured states

    * X state vector (n,1)
    * Y output vector (p,1)
    * u input (or control) vector (m,1)

    The matrices A, B, C, D, Q, R are provided with attributes,
    and A and B matrices are those of the *continuous time* system

    Attributes:
        matA: (n x n) Continuous state matrix
        matB:  (n x m) Continuous input matrix
        matC: (p x n) Output matrix
        matD: (p x m) Feedthrough (or feedforward) matrix
        matQ: (n x n) N_x covariance
        matR: (p x p) N_y covariance

    Args:
        name: Name of the element
        shape_cmd: Shape of the command
        snames_state: Name of each of the scalar components of the state.
          Its shape defines the shape of the data
        snames_output: Name of each of the scalar components of the output.
          Its shape defines the shape of the data

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        shape_cmd: tuple,
        snames_state: Iterable[str],
        snames_output: Iterable[str],
        dtype=np.float64,
    ):
        AKalmanFilter.__init__(
            self,
            name=name,
            shape_cmd=shape_cmd,
            snames_state=snames_state,
            snames_output=snames_output,
            dtype=dtype,
        )
        self.createParameter("matA", value=0)
        self.createParameter("matB", value=0)
        self.createParameter("matC", value=0)
        self.createParameter("matD", value=0)
        self.createParameter("matQ", value=0)
        self.createParameter("matR", value=0)

    @lru_cache(maxsize=None)
    def discretize(
        self, dt: float, method: str = "zoh", alpha: float = None
    ) -> Iterable[np.array]:
        """Turns the continous system into a discrete one

        Args:
            dt: Fixed time step of the simulation (s)
            method: Discretization method:

              * gbt: generalized bilinear transformation
              * bilinear: Tustin’s approximation (“gbt” with alpha=0.5)
              * euler: Euler (or forward differencing) method (“gbt” with alpha=0)
              * backward_diff: Backwards differencing (“gbt” with alpha=1.0)
              * zoh: zero-order hold (default)
            alpha: Parameter for the gbt method, within [0, 1]
              The generalized bilinear transformation weighting parameter, which should only be specified with method=”gbt”, and is ignored otherwise

        Returns:
            A tuple of 4 matrices:

            * Ad: Discrete state matrix
            * Bd: Discrete input matrix
            * Cd: Discrete output matrix
            * Dd: Discrete feedthrough matrix

        """
        sys = (self.matA, self.matB, self.matC, self.matD)
        Ad, Bd, Cd, Dd, dt = cont2discrete(sys, dt, method=method, alpha=alpha)
        return Ad, Bd, Cd, Dd

    def A(self, t1: float, t2: float) -> NDArray[Any, Any]:
        dt = t2 - t1
        Ad, Bd, Cd, Dd = self.discretize(dt)
        return Ad

    def B(self, t1: float, t2: float) -> NDArray[Any, Any]:
        dt = t2 - t1
        Ad, Bd, Cd, Dd = self.discretize(dt)
        return Bd

    def C(self, t: float) -> NDArray[Any, Any]:
        return self.matC

    def D(self, test_ss_kal: float) -> NDArray[Any, Any]:
        return self.matD

    def Q(self, t: float) -> NDArray[Any, Any]:
        return self.matQ

    def R(self, t: float) -> NDArray[Any, Any]:
        return self.matR


class SteadyStateKalmanFilter(TimeInvariantKalmanFilter):
    """Definition of the matrices which define the filter

    The inputs of the element are **command** and **measurement**
    The outputs of the computer are **state**, **output**, **statecov** and **matK**

    https://en.wikipedia.org/wiki/State-space_representation#Linear_systems

    $$ X_{k+1} = A.X_k + B.u_k + N_x $$
    $$ Y_k = C.X_k + D.u_k + N_y $$

    with :

    * n number of states
    * m number of commands
    * p number of measured states

    * X state vector (n,1)
    * Y output vector (p,1)
    * u input (or control) vector (m,1)

    The matrices A, B, C, D, Q, R are provided with attributes,
    and A and B matrices are those of the *continuous time* system
    SteadyStateKalmanFilter also takes dt as attribute, since the converged gain matrix depends on
    the time step, which shall be constant.
    After the simulation (or a call to SteadyStateKalmanFilter.reset),
    the attribute matK contains the gain matrix in steady state

    Attributes:
        dt: Time step of the discretized Kalman filter
        matA: (n x n) Continuous state matrix
        matB: (n x m) Continuous input matrix
        matC: (p x n) Output matrix
        matD: (p x m) Feedthrough (or feedforward) matrix
        matQ: (n x n) N_x covariance
        matR: (p x p) N_y covariance

    Args:
        name: Name of the element
        dt: Time step used for solving the Discrete Algebraic Riccati Equation (s)
        shape_cmd: Shape of the command
        snames_state: Name of each of the scalar components of the state.
          Its shape defines the shape of the data
        snames_output: Name of each of the scalar components of the output.
          Its shape defines the shape of the data

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        dt: float,
        shape_cmd: tuple,
        snames_state: Iterable[str],
        snames_output: Iterable[str],
        dtype=np.float64,
    ):
        TimeInvariantKalmanFilter.__init__(
            self,
            name=name,
            shape_cmd=shape_cmd,
            snames_state=snames_state,
            snames_output=snames_output,
            dtype=dtype,
        )

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

        self.createParameter("dt", value=dt)

    def getConvergedStateCovariance(self) -> NDArray[Any, Any]:
        """Returns the offline covariance matrix P

        Returns:
            The offline covariance matrix P

        """
        otp = self.getOutputByName("statecov")
        return otp.getConvergedStateCovariance()

    def getConvergedGainMatrix(self) -> NDArray[Any, Any]:
        """Returns the offline gain matrix K

        Returns:
            The offline gain matrix K

        """
        otp = self.getOutputByName("matK")
        return otp.getConvergedGainMatrix()

    def update(
        self,
        t1: float,
        t2: float,
        command: NDArray[Any, Any],
        measurement: NDArray[Any, Any],
        state: NDArray[Any, Any],
        output: NDArray[Any, Any],
        statecov: NDArray[Any, Any],
        matK: NDArray[Any, Any],
    ) -> dict:
        xest_pred, meas_pred, P_pred = self._prediction(
            state, statecov, command, t1, t2
        )

        # Modified update with converged gain matrix
        y = measurement - meas_pred
        xest = xest_pred + matK @ y
        # [END]Modified update with converged gain matrix

        outputs = {}
        outputs["output"] = meas_pred
        outputs["statecov"] = statecov.copy()
        outputs["matK"] = matK.copy()
        outputs["state"] = xest

        return outputs


class SpectrumEstimator(SteadyStateKalmanFilter):
    r"""Frequency tracker based on a Kalman filter.
    The number of frequencies to be tracked in *tracks* and the number of states shall be equal

    The associated Kalman system is:

    $$ \dot{X} = A.X $$
    $$ Y=C.X $$

    With

    $$ X = (A_1.exp(i.\omega_1.t),A_2.exp(i.\omega_2.t),\dots,A_n.exp(i.\omega_n.t))^T $$

    $$
        A=\begin{pmatrix}
        i.\omega_1 & 0   & 0 & \dots  & 0 \\
        0   & i.\omega_2 & 0 & \dots & 0 \\
        \vdots & \vdots & \ddots & \vdots \\
        0 & 0   & 0 & \dots  & i.\omega_n
        \end{pmatrix}
    $$

    $$
        C=\begin{pmatrix}
        1 & 1   & 1 & \dots  & 1
        \end{pmatrix}
    $$

    At each time step, the measurement is one complex signal sample.
    Applying Kalman filter theory to this system, we get at each time step an updated estimate of X.
    We can extract from \( X \) the complex coefficients \( A_k \), which are amplitude and phase for each of the pulsations \( \omega_k \)

    The input of the element is **measurement**

    Args:
        name: Name of the system
        name_of_outputs: Names of the outputs of the element
        name_of_states: Names of the states of the element
        tracks: List of the frequencies to be tracked (Hz)

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        dt: float,
        snames_state: Iterable[str],
        sname_output: str,
        tracks: NDArray[Any, Any],
    ):
        SteadyStateKalmanFilter.__init__(
            self,
            name=name,
            dt=dt,
            shape_cmd=(1,),
            snames_state=snames_state,
            snames_output=[sname_output],
            dtype=np.complex128,
        )
        self.createParameter("tracks", tracks)
        self.removeInput("command")

        nb_tracks = len(self.tracks)

        self.matA = 1j * 2 * np.pi * np.diag(self.tracks)
        self.matB = np.zeros((nb_tracks, 1), dtype=np.complex128)
        self.matC = np.ones((1, nb_tracks), dtype=np.complex128)
        self.matD = np.zeros((1, 1), dtype=np.complex128)

    def to_dlti(self, ma_freq: Iterable[int] = None) -> TransferFunction:
        """Creates a scipy TransferFunction instance.
        See https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.TransferFunction.html

        Args:
            ma_freq: List of index of frequencies that shall be selected by the ArbitraryDSPFilter.
                By default, all the frequencies specified in **tracks** are taken

        Returns:
            The TransferFunction instance

        """
        from scipy.signal import StateSpace

        matK = self.getOutputByName("matK")
        matK.resetCallback(None)
        K = self.getConvergedGainMatrix()

        n = len(self.tracks)
        if ma_freq is None:
            ma_freq = list(range(n))

        Ad, Bd, Cd, Dd = self.discretize(self.dt)
        Cf = np.zeros((1, n))
        Cf[0, ma_freq] = 1.0
        sys = StateSpace(Ad - K @ Cd, K, Cf, Dd, dt=self.dt)
        sys2 = sys.to_tf()

        return sys2

    def getEstimatingFilter(
        self, name: str, ma_freq: Iterable[int] = None
    ) -> ArbitraryDSPFilter:
        """Returns the filter equivalent to the SISO system
        that takes as input the measured sample, and as output the estimated signal

        Args:
            name: Name of the created filter
            ma_freq: List of index of frequencies that shall be selected by the ArbitraryDSPFilter.
                By default, all the frequencies specified in **tracks** are taken

        Returns:
            The ArbitraryDSPFilter instance

        """
        sys = self.to_dlti(ma_freq=ma_freq)

        filt = ArbitraryDSPFilter(
            name=name, samplingPeriod=self.dt, num=sys.num, den=sys.den
        )

        return filt

    def getSpectrogram(self, log: Logger) -> DSPRectilinearMap:
        """Gets the map from the Logger after simulation

        Args:
            log: The Logger after simulation

        Returns:
            The map

        """
        nb_tracks = len(self.tracks)
        t_sim = log.getValue("t")

        img = np.empty((nb_tracks, len(t_sim)), dtype=np.complex128)
        otp = self.getOutputByName("state")
        ns = otp.getScalarNames()
        for k in range(nb_tracks):
            f = self.tracks[k]
            vname = "%s_%s_%s" % (self.getName(), otp.getName(), ns[k])
            x = log.getValue(vname)
            y = x * exp(-1j * 2 * pi * t_sim * f)
            img[k, :] = y

        df = self.tracks[1] - self.tracks[0]

        spg = DSPRectilinearMap(
            name="map",
            samplingXStart=t_sim[0] - nb_tracks * self.dt / 2,
            samplingXPeriod=self.dt,
            samplingYStart=self.tracks[0],
            samplingYPeriod=df,
            img=img,
        )
        spg.name_of_x_var = "Time"
        spg.unit_of_x_var = "s"
        spg.name_of_y_var = "Frequency"
        spg.unit_of_y_var = "Hz"

        return spg

    def update(
        self,
        t1: float,
        t2: float,
        measurement: NDArray[Any, Any],
        state: NDArray[Any, Any],
        output: NDArray[Any, Any],
        statecov: NDArray[Any, Any],
        matK: NDArray[Any, Any],
    ) -> dict:
        xest_pred, meas_pred, P_pred = self._prediction(
            state, statecov, np.zeros(1), t1, t2
        )

        # Modified update with converged gain matrix
        y = measurement - meas_pred
        xest = xest_pred + matK @ y
        # [END]Modified update with converged gain matrix

        outputs = {}
        outputs["output"] = meas_pred
        outputs["statecov"] = statecov.copy()
        outputs["matK"] = matK.copy()
        outputs["state"] = xest

        return outputs


class MadgwickFilter(AComputer):
    """Madgwick filter

    Estimates roll, pitch, yaw (rad)

    The inputs of the element are **command** and **measurement**

    Attributes:
        beta: Proportional gain of the Madgwick algorithm
        mag_softiron_matrix: Soft iron error compensation matrix
        mag_offsets: Offsets applied to raw x/y/z values (uTesla)

    Args:
        name: Name of the element
        beta: Proportional gain of the Madgwick algorithm

    """

    __slots__ = []

    def __init__(self, name: str, beta: float = 2.0, dtype=np.float64):
        AComputer.__init__(self, name)
        self.defineInput("measurement", shape=(9,), dtype=np.float64)
        self.defineOutput("state", snames=["q0", "q1", "q2", "q3"], dtype=np.float64)
        self.defineOutput("euler", snames=["roll", "pitch", "yaw"], dtype=np.float64)
        self.setInitialStateForOutput(np.array([1, 0, 0, 0]), "state")

        # Parametres du filtre Madgwick
        self.createParameter(name="beta", value=beta)
        self.createParameter(name="mag_softiron_matrix", value=np.eye(3))
        self.createParameter(name="mag_offsets", value=np.zeros(3))

    def setMagnetometerCalibration(
        self, offset: NDArray[Any, Any], softiron_matrix: NDArray[Any, Any]
    ):
        """Sets the magnetometer calibration

        Args:
            offsets: Offsets applied to raw x/y/z values (uTesla)
            softiron_matrix: Soft iron error compensation matrix

        """
        self.mag_offsets = assignVector(
            offset,
            expected_shape=(3,),
            dst_name=self.getName(),
            src_name="offset",
            dtype=np.float64,
        )
        self.mag_softiron_matrix = assignVector(
            softiron_matrix,
            expected_shape=(3, 3),
            dst_name=self.getName(),
            src_name="softiron_matrix",
            dtype=np.float64,
        )

    def getMagnetometerCalibration(self) -> Iterable[np.array]:
        """Returns a copy of the elements of the magnetometer calibration

        Returns:
            A tuple containing:

            * The offsets applied to raw x/y/z values (uTesla)
            * The soft iron error compensation matrix

        """
        return self.mag_offsets.copy(), self.mag_softiron_matrix.copy()

    def update(
        self,
        t1: float,
        t2: float,
        euler: NDArray[Any, Any],
        measurement: NDArray[Any, Any],
        state: NDArray[Any, Any],
    ) -> dict:
        q0, q1, q2, q3 = state
        gx, gy, gz, ax, ay, az, mx, my, mz = measurement
        dt = t2 - t1

        acc = np.array([ax, ay, az])
        if lin.norm(acc) < 1e-6:
            raise TooWeakAcceleration(self.getName(), acc)

        mag = np.array([mx, my, mz])
        if lin.norm(mag) < 1e-6:
            raise TooWeakMagneticField(self.getName(), mag)

        # Apply mag offset compensation (base values in uTesla)
        x = mx - self.mag_offsets[0]
        y = my - self.mag_offsets[1]
        z = mz - self.mag_offsets[2]

        # Apply mag soft iron error compensation
        mx = (
            x * self.mag_softiron_matrix[0, 0]
            + y * self.mag_softiron_matrix[0, 1]
            + z * self.mag_softiron_matrix[0, 2]
        )
        my = (
            x * self.mag_softiron_matrix[1, 0]
            + y * self.mag_softiron_matrix[1, 1]
            + z * self.mag_softiron_matrix[1, 2]
        )
        mz = (
            x * self.mag_softiron_matrix[2, 0]
            + y * self.mag_softiron_matrix[2, 1]
            + z * self.mag_softiron_matrix[2, 2]
        )

        # Rate of change of quaternion from gyroscope
        qDot1 = 0.5 * (-q1 * gx - q2 * gy - q3 * gz)
        qDot2 = 0.5 * (q0 * gx + q2 * gz - q3 * gy)
        qDot3 = 0.5 * (q0 * gy - q1 * gz + q3 * gx)
        qDot4 = 0.5 * (q0 * gz + q1 * gy - q2 * gx)

        # Compute feedback only if accelerometer measurement valid (avoids NaN in accelerometer normalisation)

        # Normalise accelerometer measurement
        recipNorm = 1.0 / np.sqrt(ax * ax + ay * ay + az * az)
        ax *= recipNorm
        ay *= recipNorm
        az *= recipNorm

        # Normalise magnetometer measurement
        recipNorm = 1.0 / np.sqrt(mx * mx + my * my + mz * mz)
        mx *= recipNorm
        my *= recipNorm
        mz *= recipNorm

        # Auxiliary variables to avoid repeated arithmetic
        _2q0mx = 2.0 * q0 * mx
        _2q0my = 2.0 * q0 * my
        _2q0mz = 2.0 * q0 * mz
        _2q1mx = 2.0 * q1 * mx
        _2q0 = 2.0 * q0
        _2q1 = 2.0 * q1
        _2q2 = 2.0 * q2
        _2q3 = 2.0 * q3
        _2q0q2 = 2.0 * q0 * q2
        _2q2q3 = 2.0 * q2 * q3
        q0q0 = q0 * q0
        q0q1 = q0 * q1
        q0q2 = q0 * q2
        q0q3 = q0 * q3
        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q2q2 = q2 * q2
        q2q3 = q2 * q3
        q3q3 = q3 * q3

        # Reference direction of Earth's magnetic field
        hx = (
            mx * q0q0
            - _2q0my * q3
            + _2q0mz * q2
            + mx * q1q1
            + _2q1 * my * q2
            + _2q1 * mz * q3
            - mx * q2q2
            - mx * q3q3
        )
        hy = (
            _2q0mx * q3
            + my * q0q0
            - _2q0mz * q1
            + _2q1mx * q2
            - my * q1q1
            + my * q2q2
            + _2q2 * mz * q3
            - my * q3q3
        )
        _2bx = np.sqrt(hx * hx + hy * hy)
        _2bz = (
            -_2q0mx * q2
            + _2q0my * q1
            + mz * q0q0
            + _2q1mx * q3
            - mz * q1q1
            + _2q2 * my * q3
            - mz * q2q2
            + mz * q3q3
        )
        _4bx = 2.0 * _2bx
        _4bz = 2.0 * _2bz

        # Gradient decent algorithm corrective step
        s0 = (
            -_2q2 * (2.0 * q1q3 - _2q0q2 - ax)
            + _2q1 * (2.0 * q0q1 + _2q2q3 - ay)
            - _2bz * q2 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
            + (-_2bx * q3 + _2bz * q1)
            * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
            + _2bx * q2 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
        )
        s1 = (
            _2q3 * (2.0 * q1q3 - _2q0q2 - ax)
            + _2q0 * (2.0 * q0q1 + _2q2q3 - ay)
            - 4.0 * q1 * (1 - 2.0 * q1q1 - 2.0 * q2q2 - az)
            + _2bz * q3 * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
            + (_2bx * q2 + _2bz * q0)
            * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
            + (_2bx * q3 - _4bz * q1)
            * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
        )
        s2 = (
            -_2q0 * (2.0 * q1q3 - _2q0q2 - ax)
            + _2q3 * (2.0 * q0q1 + _2q2q3 - ay)
            - 4.0 * q2 * (1 - 2.0 * q1q1 - 2.0 * q2q2 - az)
            + (-_4bx * q2 - _2bz * q0)
            * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
            + (_2bx * q1 + _2bz * q3)
            * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
            + (_2bx * q0 - _4bz * q2)
            * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
        )
        s3 = (
            _2q1 * (2.0 * q1q3 - _2q0q2 - ax)
            + _2q2 * (2.0 * q0q1 + _2q2q3 - ay)
            + (-_4bx * q3 + _2bz * q1)
            * (_2bx * (0.5 - q2q2 - q3q3) + _2bz * (q1q3 - q0q2) - mx)
            + (-_2bx * q0 + _2bz * q2)
            * (_2bx * (q1q2 - q0q3) + _2bz * (q0q1 + q2q3) - my)
            + _2bx * q1 * (_2bx * (q0q2 + q1q3) + _2bz * (0.5 - q1q1 - q2q2) - mz)
        )
        recipNorm = 1.0 / np.sqrt(
            s0 * s0 + s1 * s1 + s2 * s2 + s3 * s3
        )  # normalise step magnitude
        s0 *= recipNorm
        s1 *= recipNorm
        s2 *= recipNorm
        s3 *= recipNorm

        # Apply feedback step
        qDot1 -= self.beta * s0
        qDot2 -= self.beta * s1
        qDot3 -= self.beta * s2
        qDot4 -= self.beta * s3

        # Integrate rate of change of quaternion to yield quaternion
        q0 += qDot1 * dt
        q1 += qDot2 * dt
        q2 += qDot3 * dt
        q3 += qDot4 * dt

        # Normalise quaternion
        recipNorm = 1.0 / np.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        q0 *= recipNorm
        q1 *= recipNorm
        q2 *= recipNorm
        q3 *= recipNorm

        outputs = {}
        state = np.array([q0, q1, q2, q3])
        outputs["state"] = state
        outputs["euler"] = np.array(quat_to_euler(*state))

        return outputs


class MahonyFilter(AComputer):
    """Mahony filter

    Estimates roll, pitch, yaw (rad)

    The inputs of the element are **command** and **measurement**

    Attributes:
        Kp: Proportional gain of the Mahony algorithm
        Ki: Integral gain of the Mahony algorithm
        mag_softiron_matrix: Soft iron error compensation matrix
        mag_offsets: Offsets applied to raw x/y/z values (uTesla)

    Args:
        name: Name of the element
        Kp: Proportional gain of the Mahony algorithm
        Ki: Integral gain of the Mahony algorithm

    """

    __slots__ = ["__integralFBx", "__integralFBy", "__integralFBz"]

    def __init__(self, name: str, Kp: float = 0.5, Ki: float = 0.0, dtype=np.float64):
        AComputer.__init__(self, name)
        self.defineInput("measurement", shape=(9,), dtype=np.float64)
        self.defineOutput("state", snames=["q0", "q1", "q2", "q3"], dtype=np.float64)
        self.defineOutput("euler", snames=["roll", "pitch", "yaw"], dtype=np.float64)
        self.setInitialStateForOutput(np.array([1, 0, 0, 0]), "state")

        # Paramtres du filtre Mahony
        self.createParameter(name="Kp", value=Kp)
        self.createParameter(name="Ki", value=Ki)

        self.__integralFBx = 0.0
        self.__integralFBy = 0.0
        self.__integralFBz = 0.0

        self.createParameter(name="mag_softiron_matrix", value=np.eye(3))
        self.createParameter(name="mag_offsets", value=np.zeros(3))

    def setMagnetometerCalibration(
        self, offset: NDArray[Any, Any], softiron_matrix: NDArray[Any, Any]
    ):
        """Sets the magnetometer calibration

        Args:
            offsets: Offsets applied to raw x/y/z values (uTesla)
            softiron_matrix: Soft iron error compensation matrix

        """
        self.mag_offsets = assignVector(
            offset,
            expected_shape=(3,),
            dst_name=self.getName(),
            src_name="offset",
            dtype=np.float64,
        )
        self.mag_softiron_matrix = assignVector(
            softiron_matrix,
            expected_shape=(3, 3),
            dst_name=self.getName(),
            src_name="softiron_matrix",
            dtype=np.float64,
        )

    def getMagnetometerCalibration(self) -> Iterable[np.array]:
        """Returns a copy of the elements of the magnetometer calibration

        Returns:
            A tuple containing:

            * Offsets applied to raw x/y/z values (uTesla)
            * Soft iron error compensation matrix

        """
        return self.mag_offsets.copy(), self.mag_softiron_matrix.copy()

    def update(
        self,
        t1: float,
        t2: float,
        euler: NDArray[Any, Any],
        measurement: NDArray[Any, Any],
        state: NDArray[Any, Any],
    ) -> dict:
        q0, q1, q2, q3 = state
        gx, gy, gz, ax, ay, az, mx, my, mz = measurement
        dt = t2 - t1

        acc = np.array([ax, ay, az])
        if lin.norm(acc) < 1e-6:
            raise TooWeakAcceleration(self.getName(), acc)

        mag = np.array([mx, my, mz])
        if lin.norm(mag) < 1e-6:
            raise TooWeakMagneticField(self.getName(), mag)

        # Apply mag offset compensation (base values in uTesla)
        x = mx - self.mag_offsets[0]
        y = my - self.mag_offsets[1]
        z = mz - self.mag_offsets[2]

        # Apply mag soft iron error compensation
        mx = (
            x * self.mag_softiron_matrix[0, 0]
            + y * self.mag_softiron_matrix[0, 1]
            + z * self.mag_softiron_matrix[0, 2]
        )
        my = (
            x * self.mag_softiron_matrix[1, 0]
            + y * self.mag_softiron_matrix[1, 1]
            + z * self.mag_softiron_matrix[1, 2]
        )
        mz = (
            x * self.mag_softiron_matrix[2, 0]
            + y * self.mag_softiron_matrix[2, 1]
            + z * self.mag_softiron_matrix[2, 2]
        )

        # Normalise accelerometer measurement
        recipNorm = 1.0 / np.sqrt(ax * ax + ay * ay + az * az)
        ax *= recipNorm
        ay *= recipNorm
        az *= recipNorm

        # Normalise magnetometer measurement
        recipNorm = 1.0 / np.sqrt(mx * mx + my * my + mz * mz)
        mx *= recipNorm
        my *= recipNorm
        mz *= recipNorm

        # Auxiliary variables to avoid repeated arithmetic
        q0q0 = q0 * q0
        q0q1 = q0 * q1
        q0q2 = q0 * q2
        q0q3 = q0 * q3
        q1q1 = q1 * q1
        q1q2 = q1 * q2
        q1q3 = q1 * q3
        q2q2 = q2 * q2
        q2q3 = q2 * q3
        q3q3 = q3 * q3

        # Reference direction of Earth's magnetic field
        hx = 2.0 * (mx * (0.5 - q2q2 - q3q3) + my * (q1q2 - q0q3) + mz * (q1q3 + q0q2))
        hy = 2.0 * (mx * (q1q2 + q0q3) + my * (0.5 - q1q1 - q3q3) + mz * (q2q3 - q0q1))
        bx = np.sqrt(hx * hx + hy * hy)
        bz = 2.0 * (mx * (q1q3 - q0q2) + my * (q2q3 + q0q1) + mz * (0.5 - q1q1 - q2q2))

        # Estimated direction of gravity and magnetic field
        halfvx = q1q3 - q0q2
        halfvy = q0q1 + q2q3
        halfvz = q0q0 - 0.5 + q3q3
        halfwx = bx * (0.5 - q2q2 - q3q3) + bz * (q1q3 - q0q2)
        halfwy = bx * (q1q2 - q0q3) + bz * (q0q1 + q2q3)
        halfwz = bx * (q0q2 + q1q3) + bz * (0.5 - q1q1 - q2q2)

        # Error is sum of cross product between estimated direction
        # and measured direction of field vectors
        halfex = (ay * halfvz - az * halfvy) + (my * halfwz - mz * halfwy)
        halfey = (az * halfvx - ax * halfvz) + (mz * halfwx - mx * halfwz)
        halfez = (ax * halfvy - ay * halfvx) + (mx * halfwy - my * halfwx)

        # Compute and apply integral feedback if enabled
        if self.Ki > 0.0:
            # integral error scaled by Ki
            self.__integralFBx += 2 * self.Ki * halfex * dt
            self.__integralFBy += 2 * self.Ki * halfey * dt
            self.__integralFBz += 2 * self.Ki * halfez * dt
            gx += self.__integralFBx  # apply integral feedback
            gy += self.__integralFBy
            gz += self.__integralFBz
        else:
            self.__integralFBx = 0.0  # prevent integral windup
            self.__integralFBy = 0.0
            self.__integralFBz = 0.0

        # Apply proportional feedback
        gx += 2 * self.Kp * halfex
        gy += 2 * self.Kp * halfey
        gz += 2 * self.Kp * halfez

        # Integrate rate of change of quaternion
        gx *= 0.5 * dt  # pre-multiply common factors
        gy *= 0.5 * dt
        gz *= 0.5 * dt
        qa = q0
        qb = q1
        qc = q2
        q0 += -qb * gx - qc * gy - q3 * gz
        q1 += qa * gx + qc * gz - q3 * gy
        q2 += qa * gy - qb * gz + q3 * gx
        q3 += qa * gz + qb * gy - qc * gx

        # Normalise quaternion
        recipNorm = 1.0 / np.sqrt(q0 * q0 + q1 * q1 + q2 * q2 + q3 * q3)
        q0 *= recipNorm
        q1 *= recipNorm
        q2 *= recipNorm
        q3 *= recipNorm

        outputs = {}
        state = np.array([q0, q1, q2, q3])
        outputs["state"] = state
        outputs["euler"] = np.array(quat_to_euler(*state))

        return outputs
