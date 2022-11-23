"""A set of useful functions

"""
from datetime import datetime
import os
import sys
import glob
import re
from typing import Iterable, Tuple, Any, List
import importlib
from dataclasses import dataclass

from scipy import linalg as lin
from scipy.interpolate import interp1d, interp2d
from scipy.optimize import root_scalar, minimize
from nptyping import NDArray
import numpy as np
from numpy import pi, arcsin, arccos, arctan, arctan2, tan, sin, cos, sqrt, exp
from skyfield.api import load, utc
from skyfield.timelib import Time
from skyfield.sgp4lib import theta_GMST1982
from scipy.special import ellipe, ellipkinc, ellipeinc

from . import logger
from .constants import *
from .exceptions import *
from .constants import Req


__all__ = [
    "Peak",
    "find1dpeak",
    "find2dpeak",
    "casedpath",
    "resource_path",
    "calc_cho",
    "deg",
    "rad",
    "assignVector",
    "quat_to_matrix",
    "matrix_to_quat",
    "matrix_to_euler",
    "euler_to_matrix",
    "quat_to_euler",
    "euler_to_quat",
    "vecBodyToEarth",
    "vecEarthToBody",
    "q_function",
    "anomaly_mean_to_ecc",
    "anomaly_ecc_to_mean",
    "anomaly_ecc_to_true",
    "anomaly_true_to_ecc",
    "anomaly_mean_to_true",
    "anomaly_true_to_mean",
    "build_local_matrix",
    "geodetic_to_itrf",
    "orbital_to_teme",
    "teme_to_orbital",
    "itrf_to_geodetic",
    "time_to_jd_fraction",
    "rotation_matrix",
    "teme_to_itrf",
    "itrf_to_teme",
    "itrf_to_azeld",
    "azeld_to_itrf",
    "azelalt_to_itrf",
    "itrf_to_llavpa",
    "llavpa_to_itrf",
    "datetime_to_skyfield",
    "skyfield_to_datetime",
    "pdot",
    "cexp",
    "load_antenna_config",
    "mean_variance_2dgauss_norm",
    "mean_variance_3dgauss_norm",
]


@dataclass(init=True)
class Peak:
    """Represents a peak in a plot or a 3D plot

    Examples:
        >>> p = Peak(coord_label=("x",), coord_unit=("m",),coord=(1,), value=3.5)
        >>> print(p)
        Peak(x=1 m, value=3.5)

    """

    coord_label: tuple
    coord_unit: tuple
    coord: tuple
    value: float

    def __repr__(self):
        from .graphics import format_parameter

        res = "%s(" % self.__class__.__name__
        for lbl, c, unt in zip(self.coord_label, self.coord, self.coord_unit):
            txt = format_parameter(c, unt)
            res += "%s=%s, " % (lbl, txt)
        res += "value=%.3g)" % self.value

        return res


def casedpath(path):
    r = glob.glob(re.sub(r"([^:/\\])(?=[/\\]|$)", r"[\1]", path))
    return r and r[0] or path


def resource_path(resource: str, package: str = "blocksim") -> str:
    """Returns the full path to a resource of the package specified in argument

    Args:
        resource: Name of the resource file
        package: Name of the package where the resource is

    Returns:
        The path to the resource

    Examples:
        >>> pth = resource_path('dummy.txt')

    """
    from importlib import import_module

    module = import_module(package)
    spec = module.__spec__
    if spec.submodule_search_locations is None:
        raise TypeError("{!r} is not a package".format(package))

    for root in spec.submodule_search_locations:
        path = os.path.join(root, "resources", resource)
        if os.path.exists(path):
            path = casedpath(path)
            path = path.replace("\\", "/")
            if path[1] == ":":
                path = "/%s/%s" % (path[0].lower(), path[3:])
            return path

    raise FileExistsError(resource)


def find1dpeak(
    nb_peaks: int,
    xd: NDArray[Any, Any],
    yd: NDArray[Any, Any],
    name_of_x_var: str,
    unit_of_x_var: str,
) -> List[Peak]:
    """Finds the peaks in a serie.

    Args:
        nb_peaks: Max number of peaks to seach. Only the highest are kept
        xd: Array of X samples
        yd: Array of Y samples. Cannot be complex samples
        name_of_x_var: Name of the X coordinate
        unit_of_x_var: Unit of the X coordinate

    Returns:
        The list of detected peaks, sorted by descreasing value of the peak

    """
    if nb_peaks == 0:
        return []

    n = len(yd)
    lpeak = []
    _x_itp = interp1d(
        x=np.arange(n),
        y=xd,
        kind="linear",
        copy=False,
        bounds_error=True,
        assume_sorted=True,
    )

    b = (yd[2:] - yd[:-2]) / 2
    a = (yd[2:] + yd[:-2] - 2 * yd[1:-1]) / 2

    ip = np.where(a < 0)[0]
    ip = np.intersect1d(ip, np.where(np.abs(b) < -a)[0])

    dp = -b[ip] / (2 * a[ip])
    dval = -b[ip] ** 2 / (4 * a[ip])

    for kp in range(len(ip)):
        p0 = 1 + ip[kp]
        x0 = _x_itp(p0 + dp[kp])
        p = Peak(
            coord_label=(name_of_x_var,),
            coord_unit=(unit_of_x_var,),
            coord=(x0,),
            value=yd[p0] + dval[kp],
        )
        lpeak.append(p)

    lpeak.sort(key=lambda x: x.value, reverse=True)

    if len(lpeak) > nb_peaks:
        lpeak = lpeak[:nb_peaks]

    return lpeak


def find2dpeak(
    nb_peaks: int,
    xd: NDArray[Any, Any],
    yd: NDArray[Any, Any],
    zd: NDArray[Any, Any],
    name_of_x_var: str,
    unit_of_x_var: str,
    name_of_y_var: str,
    unit_of_y_var: str,
) -> List[Peak]:
    """Finds the peaks

    Args:
        nb_peaks: Max number of peaks to seach. Only the highest are kept
        xd: Array of X samples
        yd: Array of Y samples
        zd: Matrix of Z samples
        name_of_x_var: Name of the X coordinate
        unit_of_x_var: Unit of the X coordinate
        name_of_y_var: Name of the Y coordinate
        unit_of_y_var: Unit of the Y coordinate

    Returns:
        The list of detected peaks, sorted by descreasing value of the peak

    """
    if nb_peaks == 0:
        return []

    ep = 2
    eq = 2
    Np, Nq = zd.shape
    _x_itp = interp1d(
        x=np.arange(Nq),
        y=xd,
        kind="linear",
        copy=False,
        bounds_error=True,
        assume_sorted=True,
    )
    _y_itp = interp1d(
        x=np.arange(Np),
        y=yd,
        kind="linear",
        copy=False,
        bounds_error=True,
        assume_sorted=True,
    )
    iDtr = 1 / (2 * ep**2 * eq**2)
    iA = iDtr * np.array(
        [
            [0, -ep * eq**2, 0, 0, ep * eq**2],
            [0, 0, -(ep**2) * eq, ep**2 * eq, 0],
            [-2 * eq**2, eq**2, 0, 0, eq**2],
            [-2 * ep**2, 0, ep**2, ep**2, 0],
        ]
    )

    if nb_peaks == 1:
        ir, ic = zd.shape
        ipeak = np.argmax(zd)

        q0 = ipeak % ic
        p0 = (ipeak - q0) // ic
        p_lookup = [p0]
        q_lookup = [q0]
    else:
        p_lookup = range(ep, Np - ep)
        q_lookup = range(eq, Nq - eq)

    lpeak = []
    for p0 in p_lookup:
        for q0 in q_lookup:
            Z00 = zd[p0, q0]
            B = np.array(
                [
                    Z00,
                    zd[p0 - ep, q0],
                    zd[p0, q0 - eq],
                    zd[p0, q0 + eq],
                    zd[p0 + ep, q0],
                ]
            )
            if np.any(B[1:] >= Z00):
                continue

            X0 = iA @ B
            b, c, d, f = X0

            dp = -b / (2 * d)
            dq = -c / (2 * f)
            dval = (-(b**2) * f - c**2 * d) / (4 * d * f)

            if -ep / 2 <= dp and dp < ep / 2 and -eq / 2 <= dq and dq < eq / 2:
                p = Peak(
                    coord_label=(name_of_x_var, name_of_y_var),
                    coord_unit=(unit_of_x_var, unit_of_y_var),
                    coord=(_x_itp(q0 + dq), _y_itp(p0 + dp)),
                    value=Z00 + dval,
                )
                lpeak.append(p)
            else:
                raise AssertionError(dp, dq)

    lpeak.sort(key=lambda x: x.value, reverse=True)

    if len(lpeak) > nb_peaks:
        lpeak = lpeak[:nb_peaks]

    return lpeak


def verif_mat_diag(A: NDArray[Any, Any]) -> bool:
    """Tests if a square matrix is diagonal

    Args:
        A: Matrix to test

    Returns:
        The result of the test

    """
    m, n = A.shape
    assert m == n

    for i in range(m):
        for j in range(n):
            if i != j and A[i, j] != 0:
                return False
    return True


def calc_cho(A: NDArray[Any, Any]) -> NDArray[Any, Any]:
    """Returns the cholesky decomposition C of a symetric matrix A:

    $$ A = C.C^T $$

    Args:
        A: Matrix

    Returns:
        Cholesky decomposition C

    """
    m, n = A.shape
    assert m == n

    if verif_mat_diag(A):
        res = np.zeros(A.shape)
        for i in range(n):
            res[i, i] = np.sqrt(A[i, i])
    else:
        c, _ = lin.cho_factor(A)
        res = np.triu(c).T

    return res


def deg(x: float) -> float:
    """Converts from radians to degrees

    Args:
        x: Angle in radians

    Returns:
        Angle in degrees

    """
    return x * 180 / np.pi


def rad(x: float) -> float:
    """Converts from degrees to radians

    Args:
        x: Angle in degrees

    Returns:
        Angle in radians

    """
    return x * np.pi / 180


def assignVector(
    v: NDArray[Any, Any], expected_shape: tuple, dst_name: str, src_name: str, dtype
) -> NDArray[Any, Any]:
    """

    Args:
        v: NDArray[Any,Any] to assign
        expected_shape: Expected shape for v
        dst_name: Name of the element where the assignement will take place. (To allow meaningfull error messages)
        src_name: Name of the source vector. (To allow meaningfull error messages)
        dtype: Type of the assigned vector

    Returns:
        Copy of the vector v if no problem encountered

    Raises:
        ValueError: If the vector is not a np.array or not with the correct shape

    Examples:
        >>> v = np.arange(5)
        >>> assignVector(v, (5,), 'elem', 'v', np.float64)
        array([0., 1., 2., 3., 4.])

    """
    # if np.any(np.isnan(v)):
    #     txt = "Element '%s' : Argument '%s'=%s has NaN" % (
    #         dst_name,
    #         src_name,
    #         v,
    #     )
    #     logger.warning(txt)

    if expected_shape is None:
        expected_shape = len(v)

    if isinstance(expected_shape, int):
        expected_shape = (expected_shape,)

    if not hasattr(v, "__iter__") and expected_shape[0] == 1:
        v = np.array([v])

    if isinstance(v.shape, int):
        vshape = (v.shape,)
    else:
        vshape = v.shape

    if type(v) != type(np.empty(1)):
        txt = "Element '%s' : Argument '%s'=%s is not a vector" % (
            dst_name,
            src_name,
            v,
        )
        logger.error(txt)
        raise InvalidAssignedVector(txt)

    elif vshape != expected_shape:
        txt = "Element '%s' : Array '%s'=%s has shape %s; expected %s" % (
            dst_name,
            src_name,
            v,
            vshape,
            expected_shape,
        )
        logger.error(txt)
        raise InvalidAssignedVector(txt)

    elif not (dtype == np.complex64 or dtype == np.complex128) and (
        v.dtype == np.complex64 or v.dtype == np.complex128
    ):
        txt = (
            "Element '%s' : Argument '%s' - trying to affect a complex vector into a real or integer vector"
            % (
                dst_name,
                src_name,
            )
        )
        raise WrongDataType(txt)

    else:
        try:
            res = np.array(v.copy(), dtype=dtype)
        except Exception as e:
            txt = "Element '%s' : Argument '%s' - impossible to instantiate array:\n%s" % (
                dst_name,
                src_name,
                str(e),
            )
            raise WrongDataType(txt)

    return res


def quat_to_matrix(qr: float, qi: float, qj: float, qk: float) -> NDArray[Any, Any]:
    """
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix

    Args:
        qr: Real part of the quaternion
        qi: First imaginary of the quaternion
        qj: Second imaginary of the quaternion
        qk: Third imaginary of the quaternion

    Returns:
        Rotation matrix

    Examples:
        >>> R = quat_to_matrix(1.,0.,0.,0.)
        >>> lin.norm(R - np.eye(3)) # doctest: +ELLIPSIS
        0.0...

    """
    res = np.empty((3, 3))

    res[0, 0] = qr**2 + qi**2 - qj**2 - qk**2
    res[1, 1] = qr**2 - qi**2 + qj**2 - qk**2
    res[2, 2] = qr**2 - qi**2 - qj**2 + qk**2
    res[0, 1] = 2 * (qi * qj - qk * qr)
    res[0, 2] = 2 * (qi * qk + qj * qr)
    res[1, 0] = 2 * (qi * qj + qk * qr)
    res[1, 2] = 2 * (qj * qk - qi * qr)
    res[2, 0] = 2 * (qi * qk - qj * qr)
    res[2, 1] = 2 * (qj * qk + qi * qr)

    return res


def matrix_to_quat(R: NDArray[Any, Any]) -> Iterable[float]:
    """

    Examples:
        >>> q = np.array([2, -3, 4, -5])
        >>> q = q/lin.norm(q)
        >>> qr,qx,qy,qz = q
        >>> R = quat_to_matrix(qr,qx,qy,qz)
        >>> q2 = matrix_to_quat(R)
        >>> lin.norm(q-q2) # doctest: +ELLIPSIS
        0.0...

    """
    K = np.empty((4, 4))
    K[0, 0] = R[0, 0] - R[1, 1] - R[2, 2]
    K[1, 0] = K[0, 1] = R[1, 0] + R[0, 1]
    K[2, 0] = K[0, 2] = R[2, 0] + R[0, 2]
    K[3, 0] = K[0, 3] = R[1, 2] - R[2, 1]

    K[1, 1] = R[1, 1] - R[0, 0] - R[2, 2]
    K[1, 2] = K[2, 1] = R[2, 1] + R[1, 2]
    K[1, 3] = K[3, 1] = R[2, 0] - R[0, 2]

    K[2, 2] = R[2, 2] - R[0, 0] - R[1, 1]
    K[2, 3] = K[3, 2] = R[0, 1] - R[1, 0]

    K[3, 3] = R[0, 0] + R[1, 1] + R[2, 2]

    w, v = lin.eig(K / 3)
    # w,v = lin.eigh(K/3)
    i = np.argmax(w)
    qi, qj, qk, qr = v[:, i]

    return np.array([-qr, qi, qj, qk])


def matrix_to_euler(R: NDArray[Any, Any]) -> Iterable[float]:
    """

    https://www.learnopencv.com/rotation-matrix-to-euler-angles/

    Examples:
        >>> q = np.array([2, -3, 4, -5])
        >>> q = q/lin.norm(q)
        >>> qr,qx,qy,qz = q
        >>> R = quat_to_matrix(qr,qx,qy,qz)
        >>> qr,qi,qj,qk = matrix_to_quat(R)
        >>> r0,p0,y0 = quat_to_euler(qr,qi,qj,qk)
        >>> r,p,y = matrix_to_euler(R)
        >>> np.abs(r-r0) < 1e-10
        True
        >>> np.abs(p-p0) < 1e-10
        True
        >>> np.abs(y-y0) < 1e-10
        True

    """
    sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        r = np.arctan2(R[2, 1], R[2, 2])
        p = np.arctan2(-R[2, 0], sy)
        y = np.arctan2(R[1, 0], R[0, 0])
    else:
        r = np.arctan2(-R[1, 2], R[1, 1])
        p = np.arctan2(-R[2, 0], sy)
        y = 0

    return r, p, y


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> NDArray[Any, Any]:
    """

    https://www.learnopencv.com/rotation-matrix-to-euler-angles/

    Examples:
        >>> roll = 1; pitch = -2; yaw = 3
        >>> qr,qi,qj,qk = euler_to_quat(roll, pitch, yaw)
        >>> R0 = quat_to_matrix(qr,qi,qj,qk)
        >>> R = euler_to_matrix(roll, pitch, yaw)
        >>> lin.norm(R-R0) < 1e-10
        True

    """
    Ry = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]])
    Rp = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rr = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]])
    R = Ry @ Rp @ Rr
    return R


def quat_to_euler(
    qr: float, qi: float, qj: float, qk: float, normalize: bool = False
) -> Iterable[float]:
    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Args:
        qr: Real part of the quaternion
        qi:  First imaginary of the quaternion
        qj: Second imaginary of the quaternion
        qk: Third imaginary of the quaternion
        normalize: Pass True to force normalization

    Returns:
        A tuple containing:

        * Roll angle (rad)
        * Pitch angle (rad)
        * Yaw angle (rad)

    Examples:
        >>> quat_to_euler(1.,0.,0.,0.)
        (0.0, -0.0, 0.0)

    """
    if normalize:
        q = np.array([qr, qi, qj, qk])
        qr, qi, qj, qk = q / lin.norm(q)

    roll = np.arctan2(qr * qi + qj * qk, 0.5 - qi * qi - qj * qj)
    t2 = -2.0 * (qi * qk - qr * qj)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    pitch = np.arcsin(t2)
    yaw = np.arctan2(qi * qj + qr * qk, 0.5 - qj * qj - qk * qk)

    return roll, pitch, yaw


def euler_to_quat(roll: float, pitch: float, yaw: float) -> Iterable[float]:
    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Args:
        roll: Roll angle (rad)
        pitch: Pitch angle (rad)
        yaw: Yaw angle (rad)

    Returns:
        A tuple representing the quaternion:

        * Real part of the quaternion
        * First imaginary of the quaternion
        * Second imaginary of the quaternion
        * Third imaginary of the quaternion

    Examples:
        >>> qr,qi,qj,qk = euler_to_quat(10.*np.pi/180., 20.*np.pi/180., 30.*np.pi/180.)
        >>> r,p,y = quat_to_euler(qr,qi,qj,qk)
        >>> r*180/np.pi # doctest: +ELLIPSIS
        10.0...
        >>> p*180/np.pi # doctest: +ELLIPSIS
        20.0...
        >>> y*180/np.pi # doctest: +ELLIPSIS
        29.999...

    """
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)

    qr = cy * cr * cp + sy * sr * sp
    qi = cy * sr * cp - sy * cr * sp
    qj = cy * cr * sp + sy * sr * cp
    qk = sy * cr * cp - cy * sr * sp

    return qr, qi, qj, qk


def vecBodyToEarth(attitude: NDArray[Any, Any], x: NDArray[Any, Any]) -> NDArray[Any, Any]:
    """Expresses a vector from the body frame to the Earth's frame

    Args:
        attitude: If 3 elements array, roll, pitch, yaw (rad)
          If 4 elements array, qw, qx, qy, qz
        x: Vector expressed in the body frame

    Returns:
        Vector x expressed in Earth's frame

    """
    if len(attitude) == 3:
        R = euler_to_matrix(*attitude)
    elif len(attitude) == 4:
        R = quat_to_matrix(*attitude)
    else:
        raise ValueError

    return R @ x


def vecEarthToBody(attitude: NDArray[Any, Any], x: NDArray[Any, Any]) -> NDArray[Any, Any]:
    """Expresses a vector from Earth's frame to the body's frame

    Args:
        attitude: If 3 elements array, roll, pitch, yaw (rad)
                  If 4 elements array, qw, qx, qy, qz
        x: Vector expressed in Earth's frame

    Returns:
        Vector x expressed in the body frame

    """
    if len(attitude) == 3:
        R = euler_to_matrix(*attitude)
    elif len(attitude) == 4:
        R = quat_to_matrix(*attitude)
    else:
        raise ValueError

    return R.T @ x


def anomaly_mean_to_ecc(ecc: float, M: float) -> float:
    def _fun(E, ecc, M):
        return E - ecc * sin(E) - M

    def _dfun(E, ecc, M):
        return 1 - ecc * cos(E)

    res = root_scalar(f=_fun, args=(ecc, M), bracket=(-pi, pi), fprime=_dfun, x0=M)
    if not res.converged:
        raise AssertionError("%s" % res)

    return res.root


def anomaly_ecc_to_mean(ecc: float, E: float) -> float:
    return E - ecc * sin(E)


def anomaly_ecc_to_true(ecc: float, E: float) -> float:
    tv2 = sqrt((1 + ecc) / (1 - ecc)) * tan(E / 2)
    return 2 * arctan(tv2)


def anomaly_true_to_ecc(ecc: float, v: float) -> float:
    tE2 = sqrt((1 - ecc) / (1 + ecc)) * tan(v / 2)
    return 2 * arctan(tE2)


def anomaly_mean_to_true(ecc: float, M: float) -> float:
    E = anomaly_mean_to_ecc(ecc, M)
    v = anomaly_ecc_to_true(ecc, E)
    return v


def anomaly_true_to_mean(ecc: float, v: float) -> float:
    E = anomaly_true_to_ecc(ecc, v)
    M = anomaly_ecc_to_mean(ecc, E)
    return M


def build_local_matrix(pos: NDArray[Any, Any], xvec: NDArray[Any, Any] = None) -> NDArray[Any, Any]:
    """Builds a ENV frame at a given position

    Args:
        pos: Position (m) of a point in ITRF
        xvec: If provided, the first column will be parallel to xvec. Else, the first column will be pointing to the East

    Returns:
        Matrix whose columns are:

        * Local East vector, or xvec if not None
        * Local North vector
        * Local Vertical vector, orthogonal to xvec

    """
    # Local ENV for the observer
    vert = pos.copy()
    vert /= lin.norm(vert)

    if xvec is None:
        east = np.cross(np.array([0, 0, 1]), pos)
        east /= lin.norm(east)
    else:
        east = xvec / lin.norm(xvec)

    north = np.cross(vert, east)

    env = np.empty((3, 3))
    env[:, 0] = east
    env[:, 1] = north
    env[:, 2] = vert

    return env


def geodetic_to_itrf(lon: float, lat: float, h: float) -> NDArray[Any, Any]:
    """
    Compute the Geocentric (Cartesian) Coordinates X, Y, Z
    given the Geodetic Coordinates lat, lon + Ellipsoid Height h

    Args:
        lon: Longitude (rad)
        lat: Latitude (rad)
        h: Altitude (m)

    Returns:
        A array of x, y, z coordinates (m)

    Examples:
        >>> x,y,z = geodetic_to_itrf(0,0,0)

    """
    N = Req / sqrt(1 - (1 - (1 - 1 / rf) ** 2) * (sin(lat)) ** 2)
    X = (N + h) * cos(lat) * cos(lon)
    Y = (N + h) * cos(lat) * sin(lon)
    Z = ((1 - 1 / rf) ** 2 * N + h) * sin(lat)

    return np.array([X, Y, Z])


def __Iter_phi_h(x: float, y: float, z: float, eps: float = 1e-6) -> Tuple[float, float]:
    r = lin.norm((x, y, z))
    p = sqrt(x**2 + y**2)

    N = Req
    hg = r - sqrt(Req - Rpo)
    e = sqrt(1 - Rpo**2 / Req**2)
    phig = arctan(z * (N + hg) / (p * (N * (1 - e**2) + hg)))

    cont = True
    niter = 0
    while cont:
        hgp = hg
        phigp = phig

        N = Req / sqrt(1 - e**2 * sin(phigp) ** 2)
        hg = p / cos(phigp) - N
        phig = arctan(z * (N + hg) / (p * (N * (1 - e**2) + hg)))

        if eps > max(abs(phigp - phig), abs(hgp - hg)):
            cont = False

        if niter > 50:
            raise ValueError("Too many iterations in __Iter_phi_h")

        niter += 1

    return phig, hgp


def time_to_jd_fraction(t_epoch: float) -> Tuple[float, float]:
    """

    Args:
        t_epoch: Time since 31/12/1949 00:00 UT (s)

    """
    epoch = t_epoch / 86400

    whole, fraction = divmod(epoch, 1.0)
    whole_jd = whole + 2433281.5

    jd = whole_jd
    fraction = fraction

    return jd, fraction


def rotation_matrix(angle: float, axis: NDArray[Any, Any]):
    """Builds the 3D rotation matrix from axis and angle

    Args:
        angle: Rotation angle (rad)
        axis: Rotation axis

    Returns:
        The rotation matrix

    """
    kx, ky, kz = axis / lin.norm(axis)
    K = np.array([[0, -kz, ky], [kz, 0, -kx], [-ky, kx, 0]])
    R = np.eye(3) + sin(angle) * K + (1 - cos(angle)) * K @ K
    return R


def teme_to_itrf(t_epoch: float, pv_teme: NDArray[Any, Any]) -> NDArray[Any, Any]:
    jd, fraction = time_to_jd_fraction(t_epoch)

    theta, theta_dot = theta_GMST1982(jd, fraction)
    uz = np.array((0.0, 0.0, 1.0))
    angular_velocity = -theta_dot * uz / 86400.0

    R = rotation_matrix(-theta, uz)

    rTEME = pv_teme[:3]
    vTEME = pv_teme[3:]

    rITRF = R @ rTEME
    vITRF = R @ vTEME + np.cross(angular_velocity, rITRF)

    pv = np.empty(6)
    pv[:3] = rITRF
    pv[3:] = vITRF

    return pv


def itrf_to_teme(t_epoch: float, pv_itrf: NDArray[Any, Any]) -> NDArray[Any, Any]:
    jd, fraction = time_to_jd_fraction(t_epoch)

    theta, theta_dot = theta_GMST1982(jd, fraction)
    uz = np.array((0.0, 0.0, 1.0))
    angular_velocity = -theta_dot * uz / 86400.0

    R = rotation_matrix(theta, uz)

    rITRF = pv_itrf[:3]
    vITRF = pv_itrf[3:]

    rTEME = R @ rITRF
    vTEME = R @ vITRF - R @ np.cross(angular_velocity, rITRF)

    return np.hstack((rTEME, vTEME))


def teme_to_orbital(pv: NDArray[Any, Any]):
    pos = np.array(pv[:3])
    x, y, z = pos
    vel = np.array(pv[3:])
    r = lin.norm(pos)
    v2 = vel @ vel
    W = v2 / 2 - mu / r
    a = -mu / (2 * W)
    v = sqrt(v2)
    h = np.cross(pos, vel)
    hx, hy, hz = h
    h2 = h @ h
    nh = sqrt(h2)
    p = h2 / mu
    asqr = 1 - p / a
    inc = arccos(hz / nh)
    if asqr <= 0:
        e = 0
        tano = 0
    else:
        e = sqrt(asqr)
        tano = arccos(1 / e * (p / r - 1))
    if vel @ pos < 0:
        tano = -tano
    node = arctan2(hx, -hy)
    argp = arctan2((y * cos(node) - x * sin(node)) / cos(inc), x * cos(node) + y * sin(node)) - tano
    mano = anomaly_true_to_mean(e, tano)
    return a, e, argp, inc, mano, node


def orbital_to_teme(
    a: float,
    ecc: float,
    argp: float,
    inc: float,
    mano: float,
    node: float,
) -> NDArray[Any, Any]:
    """

    Args:
        a: Semi-major axis (m)
        ecc: Eccentricity
        argp: Argument of periapsis (rad)
        inc: Inclination (rad)
        mano: Mean anomaly (rad)
        node: Longitude of the ascending node (rad)

    Returns:
        An array with position (m) and velocity (m/s) in TEME frame

    Examples:
        >>> pv = orbital_to_teme(7e6, 0.01, 0, 1, 1, 0)

    """
    # https://en.wikipedia.org/wiki/True_anomaly#From_the_mean_anomaly
    p = a * (1 - ecc**2)
    tano = anomaly_mean_to_true(ecc, mano)
    r = p / (1 + ecc * cos(tano))
    n = sqrt(mu / a**3)

    x = r * (cos(node) * cos(argp + tano) - sin(node) * cos(inc) * sin(argp + tano))
    y = r * (sin(node) * cos(argp + tano) + cos(node) * cos(inc) * sin(argp + tano))
    z = r * sin(inc) * sin(argp + tano)

    E = anomaly_true_to_ecc(ecc, tano)
    # M = E - e.sin(E)
    dE = n / (1 - cos(E) * ecc)
    rr = sqrt((1 + ecc) / (1 - ecc))
    dtano = dE * rr * (cos(tano / 2) / cos(E / 2)) ** 2
    vr = ecc * p * sin(tano) * dtano / (ecc * cos(tano) + 1) ** 2

    vx = r * (
        -cos(node) * dtano * sin(tano + argp) - cos(inc) * sin(node) * dtano * cos(tano + argp)
    ) + vr * (cos(node) * cos(tano + argp) - cos(inc) * sin(node) * sin(tano + argp))
    vy = r * (
        cos(inc) * cos(node) * dtano * cos(tano + argp) - sin(node) * dtano * sin(tano + argp)
    ) + vr * (cos(inc) * cos(node) * sin(tano + argp) + sin(node) * cos(tano + argp))
    vz = sin(inc) * vr * sin(tano + argp) + sin(inc) * r * dtano * cos(tano + argp)

    return np.array([x, y, z, vx, vy, vz])


def itrf_to_geodetic(position: NDArray[Any, Any]) -> Tuple[float, float, float]:
    """Converts the ITRF coordinates into latitude, longitude, altitude (WGS84)

    Args:
        position: x, y, z position in ITRF frame (m)

    Returns:
        A tuple containing:

        * Longitude (rad)
        * Latitude (rad)
        * Altitude (m)

    Examples:
        >>> pos = geodetic_to_itrf(2,1,3)
        >>> lon,lat,alt = itrf_to_geodetic(pos)
        >>> lon # doctest: +ELLIPSIS
        2.0...
        >>> lat # doctest: +ELLIPSIS
        1.0...
        >>> alt # doctest: +ELLIPSIS
        3.0...

    """
    x = position[0]
    y = position[1]
    z = position[2]
    p = sqrt(x**2 + y**2)
    cl = x / p  # cos(lambda)
    sl = y / p  # sin(lambda)
    lon = arctan2(sl, cl)
    lat, alt = __Iter_phi_h(x, y, z)

    return lon, lat, alt


def azeld_to_itrf(azeld: NDArray[Any, Any], obs: NDArray[Any, Any]) -> NDArray[Any, Any]:
    """Converts an ITRF position & velocity into
    azimut, elevation, distance, radial velocity, slope of velocity, azimut of velocity

    Args:
        azeld: Array with:

            * Azimut (rad)
            * Elevation (rad)
            * Distance (m)
            * Radial velocity (m/s)
            * Slope of velocity (rad)
            * Azimut of velocity (rad)
        obs: Position (m) & velocity (m/s) of terrestrial observer in ITRF

    Returns:
        Position (m) & velocity (m/s) of the observed satellite in ITRF

    """
    # Local ENV for the observer
    obs_env = build_local_matrix(obs[:3])

    az, el, dist, azr, elr, vr = azeld
    x = dist * cos(el) * sin(az)
    y = dist * cos(el) * cos(az)
    z = dist * sin(el)
    deltap = obs_env @ np.array([x, y, z])
    sat = obs[:3] + deltap

    # Local ENV for the satellite
    rxy2 = x**2 + y**2
    rxy = sqrt(rxy2)
    vz = rxy * elr + vr * z / dist
    vx = -(vz * x * z - azr * rxy2 * y - dist * vr * x) / rxy2
    vy = -(vz * y * z - dist * vr * y + azr * rxy2 * x) / rxy2
    deltav = obs_env @ np.array([vx, vy, vz])
    vsat = obs[3:] + deltav

    return np.hstack((sat, vsat))


def azelalt_to_itrf(azelalt: NDArray[Any, Any], sat: NDArray[Any, Any]) -> NDArray[Any, Any]:
    """Converts to an ITRF position & velocity from azimut, elevation
    so that the observer is at the specified altitude

    Args:
        azelalt: Array with:

            * Azimut (rad)
            * Elevation (rad)
            * Altitude (m)
        sat: Position (m) & velocity (m/s) of the observed satellite in ITRF

    Returns:
        Position (m) of the observed satellite in ITRF

    """
    from blocksim.utils import deg
    from cartopy.geodesic import Geodesic

    class _tmp:
        success = False

    def _fun(C, az, el, alt):
        lon, lat = C + c0
        X = geodetic_to_itrf(lon, lat, alt)
        obs = np.hstack((X, np.zeros(3)))
        azj, elj, _, _, _, _ = itrf_to_azeld(obs, sat)
        J = (azj - az) ** 2 + (elj - el) ** 2
        return J

    az, el, alt = azelalt

    # Quick'n dirty search for an initial guess:
    # 1. Get the satellite subpoint
    lon, lat, hs = itrf_to_geodetic(sat)

    # 2. Find the radius of the ground circle such that
    # an observer on this circle sees the satellite at elevation el
    r = Req + hs
    d_lim = sqrt(r**2 - Req**2 * cos(el) ** 2) - Req * sin(el)
    alpha_lim = np.arccos((Req**2 + r**2 - d_lim**2) / (2 * r * Req))
    rad = alpha_lim * Req

    # 3. Compute the ground circle
    g = Geodesic()
    val = g.circle(lon * 180 / pi, lat * 180 / pi, radius=rad, n_samples=360)

    # 4. Find on the ground circle the position that corresponds to the azimut
    if az >= pi:
        az -= 2 * pi * int((az + pi) / (2 * pi))
    if az < -pi:
        az += 2 * pi * int(-(az + pi) / (2 * pi))
    iaz = int((az + pi) * 180 / pi)
    c_lon = val[iaz, 0] * pi / 180
    c_lat = val[iaz, 1] * pi / 180

    # Check
    # Xc = geodetic_to_itrf(c_lon, c_lat, alt)
    # azeld = itrf_to_azeld(obs=np.hstack((Xc, np.zeros(3))), sat=sat)

    c0 = np.array([c_lon, c_lat])
    sol = _tmp()

    J0 = _fun(np.zeros(2), az, el, alt)
    J = J0 + 1

    c1 = np.zeros(2)
    # Nelder-Mead, L-BFGS-B, TNC, SLSQP, Powell, and trust-constr
    while J > J0:
        sol = minimize(
            method="SLSQP",
            fun=_fun,
            x0=c1,
            bounds=[(0, 2 * pi), (-pi / 2, pi / 2)],
            args=(az, el, alt),
        )
        c1 = sol.x / 2
        J = _fun(sol.x, az, el, alt)

    lon, lat = sol.x + c0
    X = geodetic_to_itrf(lon, lat, alt)

    return X


def itrf_to_azeld(
    obs: NDArray[Any, Any], sat: NDArray[Any, Any]
) -> Tuple[float, float, float, float, float, float]:
    """Converts an ITRF position & velocity into
    azimut, elevation, distance, azimut rate, elevation rate, radial velocity

    Args:
        obs: Position (m) & velocity (m/s) of terrestrial observer in ITRF
        sat: Position (m) & velocity (m/s) of the observed satellite in ITRF

    Returns:
        A tuple containing:

        * Azimut (rad)
        * Elevation (rad)
        * Range (m)
        * Azimut rate (rad/s)
        * Elevation rate (rad/s)
        * Range rate (m/s)

    """
    # Local ENV for the observer
    obs_env = build_local_matrix(obs[:3])
    deltap = sat[:3] - obs[:3]
    deltav = sat[3:] - obs[3:]
    x, y, z = obs_env.T @ deltap
    vx, vy, vz = obs_env.T @ deltav

    dist = lin.norm(deltap)

    if z > dist:
        z = dist
        logger.warning("Near zenith elevation")
    elif z < -dist:
        z = -dist
        logger.warning("Near nadir elevation")

    el = arcsin(z / dist)
    az = arctan2(x, y)

    # Local ENV for the satellite
    rxy = sqrt(x**2 + y**2)
    deltav = sat[3:] - obs[3:]
    vr = deltav @ deltap / dist
    azr = (-x * vy + y * vx) / rxy**2
    elr = (vz - vr * z / dist) / rxy

    return az, el, dist, azr, elr, vr


def itrf_to_llavpa(pv: NDArray[Any, Any]) -> Tuple[float, float, float, float, float, float]:
    """Converts an ITRF position & velocity into
    longitude, latitude, altitude (WGS84) and velocity, slope of velocity, azimut of velocity
    Velocity is the speed of sat in ITRF frame

    Args:
        pv: Position (m) & velocity (m/s) of the object in ITRF

    Returns:
        A tuple containing:

        * Longitude (rad)
        * Latitude (rad)
        * Altitude (m)
        * Velocity (m/s)
        * Slope of velocity (rad)
        * Azimut of velocity (rad)

    """
    lon, lat, alt = itrf_to_geodetic(pv)

    # Local ENV for the satellite
    M = build_local_matrix(pv[:3])
    ve, vn, vv = M.T @ pv[3:]
    nv = lin.norm((ve, vn, vv))

    if np.abs(nv) < 1e-6:
        vs = 0.0
    else:
        vs = arcsin(vv / nv)
    va = arctan2(ve, vn)

    return np.array([lon, lat, alt, nv, vs, va])


def llavpa_to_itrf(llavpa: NDArray[Any, Any]) -> NDArray[Any, Any]:
    """Converts a LLAVPA into ITRF
    A LLAVPA is an array with
    longitude, latitude, altitude (WGS84) and velocity, slope of velocity, azimut of velocity
    Velocity is the speed of sat in ITRF frame

    Args:
        llavpa: lon,lat,alt,vel,vs,va

    Returns:
        An ITRF position and velocity

    """
    lon, lat, alt = llavpa[:3]
    pos = geodetic_to_itrf(lon, lat, alt)

    # Local ENV for the satellite
    M = build_local_matrix(pos)

    nv, vs, va = llavpa[3:]
    ve = cos(vs) * sin(va) * nv
    vn = cos(vs) * cos(va) * nv
    vv = sin(vs) * nv

    vel = M @ np.array([ve, vn, vv])

    return np.hstack((pos, vel))


def datetime_to_skyfield(td: datetime) -> Time:
    """
    Converts a datetime struct to a skyfield Time struct

    Args:
        td: a datetime instance or array of datetime to convert

    Returns:
        Skyfield date and time structure. See https://rhodesmill.org/skyfield/api-time.html#skyfield.timelib.Time

    Examples:
        >>> fmt = "%Y/%m/%d %H:%M:%S.%f"
        >>> sts = "2021/04/15 09:29:54.996640"
        >>> tsync = datetime.strptime(sts, fmt)
        >>> tsync = tsync.replace(tzinfo=utc)
        >>> datetime_to_skyfield(tsync)
        <Time tt=2459319.8965761648>

    """
    ts = load.timescale(builtin=True)
    t = ts.utc(td)
    return t


def skyfield_to_datetime(t: Time) -> datetime:
    """
    Converts a skyfield Time struct to a datetime struct

    Args:
        t: Skyfield date and time structure. See https://rhodesmill.org/skyfield/api-time.html#skyfield.timelib.Time

    Returns:
        A datetime instance

    """
    return t.utc_datetime()


def pdot(u: NDArray[Any, Any], v: NDArray[Any, Any]) -> float:
    """Pseudo scalar product :

    $$ x.x'+y.y'+z.z'-t.t' $$

    Args:
        u: First quadri-vector
        v: Second quadri-vector

    Returns:
        Pseudo scalar product

    """
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2] - u[3] * v[3]


def q_function(x):
    """
    https://en.wikipedia.org/wiki/Q-function
    """
    return 0.5 * np.erfc(x / np.sqrt(2))


def cexp(x):
    """Function defined by:

    $$ cexp(x) = exp(2.\pi.i.x) $$

    """
    return exp(2 * pi * 1j * x)


def load_antenna_config(config: str):
    """Loads a module located at the given path
    Used in `blocksim.dsp.AntennaNetwork.AntennaNetwork`

    Args:
        config: path to a python file

    Returns:
        A loaded python module

    """
    pth = os.path.abspath(config)

    mod = os.path.basename(pth)[:-3]
    spec = importlib.util.spec_from_file_location(mod, pth)
    ac = importlib.util.module_from_spec(spec)
    sys.modules[mod] = ac
    spec.loader.exec_module(ac)

    return ac


def mean_variance_2dgauss_norm(cov):
    r"""If \(X\) is a 2D random variable that follows a 2D-gaussian law
    with 0 mean and covariance \(\Sigma\), computes the mean and variance of \(\| X \|\)

    Args:
        cov: Covariance of \(X\), noted \(\Sigma\) above. Must be symmetric, positive-definite

    Returns:

        * The mean of \(\| X \|\)
        * The variance of \(\| X \|\)

    """
    sp = np.real(lin.eigvals(cov))
    sp.sort()
    l2, l1 = sp

    esq = np.trace(cov)

    m = (l1 - l2) / l1
    sqe = 2 * l1 / pi * ellipe(m) ** 2

    return sqrt(sqe), esq - sqe


def mean_variance_3dgauss_norm(cov):
    r"""If \(X\) is a 3D random variable that follows a 3D-gaussian law
    with 0 mean and covariance \(\Sigma\), computes the mean and variance of \(\| X \|\).

    Args:
        cov: Covariance of \(X\), noted \(\Sigma\) above. Must be symmetric, positive-definite

    Returns:

        * The mean of \(\| X \|\)
        * The variance of \(\| X \|\)

    """
    sp = np.real(lin.eigvals(cov))
    sp.sort()
    l3, l2, l1 = sp

    esq = np.trace(cov)

    Ae = sqrt(l1 - l3)
    Af = l3 / Ae
    m = (l2 - l1) / (l3 - l1)
    th0 = arccos(sqrt(l3 / l1))  # c/a
    sqe = 4 / (2 * pi) * (sqrt(l3 * l2 / l1) + Af * ellipkinc(th0, m) + Ae * ellipeinc(th0, m)) ** 2

    return sqrt(sqe), esq - sqe
