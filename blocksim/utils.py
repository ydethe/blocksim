from datetime import datetime, timedelta, timezone
from typing import Iterable, Tuple

from scipy import linalg as lin
import numpy as np
import astropy.coordinates as coord
from skyfield.api import Topos, load
from skyfield.timelib import Time

from . import logger
from .constants import *
from .exceptions import *
from . import logger


__all__ = ["deg", "rad"]


def deg(x: float) -> float:
    """Converts from radians to degrees

    Args:
      x
        Angle in radians

    Returns:
      Angle in degrees

    """
    return x * 180 / np.pi


def rad(x: float) -> float:
    """Converts from degrees to radians

    Args:
      x
        Angle in degrees

    Returns:
      Angle in radians

    """
    return x * np.pi / 180


def assignVector(
    v: np.array, expected_shape: tuple, dst_name: str, src_name: str, dtype
) -> np.array:
    """

    Args:
      v
        np.array to assign
      expected_shape
        Expected shape for v
      dst_name
        Name of the element where the assignement will take place. (To allow meaningfull error messages)
      src_name
        Name of the source vector. (To allow meaningfull error messages)
      dtype
        Type of the assigned vector

    Returns:
      Copy of the vector v if no problem encountered

    Raises:
      ValueError
        If the vector is not a np.array or not with the correct shape

    Examples:
      >>> v = np.arange(5)
      >>> assignVector(v, (5,), 'elem', 'v', np.float64)
      array([0., 1., 2., 3., 4.])

    """
    if np.any(np.isnan(v)):
        txt = "Element '%s' : Argument '%s'=%s has NaN" % (
            dst_name,
            src_name,
            v,
        )
        logger.error(txt)
        raise InvalidAssignedVector(txt)

    if isinstance(v.shape, int):
        vshape = (v.shape,)
    else:
        vshape = v.shape

    if isinstance(expected_shape, int):
        expected_shape = (expected_shape,)

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
        txt = "Element '%s' : Argument '%s' - trying to affect a complex vector into a real or integer vector" % (
            dst_name,
            src_name,
        )
        raise WrongDataType(txt)

    else:
        return np.array(v.copy(), dtype=dtype)


def datetime_to_skyfield(td: datetime) -> Time:
    """
    Converts a datetime struct to a skyfield Time struct

    Args:
      td : a datetime instance or array of datetime
        Time to convert

    Returns:
      Skyfield date and time structure

    """
    ts = load.timescale(builtin=True)
    t = ts.utc(td)
    return t


def skyfield_to_datetime(t: Time) -> datetime:
    """
    Converts a skyfield Time struct to a datetime struct

    Args:
      t
        Skyfield date and time structure

    Returns:
      A datetime instance

    """
    return t.utc_datetime()


def quat_to_matrix(qr: float, qi: float, qj: float, qk: float) -> np.array:
    """
    https://en.wikipedia.org/wiki/Quaternions_and_spatial_rotation#Quaternion-derived_rotation_matrix

    Args:
      qr
        Real part of the quaternion
      qi
        First imaginary of the quaternion
      qj
        Second imaginary of the quaternion
      qk
        Third imaginary of the quaternion

    Returns:
      Rotation matrix

    Examples:
      >>> R = quat_to_matrix(1.,0.,0.,0.)
      >>> lin.norm(R - np.eye(3)) # doctest: +ELLIPSIS
      0.0...

    """
    res = np.empty((3, 3))

    res[0, 0] = qr ** 2 + qi ** 2 - qj ** 2 - qk ** 2
    res[1, 1] = qr ** 2 - qi ** 2 + qj ** 2 - qk ** 2
    res[2, 2] = qr ** 2 - qi ** 2 - qj ** 2 + qk ** 2
    res[0, 1] = 2 * (qi * qj - qk * qr)
    res[0, 2] = 2 * (qi * qk + qj * qr)
    res[1, 0] = 2 * (qi * qj + qk * qr)
    res[1, 2] = 2 * (qj * qk - qi * qr)
    res[2, 0] = 2 * (qi * qk - qj * qr)
    res[2, 1] = 2 * (qj * qk + qi * qr)

    return res


def matrix_to_quat(R: np.array) -> Iterable[float]:
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


def matrix_to_euler(R: np.array) -> Iterable[float]:
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


def euler_to_matrix(roll: float, pitch: float, yaw: float) -> np.array:
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
    Ry = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]]
    )
    Rp = np.array(
        [
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)],
        ]
    )
    Rr = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]]
    )
    R = Ry @ Rp @ Rr
    return R


def quat_to_euler(
    qr: float, qi: float, qj: float, qk: float, normalize: bool = False
) -> Iterable[float]:
    """
    https://en.wikipedia.org/wiki/Conversion_between_quaternions_and_Euler_angles

    Args:
      qr
        Real part of the quaternion
      qi
        First imaginary of the quaternion
      qj
        Second imaginary of the quaternion
      qk
        Third imaginary of the quaternion
      normalize
        Pass True to force normalization

    Returns:
      Roll angle (rad)

      Pitch angle (rad)

      Yaw angle (rad)

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
      roll
        Roll angle (rad)
      pitch
        Pitch angle (rad)
      yaw
        Yaw angle (rad)

    Returns:
      Real part of the quaternion

      First imaginary of the quaternion

      Second imaginary of the quaternion

      Third imaginary of the quaternion

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


def vecBodyToEarth(attitude: np.array, x: np.array) -> np.array:
    """Expresses a vector from the body frame to the Earth's frame

    Args:
      attitude
        If 3 elements array, roll, pitch, yaw (rad)
        If 4 elements array, qw, qx, qy, qz
      x
        Vector expressed in the body frame

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


def vecEarthToBody(attitude: np.array, x: np.array) -> np.array:
    """Expresses a vector from Earth's frame to the body's frame

    Args:
      attitude
        If 3 elements array, roll, pitch, yaw (rad)
        If 4 elements array, qw, qx, qy, qz
      x
        Vector expressed in Earth's frame

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


def q_function(x):
    """
    https://en.wikipedia.org/wiki/Q-function
    """
    return 0.5 * np.erfc(x / np.sqrt(2))
