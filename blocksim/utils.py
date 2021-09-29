import os
import glob
import re
from datetime import datetime, timedelta, timezone
from typing import Iterable, Tuple

from scipy import linalg as lin
import numpy as np
from numpy import pi, arcsin, arctan2, sin, cos, sqrt
import astropy.coordinates as coord
from skyfield.api import Topos, load, utc
from skyfield.timelib import Time

from . import logger
from .constants import *
from .exceptions import *
from . import logger


__all__ = [
    "casedpath",
    "resource_path",
    "geodetic_to_itrf",
    "build_env",
    "itrf_to_azeld",
    "test_diag",
    "calc_cho",
    "deg",
    "rad",
    "assignVector",
    "datetime_to_skyfield",
    "skyfield_to_datetime",
    "pdot",
    "quat_to_matrix",
    "matrix_to_quat",
    "matrix_to_euler",
    "euler_to_matrix",
    "quat_to_euler",
    "euler_to_quat",
    "vecBodyToEarth",
    "vecEarthToBody",
    "q_function",
]


def casedpath(path):
    r = glob.glob(re.sub(r"([^:/\\])(?=[/\\]|$)", r"[\1]", path))
    return r and r[0] or path


def resource_path(resource: str) -> str:
    """

    Examples:
      >>> resource_path('8081_earthmap4k.jpg') # doctest: +ELLIPSIS
      '.../blocksim/blocksim/resources/8081_earthmap4k.jpg'

    """
    from importlib import import_module

    package = "blocksim"
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


def build_env(pos: np.array) -> np.array:
    """Builds a ENV frame at a given position

    Args:
      pos
        Position (m) of a point in ITRF

    Returns:
      Matrix :
      * Local North vector
      * Local East vector
      * Local Vertical vector

    """
    # Local ENV for the observer
    vert = pos.copy()
    vert /= lin.norm(vert)

    east = np.cross(np.array([0, 0, 1]), pos)
    east /= lin.norm(east)

    north = np.cross(vert, east)

    env = np.empty((3, 3))
    env[:, 0] = east
    env[:, 1] = north
    env[:, 2] = vert

    return env


def geodetic_to_itrf(lon, lat, h) -> np.array:
    """
    Compute the Geocentric (Cartesian) Coordinates X, Y, Z
    given the Geodetic Coordinates lat, lon + Ellipsoid Height h

    Args:
      lon (rad)
        L
      lat (rad)
        Latitude
      h (m)
        Altitude

    Returns:
      x, y, z (m) : geocentric position as numpy array

    Examples:
      >>> x,y,z = geodetic_to_itrf(0,0,0)

    """
    N = Req / sqrt(1 - (1 - (1 - 1 / rf) ** 2) * (sin(lat)) ** 2)
    X = (N + h) * cos(lat) * cos(lon)
    Y = (N + h) * cos(lat) * sin(lon)
    Z = ((1 - 1 / rf) ** 2 * N + h) * sin(lat)

    return np.array([X, Y, Z])


def itrf_to_azeld(obs: np.array, sat: np.array) -> np.array:
    """Converts an ITRF position & velocity into
    azimut, elevation, distance, radial velocity, slope of velocity, azimut of velocity

    Args:
      obs
        Position (m) & velocity (m/s) of terrestrial observer in ITRF
      sat
        Position (m) & velocity (m/s) of the observed satellite in ITRF

    Returns:
      Azimut (deg)
      Elevation (deg)
      Distance (m)
      Radial velocity (m/s)
      Slope of velocity (deg)
      Azimut of velocity (deg)

    """
    # Local ENV for the observer
    obs_env = build_env(obs[:3])

    x, y, z = obs_env.T @ (sat[:3] - obs[:3])

    dist = lin.norm((x, y, z))

    if z > dist:
        z = dist
        logger.warning("Near zenith elevation")
    elif z < -dist:
        z = -dist
        logger.warning("Near nadir elevation")

    el = arcsin(z / dist) * 180 / pi
    az = arctan2(x, y) * 180 / pi

    # Local ENV for the satellite
    sat_env = build_env(sat[:3])
    vx, vy, vz = sat_env.T @ (sat[3:] - obs[3:])

    vr = lin.norm((vx, vy, vz))

    if vz > vr:
        vz = vr
        logger.warning("Near zenith velocity")
    elif vz < -vr:
        vz = -vr
        logger.warning("Near nadir velocity")

    vs = arcsin(vz / vr) * 180 / pi
    va = arctan2(vx, vy) * 180 / pi

    return az, el, dist, vr, vs, va


def test_diag(A: np.array) -> bool:
    """Tests if a square matrix is diagonal

    Args:
      A
        Matrix to test

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


def calc_cho(A: np.array) -> np.array:
    """Returns the cholesky decomposition C of a symetric matrix A:

    :math:`A = C.C^T`

    Args:
      A
        Matrix

    Returns:
      Cholesky decomposition C

    """
    m, n = A.shape
    assert m == n

    if test_diag(A):
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
    # if np.any(np.isnan(v)):
    #     txt = "Element '%s' : Argument '%s'=%s has NaN" % (
    #         dst_name,
    #         src_name,
    #         v,
    #     )
    #     logger.warning(txt)

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
      t
        Skyfield date and time structure

    Returns:
      A datetime instance

    """
    return t.utc_datetime()


def pdot(u: np.array, v: np.array) -> float:
    """Pseudo scalar product :

    :math:`x.x'+y.y'+z.z'-t.t'`

    Args:
      u
        First quadri-vector
      v
        Second quadri-vector

      Returns:
        Pseudo scalar product

    """
    return u[0] * v[0] + u[1] * v[1] + u[2] * v[2] - u[3] * v[3]


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
