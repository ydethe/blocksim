from typing import Tuple, Any

from nptyping import NDArray
import numpy as np
from numpy import pi
import scipy.linalg as lin

from ..utils import build_local_matrix, itrf_to_azeld


__all__ = ["computeDOP"]


def computeDOP(
    algo: str, ephem: NDArray[Any, Any], pv_ue: NDArray[Any, Any], elev_mask: float = 0
) -> Tuple[complex, complex, complex, complex, complex]:
    """Computes the DOPs

    Args:
        algo: Type of receiver algorithm
        elev_mask: Elevation mask to determine if a satellite is visible (rad)
        ephem: Ephemeris vector
        pv_ue: UE 3D position/velocity (ITRF) without velocity (m)

    Returns:
        A tuple containing:

        * DOP for X axis (ENV)
        * DOP for Y axis (ENV)
        * DOP for Z axis (ENV)
        * DOP for distance error
        * DOP for velocity error

    """
    if elev_mask is None:
        elev_mask = 0.0

    pos = pv_ue[:3]
    nsat = len(ephem) // 6
    nval = 0
    if algo == "doppler-ranging":
        P = np.zeros((nsat, 5))
        V = np.zeros((nsat, 5))
    else:
        P = np.zeros((nsat, 4))
        V = np.zeros((nsat, 4))

    Penv = build_local_matrix(pos)

    for k in range(nsat):
        spos = ephem[6 * k : 6 * k + 3]
        svel = ephem[6 * k + 3 : 6 * k + 6]
        spv = ephem[6 * k : 6 * k + 6]
        if np.isnan(spos[0]):
            continue
        else:
            _, el, _, _, _, _ = itrf_to_azeld(pv_ue, spv)
            if el < elev_mask:
                continue

        R = spos - pos
        d = lin.norm(R)

        P[nval, :3] = -(Penv.T @ R) / d
        V[nval, :3] = -(Penv.T @ svel) / d + (Penv.T @ R) / d**3 * (svel @ R)

        if algo == "ranging":
            P[nval, 3] = 1

        elif algo == "doppler":
            V[nval, 3] = 1

        elif algo == "doppler-ranging":
            P[nval, 3] = 1
            P[nval, 4] = 0

            V[nval, 3] = 0
            V[nval, 4] = 1

        nval += 1

    P = P[:nval, :]
    V = V[:nval, :]
    nsat, n = P.shape
    if nsat < n:
        return None

    H = np.zeros((n, n))
    if algo == "ranging" or algo == "doppler-ranging":
        _, Rp = lin.qr(P, mode="full")
        PtP = Rp.T @ Rp
        H += PtP

    if algo == "doppler" or algo == "doppler-ranging":
        _, Rv = lin.qr(V, mode="full")
        VtV = Rv.T @ Rv
        H += VtV

    if algo == "ranging":
        Q = lin.inv(H)

    elif algo == "doppler":
        Q = lin.inv(H)

    elif algo == "doppler-ranging":
        iH = lin.inv(H)
        Q1 = iH @ PtP @ iH
        Q2 = iH @ VtV @ iH
        Q = Q1 + 1j * Q2

    return Q
