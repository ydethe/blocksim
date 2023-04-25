from typing import Tuple
from datetime import datetime, timezone


import numpy as np
import scipy.linalg as lin
import fortranformat as ff

from ..utils import FloatArr, build_local_matrix, itrf_to_azeld


__all__ = ["computeDOP", "read_ionex_metadata"]


def computeDOP(
    algo: str, ephem: FloatArr, pv_ue: FloatArr, elev_mask: float = 0
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


def read_ionex_metadata(header: str) -> dict:
    """Read ionex file meta data. All dates converted in a aware datetime python object.

    Args:
        header: Portion of the ionex file between the beginning of file
        and the first occurence of 'START OF TEC MAP'

    Returns:
        A dictionary with following keys :

        * ionex_version
        * file_type
        * ionex_model
        * run_date
        * epoch_first_map
        * epoch_last_map
        * interval
        * num_of_maps
        * mapping_function
        * elevation_cutoff
        * observables_used
        * num_of_stations
        * num_of_satellites
        * base_radius
        * map_dimension
        * hgt1
        * hgt2
        * dhgt
        * lat1
        * lat2
        * dlat
        * lon1
        * lon2
        * dlon

    """
    elem = header.strip().split("\n")
    desc = ""
    metadata = dict(exponent=-1)
    for kth_elem in elem:
        ls = kth_elem.strip()
        if ls.endswith("IONEX VERSION / TYPE"):
            ff_fmt = ff.FortranRecordReader("(F8.1,7X,A1,19X,A3,17X)")
            ionex_version, file_type, ionex_model = ff_fmt.read(kth_elem)
            metadata["ionex_version"] = ionex_version
            metadata["file_type"] = file_type
            metadata["ionex_model"] = ionex_model

        elif ls.endswith("PGM / RUN BY / DATE"):
            ff_fmt = ff.FortranRecordReader("(A20,A20,A20)")
            raw = ff_fmt.read(kth_elem)
            pgm, run_by, date = [x.strip() for x in raw]

            run_date = datetime.strptime(date, "%d-%b-%y %H:%M").replace(tzinfo=timezone.utc)
            metadata["pgm"] = pgm
            metadata["run_by"] = run_by
            metadata["run_date"] = run_date

        elif ls.endswith("DESCRIPTION"):
            ff_fmt = ff.FortranRecordReader("(A60)")
            s = ff_fmt.read(kth_elem)[0].strip()

            desc += f" {s}"
        elif ls.endswith("EPOCH OF FIRST MAP"):
            ff_fmt = ff.FortranRecordReader("(6I6,24X)")
            yr, mo, da, hr, mn, sc = ff_fmt.read(kth_elem)

            epoch_first_map = datetime(yr, mo, da, hr, mn, sc, tzinfo=timezone.utc)
            metadata["epoch_first_map"] = epoch_first_map

        elif ls.endswith("EPOCH OF LAST MAP"):
            ff_fmt = ff.FortranRecordReader("(6I6,24X)")
            yr, mo, da, hr, mn, sc = ff_fmt.read(kth_elem)

            epoch_last_map = datetime(yr, mo, da, hr, mn, sc, tzinfo=timezone.utc)
            metadata["epoch_last_map"] = epoch_last_map

        elif ls.endswith("INTERVAL"):
            ff_fmt = ff.FortranRecordReader("(I6,54X)")
            interval = ff_fmt.read(kth_elem)[0]
            metadata["interval"] = interval

        elif ls.endswith("# OF MAPS IN FILE"):
            ff_fmt = ff.FortranRecordReader("(I6,54X)")
            num_of_maps = ff_fmt.read(kth_elem)[0]
            metadata["num_of_maps"] = num_of_maps

        elif ls.endswith("MAPPING FUNCTION"):
            ff_fmt = ff.FortranRecordReader("(2X,A4,54X)")
            mapping_function = ff_fmt.read(kth_elem)[0]
            metadata["mapping_function"] = mapping_function

        elif ls.endswith("ELEVATION CUTOFF"):
            ff_fmt = ff.FortranRecordReader("(F8.1)")
            elevation_cutoff = ff_fmt.read(kth_elem)[0]
            metadata["elevation_cutoff"] = elevation_cutoff

        elif ls.endswith("OBSERVABLES USED"):
            ff_fmt = ff.FortranRecordReader("(A60)")
            observables_used = ff_fmt.read(kth_elem)[0].strip()
            metadata["observables_used"] = observables_used

        elif ls.endswith("# OF STATIONS"):
            ff_fmt = ff.FortranRecordReader("(I6,54X)")
            num_of_stations = ff_fmt.read(kth_elem)[0]
            metadata["num_of_stations"] = num_of_stations

        elif ls.endswith("# OF SATELLITES"):
            ff_fmt = ff.FortranRecordReader("(I6,54X)")
            num_of_satellites = ff_fmt.read(kth_elem)[0]
            metadata["num_of_satellites"] = num_of_satellites

        elif ls.endswith("BASE RADIUS"):
            ff_fmt = ff.FortranRecordReader("(F8.1)")
            base_radius = ff_fmt.read(kth_elem)[0]
            metadata["base_radius"] = base_radius

        elif ls.endswith("MAP DIMENSION"):
            ff_fmt = ff.FortranRecordReader("(I6)")
            map_dimension = ff_fmt.read(kth_elem)[0]
            metadata["map_dimension"] = map_dimension

        elif ls.endswith("HGT1 / HGT2 / DHGT"):
            ff_fmt = ff.FortranRecordReader("(2X,3F6.1)")
            hgt1, hgt2, dhgt = ff_fmt.read(kth_elem)
            metadata["hgt1"] = hgt1
            metadata["hgt2"] = hgt2
            metadata["dhgt"] = dhgt

        elif ls.endswith("LAT1 / LAT2 / DLAT"):
            ff_fmt = ff.FortranRecordReader("(2X,3F6.1)")
            lat1, lat2, dlat = ff_fmt.read(kth_elem)
            metadata["lat1"] = lat1
            metadata["lat2"] = lat2
            metadata["dlat"] = dlat

        elif ls.endswith("LON1 / LON2 / DLON"):
            ff_fmt = ff.FortranRecordReader("(2X,3F6.1)")
            lon1, lon2, dlon = ff_fmt.read(kth_elem)
            metadata["lon1"] = lon1
            metadata["lon2"] = lon2
            metadata["dlon"] = dlon

        elif ls.endswith("EXPONENT"):
            ff_fmt = ff.FortranRecordReader("(I6)")
            exponent = ff_fmt.read(kth_elem)[0]
            metadata["exponent"] = exponent

    metadata["description"] = desc

    return metadata
