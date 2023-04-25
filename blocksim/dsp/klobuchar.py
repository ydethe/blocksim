from numpy import cos, sin, pi, abs


def klobuchar(phi, lbd, elev, azimuth, tow, alpha, beta) -> float:
    # https://fr.mathworks.com/matlabcentral/fileexchange/59530-klobuchar-ionospheric-delay-model
    """Function for computing an Ionospheric range correction for the
    GPS L1 frequency from the parameters broadcasted in the GPS Navigation Message.

    References:
      Klobuchar, J.A., (1996) "Ionosphercic Effects on GPS", in
        Parkinson, Spilker (ed), "Global Positioning System Theory and
        Applications, pp.513-514.
      ICD-GPS-200, Rev. C, (1997), pp. 125-128
      NATO, (1991), "Technical Characteristics of the NAVSTAR GPS",
        pp. A-6-31   -   A-6-33

    Args:
        phi: Geodetic latitude of receiver (rad)
        lbd: Geodetic longitude of receiver (rad)
        elev: Elevation angle of satellite (rad)
        azimuth: Geodetic azimuth of satellite (rad)
        tow: Time of Week (s)
        alpha[3]: The coefficients of a cubic equation
          representing the amplitude of the vertical
          delay (4 coefficients)
        beta[3]: The coefficients of a cubic equation
          representing the period of the model
          (4 coefficients)

    Returns:
        Ionospheric delay for the L1 frequency (sec)

    Examples:
        >>> alpha = np.array([0.3820e-7, 0.1490e-7, -0.1790e-6, 0.0000])
        >>> beta = np.array([0.1430e6, 0.0000, -0.3280e6, 0.1130e6])
        >>> elev = 20*pi/180
        >>> azimuth = 210*pi/180
        >>> phi = 40*pi/180
        >>> lbd = 260*pi/180
        >>> tow = 0.0
        >>> dIon1 = klobuchar(phi, lbd, elev, azimuth, tow, alpha, beta)
        >>> dIon1 # doctest: +ELLIPSIS
        6.93560658...e-08

    """
    rad2semi = 1.0 / pi  # radians to semisircles
    semi2rad = pi  # semisircles to radians
    a = azimuth  # asimuth in radians
    e = elev * rad2semi  # elevation angle in semicircles
    psi = 0.0137 / (e + 0.11) - 0.022  # Earth Centered angle
    lat_i = phi * rad2semi + psi * cos(a)  # Subionospheric lat
    if lat_i > 0.416:
        lat_i = 0.416
    elif lat_i < -0.416:
        lat_i = -0.416

    # Subionospheric long
    long_i = lbd * rad2semi + (psi * sin(a) / cos(lat_i * semi2rad))

    # Geomagnetic latitude
    lat_m = lat_i + 0.064 * cos((long_i - 1.617) * semi2rad)

    t = 4.32e4 * long_i + tow
    t = t % 86400.0  # Seconds of day
    if t > 86400.0:
        t = t - 86400.0

    if t < 0.0:
        t = t + 86400.0

    sF = 1.0 + 16.0 * (0.53 - e) ** 3  # Slant factor
    # Period of model
    PER = beta[0] + beta[1] * lat_m + beta[2] * lat_m**2 + beta[3] * lat_m**3
    if PER < 72000.0:
        PER = 72000.0

    x = 2.0 * pi * (t - 50400.0) / PER  # Phase of the model
    # (Max at 14.00 =
    # 50400 sec local time)
    # Amplitud of the model
    AMP = alpha[0] + alpha[1] * lat_m + alpha[2] * lat_m**2 + alpha[3] * lat_m**3
    if AMP < 0.0:
        AMP = 0.0

    # Ionospheric corr.
    if abs(x) > 1.57:
        dIon1 = sF * (5.0e-9)
    else:
        dIon1 = sF * (5.0e-9 + AMP * (1.0 - x * x / 2.0 + x * x * x * x / 24.0))

    return dIon1
