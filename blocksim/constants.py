from numpy import pi


#: Boltzmann constant (J/K)
kb = 1.380649e-23

#: Speed of light in vacuum (m/s)
c = 299792458.0

#: Equatorial Earth radius in WGS84 (m)
Req = 6378137.0

#: Polar Earth radius in WGS84 (m)
Rpo = 6356752.3

#: Inverse of flattening in WGS84 (-)
rf = 298.257223563

#: Earth sideral revolution pulsation (rad/s)
omega = 7.2921151467064e-5

#: Gravitationnal parameter (m^3/s^2)
mu = 3.986004418e14

#: Earth sideral day (s)
jour_sideral = 2 * pi / omega

#: Earth stellar day (s)
jour_stellaire = 86400.0

#: Earth J2 coefficient
J2 = 1.08263e-3

#: Astronomical Unit (m)
AU_M = 149597870700.0  # per IAU 2012 Resolution B2

#: Astronomical Unit (km)
AU_KM = 149597870.700

#: Number of arcseconds in a full circle
ASEC360 = 1296000.0

#: Stellar day (s)
DAY_S = 86400.0
