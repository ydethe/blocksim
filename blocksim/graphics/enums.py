from enum import Enum


class AxeProjection(Enum):
    #: For rectilinear plots (the most frequent use case)
    RECTILINEAR = 0
    #: For trigonometric polar plots
    POLAR = 1
    #: For north azimuthal plots
    NORTH_POLAR = 2
    #: For Mercator cartography
    PLATECARREE = 3
    #: For 3D plots
    DIM3D = 4


class FigureProjection(Enum):
    #: For matplotlib plots
    MPL = 0
    #: For panda3d 3d plots
    EARTH3D = 1
