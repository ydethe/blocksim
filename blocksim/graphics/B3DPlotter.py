from datetime import datetime

from numpy.typing import ArrayLike
import numpy as np
from numpy import pi
from scipy import linalg as lin

from skyfield.api import Loader

import pytest

try:
    panda3d = pytest.importorskip("panda3d")

    from direct.showbase.ShowBase import ShowBase
    from panda3d.core import (
        GeomVertexFormat,
        GeomVertexData,
        GeomVertexWriter,
        Geom,
        GeomNode,
        Texture,
        PointLight,
        TextureStage,
        GeomTristrips,
        GeomLinestrips,
        NodePath,
        LVecBase3,
    )
except BaseException as e:
    ShowBase = object

from ..utils import resource_path
from ..constants import Req, rf
from ..utils import datetime_to_skyfield, geodetic_to_itrf
from ..satellite.Trajectory import Trajectory


class B3DPlotter(ShowBase):
    """Panda3d application that shows the Earth with trajectories

    Usage::

        from datetime import datetime, timedelta, timezone

        import numpy as np
        from numpy import sqrt, cos, sin, pi

        from blocksim.constants import Req
        from blocksim.satellite.Satellite import SGP4Satellite
        from blocksim.B3DPlotter import B3DPlotter

        sat = SGP4Satellite.fromTLE('tests/rigidsphere.tle')

        traj = sat.geocentricITRFTrajectory()

        app = B3DPlotter()
        app.plotEarth()
        app.plotLine(color=(1, 0, 0, 1), itrf_positions=list(zip(*traj)))
        app.plotTrajectory(traj)

        app.run()

    """

    def __init__(self):
        ShowBase.__init__(self)

        self.setBackgroundColor(0.0, 0.0, 0.0)
        # self.disableMouse()
        self.useTrackball()
        self.trackball.getNode(0).setControlMode(2)

        # self.trackball.getNode(0).setOrigin(LVecBase3(0,0,0))
        # self.trackball.getNode(0).setPos(0.0, -20.0, 20.0)

        self.camLens.setNearFar(1.0, 500.0)
        self.camLens.setFov(40.0)

        self.camera.setPos(0.0, -20.0, 20.0)
        self.camera.lookAt(0.0, 0.0, 0.0)

        # props = WindowProperties()
        # props.setCursorHidden(True)
        # props.setMouseMode(WindowProperties.M_relative)
        # self.win.requestProperties(props)

        self.sun_light = None

    def buildSunLight(self, t: datetime):
        load = Loader("skyfield-data")
        ts = datetime_to_skyfield(t)

        eph = load("de421.bsp")
        sun = eph["sun"]
        earth = eph["earth"]

        v = (sun - earth).at(ts)
        pos = v.itrf_xyz().au
        d = lin.norm(pos)
        pos = LVecBase3(*(pos / d * 50))

        plight = PointLight("sun light")
        plight.setColor((1, 1, 1, 1))
        self.sun_light = self.render.attachNewNode(plight)
        self.sun_light.setPos(pos)
        self.render.setLight(self.sun_light)

    def plotTrajectory(self, traj: Trajectory):
        """Plots a Trajectory around the 3D Earth

        Args:
            traj: The Trajectory to plot

        """
        self.plotCube(
            itrf_position=(traj.x[0], traj.y[0], traj.z[0]),
            size=100000,
            color=traj.color,
        )
        if len(traj) > 1:
            self.plotLine(
                color=traj.color, itrf_positions=list(zip(traj.x, traj.y, traj.z))
            )

    def plotLine(self, color: list, itrf_positions: list) -> "NodePath":
        """Plots a custom 3D line

        Args:
            color: The color as a 4-elements tuple:

                * r between 0 and 1
                * g between 0 and 1
                * b between 0 and 1
                * alpha between 0 and 1
            itrf_positions: A list of (x,y,z) positions in the geocentric ITRF coordinate system

        Returns:
            A panda3d NodePath

        """
        format = GeomVertexFormat.get_v3c4()
        format = GeomVertexFormat.registerFormat(format)

        number_of_vertices = len(itrf_positions)

        vdata = GeomVertexData("line", format, Geom.UHStatic)
        vdata.setNumRows(number_of_vertices)

        vertex = GeomVertexWriter(vdata, "vertex")
        colors = GeomVertexWriter(vdata, "color")

        prim = GeomLinestrips(Geom.UHStatic)

        for k in range(number_of_vertices):
            x, y, z = itrf_positions[k]
            vertex.addData3(x / Req, y / Req, z / Req)
            colors.addData4(*color)

            prim.add_vertex(k)

        prim.close_primitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        node = GeomNode("gnode")
        node.addGeom(geom)

        nodePath = self.render.attachNewNode(node)
        nodePath.reparentTo(self.render)

        nodePath.setLightOff()

        return nodePath

    def plotCube(
        self, itrf_position: list, size: float, color: list = None
    ) -> "NodePath":
        """
        Plots a cube

        Args:
            itrf_position: A (x,y,z) positions in the geocentric ITRF coordinate system (m)
            size: The cube's size (m)
            color: The color as a 4-elements tuple

                * r between 0 and 1
                * g between 0 and 1
                * b between 0 and 1
                * alpha between 0 and 1

        Returns:
            A panda3d NodePath

        """
        format = GeomVertexFormat.get_v3c4()
        format = GeomVertexFormat.registerFormat(format)

        vdata = GeomVertexData("cube", format, Geom.UHStatic)
        vdata.setNumRows(8)

        vertex = GeomVertexWriter(vdata, "vertex")
        colors = GeomVertexWriter(vdata, "color")

        for _ in range(8):
            colors.addData4(*color)

        x, y, z = itrf_position
        pos = np.array(itrf_position)
        ap = np.array([0, 0, 1])
        ux = np.cross(ap, pos)
        uy = np.cross(pos, ux)

        # Repere ENV (Est, Nord, Vertical)
        ux = ux / lin.norm(ux)
        uy = uy / lin.norm(uy)
        uz = pos / lin.norm(pos)

        s2 = size / 2
        vertex.addData3(*((pos + s2 * (ux - uy + uz)) / Req))
        vertex.addData3(*((pos + s2 * (ux - uy - uz)) / Req))
        vertex.addData3(*((pos + s2 * (ux + uy + uz)) / Req))
        vertex.addData3(*((pos + s2 * (ux + uy - uz)) / Req))
        vertex.addData3(*((pos + s2 * (-ux + uy + uz)) / Req))
        vertex.addData3(*((pos + s2 * (-ux + uy - uz)) / Req))
        vertex.addData3(*((pos + s2 * (-ux - uy + uz)) / Req))
        vertex.addData3(*((pos + s2 * (-ux - uy - uz)) / Req))

        prim = GeomTristrips(Geom.UHStatic)
        prim.add_vertex(0)
        prim.add_vertex(1)
        prim.add_vertex(2)
        prim.add_vertex(3)
        prim.add_vertex(4)
        prim.add_vertex(5)
        prim.add_vertex(6)
        prim.add_vertex(7)

        prim.add_vertex(0)
        prim.add_vertex(2)
        prim.add_vertex(6)
        prim.add_vertex(4)

        prim.add_vertex(1)
        prim.add_vertex(3)
        prim.add_vertex(7)
        prim.add_vertex(5)

        prim.close_primitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        node = GeomNode("gnode")
        node.addGeom(geom)

        nodePath = self.render.attachNewNode(node)
        nodePath.reparentTo(self.render)

        nodePath.setLightOff()

        return nodePath

    def plotEarth(self) -> "NodePath":
        """Plots a 3D Earth"""
        tex_path = resource_path("8081_earthmap4k.jpg", package="blocksim")
        return self.plotSphere(
            texture=tex_path,
            number_of_meridians=180,
            number_of_latcircles=45,
        )

    def plotSphere(
        self, texture: str, number_of_meridians, number_of_latcircles
    ) -> "NodePath":
        """
        Plots a textured sphere

        Args:
            texture: The texture file to apply
            number_of_meridians: Number of meridians in the mesh
            number_of_latcircles: Number of latitude circles in the mesh

        Returns:
            A panda3d NodePath

        """
        format = GeomVertexFormat.get_v3n3t2()
        format = GeomVertexFormat.registerFormat(format)

        number_of_vertices = 2 + (number_of_meridians + 1) * number_of_latcircles

        vdata = GeomVertexData("sphere", format, Geom.UHStatic)
        vdata.setNumRows(number_of_vertices)

        vertex = GeomVertexWriter(vdata, "vertex")
        normal = GeomVertexWriter(vdata, "normal")
        texcoord = GeomVertexWriter(vdata, "texcoord")

        def vert_index(klon, klat):
            return klon * number_of_latcircles + klat

        for klon in range(number_of_meridians + 1):
            lon = -pi + klon * 2 * pi / number_of_meridians
            for klat in range(number_of_latcircles):
                lat = (klat + 1) * pi / (number_of_latcircles + 1) - pi / 2

                x, y, z = geodetic_to_itrf(lon, lat, 0) / Req

                vertex.addData3(x, y, z)
                normal.addData3(x, y, z)
                texcoord.addData2((lon + pi) / (2 * pi), (lat + pi / 2) / pi)

        x, y, z = geodetic_to_itrf(lon, -pi / 2, 0) / Req
        vertex.addData3(x, y, z)
        normal.addData3(x, y, z)
        texcoord.addData2(0.5, 0)
        k_south_pole = (number_of_meridians + 1) * number_of_latcircles

        x, y, z = geodetic_to_itrf(lon, pi / 2, 0) / Req
        vertex.addData3(x, y, z)
        normal.addData3(x, y, z)
        texcoord.addData2(0.5, 1)
        k_north_pole = 1 + (number_of_meridians + 1) * number_of_latcircles

        prim = GeomTristrips(Geom.UHStatic)

        for klon in range(number_of_meridians):
            prim.add_vertex(k_south_pole)
            for klat in range(number_of_latcircles):
                prim.add_vertex(vert_index(klon + 1, klat))
                prim.add_vertex(vert_index(klon, klat))
            prim.add_vertex(k_north_pole)

        prim.close_primitive()

        geom = Geom(vdata)
        geom.addPrimitive(prim)

        node = GeomNode("gnode")
        node.addGeom(geom)

        nodePath = self.render.attachNewNode(node)
        nodePath.reparentTo(self.render)

        texture = loader.loadTexture(texture)
        texture.setWrapU(Texture.WMClamp)
        texture.setWrapV(Texture.WMClamp)

        stage = TextureStage("Circle")
        stage.setSort(2)

        nodePath.setTexture(stage, texture)

        if not self.sun_light is None:
            nodePath.setLight(self.sun_light)

        return nodePath
