import unittest
import pytest

import numpy as np
from numpy import pi, sqrt
from blocksim.dsp.DSPLine import DSPLine
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.dsp.DSPSpectrogram import DSPSpectrogram
from blocksim.graphics.wip import (
    AxeFactory,
    FigureFactory,
    AxeProjection,
    FigureProjection,
    showFigures,
)


class TestGraphics2(unittest.TestCase):
    def setUp(self):
        np.random.seed(1883647)

        self.sig = DSPSignal.fromLinearFM(
            name="sig",
            samplingPeriod=1e-2 / 100,
            samplingStart=0,
            tau=1e-2,
            fstart=-2000,
            fend=2000,
        )

        img = np.zeros((50, 100))
        xx = np.arange(-5, 5, 0.1)
        yy = np.arange(-5, 5, 0.2)
        X, Y = np.meshgrid(xx, yy)
        R2 = X**2 + Y**2
        img = np.sinc(sqrt(R2))
        self.spg = DSPSpectrogram(
            name="spg",
            samplingXStart=-5,
            samplingXPeriod=0.1,
            samplingYStart=-5,
            samplingYPeriod=0.2,
            img=img,
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_t1_2dxy_platecarree(self):
        # ==========================
        # T1
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.PLATECARREE
        )

        lon = np.linspace(-pi, pi, 50)
        lat = np.cos(lon)
        axe.plot(plottable=(lon, lat), color="red")

        self.assertRaises(AssertionError, axe.plot, self.sig)

        return fig.render()

    def test_t2_2dtr_platecarree(self):
        # ==========================
        # T2
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.PLATECARREE
        )

        theta = np.linspace(-pi, pi, 50)
        r = np.cos(theta)
        sig = DSPLine(
            name="2DTR",
            samplingStart=theta[0],
            samplingPeriod=theta[1] - theta[0],
            y_serie=r,
            projection="polar",
            default_transform=lambda x: x,
        )
        self.assertRaises(
            AssertionError,
            axe.plot,
            plottable=sig,
            color="red",
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_t3_3dxyz_platecarree(self):
        # ==========================
        # T3
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.PLATECARREE
        )

        xx = np.linspace(-10, 0, 50)
        yy = np.linspace(40, 50, 100)
        X, Y = np.meshgrid(xx, yy)
        R2 = (X + 5) ** 2 + (Y - 45) ** 2
        img = np.sinc(sqrt(R2))
        spg = DSPSpectrogram(
            name="spg",
            samplingXStart=xx[0] * pi / 180,
            samplingXPeriod=(xx[1] - xx[0]) * pi / 180,
            samplingYStart=yy[0] * pi / 180,
            samplingYPeriod=(yy[1] - yy[0]) * pi / 180,
            img=img,
            projection="rectilinear",
        )
        axe.plot(plottable=spg)

        return fig.render()

    def test_t4_3dtrz_platecarree(self):
        # ==========================
        # T4
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.PLATECARREE
        )

        xx = np.linspace(-10, 0, 50)
        yy = np.linspace(40, 50, 100)
        X, Y = np.meshgrid(xx, yy)
        R2 = (X + 5) ** 2 + (Y - 45) ** 2
        img = np.sinc(sqrt(R2))
        spg = DSPSpectrogram(
            name="spg",
            samplingXStart=xx[0] * pi / 180,
            samplingXPeriod=(xx[1] - xx[0]) * pi / 180,
            samplingYStart=yy[0] * pi / 180,
            samplingYPeriod=(yy[1] - yy[0]) * pi / 180,
            img=img,
            projection="polar",
        )
        self.assertRaises(AssertionError, axe.plot, plottable=spg)

    def test_t5_2dxy_polar(self):
        # ==========================
        # T5
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(2, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.POLAR
        )

        theta = np.linspace(-pi, pi, 50)
        r = np.cos(theta)
        sig = DSPLine(
            name="2DTR",
            samplingStart=theta[0],
            samplingPeriod=theta[1] - theta[0],
            y_serie=r,
            projection="rectilinear",
            default_transform=lambda x: x,
        )
        self.assertRaises(AssertionError, axe.plot, plottable=sig)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_t6_2dtr_polar(self):
        # ==========================
        # T6
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(2, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="North polar axe", projection=AxeProjection.NORTH_POLAR
        )

        theta = np.linspace(0, 2 * pi, 50)
        r = theta / (2 * pi)
        axe.plot(plottable=(theta, r), color="red")

        axe = AxeFactory.create(
            spec=gs[1, 0], title="Polar axe", projection=AxeProjection.POLAR
        )

        theta = np.linspace(0, 2 * pi, 50)
        r = theta / (2 * pi)
        axe.plot(plottable=(theta, r), color="blue")

        return fig.render()

    def test_t7_3dxyz_polar(self):
        # ==========================
        # T7
        # ==========================
        xx = np.linspace(-10, 0, 50)
        yy = np.linspace(40, 50, 100)
        X, Y = np.meshgrid(xx, yy)
        R2 = (X + 5) ** 2 + (Y - 45) ** 2
        img = np.sinc(sqrt(R2))
        spg = DSPSpectrogram(
            name="spg",
            samplingXStart=xx[0] * pi / 180,
            samplingXPeriod=(xx[1] - xx[0]) * pi / 180,
            samplingYStart=yy[0] * pi / 180,
            samplingYPeriod=(yy[1] - yy[0]) * pi / 180,
            img=img,
            projection="rectilinear",
        )

        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(2, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.POLAR
        )
        self.assertRaises(AssertionError, axe.plot, plottable=spg)

        axe = AxeFactory.create(
            spec=gs[1, 0], title="axe", projection=AxeProjection.NORTH_POLAR
        )
        self.assertRaises(AssertionError, axe.plot, plottable=spg)

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_t8_3dtrz_polar(self):
        # ==========================
        # T8
        # ==========================
        xx = np.linspace(0, 2 * pi, 50)
        yy = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(xx, yy)
        R2 = (X - pi) ** 2 + (Y - 5) ** 2
        img = np.sinc(sqrt(R2))
        spg = DSPSpectrogram(
            name="spg",
            samplingXStart=xx[0],
            samplingXPeriod=xx[1] - xx[0],
            samplingYStart=yy[0],
            samplingYPeriod=yy[1] - yy[0],
            img=img,
            projection="polar",
        )

        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 2)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="Polar plot", projection=AxeProjection.POLAR
        )
        axe.plot(plottable=spg)

        axe = AxeFactory.create(
            spec=gs[0, 1],
            title="North polar plot",
            projection=AxeProjection.NORTH_POLAR,
        )
        axe.plot(plottable=spg)

        return fig.render()

    def test_t9_2dxy_dim3d(self):
        # ==========================
        # T9
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(2, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.DIM3D
        )

        theta = np.linspace(-pi, pi, 50)
        r = np.cos(theta)
        sig = DSPLine(
            name="2DXY",
            samplingStart=theta[0],
            samplingPeriod=theta[1] - theta[0],
            y_serie=r,
            projection="rectilinear",
            default_transform=lambda x: x,
        )
        self.assertRaises(AssertionError, axe.plot, plottable=sig)

    def test_t10_2dtr_dim3d(self):
        # ==========================
        # T10
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.DIM3D
        )

        theta = np.linspace(-pi, pi, 50)
        r = np.cos(theta)
        sig = DSPLine(
            name="2DTR",
            samplingStart=theta[0],
            samplingPeriod=theta[1] - theta[0],
            y_serie=r,
            projection="polar",
            default_transform=lambda x: x,
        )
        self.assertRaises(
            AssertionError,
            axe.plot,
            plottable=sig,
            color="red",
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_t11_3dxyz_dim3d(self):
        # ==========================
        # T11
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.DIM3D
        )

        axe.plot(plottable=self.spg)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_t12_3dtr_dim3d(self):
        # ==========================
        # T12
        # ==========================
        xx = np.linspace(0, 2 * pi, 50)
        yy = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(xx, yy)
        R2 = (X - pi) ** 2 + (Y - 5) ** 2
        img = np.sinc(sqrt(R2))
        spg = DSPSpectrogram(
            name="spg",
            samplingXStart=xx[0],
            samplingXPeriod=xx[1] - xx[0],
            samplingYStart=yy[0],
            samplingYPeriod=yy[1] - yy[0],
            img=img,
            projection="polar",
        )

        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="Polar plot", projection=AxeProjection.DIM3D
        )
        axe.plot(plottable=spg)

        # return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_t13_2dxy_rect(self):
        # ==========================
        # T13
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.RECTILINEAR
        )

        theta = np.linspace(-pi, pi, 50)
        r = np.cos(theta)
        sig = DSPLine(
            name="2DXY",
            samplingStart=theta[0],
            samplingPeriod=theta[1] - theta[0],
            y_serie=r,
            projection="rectilinear",
            default_transform=lambda x: x,
        )
        axe.plot(plottable=sig, color="red")

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_t14_2dtr_rect(self):
        # ==========================
        # T14
        # ==========================
        theta = np.linspace(0, 2 * pi, 50)
        r = theta / (2 * pi)

        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 2)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="Polar plot", projection=AxeProjection.POLAR
        )
        axe.plot(plottable=(theta, r), color="red")

        axe = AxeFactory.create(
            spec=gs[0, 1],
            title="North polar plot",
            projection=AxeProjection.NORTH_POLAR,
        )
        axe.plot(plottable=(theta, r), color="red")

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_t15_1_3dxyz_2d(self):
        # ==========================
        # T15
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0],
            title="axe",
            projection=AxeProjection.RECTILINEAR,
            aspect="equal",
        )

        axe.plot(plottable=self.spg, fill="contour", levels=20)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_t15_2_3dxyz_2d(self):
        # ==========================
        # T15
        # ==========================
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.RECTILINEAR
        )

        axe.plot(plottable=self.spg, find_peaks=1)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_t16_3dtr_rect(self):
        # ==========================
        # T16
        # ==========================
        xx = np.linspace(0, 2 * pi, 50)
        yy = np.linspace(0, 10, 100)
        X, Y = np.meshgrid(xx, yy)
        R2 = (X - pi) ** 2 + (Y - 5) ** 2
        img = np.sinc(sqrt(R2))
        spg = DSPSpectrogram(
            name="spg",
            samplingXStart=xx[0],
            samplingXPeriod=xx[1] - xx[0],
            samplingYStart=yy[0],
            samplingYPeriod=yy[1] - yy[0],
            img=img,
            projection="polar",
        )

        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="Polar plot", projection=AxeProjection.RECTILINEAR
        )
        axe.plot(plottable=spg, fill="pcolormesh")

        return fig.render()

    def test_3d_plot(self):
        fig = FigureFactory.create(title="Figure", projection=FigureProjection.EARTH3D)
        gs = fig.add_gridspec(
            2, 1
        )  # 2 rows is impossible with FigureProjection.EARTH3D
        self.assertRaises(
            AssertionError,
            AxeFactory.create,
            spec=gs[0, 0],
            title="axe",
            projection=AxeProjection.RECTILINEAR,
        )

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_multiple_plots(self):
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(2, 1)

        axe = AxeFactory.create(
            spec=gs[0, 0], title="axe", projection=AxeProjection.RECTILINEAR
        )

        theta = np.linspace(0, 0.5, 50)
        r = theta * 2
        axe.plot(plottable=(theta, r), color="red")
        axe.plot(plottable=self.sig, color="red")

        axe = AxeFactory.create(
            spec=gs[1, 0], title="axe", sharex=axe, projection=AxeProjection.RECTILINEAR
        )
        axe.plot(plottable=(theta, np.cos(theta * 6 * pi)), color="blue", find_peaks=2)

        return fig.render()


if __name__ == "__main__":
    # unittest.main()

    a = TestGraphics2()
    a.setUp()
    # a.test_t8_3dtrz_polar()
    # a.test_t9_2dxy_3d()
    # a.test_t10_2dtr_dim3d()
    # a.test_t11_3dxyz_dim3d()
    a.test_t12_3dtr_dim3d()
    # a.test_t13_2dxy_rect()
    # a.test_t14_2dtr_rect()
    # a.test_t16_3dtr_rect()

    showFigures()
