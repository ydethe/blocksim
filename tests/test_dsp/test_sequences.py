import sys
from pathlib import Path
import unittest

import pytest
from blocksim.dsp.DSPSignal import DSPSignal

from blocksim.graphics.BFigure import FigureFactory
from blocksim.dsp import createGNSSSequence, createZadoffChu

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestSignal(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_zadoff_chu_crosscorr(self):
        s1 = createZadoffChu(name="s1", n_zc=1021, u=1, sampling_freq=1e6)
        s2 = createZadoffChu(name="s2", n_zc=1021, u=75, sampling_freq=1e6)

        y = s1.correlate(s2)
        y.setDefaultTransform(y.to_db_lim(-100), unit_of_y_var="dB")

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(y)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_zadoff_chu_autocorr(self):
        s1 = createZadoffChu(name="s1", n_zc=1021, u=1, sampling_freq=1e6)

        y = s1.correlate(s1)
        y.setDefaultTransform(y.to_db_lim(-100), unit_of_y_var="dB")

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(y)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_gold_crosscorr(self):
        s1 = createGNSSSequence(
            name="s1", modulation="L1CA", sv=1, repeat=1, chip_rate=1.023e6, samples_per_chip=10
        )
        s2 = createGNSSSequence(
            name="s2", modulation="L1CA", sv=2, repeat=1, chip_rate=1.023e6, samples_per_chip=10
        )

        y: DSPSignal = s1.correlate(s2)
        y.setDefaultTransform(y.to_db_lim(-100), unit_of_y_var="dB")

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(y)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_gold_autocorr(self):
        s1 = createGNSSSequence(
            name="s1", modulation="L1CA", sv=1, repeat=1, chip_rate=1.023e6, samples_per_chip=10
        )

        y = s1.correlate(s1)
        y.setDefaultTransform(y.to_db_lim(-100), unit_of_y_var="dB")

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(y)

        return fig.render()

    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_gold_corr_integ(self):
        # Reference Gold sequence
        y1 = createGNSSSequence(
            name="s1", modulation="L1CA", sv=1, repeat=1, chip_rate=1.023e6, samples_per_chip=10
        )

        # Noisy received signal
        y = createGNSSSequence(
            name="s1", modulation="L1CA", sv=1, repeat=20, chip_rate=1.023e6, samples_per_chip=10
        )
        y = y.applyGaussianNoise(pwr=200)

        # Correlation
        z = y.correlate(y1)
        z.setDefaultTransform(z.to_db_lim(-100), unit_of_y_var="dB")

        # Integration
        zi = z.integrate(period=1e-3, offset=511 / (1.023e6))

        # Plotting
        fig = FigureFactory.create()
        gs = fig.add_gridspec(3, 1)

        axe = fig.add_baxe(title="Brut", spec=gs[0, 0])
        axe.plot(y)

        axe = fig.add_baxe(title="Corrélation", spec=gs[1, 0])
        axe.plot(z)

        axe = fig.add_baxe(title="Intégration", spec=gs[2, 0])
        axe.plot(zi)

        return fig.render()


if __name__ == "__main__":
    from blocksim.graphics import showFigures

    a = TestSignal()
    # a.test_zadoff_chu_crosscorr()
    # a.test_zadoff_chu_autocorr()
    a.test_gold_autocorr()
    a.test_gold_crosscorr()

    showFigures()
