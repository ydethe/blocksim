import pytest
import numpy as np

from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.graphics.BFigure import FigureFactory
from blocksim.dsp import delay_doppler_analysis


from blocksim.testing import TestBase


class TestRadar(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=7, savefig_kwargs={"dpi": 150})
    def test_analyse_dv(self):
        tau = 10e-6
        bp = 5e6
        fs = bp * 3
        eta = 0.1
        nrep = 50
        fdop = 1e3

        rep = DSPSignal.fromLinearFM(
            name="rep",
            samplingStart=0,
            samplingPeriod=1 / fs,
            tau=tau,
            fstart=-bp / 2,
            fend=bp / 2,
        )

        ns_rep = len(rep)

        ns = int(ns_rep / eta)
        tps = np.arange(nrep * ns) / fs
        y_sig = np.zeros(nrep * ns, dtype=np.complex128)
        for k in range(nrep):
            y_sig[k * ns : k * ns + ns_rep] = rep.y_serie * k
        sig = (
            DSPSignal.fromTimeAndSamples(name="sig", tps=tps, y_serie=y_sig)
            .applyDelay(tau * 1.5)
            .applyDopplerFrequency(fdop=fdop)
            .applyGaussianNoise(pwr=5)
        )

        Tr = tau / eta
        doppler_search_win = 1 / 2 / Tr
        delay_search_win = 2 * tau

        spg = delay_doppler_analysis(
            period=Tr,
            delay_search_center=tau,
            delay_search_win=delay_search_win,
            doppler_search_center=fdop,
            doppler_search_win=doppler_search_win,
            seq=rep,
            rxsig=sig,
            coherent=True,
            ndop=100,
            corr_window="hamming",
        )

        trf = DSPSignal.to_db_lim(-80)
        (peak,) = spg.findPeaksWithTransform(transform=trf, nb_peaks=1)
        self.assertAlmostEqual(peak.coord[0], fdop, delta=0.2)  # doppler
        self.assertAlmostEqual(peak.coord[1], 15e-6, delta=5e-10)  # delay
        self.assertAlmostEqual(peak.value, 27.85, delta=0.1)

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="Power (dB)", spec=gs[0, 0])
        axe.plot(
            spg,
            transform=trf,
            find_peaks=1,
        )

        return fig.render()


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    from blocksim.graphics import showFigures

    a = TestRadar()
    a.setUp()
    a.test_analyse_dv()

    showFigures()
