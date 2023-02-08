import sys
from pathlib import Path

import numpy as np
import numpy.ma as ma
from numpy import log10
import pytest

from blocksim.dsp import createGNSSSequence, delay_doppler_analysis
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.graphics.BFigure import FigureFactory
from blocksim.graphics import showFigures

sys.path.insert(0, str(Path(__file__).parent.parent))
from TestBase import TestBase


class TestRealGNSS(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=10, savefig_kwargs={"dpi": 150})
    def test_real_gnss_samples(
        self,
        pth: Path = Path("tests/test_gnss/gnss_samples.bsline"),
        bandwidth: float = 2e6,
        n_integration: int = 3,
        doppler_win: float = 5e3,
    ):
        sig = DSPSignal.fromBsline(pth)
        sig = sig.superheterodyneIQ(carrier_freq=sig.intermediate_frequency, bandwidth=bandwidth)

        rep: DSPSignal = createGNSSSequence(name="rep", modulation="L1CA", sv=1)
        code_duration = rep.duration

        acq = delay_doppler_analysis(
            period=code_duration,
            delay_search_center=code_duration / 2,
            delay_search_win=code_duration,
            doppler_search_center=0,
            doppler_search_win=doppler_win,
            seq=rep,
            rxsig=sig,
            ndop=int(doppler_win * sig.duration),
            n_integration=n_integration,
        )
        p = acq.findPeaksWithTransform(nb_peaks=1)[0]
        fdop, _ = p.coord

        int_sig: DSPSignal = (
            sig.applyDopplerFrequency(-fdop)
            .correlate(rep)
            .integrate(period=code_duration, n_integration=n_integration)
        )
        p = int_sig.findPeaksWithTransform(nb_peaks=1)[0]
        (delay,) = p.coord
        pwr = p.value

        # Masking the peak region, nb_mask samples around it
        n = len(int_sig)
        k = int((delay - int_sig.samplingStart) / int_sig.samplingPeriod)
        nb_mask = 10
        iech = np.arange(n)
        mask = (iech > k - nb_mask) & (iech < k + nb_mask)

        # Getting the noise samples
        z = ma.MaskedArray(int_sig.getTransformedSamples(), mask=mask)

        # SNR & C/N0 estimation (gea-matlab)
        niveau_bruit = z.mean()
        cn0 = 1 / (n_integration * code_duration) * (pwr - niveau_bruit) / niveau_bruit
        cn0_db = 10 * log10(cn0)

        self.assertAlmostEqual(delay, -5.47950936e-05, delta=1e-9)
        self.assertAlmostEqual(fdop, 79.66117426, delta=1e-3)
        self.assertAlmostEqual(cn0_db, 46.74755161036415, delta=1e-5)

        # Plotting
        figs = FigureFactory.create(title="Correlation")
        gs = figs.add_gridspec(1, 1)

        axe = figs.add_baxe(title="", spec=gs[0, 0])
        axe.plot(int_sig.applyDelay(-delay))

        return figs.render()


if __name__ == "__main__":
    a = TestRealGNSS()
    a.test_real_gnss_samples()
    showFigures()
