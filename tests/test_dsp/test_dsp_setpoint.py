import unittest

import numpy as np
import pytest

from blocksim.graphics.BFigure import FigureFactory
from blocksim.dsp.DSPSignal import DSPSignal
from blocksim.control.System import ASystem
from blocksim.control.Controller import PIDController
from blocksim.core.Generic import GenericComputer
from blocksim.Simulation import Simulation
from blocksim.testing import TestBase
from blocksim.utils import FloatArr


class System(ASystem):
    def __init__(self, name: str):
        ASystem.__init__(self, name, shape_command=1, snames_state=["x", "v"])
        self.setInitialStateForOutput(np.zeros(2), "state")

    def transition(self, t: float, x: FloatArr, u: FloatArr) -> FloatArr:
        k = 10
        f = 5
        m = 1

        yp, vp = x
        a = (-f * vp - k * yp + u[0]) / m
        dx = np.array([vp, a])

        return dx


class TestDSPSetpoint(TestBase):
    @pytest.mark.mpl_image_compare(tolerance=5, savefig_kwargs={"dpi": 150})
    def test_dsp_setpoint(self):
        k = 10
        m = 1
        a = 8
        Kprop = -k + 3 * a**2 * m
        Kinteg = a**3 * m
        Kderiv = 3 * a * m

        tps = np.arange(0, 10, 0.02)
        y = np.exp(1j * tps * 2 * np.pi * 2)
        stp = DSPSignal(name="sig", samplingStart=0, samplingPeriod=0.01, y_serie=y)
        re = GenericComputer(
            name="re",
            shape_in=(1,),
            shape_out=(1,),
            callable=np.real,
            dtype_in=np.complex128,
            dtype_out=np.float64,
        )
        ctl = PIDController("ctl", shape_estimation=2, snames=["u"], coeffs=(Kprop, Kinteg, Kderiv))
        sys = System("sys")

        sim = Simulation()
        sim.addComputer(stp)
        sim.addComputer(re)
        sim.addComputer(ctl)
        sim.addComputer(sys)

        sim.connect(src_name="ctl.command", dst_name="sys.command")
        sim.connect(src_name="sys.state", dst_name="ctl.estimation")
        sim.connect(src_name="sig.setpoint", dst_name="re.xin")
        sim.connect(src_name="re.xout", dst_name="ctl.setpoint")

        sim.simulate(tps, progress_bar=False)

        self.log = sim.getLogger()
        sig_out = self.log.getSignal("sys_state_x")
        spectrum = sig_out.fft()

        fig = FigureFactory.create()
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(title="", spec=gs[0, 0])
        axe.plot(spectrum)

        return fig.render()


if __name__ == "__main__":
    unittest.main()
    exit(0)

    from blocksim.graphics import showFigures

    a = TestDSPSetpoint()
    a.test_dsp_setpoint()

    showFigures()
