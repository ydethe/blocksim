from typing import Any

from nptyping import NDArray, Shape
import numpy as np
from numpy import sqrt, sign, pi, exp, cos, sin, log2

from .ADSPComputer import ADSPComputer
from .DSPSpectrum import DSPSpectrum
from .DSPSignal import DSPSignal

from .. import logger


class BOCMapping(ADSPComputer):
    """
    Returns a BOC(m,n) sequence that, once convoluted with a PRN, gives the modulated signal
    Sub carrier frequency and chip frequency are given as multiples of f_ref = 1.023 MHz

    Attributes:
        f_ref: the value passed to __init__
        m: the value passed to __init__
        n: the value passed to __init__
        p_samp: the value passed to __init__
        boc_seq (blocksim.dsp.DSPSignal): The unitary BOC sequence

    Args:
        name: Name of the computer
        f_ref: Chip rate of the modulation (Hz)
        m: Sub-carrier frequency multiplier : f_s = m.f_ref
        n: Chip frequency multiplier : f_s = n.f_ref
        p_samp: Muliplier so that the sequence is sampled at f_samp = p_samp*f_s
        input_size: Number of bits modulated in parallel.

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        f_ref: float,
        m: int,
        n: int,
        p_samp: int = 1,
        input_size: int = 1,
    ):
        n_boc = m // n * 2
        ADSPComputer.__init__(
            self,
            name=name,
            input_name="input",
            output_name="output",
            input_size=input_size,
            output_size=input_size * p_samp * n_boc,
            input_dtype=np.int64,
            output_dtype=np.complex128,
        )

        self.createParameter(name="f_ref", value=f_ref, read_only=True)
        self.createParameter(name="m", value=m, read_only=True)
        self.createParameter(name="n", value=n, read_only=True)
        self.createParameter(name="p_samp", value=p_samp, read_only=True)
        self.createParameter(name="n_boc", value=n_boc, read_only=True)
        s_boc = self.createSequence()
        self.createParameter(name="boc_seq", value=s_boc, read_only=True)

    def createSequence(self) -> DSPSignal:
        """Creates the unitary BOC sequence

        Returns:
            The sequence, sampled at fs = 2 * p_samp * m * f_ref

        """
        p = self.p_samp
        n_samp = self.n_boc * p

        f_samp = 2 * p * self.m * self.f_ref

        drc = np.zeros(n_samp - p + 1)
        for i in range(self.n_boc):
            drc[p * i] = (-1) ** i

        p_tb = np.ones(self.p_samp)

        s_boc = np.convolve(p_tb, drc, mode="full")
        assert len(s_boc) == n_samp

        sig = DSPSignal(
            name="BOC(%i,%i)_seq" % (self.m, self.n),
            samplingStart=0,
            samplingPeriod=1 / f_samp,
            y_serie=s_boc,
        )

        return sig

    def adaptTimeSerie(self, tps: NDArray[Any, Any]) -> NDArray[Any, Any]:
        return tps / (self.n_boc * self.p_samp)

    def update(
        self,
        t1: float,
        t2: float,
        input: NDArray[Any, Any],
        output: NDArray[Any, Any],
    ) -> dict:
        seq = self.boc_seq.y_serie
        p = len(seq)
        prn = self.flatten(input)
        n_code = len(prn)

        sig = np.empty(n_code * p)
        for k in range(n_code):
            shard = prn[k] * seq
            sig[k * p : (k + 1) * p] = shard

        outputs = {}
        outputs["output"] = self.unflatten(sig)
        return outputs
