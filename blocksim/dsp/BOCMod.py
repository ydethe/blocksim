import numpy as np
from numpy import sqrt, sign, pi, exp, cos, sin, log2

from .. import logger
from .ADSPComputer import ADSPComputer
from .CircularBuffer import CircularBuffer
from .DSPSpectrum import DSPSpectrum
from .DSPSignal import DSPSignal


class BOCMapping(ADSPComputer):
    """
    Returns a BOC sequence that, once convoluted with a PRN, gives the modulated signal
    Sub carrier frequency and chip frequency are given as multiples of f_ref = 1.023 MHz

    Args:
      name
        Name of the computer
      f_ref (Hz)
        Chip rate of the modulation
      m
        Sub-carrier frequency multiplier : f_s = m.f_ref
      n
        Chip frequency multiplier : f_s = m.f_ref
      p_samp
        Muliplier so that the sequence is sampled at f_samp = p_samp*f_s
      input_size
        Number of bits modulated in parallel.

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        f_ref: float,
        m: int,
        n: int,
        p_samp: int = 10,
        input_size: int = 1,
    ):
        ADSPComputer.__init__(
            self,
            name=name,
            input_name="input",
            output_name="output",
            input_size=input_size,
            output_size=input_size * p_samp,
            input_dtype=np.int64,
            output_dtype=np.complex128,
        )

        self.createParameter(name="f_ref", value=1.023e6, read_only=True)
        self.createParameter(name="m", value=m, read_only=True)
        self.createParameter(name="n", value=n, read_only=True)
        self.createParameter(name="p_samp", value=p_samp, read_only=True)
        s_boc = self.createSequence()
        self.createParameter(name="boc_seq", value=s_boc, read_only=True)

    def createSequence(self) -> DSPSignal:
        Nb = (2 * self.m) // self.n
        f_c = self.n * self.f_ref
        f_s = self.m * self.f_ref
        T_c = 1 / f_c
        f_samp = self.p_samp * f_s

        n_samp = int(T_c * f_samp)
        p = int(f_samp / (2 * f_s))
        drc = np.zeros(n_samp - p + 1)
        for i in range(Nb):
            drc[p * i] = (-1) ** i

        p_tb = np.ones(p)

        s_boc = np.convolve(p_tb, drc, mode="full")

        sig = DSPSignal(
            name="BOC(%i,%i)_seq" % (self.m, self.n),
            samplingStart=0,
            samplingPeriod=1 / f_samp,
            y_serie=s_boc,
        )

        return sig

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
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
