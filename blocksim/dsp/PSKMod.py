from typing import Any

from nptyping import NDArray, Shape
import numpy as np
from numpy import sqrt, sign, pi, exp, cos, sin, log2

from .ADSPComputer import ADSPComputer
from .DSPSpectrum import DSPSpectrum
from .DSPSignal import DSPSignal
from ..graphics.BAxe import BAxe
from .. import logger


class PSKMapping(ADSPComputer):
    """Phase Shift Keying modulator.
    To determine the type of modulation, the *mapping* argument must tell what keying is used :
    For example, in 8-PSK, the block '010' in decimal is 2. So the phase is mapping[2], and the symbol is exp(1j*mapping[2])
    The number of values in *mapping* shall be a power of 2: len(mapping) == 2**mu.

    To initiate the modulator, mu-1 samples shall be processed. The mu-th sample will trigger the conversion of the mu previous samples into a symbol.
    Then the following mu-1 samples are memorised and the last symbol is repeated on the output mu-1 times.

    Args:
        name: Name of the computer
        mapping: List of phase values. For example, in QPSK, [pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4]
        output_size: Number of symbols computed in parallel. The size of the input vector shall therefore be output_size*mu
        p_samp: For one input bit, the PSK symbol is repeated p_samp times

    """

    __slots__ = []

    def __init__(self, name: str, mapping: list, output_size: int = 1, p_samp: int = 1):
        mu = int(np.round(log2(len(mapping)), 0))
        if 2**mu != len(mapping):
            raise ValueError("Mapping size must be a power of 2. Got %i" % len(mapping))

        ADSPComputer.__init__(
            self,
            name=name,
            input_name="input",
            output_name="output",
            input_size=output_size * mu,
            output_size=output_size * p_samp,
            input_dtype=np.int64,
            output_dtype=np.complex128,
        )

        self.createParameter("mu", value=mu, read_only=True)
        self.createParameter("p_samp", value=p_samp, read_only=True)
        self.createParameter("mapping", value=np.array(mapping), read_only=True)
        self.createParameter("cmapping", value=exp(1j * self.mapping), read_only=True)
        self.createParameter("bmapping", value=2 ** np.arange(mu), read_only=True)

    def update(
        self,
        t1: float,
        t2: float,
        input: NDArray[Any, Any],
        output: NDArray[Any, Any],
    ) -> dict:
        if len(input.shape) == 1:
            n = 1
            ny = self.input_size
            input = input.reshape((ny, n))
        else:
            ny, n = input.shape

        symbols = np.empty((self.output_size, n), dtype=np.complex128)

        nb_chunk = self.input_size // self.mu
        for k in range(nb_chunk):
            chunk_bits = input[k * self.mu : (k + 1) * self.mu, :]
            chunk_idx = self.bmapping @ chunk_bits
            chunk_symb = self.cmapping[chunk_idx]
            symbols[k * self.p_samp : (k + 1) * self.p_samp, :] = chunk_symb

        if n == 1:
            symbols = symbols.reshape(self.output_size)

        outputs = {}
        outputs["output"] = symbols
        return outputs

    def plotConstellation(self, axe: BAxe):
        """Plots the PSK constellation on a matplotlib axe

        Args:
            axe: The axe to draw on

        """
        theta = np.linspace(0, 2 * pi, 100)
        x_circle = cos(theta)
        y_circle = sin(theta)

        axe.kwargs["aspect"] = "equal"
        axe.plot(
            plottable=(cos(self.mapping), sin(self.mapping)),
            color="red",
            marker="o",
            label="constellation",
            linestyle="",
        )
        axe.plot(plottable=(x_circle, y_circle), color="black", linestyle="--")


class PSKDemapping(ADSPComputer):
    """Phase Shift Keying demodulator.
    To determine the type of modulation, the *mapping* argument must tell what keying is used :
    For example, in 8-PSK, the block '010' in decimal is 2. So the phase is mapping[2], and the symbol is exp(1j*mapping[2])
    The number of values in *mapping* shall be a power of 2: len(mapping) == 2**mu.

    To initiate the demodulator, 1 symbol shall be processed to store the mu bits.
    Then the following mu samples are the dequeing of the stored samples.
    The symbols on the input are not supposed to change during this period

    Args:
        name: Name of the computer
        mapping: List of phase values. For example, in QPSK, [pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4]
        output_size: Number of bits computed in parallel. The size of the input vector shall therefore be output_size//mu

    """

    __slots__ = []

    def __init__(self, name: str, mapping: list, output_size: int = 1):
        mu = int(np.round(log2(len(mapping)), 0))
        if 2**mu != len(mapping):
            raise ValueError(
                "[%s]Mapping size must be a power of 2. Got %i" % (name, len(mapping))
            )

        if output_size % mu != 0:
            raise ValueError(
                "[%s]output_size (=%i) must be divisible by mu (=%i)"
                % (name, output_size, mu)
            )

        ADSPComputer.__init__(
            self,
            name=name,
            input_name="input",
            output_name="output",
            input_size=output_size // mu,
            output_size=output_size,
            input_dtype=np.complex128,
            output_dtype=np.int64,
        )

        self.createParameter("mu", value=mu, read_only=True)
        self.createParameter("mapping", value=np.array(mapping), read_only=True)
        self.createParameter("cmapping", value=exp(1j * self.mapping), read_only=True)

    def update(
        self,
        t1: float,
        t2: float,
        input: NDArray[Any, Any],
        output: NDArray[Any, Any],
    ) -> dict:
        if len(input.shape) == 1:
            n = 1
            ny = self.input_size
            input = input.reshape((ny, n))
        else:
            ny, n = input.shape

        bits = np.empty((self.output_size, n), dtype=np.int64)

        for kech in range(n):
            for k in range(self.input_size):
                symb_in = input[k, kech]
                isymb = np.argmin(np.abs(self.cmapping - symb_in))
                ss = bin(isymb)[2:].zfill(self.mu)
                symb_bits = np.array([int(x) for x in ss[::-1]], dtype=np.int64)
                bits[k * self.mu : (k + 1) * self.mu, kech] = symb_bits

        if n == 1:
            bits = bits.reshape(self.output_size)

        outputs = {}
        outputs["output"] = bits
        return outputs
