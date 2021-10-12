import numpy as np
from numpy import sqrt, sign, pi, exp, cos, sin, log2

from .. import logger
from .ADSPComputer import ADSPComputer
from .CircularBuffer import CircularBuffer
from .DSPSpectrum import DSPSpectrum
from .DSPSignal import DSPSignal


class PSKMapping(ADSPComputer):
    """Phase Shift Keying modulator.
    To determine the type of modulation, the *mapping* argument must tell what keying is used :
    For example, in 8-PSK, the block '010' in decimal is 2. So the phase is mapping[2], and the symbol is exp(1j*mapping[2])
    The number of values in *mapping* shall be a power of 2: len(mapping) == 2**mu.

    To initiate the modulator, mu-1 samples shall be processed. The mu-th sample will trigger the conversion of the mu previous samples into a symbol.
    Then the following mu-1 samples are memorised and the last symbol is repeated on the output mu-1 times.

    Args:
      name
        Name of the computer
      mapping
        List of phase values. For example, in QPSK, [pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4]
      output_size
        Number of symbols computed in parallel. The size of the input vector shall therefore be output_size*mu

    """

    __slots__ = []

    def __init__(self, name: str, mapping: list, output_size: int = 1):
        mu = int(np.round(log2(len(mapping)), 0))
        if 2 ** mu != len(mapping):
            raise ValueError("Mapping size must be a power of 2. Got %i" % len(mapping))

        ADSPComputer.__init__(
            self,
            name=name,
            input_name="input",
            output_name="output",
            input_size=output_size * mu,
            output_size=output_size,
            input_dtype=np.int64,
            output_dtype=np.complex128,
        )

        self.createParameter("mu", value=mu, read_only=True)
        self.createParameter("mapping", value=np.array(mapping), read_only=True)
        self.createParameter("cmapping", value=exp(1j * self.mapping), read_only=True)
        self.createParameter("bmapping", value=2 ** np.arange(mu), read_only=True)

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
    ) -> dict:
        if len(input.shape) == 1:
            n = 1
            ny = self.input_size
            input = input.reshape((ny, n))
        else:
            ny, n = input.shape

        symbols = np.empty((self.output_size, n), dtype=np.complex128)

        for k in range(self.output_size):
            bits = input[k * self.mu : (k + 1) * self.mu, :]
            idx = self.bmapping @ bits
            symbols[k, :] = self.cmapping[idx]

        if n == 1:
            symbols = symbols.reshape(self.output_size)

        outputs = {}
        outputs["output"] = symbols
        return outputs

    def plotConstellation(self, axe):
        """Plots the PSK constellation on a matplotlib axe

        Args:
          axe
            The axe to draw on

        """
        theta = np.linspace(0, 2 * pi, 100)
        x_circle = cos(theta)
        y_circle = sin(theta)

        axe.grid(True)
        axe.set_aspect("equal")
        axe.scatter(
            cos(self.mapping),
            sin(self.mapping),
            color="red",
            marker="o",
            label="constellation",
        )
        axe.plot(x_circle, y_circle, color="black", linestyle="--")


class PSKDemapping(ADSPComputer):
    """Phase Shift Keying demodulator.
    To determine the type of modulation, the *mapping* argument must tell what keying is used :
    For example, in 8-PSK, the block '010' in decimal is 2. So the phase is mapping[2], and the symbol is exp(1j*mapping[2])
    The number of values in *mapping* shall be a power of 2: len(mapping) == 2**mu.

    To initiate the demodulator, 1 symbol shall be processed to store the mu bits.
    Then the following mu samples are the dequeing of the stored samples.
    The symbols on the input are not supposed to change during this period

    Args:
      name
        Name of the computer
      mapping
        List of phase values. For example, in QPSK, [pi / 4, 3 * pi / 4, 5 * pi / 4, 7 * pi / 4]
      output_size
        Number of bits computed in parallel. The size of the input vector shall therefore be output_size//mu

    """

    __slots__ = []

    def __init__(self, name: str, mapping: list, output_size: int = 1):
        mu = int(np.round(log2(len(mapping)), 0))
        if 2 ** mu != len(mapping):
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

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
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
