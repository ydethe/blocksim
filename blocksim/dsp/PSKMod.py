import numpy as np
from numpy import sqrt, sign, pi, exp, cos, sin, log2

from .. import logger
from ..core.Node import AComputer
from .CircularBuffer import CircularBuffer


class PSKMapping(AComputer):
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
        AComputer.__init__(self, name=name)

        mu = int(np.round(log2(len(mapping)), 0))
        if 2 ** mu != len(mapping):
            raise ValueError("Mapping size must be a power of 2. Got %i" % len(mapping))

        self.createParameter("mu", value=mu, read_only=True)
        self.createParameter("mapping", value=np.array(mapping), read_only=True)
        self.createParameter("cmapping", value=exp(1j * self.mapping), read_only=True)
        self.createParameter("output_size", value=output_size, read_only=True)

        self.defineInput("input", shape=output_size * mu, dtype=np.int64)
        self.defineOutput(
            "output",
            snames=["s%i" % i for i in range(output_size)],
            dtype=np.complex128,
        )

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
    ) -> dict:
        symbols = np.empty(self.output_size, dtype=np.complex128)

        for k in range(self.output_size):
            bits = input[k * self.mu : (k + 1) * self.mu]
            idx = 2 ** np.arange(self.mu) @ bits
            symbols[k] = self.cmapping[idx]

        outputs = {}
        outputs["output"] = symbols
        return outputs

    def process(self, data: np.array) -> np.array:
        """Processes a bitstream by calling compute_outputs.
        This proccessing removes the initial null symbol,
        due to the fact that the modulator is not initialised.

        Args:
          data
            A bitstream of length n, n even. Bits are either 0, or 1

        Returns:
          An array of QPSK symbols

        """
        ny, n = data.shape
        assert ny == self.output_size * self.mu

        res = np.empty((self.output_size, n), dtype=np.complex128)
        for k in range(n):
            outputs = self.compute_outputs(t1=0, t2=0, input=data[:, k], output=None)
            symb = outputs["output"]

            res[:, k] = symb

        return res

    def plotConstellation(self, axe):
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


class PSKDemapping(AComputer):
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
        AComputer.__init__(self, name=name)

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

        self.createParameter("mu", value=mu, read_only=True)
        self.createParameter("mapping", value=np.array(mapping), read_only=True)
        self.createParameter("cmapping", value=exp(1j * self.mapping), read_only=True)
        self.createParameter("output_size", value=output_size, read_only=True)

        self.defineInput("input", shape=1, dtype=np.complex128)
        self.defineOutput(
            "output", snames=["bit%i" % i for i in range(output_size)], dtype=np.int64
        )

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
    ) -> dict:
        bits = np.empty(self.output_size, dtype=np.int64)

        for k in range(self.output_size // self.mu):
            symb_in = input[k]
            isymb = np.argmin(np.abs(self.cmapping - symb_in))
            ss = bin(isymb)[2:].zfill(self.mu)
            symb_bits = np.array([int(x) for x in ss[::-1]], dtype=np.int64)
            bits[k * self.mu : (k + 1) * self.mu] = symb_bits

        outputs = {}
        outputs["output"] = bits
        return outputs

    def process(self, symbols: np.array) -> np.array:
        """Processes a bitstream by calling compute_outputs.
        This proccessing removes the initial null bits,
        due to the fact that the demodulator is not initialised.

        Args:
          symbols
            An array of QPSK symbols

        Returns:
          A bitstream of length n, n even. Bits are either 0, or 1

        """
        ny, n = symbols.shape
        assert ny == self.output_size // self.mu

        res = np.empty((self.output_size, n), dtype=np.int64)

        for k in range(n):
            outputs = self.compute_outputs(t1=0, t2=0, input=symbols[:, k], output=None)

            bits = outputs["output"]

            res[:, k] = bits

        return res
