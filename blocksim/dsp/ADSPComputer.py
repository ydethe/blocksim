import numpy as np

from ..core.Node import AComputer


class ADSPComputer(AComputer):
    """Generic DSP Computer. It processes one input into one output.
    Adds a helper method process, that batch computes a set of input

    Args:
      name
        Name of the computer
      input_name
        Name of the input
      input_size
        Size of the input vector
      input_dtype
        Type of the input vector
      output_name
        Name of the output
      output_size
        Size of the output vector
      output_dtype
        Type of the output vector

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        input_name: str = "input",
        input_size: int = 1,
        input_dtype=np.complex128,
        output_name: str = "output",
        output_size: int = 1,
        output_dtype=np.complex128,
    ):
        AComputer.__init__(self, name=name)

        self.createParameter("input_size", value=input_size, read_only=True)
        self.createParameter("output_size", value=output_size, read_only=True)
        self.createParameter("input_name", value=input_name, read_only=True)
        self.createParameter("output_name", value=output_name, read_only=True)

        self.defineInput(input_name, shape=input_size, dtype=input_dtype)
        self.defineOutput(
            output_name,
            snames=["s%i" % i for i in range(output_size)],
            dtype=output_dtype,
        )

    def process(self, data: np.array) -> np.array:
        """Batch processes an input stream by calling compute_outputs.

        Args:
          data
            A stream of input data

        Returns:
          An stream of output data

        """
        if len(data.shape) == 1:
            assert len(data) % self.input_size == 0
            data = data.reshape((self.input_size, len(data) // self.input_size))

        ny, n = data.shape
        assert ny == self.input_size

        otp = self.getOutputByName(self.output_name)
        typ = otp.getDataType()

        comp_args = dict()
        comp_args[self.output_name] = None
        comp_args[self.input_name] = data
        outputs = self.compute_outputs(t1=0, t2=0, **comp_args)

        return outputs[self.output_name].astype(typ)

    def flatten(self, data: np.array) -> np.array:
        """Given a batch block of input, returns a 1D array for processing
        The first column of the block is copied, then the second and so on

        Args:
          data
            The block of data

        Returns:
          The 1D copy

        Examples:
          >>> a = DummyDSPComputer()
          >>> data = np.arange(6).reshape((3, 2))
          >>> data
          array([[0, 1],
                 [2, 3],
                 [4, 5]])
          >>> a.flatten(data)
          array([0, 2, 4, 1, 3, 5])

        """
        if len(data.shape) == 1:
            ny = len(data)
        else:
            ny, _ = data.shape
        assert ny == self.input_size

        strm = data.flatten(order="F")

        return strm

    def unflatten(self, strm: np.array) -> np.array:
        """Given a 1D flatten data, returns the matching data block
        The attribute *output_size* gives the size of the columns

        Args:
          strm
            The 1D stream of data

        Returns:
          The data block

        Examples:
          >>> a = DummyDSPComputer()
          >>> strm = np.arange(10)
          >>> a.unflatten(strm)
          array([[0, 5],
                 [1, 6],
                 [2, 7],
                 [3, 8],
                 [4, 9]])

        """
        ny = self.output_size

        assert len(strm) % ny == 0
        n = len(strm) // ny

        if n == 1:
            data = strm
        else:
            data = strm.reshape((ny, n), order="F")

        return data


class DummyDSPComputer(ADSPComputer):
    def __init__(self):
        ADSPComputer.__init__(
            self,
            name="dum",
            input_name="input",
            input_size=3,
            input_dtype=np.int64,
            output_name="output",
            output_size=5,
            output_dtype=np.int64,
        )

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        input: np.array,
        output: np.array,
    ) -> dict:
        out = np.zeros(5, dtype=np.int64)
        out[0] = input[0]
        out[2] = input[1]
        out[4] = input[2]

        outputs = {}
        outputs["output"] = out
        return outputs
