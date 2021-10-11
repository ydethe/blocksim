from typing import Callable

import numpy as np

from blocksim.core.Node import AComputer


__all__ = ["GenericComputer"]


class GenericComputer(AComputer):
    """Generic computer, that uses a vector function to transform an input

    The input of the element is *xin*
    The output of the element is *xout*

    Args:
      name
        Name of the element
      shape_in
        Shape of the input data
      shape_out
        Shape of the input data
      callable
        A callable object, use to compute the output

    Examples:
      >>> com = GenericComputer("com", shape_in=(5,), shape_out=(1,), callable=np.sum, dtype=np.int64)
      >>> out = com.compute_outputs(t1=0, t2=1, xin=np.arange(5), xout=None)
      >>> out["xout"]
      10

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        shape_in: tuple,
        shape_out: tuple,
        callable: Callable,
        dtype=np.float64,
    ):
        AComputer.__init__(self, name)
        self.defineInput("xin", shape=shape_in, dtype=dtype)
        otp = self.defineOutput(
            "xout", snames=["y%i" % i for i in range(shape_out[0])], dtype=dtype
        )
        self.setInitialStateForOutput(
            np.zeros(otp.getDataShape(), dtype=dtype), output_name="xout"
        )
        self.createParameter("callable", value=callable)

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        xin: np.array,
        xout: np.array,
    ) -> dict:
        outputs = {}
        outputs["xout"] = self.callable(xin)

        return outputs
