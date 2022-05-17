from typing import Callable, Any

from nptyping import NDArray, Shape
import numpy as np

from .Node import AComputer


__all__ = ["GenericComputer"]


class GenericComputer(AComputer):
    """Generic computer, that uses a vector function to transform an input

    The input of the element is *xin*
    The output of the element is *xout*

    Attributes:
        callable: the callable passed to __init__

    Args:
        name: Name of the element
        shape_in: Shape of the input data
        shape_out: Shape of the input data
        callable: A callable object, use to compute the output

    Examples:
        >>> com = GenericComputer("com", shape_in=(5,), shape_out=(1,), callable=np.sum, dtype_in=np.int64, dtype_out=np.int64)
        >>> out = com.update(t1=0, t2=1, xin=np.arange(5), xout=None)
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
        dtype_in=np.float64,
        dtype_out=np.float64,
    ):
        AComputer.__init__(self, name)
        self.defineInput("xin", shape=shape_in, dtype=dtype_in)
        otp = self.defineOutput(
            "xout", snames=["y%i" % i for i in range(shape_out[0])], dtype=dtype_out
        )
        self.setInitialStateForOutput(
            np.zeros(otp.getDataShape(), dtype=dtype_out), output_name="xout"
        )
        self.createParameter("callable", value=callable)

    def update(
        self,
        t1: float,
        t2: float,
        xin: NDArray[Any, Any],
        xout: NDArray[Any, Any],
    ) -> dict:
        outputs = {}
        outputs["xout"] = self.callable(xin)

        return outputs
