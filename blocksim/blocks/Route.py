from typing import Iterable
from itertools import product

import numpy as np
from scipy import linalg as lin

from ..core.Frame import Frame
from ..core.Node import AComputer


__all__ = ["IQExtract", "Split", "Group", "Clip", "Multiplier"]


class IQExtract(AComputer):
    """From a 1 complex element input, outputs a 2 real elements output.
    The 2 components are real part and imaginary part respectively

    The input of the computer is *signal*
    The output of the computer is *iq*

    Args:
      name
        Name of the element

    Examples:
      >>> iqe=IQExtract('iqe')
      >>> out = iqe.compute_outputs(t1=0,t2=1,signal=np.array([1-2*1j]),iq=np.array([0]))
      >>> out['iq']
      array([ 1., -2.]...

    """

    __slots__ = []

    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("signal", shape=(1,), dtype=np.complex128)
        self.defineOutput("iq", snames=["s_i", "s_q"], dtype=np.float64)

    def compute_outputs(
        self, t1: float, t2: float, signal: np.array, iq: np.array
    ) -> dict:
        (z,) = signal

        outputs = {}
        outputs["iq"] = np.array([np.real(z), np.imag(z)])

        return outputs


class Clip(AComputer):
    """Clips the values of a signal

    The input of the element is *signal*
    The output of the computer is *clipped*

    Args:
      name
        Name of the element
      signal_shape
        Shape of the input data
      snames
        Name of each of the scalar components of the clipped data.
        Its shape defines the shape of the data
      clipping_values
        Dictionary of the clipping values :

        * keys : number of the input parameters that is clipped
        * values : tuple of min and max value. None means no limit
      name_of_outputs
        Names of the outputs of the element

    Examples:
      >>> clp = Clip("clp", signal_shape=(1,), clipping_values={(1,): (None, 1)}, snames=["c0", "c1"])
      >>> out = clp.compute_outputs(t1=0, t2=1, signal=np.array([3, 3]), clipped=np.zeros(2))
      >>> out["clipped"]
      array([3., 1.]...

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        signal_shape: tuple,
        snames: Iterable[str],
        clipping_values: dict,
        dtype=np.float64,
    ):
        AComputer.__init__(
            self,
            name,
        )
        self.defineInput("signal", shape=signal_shape, dtype=dtype)
        self.defineOutput("clipped", snames=snames, dtype=dtype)
        self.createParameter("clipping_values", value=clipping_values)

    def compute_outputs(
        self, t1: float, t2: float, signal: np.array, clipped: np.array
    ) -> dict:
        res = np.empty(clipped.shape, clipped.dtype)

        # Creating iterables, to handle the case where
        # the output 'setpoint' is a matrix
        it = []
        for k in clipped.shape:
            it.append(range(k))

        # Iterate over all dimensions
        for iscal in product(*it):
            if iscal in self.clipping_values.keys():
                a_min, a_max = self.clipping_values[iscal]
                x = np.clip(signal[iscal], a_min, a_max)
            else:
                x = signal[iscal]
            res[iscal] = x

        outputs = {}
        outputs["clipped"] = res

        return outputs


class Split(AComputer):
    """
    Splits one signal into n

    The input of the element is *signal*
    The output of the element is *split*

    Args:
      name
        Name of the element
      signal_shape
        Shape of the input data
      snames
        Name of each of the scalar components of the clipped data.
        Its shape defines the shape of the data
      selected_input
        Number of the input parameters that will be present in the output signal

    Examples:
      >>> spt = Split("spt", signal_shape=(5,), snames=["ss0", "ss2"], selected_input=[0, 2])
      >>> out = spt.compute_outputs(t1=0, t2=1, signal=np.arange(5), split=np.zeros(2))
      >>> out["split"]
      array([0, 2]...

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        signal_shape: tuple,
        snames: Iterable[str],
        selected_input: Iterable[int],
        dtype=np.float64,
    ):
        AComputer.__init__(self, name)
        self.defineInput("signal", shape=signal_shape, dtype=dtype)
        self.defineOutput("split", snames=snames, dtype=dtype)
        self.createParameter("selected_input", value=list(selected_input))

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        signal: np.array,
        split: np.array,
    ) -> dict:
        res = np.array(signal[self.selected_input], dtype=signal.dtype)

        outputs = {}
        outputs["split"] = res

        return outputs


class Group(AComputer):
    """
    Groups n signals into one

    Args:
      name
        Name of the element
      inputs
        Dictionary of inputs :
         * the keys are the names of the inputs
         * the values are the corresponding shape
      snames
        Name of each of the scalar components of the setpoint.
        Its shape defines the shape of the data

    Examples:
      >>> grp = Group("grp", snames=["gs1", "gs2"], inputs={"s1": (1,), "s2": (1,)})
      >>> out = grp.compute_outputs(t1=0, t2=1, grouped=np.zeros(2), s1=np.array([2]), s2=np.array([-1]))
      >>> out["grouped"]
      array([ 2., -1.]...

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        inputs: dict,
        snames: Iterable[str],
        dtype=np.float64,
    ):
        AComputer.__init__(self, name)
        for k in inputs.keys():
            shape = inputs[k]
            self.defineInput(k, shape=shape, dtype=dtype)
        self.defineOutput("grouped", snames=snames, dtype=dtype)

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        grouped: np.array,
        **inputs,
    ) -> dict:
        res = []
        for name in inputs.keys():
            u = inputs[name]
            res.extend(u.flat)

        outputs = {}
        outputs["grouped"] = np.array(res, dtype=grouped.dtype)

        return outputs


class Multiplier(AComputer):
    """
    Multiplies all the inputs by a coefficient

    The input of the element is *signal*
    The output of the element is *multiplied*

    Args:
      name
        Name of the element
      coeff
        Coefficient

    Examples:
      >>> mul = Multiplier("mul", coeff=2 * np.ones((2, 2)))
      >>> out = mul.compute_outputs(t1=0, t2=1, multiplied=np.ones((2, 2)), signal=np.ones(2))
      >>> out["multiplied"]
      array([[2., 2.],
             [2., 2.]]...

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        coeff: Iterable[float],
        dtype=np.float64,
    ):
        AComputer.__init__(self, name)
        self.createParameter("coeff", value=np.array(coeff, dtype=dtype))
        ns = self.coeff.shape
        self.defineInput("signal", shape=ns, dtype=dtype)

        # Creating iterables, to handle the case where
        # the output 'setpoint' is a matrix
        it = []
        for k in ns:
            it.append(range(k))

        # Iterate over all dimensions
        ndig = 1 + int(np.ceil(np.log10(max(ns))))
        sze = (1 + ndig) * len(ns)
        snames = np.empty(ns, dtype="<U%i" % sze)
        for iscal in product(*it):
            snames[iscal] = "m" + "_".join([str(x) for x in iscal])

        self.defineOutput("multiplied", snames=snames, dtype=dtype)

    def compute_outputs(
        self,
        t1: float,
        t2: float,
        signal: np.array,
        multiplied: np.array,
    ) -> dict:
        res = signal * self.coeff

        outputs = {}
        outputs["multiplied"] = res

        return outputs
