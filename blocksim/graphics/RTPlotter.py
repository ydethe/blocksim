import numpy as np

from .. import logger
from ..core.Node import AComputer, Input


class RTPlotter(AComputer):
    """Plots the inputs in real time

    Args:
        name: Name of the element
        axe: Matplotlib axe to draw on
        input_map: Ordered dictionary of inputs:

            * the keys are the names of the inputs
            * the values are a tuple (indices of the chosen salars in the input vector, kwargs passed to plot)

    """

    __slots__ = ["__lines"]

    def __init__(self, name: str, axe, input_map: dict):
        AComputer.__init__(self, name=name, logged=False)

        snames = []
        for k in input_map.keys():
            selected_input, kwargs = input_map[k]
            self.defineInput(name=k, shape=None, dtype=np.complex128)
            snames.extend(["%s_%s" % (k, si) for si in selected_input])
        self.defineOutput(name="data", snames=snames, dtype=np.float64)

        self.createParameter(name="input_map", value=input_map, read_only=True)
        self.createParameter(name="axe", value=axe)

    def resetCallback(self, t0: float):
        super().resetCallback(t0)

        self.axe.grid(True)

        self.__lines = []
        for k in self.input_map.keys():
            selected_input, kwargs = self.input_map[k]
            for si in selected_input:
                (line,) = self.axe.plot([], [], **kwargs)
                self.__lines.append(line)

        self.axe.legend()

    def update(self, t1, t2, **inputs):
        _ = inputs.pop("data")

        res = []
        for name in inputs.keys():
            # Current simuation values for input 'name'
            u = inputs[name]

            # Subset of scalar to be kept
            selected_input, kwargs = list(self.input_map[name])
            sset = np.real(u[selected_input])

            # Extension of res
            res.extend(sset.flat)

        ymi = None
        yma = None

        # res is the vector of the scalar values to be plotted
        for x, line in zip(res, self.__lines):
            xd, yd = line.get_data()
            xd = np.hstack((xd, [t2]))
            yd = np.hstack((yd, [x]))

            if yma is None or np.max(yd) > yma:
                yma = np.max(yd)

            if ymi is None or np.min(yd) < ymi:
                ymi = np.min(yd)

            line.set_data(xd, yd)

        if yma == ymi:
            ymi = yma - 0.5
            yma = yma + 0.5

        if xd[0] == t2:
            xmi = xd[0] - 0.5
            xma = t2 + 0.5
        else:
            xmi = xd[0]
            xma = t2

        self.axe.set_xlim((xmi, xma))
        self.axe.set_ylim((ymi, yma))

        outputs = {}
        outputs["data"] = np.array(res)
        return outputs
