from pathlib import Path


import numpy as np

from blocksim.exceptions import (
    IncompatibleShapes,
    DuplicateElement,
    InvalidAssignedVector,
    DuplicateInput,
    UnknownInput,
    DenormalizedQuaternion,
)
from blocksim.Simulation import Simulation
from blocksim.control.System import LTISystem, G6DOFSystem
from blocksim.control.SetPoint import Step
from blocksim.core.Node import AComputer
from blocksim.control.Sensors import LinearSensors
from blocksim.utils import quat_to_euler, FloatArr
from blocksim.graphics.BFigure import FigureFactory
from blocksim.testing import TestBase


class DummyTestElement(AComputer):
    __slots__ = []

    def __init__(self, name, name_of_outputs, name_of_inputs, name_of_states=[]):
        AComputer.__init__(self, name)
        self.defineOutput("output", snames=name_of_outputs, dtype=np.int64)
        self.defineOutput("state", snames=name_of_states, dtype=np.int64)
        for inp in name_of_inputs:
            self.defineInput(inp, shape=(1,), dtype=np.int64)
        self.createParameter("ns", 0)
        self.createParameter("no", 0)

    def update(self, t1: float, t2: float, output: FloatArr, state: FloatArr, **inputs) -> dict:
        n = self.getOutputByName("state").getDataShape()[0]
        if self.ns == 2:
            state = "foo"
        else:
            state = np.zeros(self.ns + n)

        n = self.getOutputByName("output").getDataShape()[0]
        output = np.zeros(self.no + n)

        outputs = {}
        outputs["state"] = state
        outputs["output"] = output

        return outputs


class TestExceptions(TestBase):
    def setUp(self):
        super().setUp()

        # Init système
        self.sys = LTISystem("sys", shape_command=(1,), snames_state=["x", "v"])
        self.sys.matA = np.zeros((2, 2))
        self.sys.matA[0, 1] = 1
        self.sys.matA[1, 0] = -((2 * np.pi) ** 2)
        self.sys.matB = np.zeros((2, 1))

        self.sys.setInitialStateForOutput(np.array([1.0, 0.0]), "state")

        self.ctrl = Step("ctrl", snames=["u"], cons=np.zeros(1))

        self.sim = Simulation()

        self.sim.addComputer(self.sys)
        self.sim.addComputer(self.ctrl)

    def test_sim_exc(self):
        tmp = DummyTestElement(
            name="wrong_name",
            name_of_outputs=["a"],
            name_of_inputs=["b"],
            name_of_states=["c"],
        )
        self.assertRaises(ValueError, self.sim.addComputer, tmp)

        tmp2 = DummyTestElement(
            name="dummy",
            name_of_outputs=["a", "b"],
            name_of_inputs=["b"],
            name_of_states=["c"],
        )
        self.sim.addComputer(tmp2)
        self.assertRaises(IncompatibleShapes, self.sim.connect, "dummy.output", "sys.command")

        self.assertRaises(KeyError, self.sim.getComputerByName, "foo")

        self.assertRaises(
            UnknownInput,
            self.sim.connect,
            "ctrl.setpoint",
            "sys.foo",
        )

        self.assertRaises(DuplicateElement, self.sim.addComputer, self.sys)

        tmp2 = DummyTestElement(
            "tmp2",
            name_of_outputs=["u"],
            name_of_inputs=["in2"],
            name_of_states=["state_tmp2"],
        )

        tmp = DummyTestElement("tmp", name_of_outputs=["out1"], name_of_inputs=["in1"])
        tmp2 = DummyTestElement("tmp2", name_of_outputs=["out2"], name_of_inputs=["in2"])
        self.sim.addComputer(tmp)
        self.sim.addComputer(tmp2)
        self.sim.connect("tmp.output", "tmp2.in2")
        self.sim.connect("tmp2.output", "tmp.in1")

    def test_sys_exc(self):
        # self.assertRaises(UnknownOutput, self.sys.getTransferFunction, 0, "foo")

        sys = G6DOFSystem("6dof")
        sys.setInitialStateForOutput(np.zeros(13), "state")
        self.assertRaises(
            DenormalizedQuaternion,
            sys.update,
            0,
            1,
            command=np.zeros(6),
            state=np.zeros(13),
            euler=None,
        )

    def test_sens_exc(self):
        cpt = LinearSensors(name="cpt", shape_command=(1,), shape_state=(2,), snames=["x", "v"])
        cpt.setMean(np.zeros(2))
        self.assertRaises(ValueError, cpt.setCovariance, np.zeros((3, 2)))
        self.assertRaises(ValueError, cpt.setMean, np.zeros(5))

    def test_logger_exc(self):
        log = self.sim.getLogger()
        self.assertRaises(IOError, log.loadLogFile, Path("_dummy"))
        self.assertRaises(FileNotFoundError, log.loadLogFile, Path("_dummy.csv"))
        self.assertRaises(SystemError, log.getValue, "tps")
        log.log(name="tps", val=0, unit="")
        log.log(name="y", val=0, unit="")
        fig = FigureFactory.create(title="Figure")
        gs = fig.add_gridspec(1, 1)
        axe = fig.add_baxe(spec=gs[0, 0], title="")
        self.assertRaises(AssertionError, axe.plot, (log, "tps", None))

    def test_aelem_exc(self):
        tmp = DummyTestElement(
            "tmp",
            name_of_outputs=["out1"],
            name_of_inputs=["in1"],
            name_of_states=["state1"],
        )
        self.assertRaises(DuplicateInput, tmp.defineInput, "in1", (1,), np.int64)

        tmp = DummyTestElement(
            "tmp",
            name_of_outputs=["out1"],
            name_of_inputs=["in1"],
            name_of_states=["state1"],
        )

        tmp = DummyTestElement(
            "tmp",
            name_of_outputs=["out1"],
            name_of_inputs=["in1"],
            name_of_states=["state1"],
        )

        tmp = DummyTestElement(
            "tmp",
            name_of_outputs=["out1"],
            name_of_inputs=["in1"],
            name_of_states=["state1"],
        )

        self.assertRaises(
            InvalidAssignedVector, tmp.setInitialStateForOutput, np.zeros(3), "output"
        )
        otp = tmp.getOutputByName("output")
        self.assertRaises(InvalidAssignedVector, otp.setData, np.zeros(3))

    def test_misc_exc(self):
        quat_to_euler(1, 0, 1, 0)
        quat_to_euler(-1, 0, 1, 0)
        quat_to_euler(-1, 0, 1, 0, normalize=True)


if __name__ == "__main__":
    # unittest.main()
    # exit(0)

    a = TestExceptions()
    a.setUp()
    # a.test_sim_exc()
    a.test_logger_exc()
