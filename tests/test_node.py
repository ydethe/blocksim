import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from blocksim.core.Node import Frame, Input, Output, AComputer
from blocksim.Simulation import Simulation


class SetPoint(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        otp = self.defineOutput("setpoint")
        otp.setInitialState(np.array([1]))

    def updateAllOutput(self, frame: Frame):
        print("update %s..." % self.getName())


class Controller(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("setpoint")
        self.defineInput("estimation")
        otp = self.defineOutput("command")
        otp.setInitialState(np.array([2]))

    def updateAllOutput(self, frame: Frame):
        print("update %s..." % self.getName())
        data = np.array([0])
        for iid in self.getListInputsIds():
            inp = self.getInputById(iid)
            idat = inp.getDataForFrame(frame)
            data += idat
            print("got data from %s.%s: %i" % (self.getName(), inp.getName(), idat[0]))

        for oid in self.getListOutputsIds():
            otp = self.getOutputById(oid)
            otp.setData(data)
            print("set data for %s.%s : %i" % (self.getName(), otp.getName(), data[0]))


class System(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("command")
        otp = self.defineOutput("output")
        otp.setInitialState(np.array([4]))

    def updateAllOutput(self, frame: Frame):
        print("update %s..." % self.getName())
        data = np.array([0])
        for iid in self.getListInputsIds():
            inp = self.getInputById(iid)
            idat = inp.getDataForFrame(frame)
            data += idat
            print("got data from %s.%s: %i" % (self.getName(), inp.getName(), idat[0]))

        for oid in self.getListOutputsIds():
            otp = self.getOutputById(oid)
            otp.setData(data)
            print("set data for %s.%s : %i" % (self.getName(), otp.getName(), data[0]))


class TestNode(TestBase):
    def test_node_dag(self):
        stp = SetPoint("stp")
        ctl = Controller("ctl")
        sys = System("sys")

        sim = Simulation()
        sim.addComputer(stp)
        sim.addComputer(ctl)
        sim.addComputer(sys)

        sim.connect(src_name="ctl.command", dst_name="sys.command")
        sim.connect(src_name="sys.output", dst_name="ctl.estimation")
        sim.connect(src_name="stp.setpoint", dst_name="ctl.setpoint")

        frame = Frame()
        sim.reset(frame)

        otp = sys.getOutputByName("output")
        oid1 = otp.getID()
        otp = ctl.getOutputByName("command")
        oid2 = otp.getID()

        ref_out = np.array([5])

        frame = Frame(start_timestamp=0, stop_timestamp=0.1)
        sys_out = sys.getDataForOutput(frame, oid1)
        ctl_out = ctl.getDataForOutput(frame, oid2)

        self.assertAlmostEqual(np.abs(sys_out - ref_out), 0, delta=1e-10)
        self.assertAlmostEqual(np.abs(ctl_out - ref_out), 0, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
