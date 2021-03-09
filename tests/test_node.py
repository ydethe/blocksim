import sys
import os
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from TestBase import TestBase

from blocksim.Node import Frame, Input, Output, AComputer, connect


class SetPoint(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        otp = self.defineOutput("setpoint", initial_state=np.array([1]))

    def updateAllOutput(self, frame: Frame):
        print("update %s..." % self.getName())


class Controller(AComputer):
    def __init__(self, name: str):
        AComputer.__init__(self, name)
        self.defineInput("setpoint")
        self.defineInput("estimation")
        otp = self.defineOutput("command", initial_state=np.array([2]))

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
        otp = self.defineOutput("output", initial_state=np.array([4]))

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

        connect(
            computer_src=ctl,
            output_name="command",
            computer_dst=sys,
            intput_name="command",
        )
        connect(
            computer_src=sys,
            output_name="output",
            computer_dst=ctl,
            intput_name="estimation",
        )
        connect(
            computer_src=stp,
            output_name="setpoint",
            computer_dst=ctl,
            intput_name="setpoint",
        )

        frame = Frame()
        stp.reset(frame)
        ctl.reset(frame)
        sys.reset(frame)

        otp = sys.getOutputByName("output")
        oid1 = otp.getID()
        otp = ctl.getOutputByName("command")
        oid2 = otp.getID()

        ref_out = np.array([5])

        frame = Frame(start_timestamp=0, stop_timestamp=0.1)
        sys_out = sys.getDataForOuput(frame, oid1)
        ctl_out = ctl.getDataForOuput(frame, oid2)

        self.assertAlmostEqual(np.abs(sys_out - ref_out), 0, delta=1e-10)
        self.assertAlmostEqual(np.abs(ctl_out - ref_out), 0, delta=1e-10)


if __name__ == "__main__":
    unittest.main()
