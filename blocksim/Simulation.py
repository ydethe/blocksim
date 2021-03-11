import numpy as np

from .core.Frame import Frame
from .core.Node import Input, Output, AComputer
from .Logger import Logger


class Simulation(object):
    def __init__(self):
        self.__computers = []
        self.__logger = Logger()

    def addComputer(self, computer: AComputer):
        # Controllers shall be updated last
        if computer.isController():
            self.__computers.append(computer)
        else:
            self.__computers.insert(0, computer)

    def getComputerByName(self, name: str) -> AComputer:
        for c in self.__computers:
            if c.getName() == name:
                return c

    def reset(self, frame: Frame):
        self.__logger.reset()

        for c in self.__computers:
            c.reset(frame)

        self.update(frame)

    def update(self, frame: Frame):
        t = frame.getStartTimeStamp()
        self.__logger.log(name="t", val=t)

        # Controllers shall be updated last
        for c in self.__computers:
            c_name = c.getName()
            for oid in c.getListOutputsIds():
                otp = c.getOutputById(oid)
                o_name = otp.getName()
                data = otp.getDataForFrame(frame)

                for k, x in enumerate(data):
                    self.__logger.log(
                        name="%s_%s_%i" % (c_name, o_name, k), val=data[k]
                    )

    def simulate(self, tps: np.array):
        frame = Frame(start_timestamp=tps[0], stop_timestamp=tps[0])
        self.reset(frame)

        ns = len(tps)
        for k in range(1, ns):
            dt = tps[k] - tps[k - 1]
            frame.updateByStep(dt)
            self.update(frame)

    def getLogger(self) -> Logger:
        return self.__logger

    def connect(self, src_name: str, dst_name: str):
        src_comp_name, src_out_name = src_name.split(".")
        dst_comp_name, dst_in_name = dst_name.split(".")

        src = self.getComputerByName(src_comp_name)
        dst = self.getComputerByName(dst_comp_name)

        otp = src.getOutputByName(src_out_name)
        inp = dst.getInputByName(dst_in_name)
        inp.setOutput(otp)
