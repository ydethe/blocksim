from .Frame import Frame
from .Node import Input, Output, AComputer


class Simulation(object):
    def __init__(self):
        self.__computers = []

    def addComputer(self, computer: AComputer):
        self.__computers.append(computer)

    def getComputerByName(self, name: str) -> AComputer:
        for c in self.__computers:
            if c.getName() == name:
                return c

    def reset(self, frame: Frame):
        for c in self.__computers:
            c.reset(frame)

    def connect(self, src_name: str, dst_name: str):
        src_comp_name, src_out_name = src_name.split(".")
        dst_comp_name, dst_in_name = dst_name.split(".")

        src = self.getComputerByName(src_comp_name)
        dst = self.getComputerByName(dst_comp_name)

        otp = src.getOutputByName(src_out_name)
        inp = dst.getInputByName(dst_in_name)
        inp.setOutput(otp)
