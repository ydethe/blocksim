from typing import Iterable
from uuid import UUID

import tqdm
import numpy as np

from .exceptions import *
from .core.Frame import Frame
from .core.Node import Input, Output, AComputer, DummyComputer
from .Logger import Logger


__all__ = ["Simulation"]


class Simulation(object):
    """Class which handles the closed-loop simulation

    Also logs all the simulated values

    """

    def __init__(self):
        self.__computers = []
        self.__logger = Logger()

    def getComputersList(self) -> Iterable[AComputer]:
        """Returns the list of all the computers of the simulation.

        Returns:
          The list of all the computers in the simulation

        Examples:
          >>> el = DummyComputer('el')
          >>> sim = Simulation()
          >>> sim.addComputer(el)
          >>> for e in sim.getComputersList():
          ...     print(e.getName())
          el

        """
        return self.__computers

    def addComputer(self, computer: AComputer):
        """Adds a computer to the simulation

        Args:
          computer
            Computer to be added

        Raises:
          DuplicateElement
            If the computer is already in the simulation

        """
        for c in self.__computers:
            if c.getName() == computer.getName():
                raise DuplicateElement(c.getName())

        # Controllers shall be updated last
        if computer.isController():
            self.__computers.append(computer)
        else:
            self.__computers.insert(0, computer)

    def getComputerByName(self, name: str) -> AComputer:
        """Returns the computer named *name*

        Args:
          name
            Name of the computer. If it does not exist, raises KeyError

        Raises:
          KeyError
            If no computer has the name *name*

        """
        for c in self.__computers:
            if c.getName() == name:
                return c

        raise KeyError(name)

    def getComputerById(self, cid: UUID) -> AComputer:
        """Returns the computer with given id

        Args:
          cid
            Id of the computer. If it does not exist, raises KeyError

        Returns
          The computer with id *cid*

        Raises:
          KeyError
            If no element has the id *cid*

        """
        for c in self.__computers:
            if c.getID() == cid:
                return c

        raise KeyError(cid)

    def update(self, frame: Frame):
        """Steps the simulation, and logs all the outputs of the computers

        Args:
          frame
            Time frame

        """
        t = frame.getStopTimeStamp()
        self.__logger.log(name="t", val=t)

        # Controllers shall be updated last
        for c in self.__computers:
            c_name = c.getName()
            for oid in c.getListOutputsIds():
                otp = c.getOutputById(oid)
                o_name = otp.getName()

                for n, x in otp.iterScalarNameValue(frame):
                    self.__logger.log(name="%s_%s_%s" % (c_name, o_name, n), val=x)

    def simulate(self, tps: np.array, progress_bar: bool = True) -> Frame:
        """Resets the simulator, and simulates the closed-loop system
        up to the date given as an argument :

        * As much calls to update as time samples in the *tps* argument

        Args:
          tps
            Dates to be simulated (s)
          progress_bar
            True to display a progress bar in the terminal

        Returns:
          The time frame used for the simulation

        """
        frame = Frame(start_timestamp=tps[0], stop_timestamp=tps[0])
        self.update(frame)

        if progress_bar:
            itr = tqdm.tqdm(range(len(tps) - 1))
        else:
            itr = range(len(tps) - 1)

        for k in itr:
            dt = tps[k + 1] - tps[k]
            frame.updateByStep(dt)
            self.update(frame)

        return frame

    def setOutputLoggerFile(self, fic: str, binary: bool = False):
        """Sets a file to write the logs in

        Args:
          fic
            File to write the logs in
          binary
            True to write a binary log

        """
        self.__logger.setOutputLoggerFile(fic, binary)

    def getLogger(self) -> Logger:
        """Gets the Logger used for the simulation

        Returns:
          The logger used by the simulation

        """
        return self.__logger

    def connect(self, src_name: str, dst_name: str):
        """Links an computer with another, so that the state of the source is connected to the input of the destination.
        Both src and dst must have been added with :class:`blocksim.Simulation.Simulation.addComputer`

        Args:
          src_name
            Source computer. Example : sys.output
          dst_name
            Target computer. Example : ctl.estimation

        """
        src_comp_name, src_out_name = src_name.split(".")
        dst_comp_name, dst_in_name = dst_name.split(".")

        src = self.getComputerByName(src_comp_name)
        dst = self.getComputerByName(dst_comp_name)

        otp = src.getOutputByName(src_out_name)
        inp = dst.getInputByName(dst_in_name)

        if inp.getDataShape() != otp.getDataShape():
            raise IncompatibleShapes(
                src_name, otp.getDataShape(), dst_name, inp.getDataShape()
            )

        inp.setOutput(otp)

    def getComputerOutputByName(self, frame: Frame, name: str) -> np.array:
        """Returns the data of the computer's output
        The *name* of the data is designated by :
        <computer>.<output>[coord]

        * computer is the name of the computer (str)
        * output is the name of the output (str)
        * coord is optional, and is the number of the scalar in the data vector

        Args:
          frame
            The current time frame
          name
            Name of the data. If it does not exist, raises KeyError

        Returns:
          The requested data

        Raises:
          KeyError
            If the data cannot be found

        Examples:
          >>> el = DummyComputer('el', with_input=False)
          >>> sim = Simulation()
          >>> sim.addComputer(el)
          >>> frame = Frame()
          >>> sim.getComputerOutputByName(frame, 'el.xout')
          array([0]...
          >>> sim.getComputerOutputByName(frame, 'el.xout[0]')
          0

        """
        comp_name, out_name = name.split(".")
        if "[" in out_name:
            out_name, idx = out_name.split("[")
            idx = int(idx[:-1])
        else:
            idx = None

        c = self.getComputerByName(comp_name)
        otp = c.getOutputByName(out_name)
        data = otp.getDataForFrame(frame)
        if idx is None:
            return data
        else:
            return data[idx]


if __name__ == "__main__":
    import doctest

    doctest.testmod()
