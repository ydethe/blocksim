from typing import Iterable
from uuid import UUID

import tqdm
import numpy as np
import matplotlib.animation as animation
from matplotlib import pyplot as plt

from .exceptions import *
from .core.Frame import Frame
from .core.Node import Input, Output, AComputer, DummyComputer
from .Logger import Logger
from . import logger

__all__ = ["Simulation"]


class Simulation(object):
    """Class which handles the closed-loop simulation

    Also logs all the simulated values

    Args:
        computers: list of AComputer to add to the simulation. Can be done later with `addComputer`

    """

    def __init__(self, computers:list=[]):
        self.__computers = []
        self.__logger = Logger()
        for c in computers:
            self.addComputer(c)

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
            computer: Computer to be added

        Raises:
            DuplicateElement: If the computer is already in the simulation

        """
        name = computer.getName()

        if "_" in name:
            raise ValueError(
                "Cannot log variables with '_' in their name (got '%s')" % name
            )

        for c in self.__computers:
            if c.getName() == name:
                raise DuplicateElement(c.getName())

        # Controllers shall be updated last
        if computer.isController():
            self.__computers.append(computer)
        else:
            self.__computers.insert(0, computer)

    def getComputerByName(self, name: str) -> AComputer:
        """Returns the computer named *name*

        Args:
            name: Name of the computer. If it does not exist, raises KeyError

        Raises:
            KeyError: If no computer has the name *name*

        """
        for c in self.__computers:
            if c.getName() == name:
                return c

        raise KeyError(name)

    def getComputerById(self, cid: UUID) -> AComputer:
        """Returns the computer with given id

        Args:
            cid: Id of the computer. If it does not exist, raises KeyError

        Returns
            The computer with id *cid*

        Raises:
            KeyError: If no element has the id *cid*

        """
        for c in self.__computers:
            if c.getID() == cid:
                return c

        raise KeyError(cid)

    def update(self, frame: Frame, error_on_unconnected: bool = True):
        """Steps the simulation, and logs all the outputs of the computers

        Args:
            frame: Time frame

        """
        # Controllers shall be updated last
        for c in self.__computers:
            c_name = c.getName()
            for oid in c.getListOutputsIds():
                otp = c.getOutputById(oid)
                o_name = otp.getName()
                for n, x in otp.iterScalarNameValue(
                    frame, error_on_unconnected=error_on_unconnected
                ):
                    if c.isLogged:
                        self.__logger.log(name="%s_%s_%s" % (c_name, o_name, n), val=x)

        t = frame.getStopTimeStamp()
        self.__logger.log(name="t", val=t)

    def simulate(
        self,
        tps: "array",
        progress_bar: bool = True,
        error_on_unconnected: bool = True,
        fig=None,
    ) -> Frame:
        """Resets the simulator, and simulates the closed-loop system
        up to the date given as an argument :

        * As much calls to update as time samples in the *tps* argument

        Args:
            tps: Dates to be simulated (s)
            progress_bar: True to display a progress bar in the terminal
            error_on_unconnected: True to raise an exception is an input is not connected. If an input is not connected and error_on_unconnected is False, the input will be padded with zeros
            fig: In the case of a realtime plot (use of RTPlotter for example), must be the figure that is updated in real time

        Returns:
            The time frame used for the simulation

        """
        # self.__logger.allocate(len(tps))

        frame = Frame(start_timestamp=tps[0], stop_timestamp=tps[0])

        for c in self.__computers:
            c.resetCallback(frame)

        self.update(frame, error_on_unconnected=error_on_unconnected)

        if progress_bar and fig is None:
            itr = tqdm.tqdm(range(len(tps) - 1))
        else:
            itr = range(len(tps) - 1)

        def _anim_func(k):
            dt = tps[k + 1] - tps[k]
            frame.updateByStep(dt)
            self.update(frame, error_on_unconnected=error_on_unconnected)

        if fig is None:
            for k in itr:
                _anim_func(k)
            ani = None
        else:
            dt = np.mean(np.diff(tps))
            ani = animation.FuncAnimation(
                fig,
                func=_anim_func,
                init_func=None,
                frames=itr,
                interval=1000 * dt,
                blit=False,
                repeat=False,
            )

        return ani

    def getLogger(self) -> Logger:
        """Gets the Logger used for the simulation

        Returns:
            The logger used by the simulation

        """
        return self.__logger

    def connect(self, src_name: str, dst_name: str):
        """Links an computer with another, so that the state of the source is connected to the input of the destination.
        Both src and dst must have been added with `Simulation.addComputer`

        Args:
            src_name: Source computer. Example : sys.output
            dst_name: Target computer. Example : ctl.estimation

        """
        src_comp_name, src_out_name = src_name.split(".")
        dst_comp_name, dst_in_name = dst_name.split(".")

        src = self.getComputerByName(src_comp_name)
        dst = self.getComputerByName(dst_comp_name)

        otp = src.getOutputByName(src_out_name)
        inp = dst.getInputByName(dst_in_name)

        if not inp.getDataShape() is None and not otp.getDataShape() is None:
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
            frame: The current time frame
            name: Name of the data. If it does not exist, raises KeyError

        Returns:
            The requested data

        Raises:
            KeyError: If the data cannot be found

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
