"""Simulation classs implementation

"""

from typing import Iterable, Tuple, Any
from uuid import UUID, uuid4
from time import time
from datetime import datetime

import tqdm
from nptyping import NDArray
import numpy as np
import matplotlib.animation as animation
import networkx as nx

from .exceptions import *
from .core.Node import AComputer
from .loggers.Logger import Logger
from . import logger

__all__ = ["Simulation"]


class Simulation(object):
    """Class which handles the closed-loop simulation

    Also logs all the simulated values

    Args:
        computers: list of `blocksim.core.Node.AComputer` to add to the simulation. Can be done later with `blocksim.Simulation.addComputer`

    """

    __slots__ = ["__logger", "__graph"]

    def __init__(self, *computers: Iterable[AComputer]):
        self.__logger = Logger()
        self.__graph = nx.MultiDiGraph()
        for c in computers:
            self.addComputer(c)

    def iterComputersList(self) -> Iterable[AComputer]:
        """Returns the list of all the computers of the simulation.

        Returns:
            The list of all the computers in the simulation

        Examples:
            >>> from blocksim.core.Node import DummyComputer
            >>> el = DummyComputer('el')
            >>> sim = Simulation()
            >>> sim.addComputer(el)
            >>> for e in sim.iterComputersList():
            ...     print(e.getName())
            el

        """
        for n, data in self.__graph.nodes(data=True):
            comp = data["computer"]
            yield comp

    def addComputer(self, *computers: Iterable[AComputer]):
        """Adds one or several computers to the simulation

        Args:
            computers: List of AComputer to be added

        Raises:
            DuplicateElement: If one of the computers is already in the simulation

        """
        for comp in computers:
            name = comp.getName()

            if "_" in name:
                raise ValueError(
                    "Cannot log variables with '_' in their name (got '%s')" % name
                )

            for c in self.iterComputersList():
                if c.getName() == name:
                    raise DuplicateElement(c.getName())

            self.__graph.add_node(name, computer=comp)

    def removeComputer(self, *computers: Iterable[AComputer]):
        """Removes one or several computers to the simulation.
        If a computer in the given list is not in the Simulation, it is ignored

        Args:
            computers: List of AComputer to be removed

        """
        for comp in computers:
            name = comp.getName()

            self.__graph.remove_node(name)

    def getComputerByName(self, name: str) -> AComputer:
        """Returns the computer named *name*

        Args:
            name: Name of the computer. If it does not exist, raises KeyError

        Raises:
            KeyError: If no computer has the name *name*

        """
        data = dict(self.__graph.nodes(data="computer", default=None))

        comp = data[name]

        return comp

    def getComputerById(self, cid: UUID) -> AComputer:
        """Returns the computer with given id

        Args:
            cid: Id of the computer. If it does not exist, raises KeyError

        Returns
            The computer with id *cid*

        Raises:
            KeyError: If no element has the id *cid*

        """
        for c in self.iterComputersList():
            if c.getID() == cid:
                return c

        raise KeyError(cid)

    def getParentComputers(self, cname: str) -> Iterable[AComputer]:
        for p in self.__graph.predecessors(cname):
            yield self.getComputerByName(p)

    def getInputStream(self, cname: str) -> Iterable[Tuple[AComputer, str, str]]:
        """Iterates over the source computer and the connected ports

        Yields:
            A (src,oport,iport) tuple, where:

            * src is the source computer
            * oport is the name of the connected source's output
            * iport is the name of the connected computer's input

        """
        for (u, v, ddict) in self.__graph.in_edges(nbunch=cname, data=True):
            comp = self.getComputerByName(u)
            yield comp, ddict["output_port"], ddict["input_port"]

    def __init_sim(
        self,
        clist: Iterable[str],
        t0: float,
        error_on_unconnected: bool = True,
    ):
        self.__logger.setStartTime(datetime.utcnow())

        for cname in clist:
            comp = self.getComputerByName(cname)
            comp.resetCallback(t0)

        self.update(
            clist,
            t0,
            t0,
            error_on_unconnected=error_on_unconnected,
            noexc=True,
            nolog=True,
        )

        # This loop keeps track of the state of the AComputer
        # Then it resets the AComputer
        # Then it restores the state of the AComputer.
        # This allows reseting the DelayLines or CircularBuffers of any other internal state of the AComputer
        for cname in clist:
            comp = self.getComputerByName(cname)
            odata = {}
            for otp in comp.getListOutputs():
                odata[otp.getName()] = otp.getData()
            comp.resetCallback(t0)
            for oname in odata.keys():
                otp = comp.getOutputByName(oname)
                otp.setData(odata[oname])

        self.update(
            clist,
            t0,
            t0,
            error_on_unconnected=error_on_unconnected,
            noexc=True,
            nolog=False,
        )

    def update(
        self,
        clist: Iterable[str],
        t1: float,
        t2: float,
        error_on_unconnected: bool = True,
        tindex: int = None,
        noexc: bool = False,
        nolog: bool = False,
    ):
        """Steps the simulation, and logs all the outputs of the computers

        Args:
            clist: List of the computers' name, topologically ordered
            t1: Current simulation time (s)
            t2: New simulation time (s)
            error_on_unconnected: True to raise an exception is an input is not connected.
                If an input is not connected and error_on_unconnected is False, the input will be padded with zeros

        """
        for cname in clist:
            comp = self.getComputerByName(cname)
            idata = {}
            for src, oport, iport in self.getInputStream(cname):
                idata[iport] = src.getDataForOutput(oport)

            # Check completness of input data
            for itp in comp.getListInputs():
                iname = itp.getName()
                if not iname in idata.keys():
                    if error_on_unconnected:
                        raise UnconnectedInput(cname, iname)
                    else:
                        idata[iname] = itp.getDefaultInputData()

            # Gives output data for computers that need self looping
            for otp in comp.getListOutputs():
                oname = otp.getName()
                idata[oname] = otp.getData()

            odata = None
            try:
                odata = comp.update(t1, t2, **idata)
            except BaseException as e:
                if noexc:
                    pass
                else:
                    logger.error(f"While updating {cname}")
                    raise e

            for otp in comp.getListOutputs():
                oname = otp.getName()
                if not odata is None:
                    otp.setData(odata[oname])
                for sname, unit, x in otp.iterScalarParameters():
                    if comp.isLogged and not nolog:
                        pname = self.__logger.buildParameterNameFromComputerElements(
                            cname, oname, sname
                        )
                        self.__logger.log(name=pname, val=x, unit=unit, tindex=tindex)

        if not nolog:
            self.__logger.log(name="t", val=t2, unit="s", tindex=tindex)

    def simulate(
        self,
        tps: NDArray[Any, Any],
        progress_bar: bool = True,
        error_on_unconnected: bool = True,
        fig=None,
    ) -> "FuncAnimation":
        """Resets the simulator, and simulates the closed-loop system
        up to the date given as an argument :

        * As much calls to update as time samples in the *tps* argument

        Args:
            tps: Dates to be simulated (s)
            progress_bar: True to display a progress bar in the terminal
            error_on_unconnected: True to raise an exception is an input is not connected. If an input is not connected and error_on_unconnected is False, the input will be padded with zeros
            fig: In the case of a realtime plot (use of RTPlotter for example), must be the figure that is updated in real time

        Returns:
            The matplotlib FuncAnimation if fig is not None

        """
        # Remove cycles in Simulation graph
        if nx.is_directed_acyclic_graph(self.__graph) and fig is None:
            sg = self.__graph
            # TODO: activate the parallelization options
            parallel = True
            progress_bar = False
        else:
            sg = self.computeAcyclicGraph()
            parallel = False
        clist = list(nx.topological_sort(sg))

        self.__logger.reset()
        self.__logger.allocate(size=len(tps))

        t0 = tps[0]

        self.__init_sim(clist, t0, error_on_unconnected=error_on_unconnected)

        if progress_bar and fig is None:
            itr = tqdm.tqdm(range(len(tps) - 1))
        else:
            itr = range(len(tps) - 1)

        def _anim_func(k):
            self.update(
                clist, tps[k], tps[k + 1], error_on_unconnected=error_on_unconnected
            )

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
        Both src and dst must have been added with `blocksim.Simulation.addComputer`

        Args:
            src_name: Source computer. Example: sys.output
            dst_name: Target computer. Example: ctl.estimation

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

        self.__graph.add_edge(
            src_comp_name,
            dst_comp_name,
            output_port=src_out_name,
            input_port=dst_in_name,
            key=uuid4(),
        )

    def computeGraph(self) -> nx.MultiDiGraph:
        """Computes the simulation graph. The result can be plotted thanks to `blocksim.graphics.plotGraph`

        Returns:
            The simulation graph as an instance of nx.MultiDiGraph

        """
        return self.__graph.copy()

    def computeAcyclicGraph(self) -> nx.MultiDiGraph:
        """Computes the simulation DAG. The result can be plotted thanks to `blocksim.graphics.plotGraph`

        Returns:
            The simulation DAG as an instance of nx.MultiDiGraph

        """
        sg = self.__graph.copy()

        while not nx.is_directed_acyclic_graph(sg):
            cycle = nx.find_cycle(sg, orientation="original")
            nb = None
            for u, v, key, _ in cycle:
                inb = sg.number_of_edges(u, v)
                if nb is None:
                    nb = inb
                if inb <= nb:
                    nb = inb
                    rkey = key
                    ru = u
                    rv = v
            sg.remove_edge(ru, rv, key=rkey)

        return sg

    def getComputerOutputByName(self, name: str) -> NDArray[Any, Any]:
        """Returns the data of the computer's output
        The *name* of the data is designated by :
        <computer>.<output>[coord]

        * computer is the name of the computer (str)
        * output is the name of the output (str)
        * coord is optional, and is the number of the scalar in the data vector

        Args:
            name: Name of the data. If it does not exist, raises KeyError

        Returns:
            The requested data

        Raises:
            KeyError: If the data cannot be found

        Examples:
            >>> from blocksim.core.Node import DummyComputer
            >>> el = DummyComputer('el', with_input=False)
            >>> sim = Simulation()
            >>> sim.addComputer(el)
            >>> sim.simulate([0], progress_bar=False)
            >>> sim.getComputerOutputByName('el.xout')
            array([0]...
            >>> sim.getComputerOutputByName('el.xout[0]')
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
        data = otp.getData()
        if idx is None:
            return data
        else:
            return data[idx]
