from abc import ABCMeta, abstractmethod
from uuid import UUID, uuid4

import numpy as np


class ABaseNode(metaclass=ABCMeta):
    """This base class is the parent class for :

    * `blocksim.core.Node.Input`
    * `blocksim.core.Node.Output`
    * `blocksim.core.Node.AComputer`

    Args:
        name: The name of the node

    """

    __slots__ = ["__name", "__id"]

    def __init__(self, name: str):
        self.__name = name
        self.__id = uuid4()

    def resetCallback(self, t0: float):
        """Resets the node

        Args:
            t0: Initial simulation time (s)

        """
        pass

    def getName(self) -> str:
        """Gets the name of the node

        Returns:
            The name of the node

        """
        return self.__name

    def getID(self) -> UUID:
        """Gets the id of the node

        Returns:
            The id of the node

        """
        return self.__id
