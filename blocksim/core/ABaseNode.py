from abc import ABCMeta, abstractmethod
from uuid import UUID, uuid4

import numpy as np

from .Frame import Frame


class ABaseNode(metaclass=ABCMeta):
    """This base class is the parent class for :

    * :class:`blocksim.core.Node.Input`
    * :class:`blocksim.core.Node.Output`
    * :class:`blocksim.core.Node.AComputer`

    Args:
      name
        The name of the node

    """

    def __init__(self, name: str):
        self.__name = name
        self.__id = uuid4()
        self.__current_frame = None

    def isController(self):
        """Checks if the element is derived from AController

        Returns:
          True if the element is derived from AController

        """
        from .Node import AController

        return isinstance(self, AController)

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

    def getCurrentFrame(self) -> Frame:
        """Gets the last frame used for update

        Returns:
          Last frame used for update

        """
        return self.__current_frame

    def setFrame(self, frame: Frame):
        """Sets the last frame used for update. The frame is duplicated with frame.copy()

        Args:
          frame
            Last frame used for update

        """
        self.__current_frame = frame.copy()

    @abstractmethod
    def updateAllOutput(self, frame: Frame):
        """Method used to update a Node.
        Useful only for :class:`blocksim.core.Node.AComputer` to update its outputs

        Args:
          frame
            Frame used for update

        """
        pass
