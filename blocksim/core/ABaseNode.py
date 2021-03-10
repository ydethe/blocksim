from abc import ABCMeta, abstractmethod
from uuid import UUID, uuid4

import numpy as np

from .Frame import Frame


class ABaseNode(metaclass=ABCMeta):
    def __init__(self, name: str):
        self.__name = name
        self.__id = uuid4()
        self.__current_frame = None
        self.__data = np.array([])

    def isController(self):
        """Checks if the element is derived from AController

        Returns:
          True if the element is derived from AController

        """
        from .Node import AController

        return isinstance(self, AController)

    def getName(self):
        return self.__name

    def getID(self) -> UUID:
        return self.__id

    def setData(self, data: np.array):
        self.__data = data

    def getCurrentFrame(self) -> UUID:
        return self.__current_frame

    def setFrame(self, frame: Frame):
        self.__current_frame = frame.copy()

    @abstractmethod
    def updateAllOutput(self, frame: Frame):
        pass
