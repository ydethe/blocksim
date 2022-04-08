from uuid import UUID, uuid4


class Frame(object):
    """Time frame. Describes a time interval (start time and stop time)
    Contains also an UUID, which is used to determine if a `ABaseNode` is up to date.
    If not, the `ABaseNode` is update with a call to `ABaseNode.updateAllOutput`

    Args:
        start_timestamp: Time start of the frame
        stop_timestamp: Time stop of the frame

    """

    def __init__(self, start_timestamp: float = 0, stop_timestamp: float = 0):
        self.__start_timestamp = start_timestamp
        self.__stop_timestamp = stop_timestamp
        self.__id = uuid4()

    def __repr__(self):
        s = "<%s %s> start=%g, stop=%g <end>" % (
            self.__class__.__name__,
            self.getFrameID(),
            self.getStartTimeStamp(),
            self.getStopTimeStamp(),
        )
        return s

    def getStartTimeStamp(self) -> float:
        """Gets the start time of the frame

        Returns:
            Start time

        """
        return self.__start_timestamp

    def getStopTimeStamp(self) -> float:
        """Gets the stop time of the frame

        Returns:
            Stop time

        """
        return self.__stop_timestamp

    def getTimeStep(self) -> float:
        """Gets the time step of the frame :
        stop_time - start_time

        Returns:
            Time step

        """
        return self.__stop_timestamp - self.__start_timestamp

    def getFrameID(self) -> UUID:
        """Gets the id of the frame. This elements is used to test the equality between 2 frames

        Returns:
            The id of the frame

        """
        return self.__id

    def updateByStep(self, step: float):
        """Updates the frame by step
        This operation changes the frame's id, so that it triggers update when used to retrieve computers' data

        Args:
            The time step

        """
        if step == 0:
            return

        self.__start_timestamp = self.__stop_timestamp
        self.__stop_timestamp += step
        self.__id = uuid4()

    def copy(self) -> "Frame":
        """Copies the frame
        It creates a new instance of Frame

        Returns:
            The duplicated frame

        """
        res = Frame(
            start_timestamp=self.getStartTimeStamp(),
            stop_timestamp=self.getStopTimeStamp(),
        )
        res.__id = self.getFrameID()
        return res

    def __eq__(self, y: "Frame") -> bool:
        return self.getFrameID() == y.getFrameID()

    def __hash__(self):
        return self.getFrameID().int
