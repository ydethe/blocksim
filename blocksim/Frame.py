from uuid import UUID, uuid4


class Frame(object):
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
        return self.__start_timestamp

    def getStopTimeStamp(self) -> float:
        return self.__stop_timestamp

    def getTimeStep(self) -> float:
        return self.__stop_timestamp - self.__start_timestamp

    def getFrameID(self) -> UUID:
        return self.__id

    def updateByStep(self, step: float):
        if step == 0:
            return

        self.__start_timestamp = self.__stop_timestamp
        self.__stop_timestamp += step
        self.__id = uuid4()

    def copy(self) -> "Frame":
        res = Frame(
            start_timestamp=self.getStartTimeStamp(),
            stop_timestamp=self.getStopTimeStamp(),
        )
        res.__id = self.getFrameID()
        return res

    def __eq__(self, y: "Frame") -> bool:
        return self.getFrameID() == y.getFrameID()
