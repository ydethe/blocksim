from collections import deque

import numpy as np

__all__ = ["DSPFilter"]


class CircularBuffer(deque):
    def __init__(self, size=0):
        super(CircularBuffer, self).__init__(maxlen=size)
        for _ in range(size):
            self.append(0.0)

    def getAsArray(self, dtype) -> np.array:
        return np.array(self, dtype=dtype)
