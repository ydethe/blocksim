from collections import deque

import numpy as np

__all__ = ["DSPFilter"]


class CircularBuffer(deque):
    """Circular buffer. Initially filled with 0

    Args:
      size
        Number of elements

    """

    def __init__(self, size: int):
        super(CircularBuffer, self).__init__(maxlen=size)
        for _ in range(size):
            self.append(0.0)

    def getAsArray(self, dtype) -> np.array:
        """Returns the content of the buffer in chronological order

        Args:
          dtype
            Type to cast the elements to. Ex. : np.int64

        Returns:
          A numpy array of the elements

        Examples:
          >>> a = CircularBuffer(size=5)
          >>> a.append(1)
          >>> a.append(2)
          >>> a.getAsArray(dtype=np.int64)
          array([0, 0, 0, 1, 2], dtype=int64)

        """
        return np.array(self, dtype=dtype)
