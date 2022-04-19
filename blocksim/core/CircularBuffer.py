from collections import deque
import logging

import numpy as np

__all__ = ["CircularBuffer"]


class CircularBuffer(object):
    """Circular buffer. Initially filled with 0

    Args:
        size: Number of elements
        dtype: Type of the elements to. Ex. : np.float64

    """

    __slots__ = [
        "__fill_with",
        "__size",
        "__dtype",
        "__buffer",
        "__offset",
        "__last_search_idx",
        "__nb_inserted_element",
    ]

    def __init__(self, size: int, dtype=np.float64, fill_with=0.0):
        self.__size = size
        self.__dtype = dtype
        self.__buffer = np.empty(size, dtype=dtype)
        self.__buffer[:] = fill_with
        self.__fill_with = fill_with
        self.__offset = 0  # Index where the next element will be stored
        self.__last_search_idx = -1
        self.__nb_inserted_element = 0

    def _setInternals(self, last_search_idx=None, offset=None):
        if not last_search_idx is None:
            self.__last_search_idx = last_search_idx
        if not offset is None:
            self.__offset = offset

    def _getBuffer(self) -> "array":
        return self.__buffer.copy()

    def doubleBufferSize(self):
        """Doubles the size of the buffer

        Examples:
            >>> a = CircularBuffer(size=5, dtype=np.int64, fill_with=99)
            >>> for k in range(9):
            ...    a.append(k)
            >>> a.getAsArray() # doctest: +ELLIPSIS
            array([4, 5, 6, 7, 8]...
            >>> a.doubleBufferSize()
            >>> a.getAsArray() # doctest: +ELLIPSIS
            array([99, 99, 99, 99, 99,  4,  5,  6,  7,  8]...

        """
        # logger = logging.getLogger("blocksim_logger")
        # logger.debug("doubleBufferSize: -> %s" % (self.__size * 2))

        nz = np.empty(self.__size, dtype=self.__dtype)
        nz[:] = self.__fill_with

        self.__last_search_idx += self.__size

        if self.__offset == 0:
            new_buf = np.hstack((self.__buffer, nz))
            self.__offset = self.__size
            self.__size *= 2
        else:
            new_buf = np.hstack(
                (self.__buffer[: self.__offset], nz, self.__buffer[self.__offset :])
            )
            self.__size *= 2

        self.__buffer = new_buf

    def search(self, value) -> int:
        """Searches a value in the buffer

        Args:
            The value to search

        Returns:
            The index i such that a[i] <= value < a[i+1].
            Returns -99 if the value is not in the range of the buffer

        Examples:
            >>> a = CircularBuffer(size=5, dtype=np.int64, fill_with=99)
            >>> for k in range(9):
            ...    a.append(k)
            >>> a.getAsArray() # doctest: +ELLIPSIS
            array([4, 5, 6, 7, 8]...
            >>> a.search(5.1)
            1
            >>> a.search(5.1)
            1
            >>> a.search(3.9)
            -99
            >>> a.search(4)
            0
            >>> a.search(4.1)
            0
            >>> a.search(7.9)
            3
            >>> a.search(8)
            3
            >>> a.search(8.1)
            -99

        """
        if self.inserted_elements == 0:
            return -99

        iel = min(self.inserted_elements, self.__size)

        min_tab = self[self.__size - iel]
        max_tab = self[-1]

        assert not np.isnan(min_tab)
        assert not np.isnan(max_tab)

        if value < min_tab:
            return -99
        if value > max_tab:
            return -99

        if self.__last_search_idx < 0:
            # Dichotomic search
            ka = 0
            kb = self.__size - 2
            found = False

            # La condition 'ka <= kb' permet d'éviter de faire
            # une boucle infinie si 'value' n'existe pas dans le tableau.
            while not found and ka <= kb:
                km = (ka + kb) // 2
                if self[km] <= value and value < self[km + 1]:
                    found = True
                else:
                    if value > self[km]:
                        ka = km + 1
                    else:
                        kb = km - 1

            # Affichage du résultat
            if not found:
                return -99

        else:
            km = self.__last_search_idx
            while np.isnan(self[km]):
                km += 1
            lxm = self[km]
            if km == self.__size - 2 and value >= lxm:
                search = False
            elif value > lxm or np.isnan(lxm):
                way = 1
                search = True
            elif value < lxm:
                way = -1
                search = True
            else:
                search = False

            xm = lxm
            count = 0
            while search and (way * value > way * xm or np.isnan(xm)):
                km += way
                xm = self[km]
                if km == 0:
                    search = False
                count += 1
                assert count < self.__size + 1
            if search and way == 1:
                km -= 1

        if km < 0:
            raise AssertionError("km < 0: %i" % km)

        if km > self.__size - 2:
            raise AssertionError("km > size-2: km=%i, size=%i" % (km, self.__size))

        self.__last_search_idx = km

        return km

    @property
    def inserted_elements(self) -> int:
        return self.__nb_inserted_element

    def append(self, val):
        """Appends an element in the buffer

        Args:
            val: The element to be inserted

        """
        self.__nb_inserted_element += 1
        self.__buffer[self.__offset] = val
        self.__offset = (self.__offset + 1) % self.__size

    def __len__(self):
        return self.__size

    def __getitem__(self, idx: int):
        """Returns the content of the buffer in chronological order

        Returns:
            An iterator over the elements

        Examples:
            >>> a = CircularBuffer(size=5, dtype=np.int64)
            >>> a.append(1)
            >>> a.append(2)
            >>> a[-2:]
            [1, 2]
            >>> a[2]
            0

        """
        if isinstance(idx, slice):
            # Get the start, stop, and step from the slice
            return [self[ii] for ii in range(*idx.indices(len(self)))]
        elif isinstance(idx, int):
            return self.__buffer[
                (idx + self.__offset) % self.__size
            ]  # Get the data from elsewhere
        else:
            raise TypeError("Invalid argument type.")

    def __iter__(self):
        """Returns the content of the buffer in chronological order

        Returns:
            An iterator over the elements

        Examples:
            >>> a = CircularBuffer(size=5, dtype=np.int64)
            >>> a.append(1)
            >>> a.append(2)
            >>> for x in a:
            ...     print(x)
            0
            0
            0
            1
            2

        """
        for k in range(self.__size):
            yield self.__buffer[(k + self.__offset) % self.__size]

    def getAsArray(self) -> "array":
        """Returns the content of the buffer in chronological order

        Returns:
            A numpy array of the elements

        Examples:
            >>> a = CircularBuffer(size=5, dtype=np.int64)
            >>> a.append(1)
            >>> a.append(2)
            >>> a.getAsArray() # doctest: +ELLIPSIS
            array([0, 0, 0, 1, 2]...

        """
        return np.roll(self.__buffer, -self.__offset)
