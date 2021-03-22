from abc import ABC, abstractmethod

import numpy as np
from numpy import exp, pi, log10


class ProcessingBlock(ABC):
    """
    Every bloc should:
    - inherit from ProcessingBlock
    - implement the 'blocksim.blocs.ProcessingBlock.__update__' method

    """

    @staticmethod
    def conv_sig_to_db(s, seuil_db=-80):
        amp = (s * np.conj(s)).real
        s_lin = 10 ** (seuil_db / 10)
        n = len(s)
        res = np.zeros(n) + seuil_db
        iok = np.where(amp > s_lin)[0]
        res[iok] = 10 * log10(amp[iok])
        return res

    @staticmethod
    def conv_sig_to_pha(s):
        pha = np.angle(s)
        return pha

    @abstractmethod
    def __update__(self, inp: np.array, *args, **kwargs) -> np.array:
        pass

    def process(self, inp: np.array = np.array([]), *args, **kwargs) -> np.array:
        self.inp = inp
        out = self.__update__(inp, *args, **kwargs)
        self.out = out
        return out
