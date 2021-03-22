import numpy as np
from matplotlib import pyplot as plt

from OFDM import logger
from OFDM.blocs.ProcessingBlock import ProcessingBlock


class SerialToParallel(ProcessingBlock):
    def __init__(self, nb_parallel):
        self.nb_parallel = nb_parallel

    def __update__(self, data: np.array) -> np.array:
        n = len(data)
        return data.reshape(n // self.nb_parallel, self.nb_parallel)


class ParallelToSerial(ProcessingBlock):
    def __init__(self, nb_parallel):
        self.nb_parallel = nb_parallel

    def __update__(self, data: np.array) -> np.array:
        n, r = data.shape
        if r != self.nb_parallel:
            raise ArgumentError("The parallel data must have %i row" % self.nb_parallel)
        return data.reshape((-1,))
