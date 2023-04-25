import numpy as np

from ..utils import FloatArr

from ..core.Node import AComputer, AWGNOutput


__all__ = ["DSPAWGN"]


class DSPAWGN(AComputer):
    """Adds gaussian noise to the inputs
    If dtype is complex, the generated noise will be complex.

    Args:
        name: Name of the spectrum
        mean: Mean of the gaussian law (biais)
        cov: Covariance matrix of the gaussian law
        dtype: Type of the generated samples (e.g. np.float64 or np.complex128)

    """

    __slots__ = []

    def __init__(
        self,
        name: str,
        mean: FloatArr,
        cov: FloatArr,
        dtype=np.float64,
    ):
        AComputer.__init__(self, name=name)

        size = len(mean)

        self.defineInput("noiseless", shape=size, dtype=dtype)
        otp = AWGNOutput(name="noisy", snames=["n%i" % i for i in range(size)], dtype=dtype)
        otp.setInitialState(np.zeros(otp.getDataShape(), dtype=otp.getDataType()))
        self.addOutput(otp)

        otp.cov = np.eye(size)
        otp.mean = np.zeros(size)

        self.setMean(mean)
        self.setCovariance(cov)

    def setCovariance(self, cov: FloatArr):
        """Sets the covariance matrix of the gaussian distribution

        Args:
            cov: Covariance matrix

        """
        otp = self.getOutputByName("noisy")
        n = otp.getDataShape()[0]
        if cov.shape != (n, n):
            raise ValueError(cov.shape, (n, n))
        otp.cov = cov

    def setMean(self, mean: FloatArr):
        """Sets the mean vector of the gaussian distribution

        Args:
            mean: Mean vector matrix

        """
        otp = self.getOutputByName("noisy")
        n = otp.getDataShape()[0]
        if mean.shape[0] != n:
            raise ValueError(mean.shape[0], n)
        otp.mean = mean

    def getCovariance(self) -> FloatArr:
        """Returns the covariance matrix of the gaussian distribution

        Returns:
            Covariance matrix

        """
        otp = self.getOutputByName("noisy")
        return otp.cov

    def getMean(self) -> FloatArr:
        """Returns the mean vector of the gaussian distribution

        Returns:
            Mean vector matrix

        """
        otp = self.getOutputByName("noisy")
        return otp.mean

    def update(
        self,
        t1: float,
        t2: float,
        noiseless: FloatArr,
        noisy: FloatArr,
    ) -> dict:
        noisy = noiseless

        outputs = {}
        outputs["noisy"] = noisy

        return outputs
