import numpy as np

from .DSPLine import DSPLine


__all__ = ["DSPSpectrogram"]


class DSPSpectrogram(object):
    """Representation of a spectrogram

    Args:
      name
        Name of the spectrum
      samplingXStart (s)
        First date of the sample of the spectrogram
      samplingXPeriod (s)
        Time spacing of the spectrogram
      samplingYStart (Hz)
        First frequency of the sample of the spectrogram
      samplingYPeriod (Hz)
        Frequency spacing of the spectrogram
      img
        Matrix of complex samples
      default_transform
        Function to apply to the samples before plotting.
        Shall be vectorized

    """

    name_of_x_var = "Time"
    unit_of_x_var = "s"
    name_of_y_var = "Frequency"
    unit_of_y_var = "Hz"

    def __init__(
        self,
        name: str,
        samplingXStart: float = None,
        samplingXPeriod: float = None,
        samplingYStart: float = None,
        samplingYPeriod: float = None,
        img: np.array = None,
        default_transform=np.abs,
    ):
        self.name = name
        self.samplingXStart = samplingXStart
        self.samplingXPeriod = samplingXPeriod
        self.samplingYStart = samplingYStart
        self.samplingYPeriod = samplingYPeriod
        self.img = img
        self.default_transform = default_transform

    def generateXSerie(self, index: int = None) -> np.array:
        """Generates the x samples of the spectrogram

        Args:
          index
            If given, returns only the x coord at the position given by index

        Returns:
          The x coordinate(s)

        """
        n = self.img.shape[1]
        if index is None:
            index = np.arange(n)
        elif index < 0:
            index += n
        x = index * self.samplingXPeriod + self.samplingXStart
        return x

    def generateYSerie(self, index: int = None) -> np.array:
        """Generates the y samples of the spectrogram

        Args:
          index
            If given, returns only the y coord at the position given by index

        Returns:
          The y coordinate(s)

        """
        n = self.img.shape[0]
        if index is None:
            index = np.arange(n)
        elif index < 0:
            index += n
        y = index * self.samplingYPeriod + self.samplingYStart
        return y

    def plot(self, axe, **kwargs):
        """Plots a line with the following refinements :

        * a callable *transform* is applied to all samples
        * the label of the plot is the name given at instanciation

        Args:
          axe
            Matplotlib axe to draw on
          kwargs
            Plotting options. The following extra keys are allowed:
            * transform for a different transform from the one given at instanciation
            * find_peaks to search peaks
            * x_unit_mult to have a more readable unit prefix

        """
        axe.grid(True)
        transform = kwargs.pop("transform", self.default_transform)
        x_unit_mult = kwargs.pop("x_unit_mult", 1)
        x_unit_lbl = DSPLine.getUnitAbbrev(x_unit_mult)
        y_unit_mult = kwargs.pop("y_unit_mult", 1)
        y_unit_lbl = DSPLine.getUnitAbbrev(y_unit_mult)
        lbl = kwargs.pop("label", self.name)

        ret = axe.imshow(
            transform(self.img),
            aspect="auto",
            extent=(
                self.generateXSerie(0),
                self.generateXSerie(-1),
                self.generateYSerie(0),
                self.generateYSerie(-1),
            ),
            origin="lower",
            label=lbl,
            **kwargs
        )
        axe.set_xlabel(
            "%s (%s%s)"
            % (self.__class__.name_of_x_var, x_unit_lbl, self.__class__.unit_of_x_var)
        )
        axe.set_ylabel(
            "%s (%s%s)"
            % (self.__class__.name_of_x_var, x_unit_lbl, self.__class__.unit_of_x_var)
        )
        return ret
