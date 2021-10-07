import numpy as np
from numpy import sqrt, sign, pi, exp
from numpy.fft import fft, ifft


def get_window(win, n: int) -> np.array:
    from scipy.signal import get_window

    w = get_window(win, n)
    nrm = np.sum(w) / n

    return w / nrm


def zadoff_chu(u, n):
    k = np.arange(n)
    return exp(-1j * pi * u * k * (k + 1) / n)


def phase_unfold(sig: np.array, eps: float = 1e-9) -> np.array:
    n = len(sig)
    pha = np.zeros(n)
    pha[0] = np.angle(sig[0])

    init_ok = False
    i = 0
    while True:
        if np.abs(sig[i]) > eps:
            init_ok = True
            pha[0 : 1 + i] = np.angle(sig[i])
            break

        i += 1
        if i == n:
            break

    if not init_ok or i == n - 1:
        return pha

    for j in range(i + 1, n):
        if np.abs(sig[j - 1]) < eps or np.abs(sig[j]) < eps:
            r = 0
        else:
            r = sig[j] / sig[j - 1]
            r /= np.abs(r)

        # Nyquist–Shannon sampling theorem garantees that |dpha| < pi
        # So we can call np.angle which will not produce any ambiguity
        dpha = np.angle(r)

        pha[j] = pha[j - 1] + dpha

    return pha


def shift(register, feedback, output):
    """GPS Shift Register

    :param list feedback: which positions to use as feedback (1 indexed)
    :param list output: which positions are output (1 indexed)
    :returns output of shift register:

    Examples:
    >>> G1 = [1,1,1,1,1,1,1,1,1,1]
    >>> out = shift(G1, [3,10], [10])
    >>> G1
    [0, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    >>> out
    1

    """

    # calculate output
    out = [register[i - 1] for i in output]
    if len(out) > 1:
        out = sum(out) % 2
    else:
        out = out[0]

    # modulo 2 add feedback
    fb = sum([register[i - 1] for i in feedback]) % 2

    # shift to the right
    for i in reversed(range(len(register[1:]))):
        register[i + 1] = register[i]

    # put feedback in position 1
    register[0] = fb

    return out


def createParallelBitstream(
    sim: "Simulation",
    number: int,
    samplingPeriod: float,
    size: int,
    names: list = None,
    seeds: list = None,
):
    from collections import OrderedDict

    from blocksim.dsp.DSPSignal import DSPSignal
    from blocksim.control.Route import Group

    if seeds is None:
        seeds = [np.random.randint(0, 198764654) for _ in range(number)]

    assert len(seeds) == number

    if names is None:
        names = ["bs%i" % i for i in range(number)]

    grp_inp = OrderedDict()
    for num in range(number):
        grp_inp["in%i" % num] = (1,)
    grp = Group(name="grp", inputs=grp_inp, snames=["g%i" % i for i in range(number)])
    sim.addComputer(grp)

    for num, name, seed in zip(range(number), names, seeds):
        bs = DSPSignal.fromBinaryRandom(
            name=name, samplingPeriod=samplingPeriod, size=size, seed=seed
        )
        sim.addComputer(bs)

        sim.connect("bs%i.setpoint" % num, "grp.in%i" % num)

    return bs.generateXSerie()
